import functools
import os
from typing import Any, Dict, List

from absl import app
from absl import flags
from absl import logging
import clrs
import jax
import numpy as np
import tensorflow as tf


flags.DEFINE_list('algorithms', ['bfs'], 'Which algorithms to evaluate.')
flags.DEFINE_string('checkpoint_path', 'checkpoints',
                    'Path to pretrained checkpoint dir.')
flags.DEFINE_string('checkpoint_name', None,
                    'Name of pretrained checkpoint. If not provided, evaluates randomly initialized model.')

flags.DEFINE_list('train_lengths', ['4', '7', '11', '13', '16'],
                  'Training sizes used during training.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_integer('batch_size', 32, 'Batch size used for evaluation.')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')

flags.DEFINE_enum('processor_type', 'triplet_gmpnn',
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
                  'Processor type to use as the network P.')

flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')

flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used?')
flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initialiser to use for the encoders.')

flags.DEFINE_boolean('shared_encoders_decoders', False,
                     'Whether to use a shared set of encoders and decoders for all algorithms')
flags.DEFINE_integer('encoder_decoder_rank', 0,
                     'If shared encoders and decoders are used what rank matrix to use for specific algorithm specialization')
flags.DEFINE_integer('num_lora_slots', 7,
                     'Total number of task slots in LoRA adapters')

flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')

FLAGS = flags.FLAGS


PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march']


def unpack(v):
  try:
    return v.item()
  except (AttributeError, ValueError):
    return v


def _iterate_sampler(sampler, batch_size):
  while True:
    yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path, download=True):
  """Download CLRS30 dataset if needed."""
  dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
  if not download and os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    return dataset_folder
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)

  import requests
  import shutil
  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
  os.makedirs(dataset_folder, exist_ok=True)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def make_sampler(length, rng, algorithm, split, batch_size, multiplier,
                 randomize_pos, enforce_pred_as_input, enforce_permutations,
                 sampler_kwargs):
  if length < 0:
    dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
    sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                     algorithm=algorithm,
                                                     batch_size=batch_size,
                                                     split=split)
    sampler = sampler.as_numpy_iterator()
  else:
    num_samples = int(clrs.CLRS30[split]['num_samples'] * multiplier)
    sampler, spec = clrs.build_sampler(
        algorithm,
        seed=rng.randint(2**32),
        num_samples=num_samples,
        length=length,
        **sampler_kwargs,
    )
    sampler = _iterate_sampler(sampler, batch_size)

  if randomize_pos:
    sampler = clrs.process_random_pos(sampler, rng)
  if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
    spec, sampler = clrs.process_pred_as_input(spec, sampler)
  spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
  return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples

  def cycle_samplers():
    while True:
      for s in ss:
        yield next(s)
  return cycle_samplers(), tot_samples, spec


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
  processed_samples = 0
  preds = []
  outputs = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


def create_samplers(rng, train_lengths, algorithms):
  val_samplers = []
  val_sample_counts = []
  test_samplers = []
  test_sample_counts = []
  spec_list = []

  for algorithm in algorithms:
    current_train_lengths = train_lengths

    with tf.device('/cpu:0'):
      if algorithm in ['naive_string_matcher', 'kmp_matcher']:
        max_length = max(current_train_lengths)
        if max_length > 0:
          max_length = (max_length * 5) // 4
        current_train_lengths = [max_length]

      logging.info('Creating samplers for algo %s', algorithm)

      p = tuple([0.1 + 0.1 * i for i in range(9)])
      if p and algorithm in ['articulation_points', 'bridges',
                             'mst_kruskal', 'bipartite_matching']:
        p = tuple(np.array(p) / 2)
      length_needle = FLAGS.length_needle
      sampler_kwargs = dict(p=p, length_needle=length_needle)
      if length_needle == 0:
        sampler_kwargs.pop('length_needle')

      common_sampler_args = dict(
          algorithm=algorithm,
          rng=rng,
          enforce_pred_as_input=FLAGS.enforce_pred_as_input,
          enforce_permutations=FLAGS.enforce_permutations,
      )

      mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
      val_args = dict(
          sizes=[np.amax(current_train_lengths)],
          split='val',
          batch_size=FLAGS.batch_size,
          multiplier=2 * mult,
          randomize_pos=FLAGS.random_pos,
          sampler_kwargs=sampler_kwargs,
          **common_sampler_args,
      )
      val_sampler, val_samples, _ = make_multi_sampler(**val_args)

      test_args = dict(
          sizes=[-1],
          split='test',
          batch_size=FLAGS.batch_size,
          multiplier=2 * mult,
          randomize_pos=False,
          sampler_kwargs={},
          **common_sampler_args,
      )
      test_sampler, test_samples, spec = make_multi_sampler(**test_args)

    spec_list.append(spec)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)
    test_samplers.append(test_sampler)
    test_sample_counts.append(test_samples)

  return (val_samplers, val_sample_counts,
          test_samplers, test_sample_counts,
          spec_list)


def main(unused_argv):
  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  train_lengths = [int(x) for x in FLAGS.train_lengths]

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32))

  (
      val_samplers,
      val_sample_counts,
      test_samplers,
      test_sample_counts,
      spec_list,
  ) = create_samplers(
      rng=rng,
      train_lengths=train_lengths,
      algorithms=FLAGS.algorithms,
  )

  processor_factory = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads,
  )
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=0.0,
      grad_clip_max_norm=0.0,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=False,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=0.0,
      hint_repred_mode='soft',
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
      shared_encoders_decoders=FLAGS.shared_encoders_decoders,
      encoder_decoder_rank=FLAGS.encoder_decoder_rank,
      num_lora_slots=FLAGS.num_lora_slots,
  )

  eval_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(t) for t in val_samplers],
      **model_params
  )

  # Init with val sampler features
  all_features = [next(t).features for t in val_samplers]
  eval_model.init(all_features, FLAGS.seed + 1)

  # Load checkpoint if provided, otherwise evaluate random init
  if FLAGS.checkpoint_name:
    logging.info('Loading checkpoint: %s', FLAGS.checkpoint_name)
    eval_model.restore_model(FLAGS.checkpoint_name, only_load_processor=False)
  else:
    logging.info('No checkpoint provided, evaluating randomly initialized model.')

  for algo_idx in range(len(FLAGS.algorithms)):
    common_extras = {'algorithm': FLAGS.algorithms[algo_idx]}

    # Validation info.
    new_rng_key, rng_key = jax.random.split(rng_key)
    val_stats = collect_and_eval(
        val_samplers[algo_idx],
        functools.partial(eval_model.predict, algorithm_index=algo_idx),
        val_sample_counts[algo_idx],
        new_rng_key,
        extras=common_extras)
    logging.info('(val) algo %s : %s', FLAGS.algorithms[algo_idx], val_stats)

    # Test info.
    new_rng_key, rng_key = jax.random.split(rng_key)
    test_stats = collect_and_eval(
        test_samplers[algo_idx],
        functools.partial(eval_model.predict, algorithm_index=algo_idx),
        test_sample_counts[algo_idx],
        new_rng_key,
        extras=common_extras)
    logging.info('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)

  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)