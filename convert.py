import pickle
import os
import numpy as np
from absl import app
from absl import flags
import clrs
import jax

FLAGS = flags.FLAGS

flags.DEFINE_string('input_checkpoint', None, 'Path to input checkpoint')
flags.DEFINE_string('output_checkpoint', None, 'Path to output checkpoint')
flags.DEFINE_string('algorithm', None, 'Target algorithm name')
flags.DEFINE_integer('num_lora_slots', 7, 'Number of LoRA slots')
flags.DEFINE_integer('lora_rank', 2, 'LoRA rank')
flags.DEFINE_integer('hidden_dim', 128, 'Hidden dimension')
flags.DEFINE_integer('batch_size', 32, 'Batch size for initialization')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'Hint mode to match training')

flags.mark_flag_as_required('input_checkpoint')
flags.mark_flag_as_required('output_checkpoint')
flags.mark_flag_as_required('algorithm')


def main(argv):
    print(f'Loading checkpoint: {FLAGS.input_checkpoint}')
    with open(FLAGS.input_checkpoint, 'rb') as f:
        checkpoint = pickle.load(f)
    
    old_params = checkpoint['params']
    
    print(f'Building model for {FLAGS.algorithm}')
    sampler, spec = clrs.build_sampler(
        FLAGS.algorithm,
        seed=42,
        num_samples=FLAGS.batch_size,
        length=16,
    )
    
    feedback = sampler.next(FLAGS.batch_size)
    
    processor_factory = clrs.get_processor_factory(
        'triplet_gmpnn',
        use_ln=True,
        nb_triplet_fts=8
    )
    
    if FLAGS.hint_mode == 'encoded_decoded':
        encode_hints = True
        decode_hints = True
    elif FLAGS.hint_mode == 'decoded_only':
        encode_hints = False
        decode_hints = True
    elif FLAGS.hint_mode == 'none':
        encode_hints = False
        decode_hints = False
    
    model = clrs.models.BaselineModel(
        spec=[spec],
        dummy_trajectory=[feedback],
        processor_factory=processor_factory,
        hidden_dim=FLAGS.hidden_dim,
        encode_hints=encode_hints,
        decode_hints=decode_hints,
        shared_encoders_decoders=True,
        encoder_decoder_rank=FLAGS.lora_rank,
        num_lora_slots=FLAGS.num_lora_slots,
    )
    
    print('Initializing model for target algorithm')
    model.init([feedback.features], 42)
    
    new_params = model.params

    print(f'\nAll param keys BEFORE copying ({len(new_params)} total):')
    for k in sorted(new_params.keys()):
        print(f'  {k}')
    
    print('Copying weights from checkpoint where available')
    copied = []
    initialized = []
    
    for module_name in new_params.keys():
        if module_name in old_params:
            new_params[module_name] = old_params[module_name]
            copied.append(module_name)
        else:
            initialized.append(module_name)
    
    new_checkpoint = {'params': new_params}
    
    print(f'\nCopied {len(copied)} modules from checkpoint')
    print(f'Randomly initialized {len(initialized)} missing modules:')
    for m in initialized:
        print(f'  {m}')
    
    print(f'\nFinal checkpoint keys ({len(new_params)} total):')
    for k in sorted(new_params.keys()):
        if '_construct_encoders_decoders' in k:
            print(f'  {k}: {list(new_params[k].keys())}')
    
    print(f'\nSaving to {FLAGS.output_checkpoint}')
    os.makedirs(os.path.dirname(FLAGS.output_checkpoint) or '.', exist_ok=True)
    with open(FLAGS.output_checkpoint, 'wb') as f:
        pickle.dump(new_checkpoint, f)
    
    print('Done')


if __name__ == '__main__':
    app.run(main)