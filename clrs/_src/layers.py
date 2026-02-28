import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp


class Linear(hk.Module):
  """Linear module."""

  def __init__(
      self,
      output_size: int,
      with_bias: bool = True,
      w_init: hk.initializers.Initializer | None = None,
      b_init: hk.initializers.Initializer | None = None,
      name: str | None = None,
      num_tasks: int = 1,
      encoder_decoder_rank: int = 0
  ):
    """Constructs the Linear module.

    Args:
      output_size: Output dimensionality.
      with_bias: Whether to add a bias to the output.
      w_init: Optional initializer for weights. By default, uses random values
        from truncated normal, with stddev ``1 / sqrt(fan_in)``. See
        https://arxiv.org/abs/1502.03167v3.
      b_init: Optional initializer for bias. By default, zero.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros
    
    self.num_tasks = num_tasks
    self.encoder_decoder_rank = encoder_decoder_rank

  def __call__(
      self,
      inputs: jax.Array,
      *,
      precision = None,
      algorithm_index: int=None
  ) -> jax.Array:
    """Computes a linear transform of the input."""
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
      stddev = 1. / np.sqrt(self.input_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

    if self.encoder_decoder_rank > 0:
      assert algorithm_index is not None

      A = hk.get_parameter(
          "A",
          [self.num_tasks, input_size, self.encoder_decoder_rank],
          dtype,
          init=hk.initializers.RandomNormal(0.02),
      )

      B = hk.get_parameter(
          "B",
          [self.num_tasks, self.encoder_decoder_rank, output_size],
          dtype,
          init=hk.initializers.RandomNormal(0.02),
      )

      delta = jnp.matmul(A[algorithm_index], B[algorithm_index])
      w = w + delta

    out = jnp.dot(inputs, w, precision=precision)

    if self.with_bias:
      b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    return out