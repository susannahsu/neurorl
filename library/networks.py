"""Basic neural networks."""
import functools

from typing import Optional, Sequence, Callable, Tuple, Dict
import chex
import haiku as hk
import jax
import jax.numpy as jnp

from acme import types
from acme.wrappers import observation_action_reward

Array = jax.Array
Image = jax.Array
Action = jax.Array
Reward = jax.Array
Task = jax.Array

@chex.dataclass(frozen=True)
class TorsoOutput:
  image: jnp.ndarray
  action: jnp.ndarray
  reward: jnp.ndarray
  task: Optional[jnp.ndarray] = None

def struct_output(image: Image, task: Task, action: Action, reward: Reward):
  return TorsoOutput(
    image=image,
    action=action,
    reward=reward,
    **(dict(task=task) if task is not None else {})
  )

def concat(image: Image, task: Task, action: Action, reward: Reward):
  pieces = (image, action, reward)
  if task is not None:
    pieces += (task, )
  return jnp.concatenate(pieces, axis=-1)

class AtariVisionTorso(hk.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self, flatten=True, conv_dim=16, out_dim=0):
    super().__init__(name='atari_torso')
    layers = [
        hk.Conv2D(32, [8, 8], 4),
        jax.nn.relu,
        hk.Conv2D(64, [4, 4], 2),
        jax.nn.relu,
        hk.Conv2D(64, [3, 3], 1),
        jax.nn.relu,
    ]
    if conv_dim:
      layers.append(hk.Conv2D(conv_dim, [1, 1], 1))
    self._network = hk.Sequential(layers)

    self.flatten = flatten
    if out_dim:
      self.out_net = hk.Linear(out_dim)
    else:
      self.out_net = lambda x: x

  def __call__(self, inputs: Image) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)

    outputs = self._network(inputs)
    if not self.flatten:
      return outputs

    if batched_inputs:
      flat = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      flat = jnp.reshape(outputs, [-1])  # [D]

    return self.out_net(flat)

class BabyAIVisionTorso(hk.Module):
  """Convolutional stack used in BabyAI codebase."""

  def __init__(self, flatten=True, conv_dim=16, out_dim=0):
    super().__init__(name='babyai_torso')
    layers = [
        hk.Conv2D(128, [8, 8], stride=8),
        hk.Conv2D(128, [3, 3], stride=1),
        jax.nn.relu,
        hk.Conv2D(128, [3, 3], stride=1),
        jax.nn.relu,
    ]
    if conv_dim > 0:
      layers.append(hk.Conv2D(conv_dim, [1, 1], stride=1))
    self._network = hk.Sequential(layers)

    self.flatten = flatten or out_dim > 0
    if out_dim:
      self.out_net = hk.Linear(out_dim)
    else:
      self.out_net = lambda x: x

  def __call__(self, inputs: Image) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)

    outputs = self._network(inputs)
    if not self.flatten:
      return outputs

    if batched_inputs:
      flat = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      flat = jnp.reshape(outputs, [-1])  # [D]

    return self.out_net(flat)


class LanguageEncoder(hk.Module):
  """Module that embed words and then runs them through GRU. The Token`0` is treated as padding and masked out."""

  def __init__(self,
               vocab_size: int,
               word_dim: int,
               sentence_dim: Optional[int] = None,
               mask_words: bool = False,
               compress: str = 'last'):
    super(LanguageEncoder, self).__init__()
    self.vocab_size = vocab_size
    self.word_dim = word_dim
    self.sentence_dim = sentence_dim or word_dim
    self.compress = compress
    self.mask_words = mask_words
    self.embedder = hk.Embed(
        vocab_size=vocab_size,
        embed_dim=word_dim)
    self.language_model = hk.GRU(sentence_dim)

  def __call__(self, x: jnp.ndarray):
    """Embed words, then run through GRU.
    
    Args:
        x (TYPE): N
    
    Returns:
        TYPE: Description
    """
    # -----------------------
    # embed words + mask
    # -----------------------
    words = self.embedder(x)  # N x D
    if self.mask_words:
      mask = (x > 0).astype(words.dtype)
      words = words*jnp.expand_dims(mask, axis=-1)

    # -----------------------
    # pass through GRU
    # -----------------------
    initial = self.language_model.initial_state(None)
    sentence, _ = hk.static_unroll(self.language_model,
                                   words, initial)

    if self.compress == "last":
      task = sentence[-1]  # embedding at end
    elif self.compress == "sum":
      task = sentence.sum(0)
    else:
      raise NotImplementedError(self.compress)

    return task

class OarTorso(hk.Module):

  def __init__(self,
               num_actions: int,
               vision_torso: hk.Module,
               task_encoder: Optional[hk.Module] = None,
               flatten_image: bool = True,
               output_fn: Callable[[Image, Task, Action, Reward], Array] = concat,
               w_init: Optional[hk.initializers.Initializer] = None,
               name='torso'):
    super().__init__(name=name)
    if task_encoder is None:
      task_encoder = lambda x: x

    self._num_actions = num_actions
    self._vision_torso = vision_torso
    self._output_fn = output_fn
    self._flatten_image = flatten_image
    self._task_encoder = task_encoder
    self._w_init = w_init

  def __call__(self, inputs: observation_action_reward.OAR):
    if len(inputs.observation['image'].shape) == 3:
      observation_fn = self.unbatched
    elif len(inputs.observation['image'].shape) == 4:
      observation_fn = jax.vmap(self.unbatched)
    else:
      raise NotImplementedError

    return observation_fn(inputs)

  def unbatched(self, inputs: observation_action_reward.OAR):
    """_no_ batch [B] dimension."""
    # compute task encoding

    task = self._task_encoder(inputs.observation)

    # get action one-hot
    action = jax.nn.one_hot(
        inputs.action, num_classes=self._num_actions)

    # compute image encoding
    inputs = jax.tree_map(lambda x: x.astype(jnp.float32), inputs)
    image = self._vision_torso(inputs.observation['image']/255.0)
    if self._flatten_image:
      image = jnp.reshape(image, (-1))

    # Map rewards -> [-1, 1].
    # reward = jnp.tanh(inputs.reward)
    reward = jnp.expand_dims(inputs.reward, axis=-1)

    return self._output_fn(
      image=image,
      task=task,
      action=action,
      reward=reward)

class DummyRNN(hk.RNNCore):
  def __call__(self, inputs: jax.Array, prev_state: hk.LSTMState
               ) -> Tuple[jax.Array, hk.LSTMState]:
    return inputs, prev_state

  def initial_state(self, batch_size: Optional[int]) -> hk.LSTMState:
    return jnp.zeros((batch_size, 1)) if batch_size else jnp.zeros((1))
