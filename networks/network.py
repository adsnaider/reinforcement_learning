"""Neural Network abstract class for reinforcement learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Type
from abc import abstractmethod

import tensorflow as tf


class Network:
  """Neural Network abstract class.

  This class defines the minimum interface needed for an object to be
  considered a Network.
  """

  outputs: tf.Tensor
  name: str

  @abstractmethod
  def __init__(self, inputs: tf.Tensor, name: str, trainable: bool) -> None:
    pass

  def make_copy_op(self, other: Type['Network']) -> tf.Operation:
    """Creates an operation that can be used to copy the variables from one network to another.

    Args:
      other: Network to get variables from.
    Returns:
      An operation that when run will copy the vars from the other network
    """
    self_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    other_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=other.name)

    if len(self_vars) != len(other_vars):
      raise ValueError("Invalid network")

    copy_ops = []
    for i, _ in enumerate(self_vars):
      copy_ops.append(tf.assign(self_vars[i], other_vars[i]))

    return tf.group(copy_ops)
