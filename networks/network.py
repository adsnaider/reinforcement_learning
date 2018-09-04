"""Neural Network abstract class for reinforcement learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import tensorflow as tf


class Network:
  """Neural Network abstract class.

  This class defines the minimum interface needed for an object to be
  considered a Network.
  """

  outputs = None  # type: tf.Tensor

  @abstractmethod
  def __init__(self, inputs: tf.Tensor) -> None:
    pass

  @abstractmethod
  def create_graph(self) -> None:
    """Creates the Network Graph.

    This method creates the TensorFlow computation graph.
    It will populate the output nodes of the graph.
    """
    pass

  @abstractmethod
  def from_copy(self, other: Network):
    """Create a copy operation from another network.

    This method creates an operation that, when run, will copy all the
    trainable variables from "other" into self.

    Args:
      other: Network object of the same class from which to copy the variables.

    Returns:
      The copy operation
    """
    pass
