from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Network(object):
  """Neural Network abstract class.

  This class defines the minimum interface needed for an object to be
  considered a Network.
  """

  def create_graph(self):
    """Creates the Network Graph.

    This method creates the TensorFlow computation graph.
    It will populate the output nodes of the graph.
    """
    raise NotImplementedError

  def from_copy(self, other):
    """Create a copy operation from another network.

    This method creates an operation that, when run, will copy all the
    trainable variables from "other" into self.

    Args:
      other: Network object of the same class from which to copy the variables.
    """
    raise NotImplementedError
