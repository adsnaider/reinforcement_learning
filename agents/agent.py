from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Agent(object):
  """Agent abstract class

  This is a simple abstract definition for the Agents.
  It provides the essential methods required for something
  to be an agent.
  """

  def train(self, steps):
    """Runs the training loop.

    This method runs the training loop "steps" times.

    Args:
      steps: Number of steps to run.
    """
    raise NotImplementedError

  def play(self, steps):
    """Play in the environment.

    Args:
      steps: For how many steps to play.
    """
    raise NotImplementedError
