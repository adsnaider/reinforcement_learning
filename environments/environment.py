from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Environment(object):
  """Environment abstract class.

    This is a simple abstract definition of the environment class.
    It provides the definition of the essential methods that need to
    be implemented by an environment to be used.
  """

  def reset(self):
    """Resets the environment to it's starting defintion.
    """
    raise NotImplementedError

  def render(self):
    """Renders the current state of the environment.

    When called, it will render the current state of the environment into
    the screen.
    """
    raise NotImplementedError

  def step(self, action):
    """Runs the next step.

    When called, it will run all the logic step into the environment based on
    the action.

    Args:
      action: Agent's action.
    """
    raise NotImplementedError
