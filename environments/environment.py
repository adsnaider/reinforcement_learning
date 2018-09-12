"""Environment abstract class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
from typing import Tuple

# Used on definition of observation_space
import numpy as np  # pylint: disable=unused-import
import gym  # pylint: disable=unused-import


class Environment:
  """Environment abstract class.

    This is a simple abstract definition of the environment class.
    It provides the definition of the essential methods that need to
    be implemented by an environment to be used.
  """

  observation_space: gym.Space
  action_space: gym.Space

  @abstractmethod
  def reset(self) -> np.array:
    """Resets the environment to it's starting defintion.

    Returns:
      The initial game state.
    """
    pass

  @abstractmethod
  def render(self) -> None:
    """Renders the current state of the environment.

    When called, it will render the current state of the environment into
    the screen.
    """
    pass

  @abstractmethod
  def step(self, action: int) -> Tuple[np.array, int, bool, str]:
    """Runs the next step.

    When called, it will run all the logic step into the environment based on
    the action.

    Args:
      action: Agent's action.
    Returns:
      (observation, reward, done, info)
    """
    pass

  @abstractmethod
  def close(self) -> None:
    """Closes the environment. Stops the rendering
    """
    pass
