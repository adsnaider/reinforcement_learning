"""Agent abstract class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional


class Agent:
  """Agent abstract class

  This is a simple abstract definition for the Agents.
  It provides the essential methods required for something
  to be an agent.
  """

  def train(self, steps: int, render: bool = True) -> None:
    """Runs the training loop.

    This method runs the training loop "steps" times.

    Args:
      steps: Number of steps to run.
    """
    raise NotImplementedError

  def play(self,
           steps: Optional[int],
           render: bool = True,
           frame_rate: Optional[int] = None) -> None:
    """Play in the environment.

    Args:
      steps: For how many steps to play. If None, then play until done.
    """
    raise NotImplementedError
