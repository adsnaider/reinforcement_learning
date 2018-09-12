"""Mountain car environment
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Type

import gym
import numpy as np

from environments.environment import Environment


class MountainCar(Environment):
  """Pong environment
  """

  def __init__(self):
    self.env: Type[gym.Env] = gym.make('MountainCar-v0').env
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space

  def reset(self):
    return self._process_observation(self.env.reset())

  def render(self):
    self.env.render()

  def step(self, action):
    if self.env.action_space.contains(action):
      state, reward, done, info = self.env.step(action)
      return self._process_observation(state), reward, done, info
    raise ValueError("Action is invalid: ", action)

  def close(self):
    self.env.close()

  @staticmethod
  def _process_observation(observation: np.array) -> np.array:
    return observation
