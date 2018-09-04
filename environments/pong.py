"""Pong environment
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import gym
import numpy as np

from environments.environment import Environment


class Pong(Environment):
  """Pong environment
  """

  def __init__(self):
    self.env = gym.make('Pong-v4')
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space

  def reset(self):
    return self._process_observation(self.env.reset())

  def render(self):
    self.env.render()

  def step(self, action):
    if self.env.action_space.contains(action):
      return self._process_observation(self.env.step(action))
    raise ValueError("Action is invalid")

  def close(self):
    self.env.close()

  @staticmethod
  def _process_observation(observation: np.array) -> np.array:
    return cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) / 255.0
