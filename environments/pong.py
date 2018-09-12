"""Pong environment
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Type

import cv2
import gym
import numpy as np

from environments.environment import Environment


class Pong(Environment):
  """Pong environment
  """

  def __init__(self):
    self.env: Type[gym.Env] = gym.make('Pong-v0')
    # We make it grayscale
    self.observation_space = gym.spaces.Box(
        0.0, 1.0, [210, 160, 1], dtype=np.float32)
    self.action_space = self.env.action_space

  def reset(self):
    return self._process_observation(self.env.reset())

  def render(self):
    self.env.render()

  def step(self, action):
    if self.env.action_space.contains(action):
      state, reward, done, info = self.env.step(action)
      return self._process_observation(state), reward, done, info
    raise ValueError("Action is invalid")

  def close(self):
    self.env.close()

  @staticmethod
  def _process_observation(observation: np.array) -> np.array:
    return np.expand_dims(
        cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) / 255.0, axis=-1)
    # return observation / 255.0
