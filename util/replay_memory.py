"""Utility to store the reinforcement learning data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Tuple
from absl import logging

import numpy as np


class ReplayMemory:
  """Stores the latest (state, action, reward, done) tuples.
  """

  def __init__(self, memory_size: int, state_shape: List[int]) -> None:
    self.memory_size = memory_size
    self.states = np.zeros(dtype=np.float32, shape=[memory_size] + state_shape)
    self.actions = np.zeros(dtype=np.int32, shape=[memory_size])
    self.rewards = np.zeros(dtype=np.int32, shape=[memory_size])
    self.done = np.zeros(dtype=np.bool, shape=[memory_size])

    self.index = 0
    self.size = 0

  def append(self, state: np.array, action: int, reward: int,
             done: bool) -> None:
    """Appends a new element into the memory.

    Args:
      state: A numpy array representing the environment state.
      action: The agent's action taken at the state.
      reward: The reward received after taking the action.
      done: Whether the following response was terminal.
    """
    logging.debug("Adding new element")

    self.states[self.index, ...] = state
    self.actions[self.index, ...] = action
    self.rewards[self.index, ...] = reward
    self.done[self.index, ...] = done

    self.size = min(self.size + 1, self.memory_size)
    self.index = (self.index + 1) % self.memory_size

  def full(self) -> bool:
    """Returns whether the memory is full.
    """
    return self.size == self.memory_size

  def sample(self, size: int
            ) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """Gets sample of states, actions, rewards, done, poststates

    Returns:
      Tuple containing the sample.
    """

    if not self.full():
      raise ValueError("Memroy must be filled before we can sample")

    if size >= self.memory_size:
      raise ValueError("Sample size is too large")

    if self.index == 0:
      choices = np.arange(0, self.size - 1)
    else:
      choices = np.concatenate(
          [np.arange(self.index - 1),
           np.arange(self.index, self.size)])
    # We must never select the last element since it doesn't have a poststate yet.
    selection = np.random.choice(choices, replace=False, size=size)

    prestates = self.states[selection]
    poststates = self.states[(selection + 1) % self.size]
    actions = self.actions[selection]
    rewards = self.rewards[selection]
    done = self.done[selection]

    return prestates, poststates, actions, rewards, done
