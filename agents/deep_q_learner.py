"""DeepQ Learner.

Implements the Deep Q Learning algorithm into a reinforcement learning agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Optional, Type

import numpy as np
import tensorflow as tf

from absl import logging

from agents.agent import Agent
from environments.environment import Environment
from networks.network import Network
from util.replay_memory import ReplayMemory


class DeepQLearner(Agent):
  """Deep Q Learning agent
  """

  # TODO(adam): Change options to proto DeepQ type.
  def __init__(self, options: Any, env: Environment,
               network_def: Type[Network]) -> None:
    """Creates a DeepQLearner agent.

    Args:
      options: Protobuf configuration for the Agent.
      env: Environment class. Must inherit from environments.Environment
      netword_def: A class that inherits from networks.Network.
                   __init__ must take 1 argument, the inputs.
    """
    logging.info("Initializing DeepQLearner")
    self.env = env

    if self.env.observation_space is None:
      raise ValueError("Observation space can't be null")
    if self.env.action_space is None:
      raise ValueError("Action space can't be null")

    # This should be the prestate during training
    self.prediction_inputs = tf.placeholder(
        shape=[None] + self.env.observation_space.size, dtype=tf.float32)

    # This should be the poststate during training
    self.target_inputs = tf.placeholder(
        shape=[None] + self.env.observation_space.size, dtype=tf.float32)

    self.prediction_network = network_def(self.prediction_inputs)
    self.target_network = network_def(self.target_inputs)
    self.copy_op = self.target_network.from_copy(self.prediction_network)

    self.options = options

    # Training placholders
    self.action_input = tf.placeholder(dtype=tf.int32, shape=[None])
    self.reward_input = tf.placeholder(dtype=tf.int32, shape=[None])
    self.done_input = tf.placeholder(dtype=tf.bool, shape=[None])

    self.optimizer = tf.train.AdamOptimizer(self.options.learning_rate)
    self.global_step = None  # type: tf.Tensor
    self.train_op = None  # type: tf.python.framework.ops.Operation

    self.memory = ReplayMemory(self.options.memory_size,
                               self.env.observation_space.shape)

  def train(self, steps: int) -> None:
    self._create_training_graph()

    hooks = []
    hooks.append(tf.train.StopAtStepHook(last_step=steps))
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=self.options.checkpoint_dir, hooks=hooks) as sess:
      state = self.env.reset()
      while not sess.should_stop():
        step = sess.run(self.global_step)
        if step % self.options.target_reset == 0:
          sess.run(self.copy_op)

        action = self._get_training_action(sess, state)
        poststate, reward, done, info = self.env.step(action)
        logging.debug(info)

        self.memory.append(state, action, reward, done)
        state = poststate

        if self.memory.full():
          prestates, poststates, actions, rewards, done = self.memory.sample(
              self.options.batch_size)

          sess.run(
              self.train_op,
              feed_dict={
                  self.prediction_inputs: prestates,
                  self.target_inputs: poststates,
                  self.action_input: actions,
                  self.reward_input: rewards,
                  self.done_input: done
              })

  def play(self, steps: Optional[int]) -> None:
    state = self.env.reset()
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess,
                    tf.train.latest_checkpoint(self.options.checkpoint_dir))
      done = False
      while not done:
        action = self._get_best_action(sess, state)
        state, _, done, info = self.env.step(action)
        logging.debug(info)
        self.env.render()
        if steps is not None:
          steps -= 1
          if steps <= 0:
            break

  def _get_best_action(self, sess: tf.Session, state: np.array) -> int:
    output = sess.run(
        [self.prediction_network.output],
        feed_dict={self.prediction_inputs: state})
    return np.argmax(output, axis=0)[0]

  def _get_training_action(self, sess: tf.Session, state: np.array) -> int:
    if np.random.uniform(0.0, 1.0) < self.options.epsilon:
      return self.env.action_space.sample()
    return self._get_training_action(sess, state)

  def _create_training_graph(self) -> None:
    self.global_step = tf.train.get_or_create_global_step()
    prediction_q = self.prediction_network.outputs

    indices_actions = tf.stack(
        tf.range(tf.size(self.action_input)), self.action_input, axis=1)

    actual_q = tf.gather_nd(prediction_q, indices_actions)

    # If that was the end of the episode, then the Q value should just be the reward.
    expected_q = self.reward_input + tf.multiply(
        self.options.gamma * tf.reduce_max(self.target_network.outputs, axis=1),
        1 - self.done_input)

    loss = tf.losses.mean_squared_error(actual_q, expected_q)
    self.train_op = self.optimizer.minimize(loss, global_step=self.global_step)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("max_Q", tf.reduce_max(prediction_q))
