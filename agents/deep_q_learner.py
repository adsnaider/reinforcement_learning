"""DeepQ Learner.

Implements the Deep Q Learning algorithm into a reinforcement learning agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Optional, Callable, Type

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

  def __init__(self,
               env: Environment,
               optimizer: tf.train.Optimizer,
               memory_size: int,
               network_def: Callable[..., Type[Network]],
               observation_time: int,
               log_step: int = 500,
               batch_size: int = 32,
               checkpoint_dir: str = "checkpoints",
               gamma: float = 0.99,
               epsilon_fn: Callable[[int], float] = lambda x: 0.05) -> None:
    """Creates a DeepQLearner agent.

    Args:
      options: Protobuf configuration for the Agent.
      env: Environment class. Must inherit from environments.Environment
      netword_def: A class that inherits from networks.Network.
                   __init__ must take 1 argument, the inputs.
    """
    logging.info("Initializing DeepQLearner")

    self.env = env
    self.optimizer = optimizer
    self.observation_time = observation_time
    self.log_step = log_step
    self.batch_size = batch_size
    self.checkpoint_dir = checkpoint_dir
    self.gamma = gamma
    self.epsilon_fn = epsilon_fn

    # This should be the prestate during training
    self.prediction_inputs = tf.placeholder(
        shape=[None] + list(self.env.observation_space.shape),
        dtype=tf.float32,
        name="prediction_input")  # type: tf.Tensor

    # This should be the poststate during training
    self.target_inputs = tf.placeholder(
        shape=[None] + list(self.env.observation_space.shape),
        dtype=tf.float32,
        name="target_input")  # type: tf.Tensor

    self.prediction_network: Type[Network] = network_def(
        self.prediction_inputs, 'predictor', True)
    self.target_network: Type[Network] = network_def(self.target_inputs,
                                                     'target', True)
    self.target_copy = self.target_network.make_copy_op(self.prediction_network)

    # Training placholders
    self.action_input: tf.Tensor = tf.placeholder(
        dtype=tf.int32, shape=[None], name="action_input")
    self.reward_input: tf.Tensor = tf.placeholder(
        dtype=tf.int32, shape=[None], name="reward_input")
    self.done_input: tf.Tensor = tf.placeholder(
        dtype=tf.bool, shape=[None], name="done_input")

    self.global_step = tf.train.get_or_create_global_step()
    self.loss: tf.Tensor
    self.train_op: tf.Operation

    self.memory = ReplayMemory(memory_size,
                               list(self.env.observation_space.shape))

  def train(self, steps: int, render: bool = True) -> None:
    self._create_training_graph()

    hooks = []
    hooks.append(tf.train.StopAtStepHook(last_step=steps))
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=self.checkpoint_dir, hooks=hooks) as sess:
      state = self.env.reset()
      while not sess.should_stop():
        step = sess._tf_sess().run(self.global_step)
        epsilon = self.epsilon_fn(step)
        action = self._get_training_action(sess._tf_sess(),
                                           np.expand_dims(state, 0), epsilon)
        poststate, reward, done, info = self.env.step(np.squeeze(action))
        logging.debug(info)
        if render:
          self.env.render()

        self.memory.append(state, action, reward, done)
        if done:
          state = self.env.reset()
        else:
          state = poststate

        if self.memory.full():
          if step % self.observation_time == 0:
            sess._tf_sess().run(self.target_copy)
          prestates, poststates, actions, rewards, dones = self.memory.sample(
              self.batch_size)

          _, loss = sess.run(
              [self.train_op, self.loss],
              feed_dict={
                  self.prediction_inputs: prestates,
                  self.target_inputs: poststates,
                  self.action_input: actions,
                  self.reward_input: rewards,
                  self.done_input: dones
              })

          if step % self.log_step == 0:
            logging.info("Step: {}\t Loss: {:.3E}\t Epsilon: {:.2E}".format(
                step, loss, epsilon))
    self.env.close()

  def play(self, steps: Optional[int], render: bool = True) -> None:
    state = self.env.reset()
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_dir))
      done = False
      while not done:
        action = self._get_best_action(sess, np.expand_dims(state, axis=0))
        state, _, done, info = self.env.step(np.squeeze(action))
        if render:
          self.env.render()
        logging.debug(info)
        self.env.render()
        if steps is not None:
          steps -= 1
          if steps <= 0:
            break

  def _get_best_action(self, sess: tf.Session, state: np.array) -> int:
    output = sess.run(
        self.prediction_network.outputs,
        feed_dict={self.prediction_inputs: state})
    action = np.argmax(output, axis=1)
    return action

  def _get_training_action(self, sess: tf.Session, state: np.array,
                           epsilon: float) -> int:
    if np.random.uniform(0.0, 1.0) < epsilon:
      return self.env.action_space.sample()
    return self._get_best_action(sess, state)

  def _create_training_graph(self) -> None:
    self.global_step = tf.train.get_or_create_global_step()
    prediction_q = self.prediction_network.outputs

    indices_actions = tf.stack(
        [tf.range(tf.size(self.action_input)), self.action_input], axis=1)

    actual_q = tf.gather_nd(prediction_q, indices_actions)

    # If that was the end of the episode, then the Q value should just be the reward.
    forward_q = tf.multiply(
        tf.reduce_max(tf.stop_gradient(self.target_network.outputs), axis=1),
        1.0 - tf.to_float(self.done_input))
    expected_q = tf.to_float(self.reward_input) + self.gamma * forward_q

    self.loss = tf.losses.mean_squared_error(actual_q, expected_q)
    self.train_op = self.optimizer.minimize(
        self.loss, global_step=self.global_step)

    tf.summary.scalar("loss", self.loss)
    tf.summary.scalar("max_Q",
                      tf.reduce_mean(tf.reduce_max(prediction_q, axis=1)))
