"""
Title: Proximal Policy Optimization
Author: [Ilias Chrysovergis](https://twitter.com/iliachry)
Date created: 2021/06/24
Last modified: 2021/06/24
Description: Implementation of a Proximal Policy Optimization agent for the CartPole-v0 environment.
"""

"""
## Introduction

This code example solves the CartPole-v0 environment using a Proximal Policy Optimization (PPO) agent.

### CartPole-v0

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The system is controlled by applying a force of +1 or -1 to the cart.
The pendulum starts upright, and the goal is to prevent it from falling over.
A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
After 200 steps the episode ends. Thus, the highest return we can get is equal to 200.

[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)

### Proximal Policy Optimization

PPO is a policy gradient method and can be used for environments with either discrete or continuous action spaces.
It trains a stochastic policy in an on-policy way. Also, it utilizes the actor critic method. The actor maps the
observation to an action and the critic gives an expectation of the rewards of the agent for the observation given.
Firstly, it collects a set of trajectories for each epoch by sampling from the latest version of the stochastic policy.
Then, the rewards-to-go and the advantage estimates are computed in order to update the policy and fit the value function.
The policy is updated via a stochastic gradient ascent optimizer, while the value function is fitted via some gradient descent algorithm.
This procedure is applied for many epochs until the environment is solved.

![Algorithm](https://i.imgur.com/rd5tda1.png)

- [PPO Original Paper](https://arxiv.org/pdf/1707.06347.pdf)
- [OpenAI Spinning Up docs - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

### Note

This code example uses Keras and Tensorflow v2. It is based on the PPO Original Paper,
the OpenAI's Spinning Up docs for PPO, and the OpenAI's Spinning Up implementation of PPO using Tensorflow v1.

[OpenAI Spinning Up Github - PPO](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/ppo.py)
"""

"""
## Libraries

For this example the following libraries are used:

1. `numpy` for n-dimensional arrays
2. `tensorflow` and `keras` for building the deep RL PPO agent
3. `gym` for getting everything we need about the environment
4. `scipy.signal` for calculating the discounted cumulative sums of vectors
"""

"""
## Functions and class
"""




import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
class PPO:
    def __init__(self, observation_dimensions, hidden_sizes, num_actions, policy_learning_rate, value_function_learning_rate, steps_per_epoch, train_policy_iterations, train_value_iterations, target_kl, clip_ratio):
        # Hyperparameters
        self.observation_dimensions = observation_dimensions
        self.num_actions = num_actions
        self.train_policy_iterations = train_policy_iterations
        self.train_value_iterations = train_value_iterations
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio

        # Initialize the buffer
        self.buffer = self.Buffer(self.observation_dimensions, steps_per_epoch)

        # Initialize the actor and the critic as keras models
        observation_input = keras.Input(
            shape=(self.observation_dimensions,), dtype=tf.float32)
        logits = self.build_feedforward_neural_network(
            observation_input, list(hidden_sizes) + [self.num_actions], tf.tanh, None)
        self.actor = keras.Model(inputs=observation_input, outputs=logits)
        value = tf.squeeze(self.build_feedforward_neural_network(
            observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
        self.critic = keras.Model(inputs=observation_input, outputs=value)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(
            learning_rate=policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(
            learning_rate=value_function_learning_rate)

    def build_feedforward_neural_network(self, layer, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            layer = layers.Dense(units=size, activation=activation)(layer)
        return layers.Dense(units=sizes[-1], activation=output_activation)(layer)

    def logprobabilities(self, logits, action):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(tf.one_hot(
            action, self.num_actions) * logprobabilities_all, axis=1)
        return logprobability

    def get_action(self, observation):
        # Get an action from the actor
        self.observation = observation
        logits = self.actor(self.observation)
        self.action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        # Get the value and log-probability of the action
        self.value_step = self.critic(self.observation)
        self.logprobability_step = self.logprobabilities(logits, self.action)

        return self.action[0].numpy()

    def store(self, reward):
        # Store the observtion, action, reward, value and the log-probability of the action
        self.reward = reward
        self.buffer.store(self.observation, self.action,
                          self.reward, self.value_step, self.logprobability_step)

    def finish_epoch(self, done, observation):
        # Finish an epoch
        if done:
            last_value = 0
        else:
            last_value = self.critic(observation)

        self.buffer.finish_trajectory(last_value)

    # Train the policy by maxizing the PPO-Clip objective:
    @tf.function
    def train_policy(
        self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(
                    observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(
            policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean(
                (return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(
            value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.critic.trainable_variables))

    def train(self):
        # Train the model

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = self.buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(self.train_policy_iterations):
            kl = self.train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer)
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(self.train_value_iterations):
            self.train_value_function(observation_buffer, return_buffer)

    class Buffer:
        # Buffer for storing trajectories
        def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
            # Buffer initialization
            self.observation_buffer = np.zeros(
                (size, observation_dimensions), dtype=np.float32
            )
            self.action_buffer = np.zeros(size, dtype=np.int32)
            self.advantage_buffer = np.zeros(size, dtype=np.float32)
            self.reward_buffer = np.zeros(size, dtype=np.float32)
            self.return_buffer = np.zeros(size, dtype=np.float32)
            self.value_buffer = np.zeros(size, dtype=np.float32)
            self.logprobability_buffer = np.zeros(size, dtype=np.float32)
            self.gamma, self.lam = gamma, lam
            self.pointer, self.trajectory_start_index = 0, 0

        def discounted_cumulative_sums(self, x, discount):
            # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

        def store(self, observation, action, reward, value, logprobability):
            # Append one step of agent-environment interaction
            self.observation_buffer[self.pointer] = observation
            self.action_buffer[self.pointer] = action
            self.reward_buffer[self.pointer] = reward
            self.value_buffer[self.pointer] = value
            self.logprobability_buffer[self.pointer] = logprobability
            self.pointer += 1

        def finish_trajectory(self, last_value=0):
            # Finish the trajectory by computing advantage estimates and rewards-to-go
            path_slice = slice(self.trajectory_start_index, self.pointer)
            rewards = np.append(self.reward_buffer[path_slice], last_value)
            values = np.append(self.value_buffer[path_slice], last_value)

            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

            self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
                deltas, self.gamma * self.lam)
            self.return_buffer[path_slice] = self.discounted_cumulative_sums(rewards, self.gamma)[
                :-1]

            self.trajectory_start_index = self.pointer

        def get(self):
            # Get all data of the buffer and normalize the advantages
            self.pointer, self.trajectory_start_index = 0, 0
            advantage_mean, advantage_std = (
                np.mean(self.advantage_buffer),
                np.std(self.advantage_buffer),
            )
            self.advantage_buffer = (
                self.advantage_buffer - advantage_mean) / advantage_std
            return (
                self.observation_buffer,
                self.action_buffer,
                self.advantage_buffer,
                self.return_buffer,
                self.logprobability_buffer,
            )


"""
## Hyperparameters
"""

# Hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
target_kl = 0.01
hidden_sizes = (64, 64)

# True if you want to render the environment
render = False

"""
## Initializations
"""

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = gym.make("CartPole-v0")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the observation, episode return and episode length
observation = env.reset()
episode_return = 0
episode_length = 0

# Initialize the PPO class
ppo = PPO(observation_dimensions, hidden_sizes, num_actions, policy_learning_rate, value_function_learning_rate,
          steps_per_epoch, train_policy_iterations, train_value_iterations, target_kl, clip_ratio)

"""
## Run
"""
# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for step in range(steps_per_epoch):
        if render:
            env.render()

        # Get the logits, action, and take one step in the environment
        _action = ppo.get_action(observation.reshape(1, -1))
        observation_new, reward, done, _ = env.step(_action)
        episode_return += reward
        episode_length += 1

        # Store the transition
        ppo.store(reward)

        # Update the observation
        observation = observation_new

        # If the episode is done, reset the environment
        if done or (step == steps_per_epoch - 1):
            ppo.finish_epoch(done, observation.reshape(1, -1))
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation_new = env.reset()
            episode_return = 0
            episode_length = 0

    ppo.train()

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )
