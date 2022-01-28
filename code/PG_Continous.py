
# ----------------------------------
# Policy Gradient for Continuous Env
# Env: Pendulum-v0
# Problem: Can't convergence
# ----------------------------------

import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'Pendulum-v0'  # environment id
RANDOM_SEED = 1  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'PG'
TRAIN_EPISODES = 500
TEST_EPISODES = 10
MAX_STEPS = 200

class PolicyGradient:

    def __init__(self, state_dim, action_dim, action_range, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.action_range = action_range
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []

        input_layer = tl.layers.Input([None, state_dim], dtype=tf.float32)
        layer = tl.layers.Dense(n_units=100, act=tf.nn.relu)(input_layer)

        a = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh)(layer)
        mu = tl.layers.Lambda(lambda x: x * action_range)(a)
        sigma = tl.layers.Dense(n_units=action_dim, act=tf.nn.softplus)(layer)

        self.model = tl.models.Model(inputs=input_layer, outputs=[mu, sigma])
        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)


    def get_action(self, state):
        s = state[np.newaxis, :].astype(np.float32)
        mu, sigma = self.model(s)
        pi = tfp.distributions.Normal(mu, sigma)
        a = tf.squeeze(pi.sample(1), axis=0)[0]
        return np.clip(a, -self.action_range, self.action_range)


    def store_transition(self, s, a, r):
        self.state_buffer.append(np.array([s], np.float32))
        self.action_buffer.append(a)
        self.reward_buffer.append(r)

    def learn(self):
        discount_reward_buffer_norm = self._discount_and_norm_reward()

        with tf.GradientTape() as tape:
            mu, sigma = self.model(np.vstack(self.state_buffer))
            pi = tfp.distributions.Normal(mu, sigma)
            action = tf.clip_by_value(pi.sample(), -self.action_range, self.action_range)
            log_prob = pi.log_prob(action)

            loss = tf.reduce_sum(- log_prob * discount_reward_buffer_norm)

        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []


    def _discount_and_norm_reward(self):
        """ compute discount_and_norm_rewards """
        discount_reward_buffer = np.zeros_like(self.reward_buffer)
        running_add = 0
        for t in reversed(range(0, len(self.reward_buffer))):
            # Gt = R + gamma * V'
            running_add = self.reward_buffer[t] +  self.gamma * running_add
            discount_reward_buffer[t] = running_add
        # normalize episode rewards
        discount_reward_buffer -= np.mean(discount_reward_buffer)
        discount_reward_buffer /= np.std(discount_reward_buffer)
        return discount_reward_buffer

    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'pg_policy.hdf5'), self.model)
        print("Succeed to save model weights !")

    def load(self):

        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'pg_policy.hdf5'), self.model)
        print("Succeed to load model weights !")

if __name__ == '__main__':
    env = gym.make(ENV_ID).unwrapped
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

    agent = PolicyGradient(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_range=env.action_space.high
    )

    t0 = time.time()

    if args.train:
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):

            state = env.reset()
            episode_reward = 0

            for step in range(MAX_STEPS):  # in one episode
                if RENDER:
                    env.render()

                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward)
                state = next_state
                episode_reward += reward
                if done: break

            agent.learn()
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward,
                    time.time() - t0
                )
            )

            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

        # agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        # test
        # agent.load()
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                env.render()
                state, reward, done, info = env.step(agent.get_action(state, True))
                episode_reward += reward
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0
                )
            )
        env.close()