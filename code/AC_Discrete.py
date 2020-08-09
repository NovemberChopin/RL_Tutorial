"""
Actor-Critic
-------------
It uses TD-error as the Advantage.

To run
------
python tutorial_AC.py --train/test
"""
import argparse
import time
import matplotlib.pyplot as plt
import os

import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'CartPole-v1'  # environment id
RANDOM_SEED = 2  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'AC'
TRAIN_EPISODES = 200  # number of overall episodes for training
TEST_EPISODES = 10  # number of overall episodes for testing
MAX_STEPS = 500  # maximum time step in one episode
LAM = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

class Actor(object):

    def __init__(self, state_dim, action_dim, lr=0.001):
        input_layer = tl.layers.Input([None, state_dim])
        layer = tl.layers.Dense(n_units=32, act=tf.nn.relu6)(input_layer)
        layer = tl.layers.Dense(n_units=action_dim)(layer)

        self.model = tl.models.Model(inputs=input_layer, outputs=layer)
        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, action, td_error):
        with tf.GradientTape() as tape:
            _logits = self.model(np.array([state]))
            _exp_v = tl.rein.cross_entropy_reward_loss(
                logits=_logits, actions=[action], rewards=td_error)
        grad = tape.gradient(_exp_v, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

    def get_action(self, state, greedy=False):
        _logits = self.model(np.array([state]))
        _prob = tf.nn.softmax(_logits).numpy()
        if greedy:
            return np.argmax(_prob.ravel())
        return tl.rein.choice_action_by_probs(_prob.ravel())


class Critic(object):

    def __init__(self, state_dim, lr=0.01):
        input_layer = tl.layers.Input([None, state_dim], name='state')
        layer = tl.layers.Dense(n_units=32, act=tf.nn.relu)(input_layer)
        layer = tl.layers.Dense(n_units=1, act=None)(layer)

        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name='Critic')
        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, reward, state_, done):
        d = 0 if done else 1

        with tf.GradientTape() as tape:
            v = self.model(np.array([state]))
            v_ = self.model(np.array([state_]))
            td_error = reward + d * LAM * v_ - v
            loss = tf.square(td_error)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return td_error


class Agent():

    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.actor = Actor(self.state_dim, self.action_dim, lr=LR_A)
        self.critic = Critic(self.state_dim, lr=LR_C)

    def train(self):
        if args.train:
            self.train_episode()
        if args.test:
            self.load()
            self.test_episode()

    def train_episode(self):
        t0 = time.time()
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state = env.reset().astype(np.float32)
            step = 0
            episode_reward = 0
            while True:
                if RENDER: env.render()
                action = self.actor.get_action(state)
                state_, reward, done, _ = env.step(action)
                state_ = state_.astype(np.float32)

                episode_reward += reward

                td_error = self.critic.learn(state, reward, state_, done)
                self.actor.learn(state, action, td_error)

                state = state_
                step += 1
                if done or step >= MAX_STEPS:
                    break
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

            print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))
            # Early Stopping for quick check
            if step >= MAX_STEPS:
                print("Early Stopping")  # Hao Dong: it is important for this task
                self.save()
                break
        env.close()

        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    def test_episode(self):
        t0 = time.time()
        for episode in range(TEST_EPISODES):
            state = env.reset().astype(np.float32)
            t = 0  # number of step in this episode
            episode_reward = 0
            while True:
                env.render()
                action = self.actor.get_action(state, greedy=True)
                state_new, reward, done, info = env.step(action)
                state_new = state_new.astype(np.float32)
                if done: reward = -20

                episode_reward += reward
                state = state_new
                t += 1

                if done or t >= MAX_STEPS:
                    print('Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                          .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))
                    break
        env.close()

    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.actor.model.trainable_weights, name=os.path.join(path, 'model_actor.npz'))
        tl.files.save_npz(self.critic.model.trainable_weights, name=os.path.join(path, 'model_critic.npz'))
        print('Succeed to save model weights')

    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_critic.npz'), network=self.critic.model)
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_actor.npz'), network=self.actor.model)
        print('Succeed to load model weights')

if __name__ == '__main__':

    env = gym.make(ENV_ID).unwrapped
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    agent = Agent(env)
    agent.train()



