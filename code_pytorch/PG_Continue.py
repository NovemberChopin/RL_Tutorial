

import argparse
import math
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import distributions
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.distributions

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'Pendulum-v1'  # environment id
RANDOM_SEED = 1  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'PG-Continue'
TRAIN_EPISODES = 200
TEST_EPISODES = 10
MAX_STEPS = 200

hidden_1 = 32
hidden_2 = 16

var = 1
var_delta = 0.999

###############################  PG  ####################################


class Net(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_1)
        self.layer2 = nn.Linear(hidden_1, hidden_2)
        self.output = nn.Linear(hidden_2, action_dim)

        self.action_bound = action_bound
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        output = torch.tanh(x) * self.action_bound
        return output


class PolicyGradient:
    def __init__(self, state_dim, action_dim, action_bound, learning_rate=0.01, gamma=0.9):
        self.gamma = gamma
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []
        self.action_bound = action_bound
        self.var = var
        self.var_delta = var_delta
        self.model = Net(state_dim, action_dim, action_bound)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
    
    def get_action(self, state, greedy=False):
        state = torch.FloatTensor(state).view(1, -1)
        action = self.model(state)
        if greedy:
            return action[0]
        with torch.no_grad():
            action = np.clip(np.random.normal(action[0], self.var), -self.action_bound, self.action_bound)
            self.var = self.var * self.var_delta
        return action
        

    def store_transition(self, s, a, r):

        self.state_buffer.append(s)
        self.action_buffer.append(a)
        self.reward_buffer.append(r)

    def learn(self):
        """
        update policy parameters via stochastic gradient ascent
        :return: None
        """
        discounted_reward = self._discount_and_norm_rewards()
        state = torch.FloatTensor(self.state_buffer)
        reward = torch.FloatTensor(discounted_reward)

        mu = self.model(state)
        pi = distributions.Normal(mu, self.var)
        action = np.clip(pi.sample(sample_shape=mu.shape), -self.action_bound, self.action_bound)
        loss = -torch.sum(pi.log_prob(action) * reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []  # empty episode data


    def _discount_and_norm_rewards(self):
        """
        compute discount_and_norm_rewards
        :return: discount_and_norm_rewards
        """
        # discount episode rewards
        discounted_reward_buffer = np.zeros_like(self.reward_buffer)
        running_add = 0
        for t in reversed(range(0, len(self.reward_buffer))):
            running_add = running_add * self.gamma + self.reward_buffer[t]
            discounted_reward_buffer[t] = running_add

        # normalize episode rewards
        discounted_reward_buffer -= np.mean(discounted_reward_buffer)
        discounted_reward_buffer /= np.std(discounted_reward_buffer)
        return discounted_reward_buffer

    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model_torch', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + '/model.pth'
        torch.save(self.model.state_dict(), path)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model_torch', '_'.join([ALG_NAME, ENV_ID]))
        path = path + '/model.pth'
        self.model.load_state_dict(torch.load(path))
        print("load parameters successed")


if __name__ == '__main__':
    env = gym.make(ENV_ID).unwrapped

    # reproducible
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

    agent = PolicyGradient(
        action_dim=env.action_space.shape[0],
        state_dim=env.observation_space.shape[0],
        action_bound=env.action_space.high[0]
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
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward)
                state = next_state
                episode_reward += reward
                if done:
                    break
            agent.learn()
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0
                )
            )

            if episode == 0: all_episode_reward.append(episode_reward)
            else:  all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

        env.close()
        agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image_torch'):
            os.makedirs('image_torch')
        plt.savefig(os.path.join('image_torch', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        # test
        agent.load()
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                env.render()
                action = agent.get_action(state, True)
                state, reward, done, info = env.step(action)
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