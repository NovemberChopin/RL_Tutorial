import argparse

import gym
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

class QLearning:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.9, e_greed=0.1):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((state_dim, action_dim))

    def sample(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.predict(state)
        return action

    def predict(self, state):
        """ 根据输入观察值，预测输出的动作值 """
        all_actions = self.Q[state, :]
        max_action = np.max(all_actions)
        # 防止最大的 Q 值有多个，找出所有最大的 Q，然后再随机选择
        # where函数返回一个 array， 每个元素为下标
        max_action_list = np.where(all_actions == max_action)[0]
        action = np.random.choice(max_action_list)
        return action

    def learn(self, state, action, reward, next_state, done):
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] += self.lr * (target_q - self.Q[state, action])

    def save(self):
        npy_file = './model/qlearning_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def load(self, npy_file='./model/qlearning_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')

class Agent:
    def __init__(self, env):
        self.env = env
        self.lr = 0.1,
        self.gamma = 0.9,
        self.e_greed = 0.1
        # type(self.state_dim): <class 'tuple'>
        # type(env.observation_space.n): <class 'int'>
        self.model = QLearning(env.observation_space.n, env.action_space.n)

    def train(self, max_episode):

        if args.train:
            for episode in range(max_episode):
                ep_reward, ep_steps = self.run_episode(render=False)
                if episode % 20 == 0:
                    print('Episode %03s: steps = %02s , reward = %.1f' % (episode, ep_steps, ep_reward))
            self.model.save()

        if args.test:
            self.model.load()
            self.test_episode(render=True)

    def run_episode(self, render=False):
        total_reward = 0
        total_steps = 0
        state = self.env.reset()
        while True:
            action = self.model.sample(state)
            next_state, reward, done, _ = self.env.step(action)
            # 训练 Q-learning算法
            self.model.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            total_steps += 1
            if render: self.env.render()
            if done: break
        return total_reward, total_steps

    def test_episode(self, render=False):
        total_reward = 0
        actions = []
        state = self.env.reset()
        while True:
            action = self.model.predict(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            total_reward += reward
            actions.append(action)
            if render: self.env.render()
            if done: break

        print('test reward = %.1f' % (total_reward))
        print('test action is: ', actions)

if __name__ == '__main__':
    # 使用gym创建迷宫环境，设置is_slippery为False降低环境难度
    env = gym.make("FrozenLake-v0", is_slippery=False)

    agent = Agent(env=env)
    agent.train(500)
