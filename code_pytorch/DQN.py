import gym
import numpy as np
import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from network import BasicQNetwork
from buffer import ReplayBUffer
from parameter import get_common_args

algo = "DQN"
env_name = "CartPole-v1"
isTrain = True
isTest = False


class Agent():
    def __init__(self, state_dim, action_dim, args, double_dqn = False):
        self.replay_buffer = ReplayBUffer(args)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.tau = args.tau
        self.learning_rate = args.lr_q
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_delta = args.epsilon_delta
        self.double_dqn = double_dqn
        self.device = args.device
        
        self.q_network = BasicQNetwork(self.state_dim, self.action_dim, args)
        self.target_q_network = BasicQNetwork(self.state_dim, self.action_dim, args)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
        for p in self.target_q_network.parameters():
            p.requires_grad = False
               
    def e_greedy_action(self,state):
        with torch.no_grad():
            state = torch.FloatTensor(state).view(1, -1).to(self.device)
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0,self.action_dim)
            else:
                action = self.q_network(state).argmax().item()
        self.epsilon = max(self.epsilon_min, self.epsilon-self.epsilon_delta)       
        return action

    def action(self,state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.q_network(state).argmax().item()
        return action

    def learn(self):
        #update the param of target network
        for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device).view(self.batch_size,1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).view(self.batch_size,1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).view(self.batch_size,1)
        
        
        with torch.no_grad():
            if self.double_dqn:
                argmax_a = self.q_network(next_state_batch).argmax(dim=1).unsqueeze(-1)
                next_q_value = self.target_q_network(next_state_batch).gather(1,argmax_a)               
            else:
                next_q_value = self.target_q_network(next_state_batch).max(1)[0].view(self.batch_size,1)			

        #self.eval_net(state_batch)得到一个batch_size x action_size的tensor
        #.gather(1,action_batch)获取得到的tensor位置上对应的动作所对应的Q值,得到一个actio_batch一样的tensor
        q_value = self.q_network(state_batch).gather(1,action_batch)
        
        target_q_value = reward_batch + (1 - done_batch) * self.gamma * next_q_value
        loss = self.loss_function(q_value,target_q_value)    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("q_network = ", print(self.q_network.state_dict()['layer1.weight']))
        return loss.item()

    def test_episode(self, test_episodes):
        for episode in range(test_episodes):
            state = self.env.reset().astype(np.float32)
            total_reward, done = 0, False
            while not done:
                action = self.model(np.array([state], dtype=np.float32))[0]
                action = np.argmax(action)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.astype(np.float32)

                total_reward += reward
                state = next_state
                self.env.render()
            print("Test {} | episode rewards is {}".format(episode, total_reward))

    def train(self, train_episodes=200):
        if isTrain:
            for episode in range(train_episodes):
                total_reward, done = 0, False
                state = self.env.reset().astype(np.float32)
                while not done:
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = next_state.astype(np.float32)
                    self.buffer.push(state, action, reward, next_state, done)
                    total_reward += reward
                    state = next_state
                    # self.render()
                if len(self.buffer.buffer) > args.batch_size:
                    self.replay()
                    self.target_update()
                print('EP{} EpisodeReward={}'.format(episode, total_reward))
            self.saveModel()
        if isTest:
            self.loadModel()
            self.test_episode(test_episodes=args.test_episodes)

    def save(self):
        if not self.double_dqn:
            dir = 'model_torch/{}_{}'.format(algo, env_name)
        else:
            dir = 'model_torch/{}'.format(algo, env_name)
        if not os.path.exists(dir): # make the path
            print("dont have this dir")
            os.mkdir(dir)
        dir = dir + '/model.pth'
        torch.save(self.q_network.state_dict(), dir)
        
    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        print('q_network load successed')

if __name__ == '__main__':
    args = get_common_args()
    env = gym.make(env_name)
    agent = Agent()
