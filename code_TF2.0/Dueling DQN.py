import argparse
import os
import random

import numpy as np

import gym
import tensorflow as tf
import tensorlayer as tl


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', default=False)
parser.add_argument('--test', dest='test', default=True)

parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=0.1)

parser.add_argument('--train_episodes', type=int, default=200)
parser.add_argument('--test_episodes', type=int, default=10)
args = parser.parse_args()

ALG_NAME = 'DuelineDQN'
ENV_ID = 'CartPole-v1'

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size = args.batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, next_state, done


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        def create_model(input_state_shape):
            input_layer = tl.layers.Input(input_state_shape)
            layer_1 = tl.layers.Dense(n_units=32, act=tf.nn.relu)(input_layer)
            layer_2 = tl.layers.Dense(n_units=16, act=tf.nn.relu)(layer_1)
            # state value
            state_value = tl.layers.Dense(n_units=1)(layer_2)
            # advantage value
            q_value = tl.layers.Dense(n_units=self.action_dim)(layer_2)
            mean = tl.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(q_value)
            advantage = tl.layers.ElementwiseLambda(lambda x, y: x-y)([q_value, mean])
            # output
            output_layer = tl.layers.ElementwiseLambda(lambda x, y: x+y)([state_value, advantage])
            return tl.models.Model(inputs=input_layer, outputs=output_layer)

        self.model = create_model([None, self.state_dim])
        self.target_model = create_model([None, self.state_dim])
        self.model.train()
        self.target_model.eval()
        self.model_optim  = tf.optimizers.Adam(lr=args.lr)

        self.epsilon = args.eps

        self.buffer = ReplayBuffer()

    def target_update(self):
        """Copy q network to target q network"""
        for weights, target_weights in zip(
                self.model.trainable_weights, self.target_model.trainable_weights):
            target_weights.assign(weights)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_value = self.model(state[np.newaxis, :])[0]
            return np.argmax(q_value)

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            target = self.target_model(states).numpy()
            # next_q_values [batch_size, action_dim]
            next_target = self.target_model(next_states)
            next_q_value = tf.reduce_max(next_target, axis=1)
            target[range(args.batch_size), actions] = rewards + (1 - done) * args.gamma * next_q_value

            # use sgd to update the network weight
            with tf.GradientTape() as tape:
                q_pred = self.model(states)
                loss = tf.losses.mean_squared_error(target, q_pred)
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.model_optim.apply_gradients(zip(grads, self.model.trainable_weights))


    def train(self, train_episodes=200):
        if args.train:
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
        if args.test:
            self.loadModel()
            self.test_episode(test_episodes=args.test_episodes)


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


    def saveModel(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'model.hdf5'), self.model)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'target_model.hdf5'), self.target_model)
        print('Saved weights.')

    def loadModel(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if os.path.exists(path):
            print('Load DQN Network parametets ...')
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'model.hdf5'), self.model)
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'target_model.hdf5'), self.target_model)
            print('Load weights!')
        else: print("No model file find, please train model first...")


if __name__ == '__main__':
    env = gym.make(ENV_ID)
    agent = Agent(env)
    agent.train(train_episodes=args.train_episodes)
    env.close()