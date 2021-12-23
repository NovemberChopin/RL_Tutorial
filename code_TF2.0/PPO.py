"""
Proximal Policy Optimization (PPO)
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.
PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

Reference
---------
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials

Environment
-----------
Openai Gym Pendulum-v0, continual action space

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_PPO.py --train/test

"""
import argparse
import os
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_false', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'Pendulum-v0'        # environment name
ALG_NAME = 'PPO'
RANDOMSEED = 1                  # random seed

EP_MAX = 1000                    # total number of episodes for training
EP_LEN = 200                    # total number of steps for each episode
GAMMA = 0.9                     # reward discount
A_LR = 0.0001                   # learning rate for actor
C_LR = 0.0002                   # learning rate for critic
BATCH = 32                      # update batchsize
A_UPDATE_STEPS = 10     # actor update steps
C_UPDATE_STEPS = 10     # critic update steps
EPS = 1e-8              # epsilon

# 注意：这里是PPO1和PPO2的相关的参数。
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty  PPO1
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better  PPO2
][1]                                                # choose the method for optimization

# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip parameters
EPSILON = 0.2

###############################  PPO  ####################################


class PPO(object):
    '''
    PPO 类
    '''
    def __init__(self, state_dim, action_dim, action_bound, method='clip'):

        def build_critic(input_state_dim):
            input_layer = tl.layers.Input(input_state_dim, tf.float32)
            l1 = tl.layers.Dense(100, tf.nn.relu)(input_layer)
            output_layer = tl.layers.Dense(1)(l1)
            return tl.models.Model(input_layer, output_layer)

        def build_actor(input_state_dim, action_dim):
            ''' actor 网络，输出mu和sigma '''
            input_layer = tl.layers.Input(input_state_dim, tf.float32)
            l1 = tl.layers.Dense(100, tf.nn.relu)(input_layer)
            a = tl.layers.Dense(action_dim, tf.nn.tanh)(l1)
            mu = tl.layers.Lambda(lambda x: x * action_bound)(a)
            sigma = tl.layers.Dense(action_dim, tf.nn.softplus)(l1)
            model = tl.models.Model(input_layer, [mu, sigma])
            return model

        # 构建critic网络, 输入state，输出V值
        self.critic = build_critic([None, state_dim])
        self.critic.train()

        # actor有两个actor 和 actor_old， actor_old的主要功能是记录行为策略的版本。
        # 输入时state，输出是描述动作分布的mu和sigma
        self.actor = build_actor([None, state_dim], action_dim)
        self.actor_old = build_actor([None, state_dim], action_dim)
        self.actor.train()
        self.actor_old.eval()
        self.actor_opt = tf.optimizers.Adam(A_LR)
        self.critic_opt = tf.optimizers.Adam(C_LR)

        self.method = method
        if method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        self.action_bound = action_bound

    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        s = s[np.newaxis, :].astype(np.float32)
        mu, sigma = self.actor(s)
        pi = tfp.distributions.Normal(mu, sigma)    # 用mu和sigma构建正态分布
        a = tf.squeeze(pi.sample(1), axis=0)[0]     # 根据概率分布随机出动作
        return np.clip(a, -self.action_bound, self.action_bound)

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def a_train(self, state, action, adv):
        '''
        更新策略网络(policy network)
        '''
        # 输入时s，a，td-error。这个和AC是类似的。
        state = np.array(state, np.float32)         #state
        action = np.array(action, np.float32)         #action
        adv = np.array(adv, np.float32)     #td-error


        with tf.GradientTape() as tape:

            # 【敲黑板】这里是重点！！！！
            # 我们需要从两个不同网络，构建两个正态分布pi，oldpi。
            mu, sigma = self.actor(state)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, sigma_old = self.actor_old(state)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            # 在新旧两个分布下，同样输出a的概率的比值
            # 除以(oldpi.prob(tfa) + EPS)，其实就是做了import-sampling。怎么解释这里好呢
            # 本来我们是可以直接用pi.prob(tfa)去跟新的，但为了能够更新多次，我们需要除以(oldpi.prob(tfa) + EPS)。
            # 在AC或者PG，我们是以1,0作为更新目标，缩小动作概率到1or0的差距
            # 而PPO可以想作是，以oldpi.prob(tfa)出发，不断远离（增大or缩小）的过程。
            ratio = pi.prob(action) / (oldpi.prob(action) + EPS)
            # 这个的意义和带参数更新是一样的。
            surr = ratio * adv

            # 我们还不能让两个分布差异太大。
            # PPO1
            if METHOD['name'] == 'kl_pen':
                tflam = METHOD['lam']
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - tflam * kl))
            else:  # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(
                        ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv)
                )
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        '''
        更新actor_old参数。
        '''
        for pi, oldpi in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldpi.assign(pi)

    def c_train(self, reward, state):
        ''' 更新Critic网络 '''
        # reward 是我们预估的 能获得的奖励
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic(state)     # td-error
            loss = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))


    def update(self):
        '''
        Update parameter with the constraint of KL divergent
        '''
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)

        self.update_old_pi()
        adv = (r - self.critic(s)).numpy()
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        # PPO2 clipping method, find this is better (OpenAI's paper)
        else:
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv)
        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()


    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done: v_s_ = 0
        else: v_s_ = self.critic(np.array([next_state], dtype=np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()


    def save_ckpt(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_NAME]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_old.hdf5'), self.actor_old)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)
        print('save weights success!')

    def load_ckpt(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_NAME]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_old.hdf5'), self.actor_old)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)
        print("load weight!")


if __name__ == '__main__':

    env = gym.make(ENV_NAME).unwrapped

    # reproducible
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    ppo = PPO(
        state_dim=env.observation_space.shape[0],
        action_dim = env.action_space.shape[0],
        action_bound = env.action_space.high,
    )

    if args.train:
        all_ep_r = []

        # 更新流程：
        for episode in range(EP_MAX):
            state = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            episode_reward = 0
            t0 = time.time()
            for t in range(EP_LEN):
                # env.render()
                action = ppo.choose_action(state)
                state_, reward, done, _ = env.step(action)
                ppo.store_transition(state, action, (reward + 8) / 8)
                state = state_
                episode_reward += reward

                # N步更新的方法，每BATCH步了就可以进行一次更新
                if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                    ppo.finish_path(state_, done)
                    ppo.update()

            if episode == 0:
                all_ep_r.append(episode_reward)
            else:
                all_ep_r.append(all_ep_r[-1] * 0.9 + episode_reward * 0.1)
            print(
                'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode, EP_MAX, episode_reward,
                    time.time() - t0
                )
            )
        ppo.save_ckpt()
        plt.plot(all_ep_r)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_NAME])))

    if args.test:
        ppo.load_ckpt()
        for episode in range(10):
            state = env.reset()
            rewards = 0
            for i in range(EP_LEN):
                env.render()
                next_state, reward, done, _ = env.step(ppo.choose_action(state))
                rewards += reward
                state = next_state
                if done:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}'.format(
                    episode + 1, 10, rewards))
        env.close()