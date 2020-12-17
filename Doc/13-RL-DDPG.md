# 强化学习13——Deep Deterministic Policy Gradient（DDPG）

上篇文章介绍了[强化学习——Actor-Critic算法详解加实战](https://blog.csdn.net/november_chopin/article/details/108170500) 介绍了Actor-Critic，本篇文章将介绍 DDPG 算法，DDPG 全称是 Deep Deterministic Policy Gradient（深度确定性策略梯度算法） 其中 PG 就是我们前面介绍了 Policy Gradient，在[强化学习10——Policy Gradient 推导](https://blog.csdn.net/november_chopin/article/details/108032626) 已经讨论过，那什么是确定性策略梯度呢？

## 一、确定性策略

与确定性策略对应的是随机性策略，就是神经网络输出的是动作的分布，在确定每一步动作时，我们需要得到的策略分布进行采样，对于某些高纬的连续值动作，频繁的在高维空间对动作进行采样，是很耗费计算能力的。

同样，对于DQN算法，其只适用于低维、离散动作的问题，对于连续动作问题，DQN要计算所有可能动作的概率，并计算可能的动作的价值，动作的数量随着自由度的数量呈指数增长，那需要非常的样本量与计算量，所以就有了确定性策略来简化这个问题。

作为随机策略，在相同的策略，在同一个状态处，采用的动作是基于一个概率分布的，即是不确定的。而确定性策略就简单的多，即使用相同的策略，在同一状态下，动作是唯一确定的：
$$
a_t = \mu(s|\theta^\mu)
$$

## 二、DDPG

首先要注意一点，**DDPG从名字上像一个策略梯度（PG）算法，但是其实它更接近DQN，或者说DDPG是使用的 Actor-Critic 架构来解决DQN不能处理连续动作控制问题的一个算法**，这点一定要注意。下面来详细解释为什么这么说

### 1、从 Q-Learning 到 DQN

我们先回忆下Q-Learning的算法流程，在 [强化学习4——时序差分控制算法](https://blog.csdn.net/november_chopin/article/details/107897225) 中已经详细介绍过Q-Learning算法。我们知道，首先我们基于状态 $S_t$，用 $\epsilon-$贪婪法选择到动作 $A_t$ 并执行，进入状态$S_{t+1}$，并得到奖励$R_{t}$，然后利用得到的$<S,A,R,S'>$ 来更新Q表格，注意在更新Q表格时，基于状态 $S_{t+1}$ 使用贪心策略选择 $A'$ 。
$$
A' = max_{a'} Q(S',a')
$$
也就是**选择使$Q(S_{t+1}, a)$ 最大的 $a$ 来作为 $A'$ 来更新价值函数**。对应到上图中就是在图下方的三个黑圆圈动作中选择一个使$Q(S', a)$ 最大的动作作为 $A'$ 。

![ar0z9A.png](https://s1.ax1x.com/2020/08/05/ar0z9A.png)

由于Q-Learning 使用Q表格存储每个状态的所有动作价值，所以面对连续状态既如果状态非常多的情况就不能胜任，所有我们就用函数逼近的方法，使用神经网络来代替 Q 表格，其余流程不变，这样就得到了DQN算法。在[强化学习7——DQN算法详解](https://blog.csdn.net/november_chopin/article/details/107912720) 中已经详细介绍过 DQN算法，下面就简单回忆下DQN算法流程：

![arwWRI.png](https://s1.ax1x.com/2020/08/05/arwWRI.png)

可以看出，DQN用神经网络代替Q表格，loss 函数就是神经网络当前的输出与target之间的差距，然后对损失函数求导更新网络 参数。

### 2、从DQN 到DDPG

而target的计算方式为 $r + \gamma\;max_{a'}\hat{q}(s',a',w)$ ，即从下一个状态使用max函数选一个最大的动作 Q，当具有有限数量的离散动作值时，计算max不是问题，因为我们可以对所有动作计算其Q值(这也能够得到能够最大化Q值的动作)。但是当动作空间连续时，我们不能穷举所有的可能，这也是DQN不能处理连续动作控制任务的原因。

那么该如何解决这个问题呢？我们知道DQN用神经网络解决了Q-Learning不能解决的连续状态空间问题。那我们可不可以也使用一个神经网络来代替 $max_aQ(s,a)$ 呢？当然是可以的，DDPG就是这样做的，即用一个函数来代替这个过程：
$$
max_aQ(s,a) \approx Q(s,\mu(s|\theta))
$$
其中的 $\theta$ 就是策略网络的参数，根据前面讲过的AC算法，可以联想到使用策略网络充当 Actor，使用DQN中的价值网络充当 Critic（注意这里估计的是 Q值，而不是 V值，不要和AC算法搞混了），Actor部分不使用带 Reward 的加权梯度更新（PG算法更新方式），而是使用Critic网络对action 的梯度对 actor更新。

DDPG既然来自DQN，那当然也有经验回放与双网络，所以在 Critic 部分增加一个目标网络去计算目标 Q值，但是还需要选取动作用来估计目标Q值，由于我们有自己的Actor策略网络，用类似的做法，再增加一个Actor目标网络，所以DDPG算法一共包含了四个网络，分别是：Actor当前网络，Actor目标网络，Critic当前网络，Critic目标网络，2个Actor网络的结构相同，2个Critic网络的结构相同，这四个网络的功能如下：

- Actor当前网络：负责策略网络参数 $\theta$ 的迭代更新，负责根据当前状态 S 选择当前动作 A，用于和环境交互生成 S'和 R。
- Actor目标网络：负责根据经验回放池中采样的下一状态 S' 选择最优下一动作 A'，网络参数 $\theta$ 定期从 $\theta$ 复制。
- Critic当前网络：负责价值网络参数ww的迭代更新，负责计算负责计算当前Q值 $Q(S,A,w)$ 。目标Q 值 $y_i = R + \gamma Q'(S',A',w')$ 。
- Critic目标网络：负责计算目标Q值中的 $Q'(S',A',w')$ 部分，网络参数 $w'$ 定期从 $w$ 复制。

注意：**Critic 网络的输入有两个：动作和状态，需要一起 输入到 Critic 中**

值得一提的是，DDPG从当前网络到目标网络的复制和我们之前讲到了DQN不一样。在DQN中是直接把将当前Q网络的参数复制到目标Q网络，即 $w' = w$，这样的更新方式为**硬更新**，与之对应的是**软更新**，DDPG就是使用的软更新，即每次参数只更新一点点，即：
$$
w' \leftarrow \tau w + (1-\tau)w' \\
\theta \leftarrow \tau \theta + (1-\tau)\theta'
$$
其中 $\tau$ 是更新系数，一般取的比较小。同时，为了学习过程可以增加一些随机性，探索潜在的更优策略，通过Ornstein-Uhlenbeck process（OU过程）为action添加噪声，最终输出的动作A为：
$$
A = \pi_\theta(s) + OU_{noise}
$$
对于损失函数，Critic部分和DQN类似，都是使用均方误差：
$$
J(w) = \frac{1}{m}\sum_{i=1}^m(y_i - Q(S_i, A_i, w))^2
$$
对于Actor部分的损失函数，[原论文](https://arxiv.org/pdf/1509.02971.pdf)定义的比较复杂，这里采取一种简单的方式来理解，我们知道 ，Actor 的作用是输出一个动作 A，这个动作 A输入到 Critic 后，可以获得最大的 Q 值。 所以我们的Actor的损失可以简单的理解为得到的反馈Q值越大损失越小，得到的反馈Q值越小损失越大，因此只要对状态估计网络返回的Q值取个负号即可，即：
$$
J(\theta) = -\frac{1}{m}\sum_{i=1}^mQ(s_i, a_i, w)
$$
关于DDPG算法完整的流程如下：

<img src="https://s1.ax1x.com/2020/08/05/asY5ng.png" alt="asY5ng.png" style="zoom:80%;" />

## 三、代码实现

代码使用 `Pendulum-v0` 连续环境，采用 `tensorflow` 学习框架

### 1、搭建网络

#### Actor 网络

```python
def get_actor(input_state_shape):
    input_layer = tl.layers.Input(input_state_shape)
    layer = tl.layers.Dense(n_units=64, act=tf.nn.relu)(input_layer)
    layer = tl.layers.Dense(n_units=64, act=tf.nn.relu)(layer)
    layer = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh)(layer)
    layer = tl.layers.Lambda(lambda x: action_range * x)(layer)
    return tl.models.Model(inputs=input_layer, outputs=layer)
```

Actor 网络输入状态 ，输出动作，注意的是，连续环境的动作一般都有一个范围，这个范围在环境中已经定以好，使用 `action_bound = env.action_space.high` 即可获取。

如果 actor 输出的动作超出范围会导致程序异常，所以在网络末端使用 `tanh` 函数把输出映射到 [-1.0, 1.0]之间。然后使用 `lamda` 表达式，把动作映射到相应的范围。

#### Critic 网络

```python
def get_critic(input_state_shape, input_action_shape):
    state_input = tl.layers.Input(input_state_shape)
    action_input = tl.layers.Input(input_action_shape)
    layer = tl.layers.Concat(1)([state_input, action_input])
    layer = tl.layers.Dense(n_units=64, act=tf.nn.relu)(layer)
    layer = tl.layers.Dense(n_units=64, act=tf.nn.relu)(layer)
    layer = tl.layers.Dense(n_units=1, name='C_out')(layer)
    return tl.models.Model(inputs=[state_input, action_input], outputs=layer)
```

在DDPG中我们把 状态和动作同时输入到 Critic 网络中，去估计 $Q(s,a)$ 。所以定义两个输入层，然后连接起来，最后模型的输入部分定义两个输入。

### 2、主流程

```python
for episode in range(TRAIN_EPISODES):
    state = env.reset()
    for step in range(MAX_STEPS):
        if RENDER: env.render()
        # Add exploration noise
        action = agent.get_action(state)
        state_, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, state_)

        if agent.pointer > MEMORY_CAPACITY:
            agent.learn()

        state = state_
        if done: break
```

可以看到，DDPG流程与DQN基本相同，重置状态，然后选择动作，与环境交互，获得S' ，R后，把数据保存起来。如多数据量足够，就对数据  进行抽样，更新网络参数。然后开始更新 s，开始下一步循环。

这里重点看一下 `get_action()` 函数：

```python
def get_action(self, s, greedy=False):
    a = self.actor(np.array([s], dtype=np.float32))[0]
    if greedy:
        return a
    return np.clip(
        np.random.normal(a, self.var), -self.action_range, self.action_range)
```

`get_action()` 函数用以选取一个动作，然后与环境交互。为了更好的探索环境，我们在训练过程中为动作加入噪声，原始的DDPG作者推荐加入与时间相关的OU噪声，但是更近的结果表明高斯噪声表现地更好。由于后者更为简单，因此其更为常用。

我们这里就采用的后者，为动作添加高斯噪声：这里我们的 Actor 输出的 `a` 作为一个正太分布的平均值，然后加上参数 `VAR`，作为正太分布的方差，然后就可以构造一个正太分布。然后从正太分布中随机选取一个动作 ，我们知道正太分布是有很大概率采样到平均值附近的点，所以利用这个方法，就可以实现一定的探索行为。此外，我们也可以控制 `VAR` 的大小，来控制探索概率的大小。

当测试的时候，选取动作时就不需要探索，因为这时 Actor 要输入有最大 `Q` 值的动作，直接输出动作就可以。所以在`get_action()` 函数中用 一个参数 `greedy` 来控制这两种情况。

### 3、网络更新

#### Critic 更新

![dahsTH.png](https://s1.ax1x.com/2020/08/22/dahsTH.png)

如上图所示，Critic 部分的更新和 DQN 是一样的，使用td-error来更新。用目标网络构造 `target`，然后和当前网络输出的 `q` 计算MSE损失，然后更新网络参数。

```python
with tf.GradientTape() as tape:
    actions_ = self.actor_target(states_)
    q_ = self.critic_target([states_, actions_])
    target = rewards + GAMMA * q_
    q_pred = self.critic([states, actions])
    td_error = tf.losses.mean_squared_error(target, q_pred)
critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))
```

#### Actor 更新

```python
with tf.GradientTape() as tape:
    actions = self.actor(states)
    q = self.critic([states, actions])
    actor_loss = -tf.reduce_mean(q)  # maximize the q
actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
```

DDPG采用梯度上升法，Actor作用就是输出一个动作，这个动作输入Critic网络能得到最大的 `Q` 值。由于和梯度下降方向相反，所以需要在 loss 函数前面加上负号。

完整代码地址： [强化学习——DDPG算法Tensorflow2.0实现](https://github.com/NovemberChopin/RL_Tutorial/blob/master/code/DDPG.py) 还望随手给个star，再次不胜感激

## 四、总结

DDPG通过借鉴AC的架构，在DQN算法的基础上引入了Actor网络，解决了连续控制问题，可以看做是DQN在连续问题上的改进算法。

下篇会介绍DDPG的进化版本的算法，就是TD3算法。

