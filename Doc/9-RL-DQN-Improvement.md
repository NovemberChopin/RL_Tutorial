# 强化学习 9 —— DQN 改进算法

上篇文章[强化学习——详解 DQN 算法](https://blog.csdn.net/november_chopin/article/details/107913103)我们介绍了 DQN 算法，但是 DQN 还存在一些问题，本篇文章介绍针对 DQN 的问题的改进算法 

## 一、Double DQN 算法

### 1、算法介绍

DQN的问题有：目标 Q 值（Q Target ）计算是否准确？全部通过 $max\;Q$ 来计算有没有问题？很显然，是有问题的，这是因为Q-Learning 本身固有的缺陷---过估计

过估计是指估计得值函数比真实值函数要大，其根源主要在于Q-Learning中的最大化操作，对于 TD Target：
$$
r + \gamma\;max_{a'}\; \hat{Q}(s', a', w)
$$
其中的 $max$ 操作使得估计的值函数比值函数的真实值大，因为DQN是一种off-policy的方法，每次学习时，不是使用下一次交互的真实动作，而是使用当前认为价值最大的动作来更新目标值函数，（**注：对于真实的策略来说并在给定的状态下并不是每次都选择使得Q值最大的动作，所以在这里目标值直接选择动作最大的Q值往往会导致目标值要高于真实值**）。Double DQN 的改进方法是将动作的**选择**和动作的**评估**分别用不同的值函数来实现，而在Nature DQN中正好我们提出了两个Q网络。所以计算 `TD Target` 的步骤可以分为下面两步：

- 1）通过当前Q估计网络（Q Estimation 网络）获得最大值函数的动作 $a$: 

$$
a_{max}(s',w) = arg\;max_{a'}Q_{estim}(s', a, w)
$$

- 2）然后利用这个选择出来的动作 $a_{max}(s',w)$ 在目标网络 (Q Target) 里面去计算目 Target Q值：

$$
r + \gamma\;max_{a'}\; Q_{target}(s', a_{max}(s',w), w)
$$

综合起来 在Double DQN 中的 TD Target 计算为：
$$
r + \gamma\;max_{a'}\; Q_{target}(s',arg\;max_{a'}Q_{estim}(s', a, w), w)
$$
除了计算 Target Q 值以外，DDQN 和 DQN 其余流程完全相同。

### 2、代码展示

由上面可知，Double DQN 和 DQN 唯一不同的地方在于Q值的估计，其余流程一样。这里附上代码：

```python
target = self.target_model(states).numpy()
# next_q_values [batch_size, action_diim]
next_target = self.target_model(next_states).numpy()
# next_q_value [batch_size, 1]
next_q_value = next_target[
    range(args.batch_size), np.argmax(self.model(next_states), axis=1)
]
# next_q_value = tf.reduce_max(next_q_value, axis=1)
target[range(args.batch_size), actions] = rewards + (1 - done) * args.gamma * next_q_value

```

完整代码[强化学习——Double DQN 代码地址](https://github.com/NovemberChopin/RL_Tutorial/blob/master/code/DDQN.py) ，劳烦点个 `star` 可好？在此谢谢了

## 二、Dueling DQN 算法

### 1、算法简介

在DQN算法中，神经网络输出的 Q 值代表动作价值，那么单纯的动作价值评估会不会不准确？我们知道，$Q(s, a)$ 的值既和 State 有关，又和 action 有关，但是这两种 “有关” 的程度不一样，或者说影响力不一样，而我们希望能反映出两个方面的差异。

Dueling-DQN 算法从网络结构上改进了DQN，神经网络输出的动作价值函数可以分为状态价值函数和**优势函数**，即：
$$
Q_\pi(s,a) = V_\pi(s) + A_\pi(s,a)
$$
然后这两个函数利用神经网络来逼近。

先来回顾一下，在前面 MDP 那节介绍过了状态价值函数 $V(s)$ 的定义：
$$
v_\pi(s) = \sum_{a\in A} \pi(a|s)\cdot q_\pi(a, s)
$$
状态价值函数就等于在该状态下**所有可能动作所对应的动作值乘以采取该动作的概率的和**。更通俗的讲，值函数 $V(s)$ 是该状态下所有动作值函数关于动作概率的**平均值**；而动作价值函数 $q(s,a)$ 表示在状态 s 下选取 动作 a 所能获得的价值。

那么什么是 优势函数？优势函数 $A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)$ 。意思是当前动作价值相对于平均价值的大小。所以，这里的优势指的是动作价值相比于当前状态的值的优势。如果优势大于零，则说明该动作比平均动作好，如果优势小于零，则说明当前动作还不如平均动作好。这样那些比平均动作好的动作将会有更大的输出，从而加速网络收敛过程。

### 2、代码展示

同样的，Dueling DQN 与DQN 的不同之处在与网络结构，其余流程完全一样。这里不再过多解释，下面附上创建模型相关代码 ：

```python
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
```

完整代码[强化学习——Dueling DQN 代码地址]([https://github.com/NovemberChopin/RL_Tutorial/blob/master/code/Dueling%20DQN.py](https://github.com/NovemberChopin/RL_Tutorial/blob/master/code/Dueling DQN.py)) ，劳烦点个 `star` 可好？在此谢谢了

