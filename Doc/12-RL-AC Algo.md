# 强化学习 12 —— Actor-Critic Algorithm

## 一、Actor-Critic 介绍

### 1、引入 Actor-Critic

我们还是从上篇[强化学习——REINFORCE Algorithm](https://blog.csdn.net/november_chopin/article/details/108033013)推导出的目标函数的梯度说起：
$$
\nabla_\theta J(\theta) = E_{\pi_\theta} \left[\sum_{t=0}^{T-1}G_t\cdot \nabla_\theta\;log\;\pi_\theta(a_t|s_t) \right]
$$
其中 $G_t$ 就表示当前采取的行为，到episode结束一共能获得的奖励。对于 $G_t$ 是使用 MC 采样得到的 sample，只有到达最终状态才能逆序计算 $G_t$ ，这也是 REINFORCE 算法效率不高的原因，那么能不能不用等到游戏结束就可以更新参数呢？当然是可以的，那就是不再使用 MC 采样的方式来更新，而是采用的 TD 方式更新。

使用 TD 就可以每隔一步更新或者每隔两步更新，但是存在一个问题：我们如何计算每一个动作的 Q 值呢？因为没有等到episode结束就要更新参数，所以只能来估计 Q 值，很自然的就想到了使用神经网络来估计。然后就可以用参数化 的 $Q$ 来代替 $G_t$ 。所以就需要两个神经网络，此时的目标梯度如下：
$$
\nabla_\theta J(\theta) = E_{\pi_\theta} \left[\sum_{t=0}^{T-1}{Q_{\color{red}w}(s_t, a_t)}\cdot \nabla_{\color{red}\theta}\;log\;\pi_\theta(a_t|s_t) \right]
$$
这样就得到了 `Actor-Critic Policy Gradient`。把 `Value Function` 和 `Policy Function` 两者结合起来的一中算法。其包含两个成分：

- Actor：Actor 就是指的 Policy Function，是用来和环境交互，做出动作，可以理解为一个”表演者“。
- Critic：Critic 就是式子中的 Q，是一个”评论者“的角色，用来评论 actor 所做出的动作实际能得到多少价值。

我们可以把 Actor-Critic 算法比喻为：Actor在台上跳舞，一开始舞姿并不好看，Critic根据Actor的舞姿打分。Actor通过Critic给出的分数，去学习：如果Critic给的分数高，那么Actor会调整这个动作的输出概率；相反，如果Critic给的分数低，那么就减少这个动作输出的概率。

下面介绍一个最简单的 Actor-Critic 算法：Sample QAC。

### 2、Sample QAC 算法

Sample QAC 算法使用线性特征组合来逼近 ：$Q_w(s,a) = \psi(s,a)^Tw$ 。通过 TD(0) 的方式来更新 参数 $w$ ，Actor使用 policy gradient来优化：

![a9aZ8g.png](https://s1.ax1x.com/2020/07/26/a9aZ8g.png)

首先根据 策略 $\pi_\theta$ 生成一系列样本数据，然后得到TD Target 进一步计算 TD Error ，来更新 价值函数的参数 $w$ ，这里因为是线性特征组合，所以经过求导后直接取 feature（特征）来更新：$w \leftarrow w + \beta\delta\psi(s,a)$ 。然后第二部分，我们得到 $Q_w(s,a)$ 后直接乘以 `score function` 通过 policy gradient 来进行更新。然后一直重复这个步骤，就得到了 Simple QAC 算法。

### 3、通过 Baseline 减少 Actor-Critic 的方差

同样的，在 AC 算法中也可以引入 Baseline 来减少方差。一般我们都会把平均值作为 Baseline 。回忆动作价值函数 Q：$Q_{\pi,\gamma}(s,a) = E_\pi[r_1+\gamma r_2+\ldots|s_1=s,a_1=a]$ ，而状态价值 $V_{\pi,\gamma}(s)$ 就是动作状态价值 Q 的期望：
$$
\begin{aligned}
V_{\pi, \gamma}(s) & = E_\pi[r_1+\gamma_2 + \ldots|s_1=s] \\
&= E_{a～\pi}[Q_{\pi,\gamma}(s,a)]
\end{aligned}
$$
也就是  说 状态价值函数 `V` 可以天然的做 Baseline，所以就得出了更新的权重：定义 为 `Advantage function`：
$$
A_{\pi, \gamma}(s,a) = Q_{\pi, \gamma}(s,a) - V_{\pi, \gamma}(s)
$$
这样我们又需要计算 V，但是贝尔曼公式告诉我们 `Q` 和 `V` 是可以互换的。也就是 `TD-Error`：
$$
TD-Error = r + gamma * V(s') - V(s)
$$
**而 `TD-Error` 就是 `Actor` 更新时的权重值。所以 `Critic` 网络不需要估计 `Q`，而是去估计 `V`。然后就可以计算出 `TD-Error`，也就是`Advantage function` ，然后最小化 `TD-Error`**。

此时的 policy gradient：
$$
\nabla_\theta J(\theta) = E_{\pi_\theta} \left[\sum_{t=0}^{T-1} \nabla_{\theta}\;log\;\pi_\theta(a_t|s_t) {\color{red}A_{\pi, \gamma}(s_t,a_t)} \right]
$$

## 二、代码分析

### 1、更新流程

本次代码我们还是采用 `CartPole-v1` 环境，在 REINFORCE算法中，agent 需要从头一直跑到尾，直到最终状态才开始进行学习，所以采用的回合更新制。 在AC中agent 采用是每步更新的方式。如下图所示

![aKHtOJ.png](https://s1.ax1x.com/2020/07/30/aKHtOJ.png)

对于每一个 episode 流程如下，智能体每走一步都要分别更新 Critic 和 Actor 网络。注意：**我们需要先更新Critic，并计算出TD-error。再用TD-error更新Actor**。

```python
while True:
    if RENDER: env.render()
        action = self.actor.get_action(state)
        state_, reward, done, _ = env.step(action)
        
        td_error = self.critic.learn(state, reward, state_, done)
        self.actor.learn(state, action, td_error)

        state = state_
        if done or step >= MAX_STEPS:
            break
```

### 2、Critic 的更新

```python
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
```

Critic 网络通过估计当前状态和下一个状态 的 V 值，来计算 TD-Error，然后更新参数。

### 3、Actor 的更新

Actor的学习本质上就是 PG 的更新，只不过在 AC 中更新的权重变成了 TD-Error。在上一篇 介绍 REINFORCE算法文章中已经详细讲过，传送门[强化学习——REINFORCE 算法推导与代码实现](https://blog.csdn.net/november_chopin/article/details/108033013) 。在此不再赘述。

```python
def learn(self, state, action, td_error):
    with tf.GradientTape() as tape:
        _logits = self.model(np.array([state]))
        _exp_v = tl.rein.cross_entropy_reward_loss(
            logits=_logits, actions=[action], rewards=td_error)
    grad = tape.gradient(_exp_v, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
```

注意的是在这里我们直接使用了 `tensorlayer` 提供的函数 `cross_entropy_reward_loss` 。其实这个函数就是把交叉熵函数包装了一下而已：

```python
def cross_entropy_reward_loss(logits, actions, rewards, name=None):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits, name=name)
    return tf.reduce_sum(tf.multiply(cross_entropy, rewards))
```

**完整代码**请查看[Actor-Critic算法](https://github.com/NovemberChopin/RL_Tutorial/blob/master/code/AC_Discrete.py) 。还望随手一个 star ，不胜感激

## 三、AC小结

要理解 AC，就要理解 TD-Error，而TD-Error 的本质就是 $Q(s,a) - V(s)$ 得来的。其中的 $Q(s,a)$ 就是采用了TD的方法 $r + \gamma V(s')$ 得来的。Critic 网络的作用就是最小化 TD-Error。Actor 网络的功能和 REINFORCE 算法中的 策略网络功能一样，就是交叉熵带权重更新。

基本版的Actor-Critic算法虽然思路很好，但是由于难收敛的原因，仍然还可以做改进。基于AC架构的改进算法还有DDPG、TD3等，都会在以后介绍。

