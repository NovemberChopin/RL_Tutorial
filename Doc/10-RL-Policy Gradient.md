# 强化学习 10 —— Policy Gradient 推导

前面几篇文章[价值函数近似](https://blog.csdn.net/november_chopin/article/details/107911868)、[DQN算法](https://blog.csdn.net/november_chopin/article/details/107912720)、[DQN改进算法DDQN和Dueling DQN](https://blog.csdn.net/november_chopin/article/details/107913317)我们学习了 DQN 算法以及其改进算法 DDQN 和 Dueling DQN 。他们都是对价值函数进行了近似表示，也就是 学习价值函数，然后从价值函数中提取策略，我们把这种方式叫做 Value Based。

## 一、Value Based 的不足

回顾我们的学习路径，我们从动态规划到蒙地卡罗，到TD到Qleaning再到DQN，一路为计算Q值和V值绞尽脑汁。但大家有没有发现，我们可能走上一个固定的思维，就是我们的学习，一定要算Q值和V值，往死里算。但算Q值和V值并不是我们最终目的呀，我们要找一个策略，能获得最多的奖励。除了这种方法之外，还有一类强化学习算法，就是 Policy Based 算法。

Value Based 强化学习方法在很多领域得到比较好的应用，但是其也有局限性。

- 1）首先就是对连续动作处理能力不足，算法 DQN 我们使用的 CartPole-v1 环境，在这个环境中只有两个动作：控制小车向左或者向右，这就是离散动作。那连续动作就是动作不光有方向，而且还有大小，对小车施加的力越大，小车的动作幅度也会越大。例如自动驾驶控制方向盘，还请读者自行体会。这个时候，使用离散的方式是不好表达的，而使用基于 Policy Based 方法就很容易。

- 2）无法解决随机策略（Stochastic Policy）问题。随机性策略就是把当前状态输入网络，输出的是不同动作的概率分布（比如 40% 的概率向左，60% 的概率向右）或者是对于连续动作输出一个高斯分布。而基于 Value Based 的算法是输出是每个动作的动作价值 $Q(s,a)$ ，然后选择一个价值最大的动作，也就是说输出的是一个确定的动作，这种我们称之为确定性策略（Deterministic Policy）。但是有些问题的最优策略是随机策略，所以此时 基于 Value Based 的方法就不再适用，这时候就可以适用基于 Policy Based 的方法。

![UvSaF0.png](https://s1.ax1x.com/2020/07/24/UvSaF0.png)

## 二、策略梯度

### 1、优化目标

类比于我们近似价值函数的过程：$\hat{q}(s, s, w) \approx q_\pi(s, a)$ 。对于 Policy Based 强化学习方法下，我们可以适用同样的方法对策略进行近似，给定一个使用参数 $\theta$ 近似的 $\pi_\theta(s,a)$ ，然后找出最好的 $\theta$ 。

那么我们如何评估策略 $\pi_\theta$ 的好坏呢？也就是我们的优化目标是什么呢？

对于离散动作的环境我们可以优化初始状态收获的期望：
$$
J_1(\theta) = V_{\pi_\theta}(s_1) = E_{\pi_\theta}[v_1]
$$
对于那些没有明确初始状态的连续环境，我们可以优化平均价值：
$$
J_{avV}(\theta) = \sum_sd_{\pi_{\theta}}(s)V_{\pi_\theta}(s)
$$
或者定义为每一段时间步的平均奖励：
$$
J_{avR}(\theta) = \sum_sd_{\pi_\theta}(s)\sum_a\pi_\theta(s,a)R(s,a)
$$

> 其中 $d_{\pi_\theta}(s)$ 是基于策略 $\pi_\theta$ 生成的马尔科夫链关于状态 的静态分布

无论我们采用上述哪一种优化 方法，最终对 $\theta$ 求导的梯度都可以表示为：
$$
\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta\;log\pi_\theta(s,a)R(s,a)]
$$

### 2、Score Function

现在假设策略 $\pi_\theta$ 是可导的，我们现在就来计算策略梯度 $\nabla_\theta \pi_\theta(s,a)$ 。在这里我们可以用到一个 叫做 `likelihood ratio` 的小技巧：
$$
\nabla_\theta \pi_\theta(s,a) = \pi_\theta(s,a)\frac{\nabla_\theta \pi_\theta(s,a)}{\pi_\theta(s,a)} \\
= \pi_\theta(s,a) \cdot \nabla_\theta log\pi_\theta(s,a)
$$
这里的 $\nabla_\theta log\pi_\theta(s,a)$ 就叫做 `score function` 。

下面介绍策略函数的形式，对于离散动作一般就是 Softmax Policy。关于 Softmax 函数，相信学过深度学习应该都比较了解，这里不在赘述。对于连续动作环境就是 Gaussian Policy。下面分别介绍

#### 1）Softmax Policy

我们把策略用线性特征组合的方式表示为：$\phi(s,a)^T\theta$ 。此时有：
$$
\pi_\theta(s,a) = \frac{exp^{\phi(s,a)^T\theta}}{\sum_{a'}exp^{\phi(s,a)^T\theta}}
$$
此时的 `score function` 为：
$$
\begin{aligned}
\nabla_\theta log\pi_\theta(s,a) & = \nabla_\theta \left[log\;exp^{\phi(s,a)^T\theta}-\sum_{a'}log\;exp^{\phi(s,a)^T\theta}\right] \\
& = \nabla_\theta\left[\phi(s,a)^T\theta - \sum_{a'}\phi(s,a)^T\theta\right] \\
& = \phi(s,a) - \sum_{a'}\phi(s,a)
\end{aligned}
$$

#### 2）Gaussian Policy

在连续动作空间中，一般使用 Gaussian policy 。其中的均值是特征的线性组合：$\mu(s) = \phi(s)^T\theta$ 。方差可以固定为 $\sigma^2$ 也可以作为一个参数。那么对于连续动作空间有：$a\;～ \;N(\mu(s), \sigma^2)$ 。

此时的 `score function` 为：
$$
\begin{aligned}
\nabla_\theta log\pi_\theta(s,a) & = \nabla_\theta log \left[ \frac{1}{\sqrt{2\pi}\sigma} exp \left(-\frac{(a-\mu(s))^2}{2\sigma^2}\right) \right] \\
& = \nabla_\theta \left[ log\;\frac{1}{\sqrt{2\pi}\sigma} - log\; exp \left(\frac{(a-\mu(s))^2}{2\sigma^2}\right) \right] \\
& = - \nabla_\theta\left(\frac{(a-\phi(s)^T\theta)^2}{2\sigma^2}\right) \\
& = \frac{(a-\mu(s))\phi(s)}{\sigma^2}
\end{aligned}
$$

### 3、Policy Gradient 推导

对于 Policy Gradient 的求导应该来说是比较复杂的，重点是理解对于一条轨迹如何表示，弄明白符号所代表的含义，推导过程也不是那么复杂。下面我们正式开始 Policy Gradient 的推导，我们首先推导对于一条 MDP 轨迹内的 Policy Gradient，然后再推导有多条轨迹的情况。

#### 1）Policy Gradient for One-Step MDPs

我们开始的状态 s 有：$s\;～\;d(s)$ 。对于这段时间的奖励 为 $r = R(s,a)$ 。

> 上文有提到其中 $d_{\pi_\theta}(s)$ 是基于策略 $\pi_\theta$ 生成的马尔科夫链关于状态 的静态分布

$$
J(\theta) = E_{\pi_\theta}[r] = \sum_{s\in S}d(s)\sum_{a\in A}\pi_\theta(s,a)\cdot r
$$

那么优化目标就是使奖励尽可能的大，然后使用 `likelihood ratio` 计算 Policy Gradient ：
$$
\begin{aligned}
\nabla J(\theta) & = \sum_{s\in S}d(s)\sum_{a\in A}{\color{red}\pi_\theta(s,a)\; \nabla_\theta log\;\pi_{\theta}(s,a)}\cdot r \\
& =E_{\pi_\theta}[\nabla_\theta log\;\pi_{\theta}(s,a)\cdot r]
\end{aligned}
$$

#### 2）Policy Gradient for Multi-step MDPs

下面介绍在多条 MDP 轨迹的情况，假设一个状态轨迹如下表示：
$$
\tau = (s_0, a_0, r_1,\ldots,s_{T-1}, a_{T-1}, r_T, s_T)\;～\;(\pi_\theta,P(s_{t+1}|s_t, a_t))
$$

> 这里我们把按照策略 $\pi_\theta$ 的状态轨迹表示为概率 P 的形式

我们一条轨迹下所获得的奖励之和表示为：$R(\tau) = \sum_{t=0}^TR(s_t, a_t)$ 。那么**轨迹的奖励期望**就是我们的优化目标。还可以把获得的奖励写成加和的形式：如果我们能表示出每条轨迹发生的概率 $P(\tau;\theta)$，那么每条轨迹的奖励期望就等于**轨迹获得奖励乘以对应轨迹发生的概率**：
$$
J(\theta) = E_{\pi_\theta}\left[\sum_{t=0}^TR(s_t, a_t)\right] = \sum_\tau P(\tau;\theta)R(\tau)
$$
此时的最优参数 $\theta^*$ 可以表示为：
$$
\theta^* = arg\;max_\theta J(\theta) = arg\;max \sum_\tau P(\tau;\theta)R(\tau)
$$
下面就来求导 $J(\theta)$ ：
$$
\begin{aligned}
\nabla_\theta J(\theta) & = \nabla_\theta \sum_\tau P(\tau;\theta)R(\tau) \\
& = \sum_\tau P(\tau;\theta)\frac{\nabla_\theta(\tau;\theta)}{P{(\tau;\theta)}} R(\tau) \\
& = \sum_\tau P(\tau;\theta) \nabla_\theta log\;P(\tau;\theta) \cdot R(\tau)
\end{aligned}
$$
这里$\nabla_\theta J(\theta)$ 是基于**轨迹**来表示的，其实轨迹 $\tau$ 的分布我们是不知道的。所以我们可以采用 MC 采样的方法来近似表示，假如我们采集到 m 条轨迹，那么我们就可以把这 m 条奖励和取平均，就得到对优化目标的近似求导结果：
$$
\color{red}\nabla_\theta J(\theta) \approx \frac{1}{m}\sum_{i=1}^mR(\tau_i)\;\nabla_\theta log\;P(\tau_i;\theta)
$$
那一条轨迹发生的概率 $P(\tau;\theta)$ 如何表示呢？若一条轨迹图如下所示，可以看到就是是一系列的概率连乘，（注意，图中下标从 1 开始，而上面公式下标从 0 开始）

![UxSJb9.png](https://s1.ax1x.com/2020/07/24/UxSJb9.png)

那么 $P(\tau;\theta)$ 就可以表示为：
$$
P(\tau;\theta) = \mu(s_0) \prod_{t=0}^{T-1}\pi_\theta(a_t|s_t)\cdot p(s_{t+1}|s_t,a_t)
$$
现在我们就把 $\nabla_\theta J(\theta)$ 中的将轨迹分解为状态和动作：
$$
\begin{aligned}
\nabla_\theta log\;P(\tau;\theta) & = \nabla_\theta log\left[P(\tau;\theta) = \mu(s_0) \prod_{t=0}^{T-1}\pi_\theta(a_t|s_t)\cdot p(s_{t+1}|s_t,a_t) \right] \\
& = \nabla_\theta\left[log\;\mu(s_0) + \sum_{t=0}^{T-1}log\;\pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1}log\;p(s_{t+1}|s_t,a_t) \right] \\
& = \sum_{t=0}^{T-1}\nabla_\theta\;log\;\pi_\theta(a_t|s_t)
\end{aligned}
$$
对于一连串的连乘形式，把 `log` 放进去就会变成连加的形式，可以看到，上面式子只有中间一项和参数 $\theta$ 有关，使得最终结果大大简化。这也是为何使用 `likelihood ratio`  的原因，这样就可以消去很多无关的变量。然后就得到了优化目标的最终求导结果：
$$
\color{red}\nabla_\theta J(\theta) \approx \frac{1}{m}\sum_{i=1}^mR(\tau_i)\;\sum_{t=0}^{T-1}\nabla_\theta\;log\;\pi_\theta(a_t^i|s_t^i)
$$

对于当前的目标函数 梯度，是基于MC采样得到的，会有比较高的方差 ，下一篇文章将会介绍两种减小方差的方法，以及 Policy Gradient 基础的算法 `REINFORCE` 

