# 强化学习 14 —— TD3 算法详解

上篇文章 [强化学习 13 —— DDPG算法详解](https://blog.csdn.net/november_chopin/article/details/108171030) 中介绍了DDPG算法，本篇介绍TD3算法。TD3的全称为 Twin Delayed Deep Deterministic Policy Gradient（双延迟深度确定性策略）。可以看出，TD3就是DDPG算法的升级版，所以如果了解了DDPG，那么TD3算法自然不在话下。

## 一、偏差与方差

在介绍TD3算法之前 ，先搞清楚偏差(bais)和方差(variance)。看下面这张图，可以类比打靶，其中红色部分是靶心，偏差(bais)就是机器学习模型的输出与真实样本的差异，

![a2H9yj.png](https://s1.ax1x.com/2020/08/06/a2H9yj.png)

<img src="https://s1.ax1x.com/2020/08/06/a2HplQ.png" alt="a2HplQ.png" style="zoom:80%;" />

## 二、算法介绍

TD3算法主要对DDPG做了三点改进，将会在下面 一一讲解，两者的代码也很相似，这里只展示改进的部分，如果对DDPG算法不太熟悉可以参考上一篇博客[强化学习 13——DDPG算法详解与实战](https://blog.csdn.net/november_chopin/article/details/108171030) 。

完整的TD3算法代码地址[强化学习——TD3算法代码地址](https://github.com/NovemberChopin/RL_Tutorial/blob/master/code/TD3.py) 还望随手一个 `star`，再此不胜感激

### 1、双 Critic 网络

我们知道，DDPG源于DQN，而DQN源于Q-Learning，这些算法都是通过估计Q值来寻找最优的策略，在强化学习中，更新Q网络的目标值target为：$y=r+\gamma\;max_{a'}Q(s',a')$ ，因为样本存在 噪声 $\epsilon$，所以真实情况下，有误差的动作价值估计的最大值通常会比真实值更大：
$$
E_\epsilon[max_{a'}(Q(s',a')+\epsilon)] \geq max_{a'}Q(s',a')
$$
这就就不可避免的降低了估值函数的准确度，由于估值方法的计算依据贝尔曼方程，即使用后续状态对估值进行更新，这种性质又加剧了精确度的下降。在每一次更新策略时，使用一个不准确的估计值将会导致错误被累加。这些被累加的错误会导致某一个不好的状态被高估，最终导致策略无法被优化到最优，并使算法无法收敛。

在DQN算法中针对Q值过估计的问题采用的是利用双网络分别实现动作的选择和评估，也就是DDQN算法。在TD3算法中，我们也使用 两个 Critic 网络来评估 Q 值，然后选取较小的那个网络的Q值来更新，这样就可以缓解Q值高估现象。这样或许会导致些许低估，低估会导致训练缓慢，但是总比高估要好得多。

注意：这里我们使用了两个 Critic 网络，每个Critic网络都有相应的 Target 网络，可以理解为这是两套独立的 Critic 网络，都在对输入的动作进行评估，然后通过 `min()` 函数求出较小值作为更新目标。所以TD3算法一共用到 6 个网络。

代码实现：

```python
self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
```

如上所示，包含两套Q网络，用来估计Q值，一套策略网络。具体的网络更新部分和DDPG是流程是一样的，唯一不同的是两个 Critic 网络算出 Q 值后，选取最小值去计算目标值：

```python
target_q_min = tf.minimum(self.target_q_net1(target_q_input), 		               self.target_q_net2(target_q_input))
target_q_value = reward + (1 - done) * gamma * target_q_min
```

然后就是分别对Critic 网络和policy 网络进行更新。

### 2、延迟 Actor 网络更新

TD3中使用的第二个技巧就是对Policy进行延时更新。在双网络中，我们让target网络与当前网络更新不同步，当前网络更新 d 次之后在对target网络进行更新（复制参数）。这样就可以减少积累误差，从而降低方差。同样的我们也可以policy网络进行延时更新，因为actor-critic方法中参数更新缓慢，进行延时更新一方面可以减少不必要的重复更新，另一方面也可以减少在多次更新中累积的误差。在降低更新频率的同时，还应使用软更新：
$$
\theta'\leftarrow \tau\theta + (1-\tau)\theta'
$$
关于 policy 网络延时更新的实现也很简单，只需要一个`if` 语句就可以实现

```python
if self.update_cnt % self.policy_target_update_interval == 0
```

其中`update_cnt` 是更新的次数，`policy_target_update_interval` 是policy 网络更新的周期，每当 critic 更新了一定次数后，再更新 policy 网络。

### 3、目标策略的平滑正则化 Target Policy Smoothing Regularization

上面我们通过延时更新policy来避免误差被过分累积，接下来我们我们再思考能不能把误差本身变小呢？那么我们首先就要弄清楚误差的来源。

误差的根源是值函数估计产生的偏差。知道了原因我们就可以去解决它，在机器学习中消除估计的偏差的常用方法就是对参数更新进行正则化，同样的，我们也可以将这种方法引入强化学习中来：

在强化学习中一个很自然的想法就是：**对于相似的action，他们应该有着相似的value。**

所以我们希望能够对action空间中target action周围的一小片区域的值能够更加平滑，从而减少误差的产生。paper中的做法是对target action的Q值加入一定的噪声 $\epsilon$ ：
$$
y = r + \gamma\;Q_{\theta'}(s', \pi_{\phi'}(s')+\epsilon)\;\; \epsilon ～clip(N(0,\sigma), -c, c)
$$
这里的噪声可以看作是一种正则化方式，这使得值函数更新更加平滑。

代码实现：

```python
def evaluate(self, state, eval_noise_scale):
    state = state.astype(np.float32)
    action = self.forward(state)
    action = self.action_range * action
    # add noise
    normal = Normal(0, 1)
    noise = normal.sample(action.shape) * eval_noise_scale
    eval_noise_clip = 2 * eval_noise_scale
    noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)
    action = action + noise
    return action
```

如代码所示，给动作加上噪音这部分在策略策略网络评估部分实现，`evaluate()`函数有两个参数，`state` 是输入的状态，参数 `eval_noise_scale` 用于调节噪声的大小。可以看到，首先经过前向计算得到 输出的动作 `action` 。下面详细说下如何给动作加上噪音：首先我们构造一个正太分布，然后根据动作的形状进行取样`normal.sample(action.shape)` ，然后乘以参数 `eval_noise_scale` 实现对噪音进行缩放，为了防止抽出的噪音很大或者很小的情况，我们对噪音进行剪切，范围相当于两倍的`eval_noise_scale` 。最后把噪音加到`action`上并输出。

**算法伪代码：**

![aRZ9c4.png](https://s1.ax1x.com/2020/08/06/aRZ9c4.png)



参考：

https://zhuanlan.zhihu.com/p/55307499

https://zhuanlan.zhihu.com/p/86297106?from_voters_page=true

