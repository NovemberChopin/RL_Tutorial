# 强化学习 2—— 用动态规划求解 MDP

在上一篇文章 [强化学习 1 —— 一文读懂马尔科夫决策过程 MDP](https://blog.csdn.net/november_chopin/article/details/106589197) 介绍了马尔科夫过程，本篇接着来介绍如何使用动态规划方法来求解。

动态规划的关键点有两个：

- 一是问题的最优解可以由若干小问题的最优解构成，即通过寻找子问题的最优解来得到问题的最优解。
- 二是可以找到子问题状态之间的递推关系，通过较小的子问题状态递推出较大的子问题的状态。

在上一篇中我们提到的状态价值的贝尔曼方程：
$$
v_{\pi}(s) = \sum_{a \in A}\pi(a|s)\left( R(s, a) + \gamma \sum_{s' \in S} P_{(s'|s, a)} \cdot v_\pi(s')\right)
$$
从这个式子我们可以看出，我们可以定义出子问题求解每个状态的状态价值函数，同时这个式子又是一个递推的式子, 意味着利用它，我们可以使用上一个迭代周期内的状态价值来计算更新当前迭代周期某状态 s 的状态价价值。可见，强化学习的问题是**满足**这两个条件的。

可以把 Decision Making 分为两个过程

- **Prediction** (策略评估)
  - 输入: MDP$<S, A, P, R, \gamma>$ 和 策略 $\pi $
  - 输出：value function $V_\pi$

- **Control** (策略控制，即寻找最优策略)
  - 输入：MDP$<S, A, P, R, \gamma> $
  - 输出：最优的价值函数 $V^*$ 和最优的策略 $\pi^*$

## 一、策略评估

首先，我们来看如何使用动态规划来求解强化学习的预测问题，即求解给定策略的状态价值函数的问题。这个问题的求解过程我们通常叫做策略评估（Policy evaluation）

策略评估的基本思路是从任意一个状态价值函数开始，依据给定的策略，结合贝尔曼期望方程、状态转移概率和奖励同步迭代更新状态价值函数，直至其收敛，得到该策略下最终的状态价值函数。

假设我们在第 t 轮迭代已经计算出了所有的状态的状态价值 $v_t(s')$ ，那么在第 t+1 轮我们可以利用第 t 轮计算出的状态价值计算出第 t+1 轮的状态价值 $v_{t+1}(s)$ 。这是通过贝尔曼方程来完成的，即 **Iteration on Bellman exception backup** 
$$
V_{t+1}(s) = \sum_{a \in A}\pi(a|s) \left(R(s, a) + \gamma \sum_{s' \in S} P_{(s'|s, a)} \cdot V_t(s')\right)
$$

### 1、策略评估实例

这是一个经典的Grid World的例子。我们有一个4x4的16宫格。只有左上和右下的格子是终止格子。该位置的价值固定为0，个体如果到达了该2个格子，则停止移动，此后每轮奖励都是0。个体在16宫格其他格的每次移动，得到的即时奖励$R$都是-1。注意个体每次只能移动一个格子，且只能上下左右4种移动选择，不能斜着走, 如果在边界格往外走，则会直接移动回到之前的边界格。衰减因子我们定义为 $\gamma=1$。这里给定的策略是随机策略，即每个格子里有25% 的概率向周围的4个格子移动。

![UoQbYq.png](https://s1.ax1x.com/2020/07/21/UoQbYq.png)

首先我们初始化所有格子的状态价值为0，如上图$k=0$的时候。现在我们开始策略迭代了。由于终止格子的价值固定为 0，我们可以不将其加入迭代过程。在 $k=1$ 的时候，我们利用上面的贝尔曼方程先计算第二行第一个格子的价值：
$$
v_1^{(21)} = \frac{1}{4}[(−1+0)+(−1+0)+(−1+0)+(−1+0)] = -1
$$
第二行第二个格子的价值：
$$
v_1^{(22)} = \frac{1}{4}[(−1+0)+(−1+0)+(−1+0)+(−1+0)] = -1
$$
其他的格子都是类似的，第一轮的状态价值迭代的结果如上图 $k=1$ 的时候。现在我们第一轮迭代完了。开始动态规划迭代第二轮了。还是看第二行第一个格子的价值：
$$
v_2^{(21)} = \frac{1}{4}[(−1+0)+(−1−1)+(−1−1)+(−1−1)] = -1.75
$$
第二行第二个格子的价值是：
$$
v_2^{(22)} = \frac{1}{4}[(−1-1)+(−1−1)+(−1−1)+(−1−1)] = -2
$$
最终得到的结果是上图 $k=2$ 的时候。第三轮的迭代如下：
$$
v_3^{(21)} = \frac{1}{4}[(−1−1.7)+(−1−2)+(−1−2)+(−1+0)] = -2.425
$$
最终得到的结果是上图 $k=3$ 的时候。就这样一直迭代下去，直到每个格子的策略价值改变很小为止。这时我们就得到了所有格子的基于随机策略的状态价值。

## 二、策略控制

### 1、Policy Iteration

对于策略控制问题，一种可行的方法就是根据我们之前基于任意一个给定策略评估得到的状态价值来及时调整我们的动作策略，这个方法我们叫做策略迭代 (Policy Iteration)。

如何调整呢？最简单的方法就是贪婪法。考虑一种如下的贪婪策略：**个体在某个状态下选择的行为是其能够到达后续所有可能的状态中状态价值最大的那个状态**。如上面的图右边。当我们计算出最终的状态价值后，我们发现，第二行第一个格子周围的价值分别是0,-18,-20，此时我们用贪婪法，则我们调整行动策略为向状态价值为0的方向移动，而不是随机移动。也就是图中箭头向上。而此时第二行第二个格子周围的价值分别是-14,-14,-20, -20。那么我们整行动策略为向状态价值为-14的方向移动，也就是图中的向左向上。

- 故 Policy Iteration 共分为两个部分
  - **Evaluate** the policy $ \pi $，（通过给定的  $ \pi $ 计算 V）
  - **Improve** the policy $ \pi $，（通过贪心策略）

$$
\pi' = greedy(v^{\pi})
$$

​	如果我们有 一个 策略 $\pi$，我们可以用策略 估计出它的状态价值函数 $v_\pi(s)$, 然后根据策略改进提取出更好的策略 $\pi'$，接着再计算 $v_{\pi'}(s)$， 然后再获得更好的 策略 $\pi''$，直到相关满足相关终止条件。

<img src="https://img-blog.csdnimg.cn/20200606161030666.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25vdmVtYmVyX2Nob3Bpbg==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 67%;" />

计算公式如下：

1、评估价值 (Evaluate)
$$
v_{i}(s) = \sum_{a\in A} \pi(a|s) \left( R(s, a) + \gamma \sum_{s' \in S} P(s'|s, a) \cdot v_{i-1}(s') \right)
$$

2、改进策略（Improve）
$$
q_i(s,a) = R(s, a) + \gamma \sum_{s' \in S} P_{(s'|s,a)} \cdot v_i(s') \\
\pi_{i+1}(s) = argmax_a \; q^{\pi_i}(s,a)
$$

> 每次迭代都是基于确定的策略，故策略 $\pi$ 不再写出，对应的加上了下标

可以把 $q^{\pi_i}(s,a)$ 想象成一个表格，其中横轴代表不同的状态，纵轴代表每种状态下不同的动作产生的价值，然后在当前状态下选择一个价值最大的行为价值作为当前的状态价值。

Policy Iteration 的具体算法为：

<img src="https://img-blog.csdnimg.cn/20200606161030653.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25vdmVtYmVyX2Nob3Bpbg==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:80%;" />

**Bellman Optimality Equation**

利用状态价值函数和动作价值函数之间的关系，得到
$$
v_*(s) = max_a q_*(s,a)
$$

$$
q_*(s,a) = R(s, a) + \gamma \sum_{s' \in S} P_{s's}^a \cdot V_*(s')
$$

> 当到达 最优的时候，一个状态的价值就等于在当前 状态下最大的那个动作价值

把上面两个式子结合起来有**Bellman Optimality Equation**

$$
v_*(s) = max_a (R(s, a) + \gamma \sum_{s' \in S} P_{s's}^a \cdot v_*(s'))
$$

$$
q_*(s,a) = R(s, a) + \gamma \sum_{s' \in S} P_{s's}^a \cdot max_{a'}q_*(s', a')
$$

### 2、Value Iteration

观察第三节的图发现，我们如果用贪婪法调整动作策略，那么当 $k=3$ 的时候，我们就已经得到了最优的动作策略。而不用一直迭代到状态价值收敛才去调整策略。那么此时我们的策略迭代优化为价值迭代。

比如当 $k=2$ 时，第二行第一个格子周围的价值分别是0,-2,-2，此时我们用贪婪法，则我们调整行动策略为向状态价值为0的方向移动，而不是随机移动。也就是图中箭头向上。而此时第二行第二个格子周围的价值分别是-1.7,-1.7,-2, -2。那么我们整行动策略为向状态价值为-1.7的方向移动，也就是图中的向左向上。

我们没有等到状态价值收敛才调整策略，而是随着状态价值的迭代及时调整策略, 这样可以大大减少迭代次数。此时我们的状态价值的更新方法也和策略迭代不同。现在的贝尔曼方程迭代式子如下：
$$
v_*(s) = max_a q_*(s,a) \\
\Downarrow \\
v_{i+1}(s) \leftarrow max_{a \in A} \; \left(R(s, a) + \gamma \sum_{s' \in S} P_{(s'|s,a)} \cdot V_i(s')\right)
$$

然后直接提取最优策略 $ \pi $
$$
\pi^*(s) = argmax_a \; q^{\pi}(s,a) \\
\Downarrow \\
\pi^*(s) \leftarrow argmax_a \; \left(R(s, a) + \gamma \sum_{s' \in S} P_{(s'|s,a)} \cdot V_{end}(s')\right)
$$

Value Iteration 的算法为：

<img src="https://img-blog.csdnimg.cn/20200606161030632.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25vdmVtYmVyX2Nob3Bpbg==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:80%;" />

## 三、实例展示

​	下面是斯坦福大学的一个网页，可以帮助更好的直观的理解策略迭代和价值迭代两种不同的迭代方式。开始时如下图左上所示，其中每一个格子代表一个状态，每个状态有都有上下左右四个动作，每个状态上都有一个数字，代表当前的状态价值，其中有些状态下半部分还有一个数字，代表进入当前状态所能获得的奖励。我们的策略是四个方向的概率都相等，即各有0.25的概率。要做的就是找出每个状态下的最优策略。

### 3.1 策略迭代

​	如图右上所示，点击$Policy \; Evaluation$ 执行一次策略评估，可以看到有些状态已经发生了变化，相应的状态价值 $V(s)$ 也已经更新，此时再点击$Policy \; Updata$ 来更新策略，如图做下所示，可与i看到，有些状态的策略已经发生了变化，已经在当前的状态价值下提高了策略。如此反复迭代，最后的结果如图右下所示，此时每个状态都有最好的状态价值和策略。

![tbPeQP.png](https://s1.ax1x.com/2020/06/11/tbPeQP.png)

### 3.2 价值迭代

​	价值迭代是不停的迭代状态价值 $V$， 然后提取出相应的动作价值$q$，然后从相应的 q 中寻找一个最大的最为当前状态价值。点击 $Toggle\; Value \; Iteration$等几秒钟就可以看到迭代 结果：

<img src="https://s1.ax1x.com/2020/06/11/tbZpp4.png" alt="tbZpp4.png" style="zoom: 67%;" />
		可以看出，无论是策略迭代还是价值迭代，最后的结果都是相同的。最后附上网址：https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html

## 四、代码理解

​	gym是 [OpenAI](http://gym.openai.com/) 开放的一个开发、比较各种强化学习算法的工具库，提供了不少内置的环境，是学习强化学习不错的一个平台。我们本次使用其中提供给的一个最简单的环境 FrozenLake-v0。如下如所示，开始时agent在 s 位置，G代表金子，我们要控制 agent 找到金子的同时获得更多的奖励，与上面例子一样，agent在每个状态下有四种动作（上下左右），F代表障碍，H代表洞，要避免调入洞中，具体描述可以访问：http://gym.openai.com/envs/FrozenLake-v0/ 查看。

![tbmNOs.png](https://s1.ax1x.com/2020/06/11/tbmNOs.png)

### 4.1、Policy Iteration

```python
import numpy as np
import gym

def extract_policy(v, gamma = 1.0):
    """ 从价值函数中提取策略 """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ 计算价值函数 """
    v = np.zeros(env.env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in 
                        env.env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    # env.env.nS: 16
    # env.env.nA: 4
    # env.env.ncol / env.env.nrow: 4
    policy = np.random.choice(env.env.nA, size=(env.env.nS))  
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

if __name__ == '__main__':

    env_name  = 'FrozenLake-v0' 
    env = gym.make(env_name)

    optimal_policy = policy_iteration(env, gamma = 1.0)
    print(optimal_policy)
    # Policy-Iteration converged at step 6.
	# [0. 3. 3. 3. 0. 0. 0. 0. 3. 1. 0. 0. 0. 2. 1. 0.]
```

### 4.2、Value Iteration

```python
def extract_policy(v, gamma = 1.0):
    """ 从状态函数中提取策略 """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        # q_sa: 在状态 s 下的 所有动作价值
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        # print("q_sa: ", q_sa)
        policy[s] = np.argmax(q_sa)
        # print('np.argmax(q_sa): ', policy[s])
    return policy

def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    # env.env.nS: 16
    # env.env.nA: 4
    # env.env.ncol / env.env.nrow: 4
    v = np.zeros(env.env.nS)  
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            # env.env.P[s][a]]: 状态 s 下动作 a 的概率
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in 
                         env.env.P[s][a]]) for a in range(env.env.nA)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v

if __name__ == '__main__':
    
    env_name  = 'FrozenLake-v0' 
    # env 中是 gym 提供的本次问题的环境信息
    env = gym.make(env_name)
    gamma = 1.0

    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(optimal_v, gamma)
    print('policy:', policy)
    # Value-iteration converged at iteration# 1373.
	# policy: [0. 3. 3. 3. 0. 0. 0. 0. 3. 1. 0. 0. 0. 2. 1. 0.]
```

## 五、总结

本篇博客介绍了如何使用动态规划来求解MDP问题，还介绍了两种迭代算法。可以发现，对于这两个算法，有一个前提条件是奖励 R 和状态转移矩阵 P 我们是知道的，因此我们可以使用策略迭代和价值迭代算法。对于这种情况我们叫做 `Model base`。同理可知，如果我们不知道环境中的奖励和状态转移矩阵，我们叫做 `Model free`。那对于 `Model_free` 情况下应该如何求解 MDP 问题呢？这就是下一篇文章要讲的 蒙特卡洛（MC）采样法。

参考资料：

1、[B站周老师的强化学习纲要第二讲下](https://www.bilibili.com/video/BV1u7411m7rh)

2、博客园 [![返回主页](https://www.cnblogs.com/skins/custom/images/logo.gif)](https://www.cnblogs.com/pinard/)[刘建平Pinard](https://www.cnblogs.com/pinard/)

