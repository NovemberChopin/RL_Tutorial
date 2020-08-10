# 强化学习 8 —— DQN 代码 Tensorflow 实现

在上一篇文章[强化学习——DQN介绍](https://blog.csdn.net/november_chopin/article/details/107912720) 中我们详细介绍了DQN 的来源，以及对于强化学习难以收敛的问题DQN算法提出的两个处理方法：经验回放和固定目标值。这篇文章我们就用代码来实现 DQN 算法

## 一、环境介绍

### 1、Gym 介绍

本算法以及以后文章要介绍的算法都会使用 由 $OpenAI$ 推出的[$Gym$](http://gym.openai.com/)仿真环境， $Gym$ 是一个研究和开发强化学习相关算法的仿真平台，了许多问题和环境（或游戏）的接口，而用户无需过多了解游戏的内部实现，通过简单地调用就可以用来测试和仿真，并兼容常见的数值运算库如 $TensorFlow$ 。

```python
import gym
env = gym.make('CartPole-v1')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```

运行结果如下：

<img src="https://s1.ax1x.com/2020/07/31/aMXZ7Q.gif" alt="aMXZ7Q.gif" style="zoom:80%;" />

以上代码中可以看出，`gym`的核心接口是`Env`。作为统一的环境接口，`Env`包含下面几个核心方法：

- `reset(self)`：重置环境的状态，返回观察。如果回合结束，就要调用此函数，重置环境信息
- `step(self, action)`：执行动作 `action` 推进一个时间步长，返回`observation`,` reward`, `done`, `info`。
  - `observation`表示环境观测，也就是`state`
  - `reward` 表示获得的奖励
  - `done`表示当前回个是否结束
  - `info` 返回一些诊断信息，一般不经常用
- `render(self, mode=‘human’, close=False)`：重新绘制环境的一帧。
- `close(self)`：关闭环境，并清除内存。

以上代码首先导入`gym`库，第2行创建`CartPole-v01`环境，并在第3行重置环境状态。在  for 循环中进行*1000*个时间步长(*timestep)的控制，第5行刷新每个时间步长环境画面，第6行对当前环境状态采取一个随机动作（0或1），最后第7行循环结束后关闭仿真环境。

### 2、CartPole-v1 环境介绍

CartPole 是gym提供的一个基础的环境，即车杆游戏，游戏里面有一个小车，上有竖着一根杆子，每次重置后的初始状态会有所不同。小车需要左右移动来保持杆子竖直，为了保证游戏继续进行需要满足以下两个条件：

- 杆子倾斜的角度 $\theta$ 不能大于15°
- 小车移动的位置 x 需保持在一定范围（中间到两边各2.4个单位长度）

对于 `CartPole-v1` 环境，其动作是两个离散的动作左移（0）和右移（1），环境包括小车位置、小车速度、杆子夹角及角变化率四个变量。如下代码所示：

```python
import gym
env = gym.make('CartPole-v0')
print(env.action_space)  # Discrete(2)
observation = env.reset()
print(observation)  # [-0.0390601  -0.04725411  0.0466889   0.02129675]
```

下面以`CartPole-v1` 环境为例，来介绍 DQN 的实现

## 二、代码实现

### 1、经验回放池的实现

```python
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
        return state, action, reward, next_state, done
```

首先定义一个经验回放池，其容量为 10000，函数 `push` 就是把智能体与环境交互的到的信息添加到经验池中，这里使用的循环队列的实现方式，注意 `position` 指针的运算。当需要用数据来更新算法 时，使用 `sample` 从经验队列中随机挑选 一个 `batch_size` 的数据，使用 zip 函数把每一条数据打包到一起：

```python
zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)]
```

然后对每一列数据使用 stack 函数转化为列表后返回

### 2、网络构造

本系列强化学习的代码，都是使用的 `tensorlayer` ，就是对 `tensorflow` 做了一些封装，使其更加易用，重点是还**专门为强化学习**内置了一些接口，下面是[官网](https://tensorlayercn.readthedocs.io/zh/latest/)介绍：

> TensorLayer 是为研究人员和工程师设计的一款基于Google TensorFlow开发的深度学习与强化学习库。 它提供高级别的（Higher-Level）深度学习API，这样不仅可以加快研究人员的实验速度，也能够减少工程师在实际开发当中的重复工作。 TensorLayer非常易于修改和扩展，这使它可以同时用于机器学习的研究与应用。

定义网络模型：

```python
def create_model(input_state_shape):
    input_layer = tl.layers.Input(input_state_shape)
    layer_1 = tl.layers.Dense(n_units=32, act=tf.nn.relu)(input_layer)
    layer_2 = tl.layers.Dense(n_units=16, act=tf.nn.relu)(layer_1)
    output_layer = tl.layers.Dense(n_units=self.action_dim)(layer_2)
    return tl.models.Model(inputs=input_layer, outputs=output_layer)

self.model = create_model([None, self.state_dim])
self.target_model = create_model([None, self.state_dim])
self.model.train()
self.target_model.eval()
```

可以看到`tensorlayer` 使用起来与`tensorflow` 大同小异，只要有`tensorflow`基础一眼就能明白，在上面代码中我们定义一个函数用来生成网络模型。然后创建一个当前网络`model`和一个目标网络`target_model` ，我们知道DQN中的目标网络是起到一个“靶子”的作用，用来评估当前的 target 值，所以我们把它设置为评估模式，调用 `eval()` 函数即可。而 `model` 网络是我们要训练的网络，调用函数 `train()` 设置为训练模式。

### 3、算法控制流程

```python
for episode in range(train_episodes):
    total_reward, done = 0, False
    while not done:
        action = self.choose_action(state)
        next_state, reward, done, _ = self.env.step(action)
        self.buffer.push(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        # self.render()
    if len(self.buffer.buffer) > args.batch_size:
        self.replay()
        self.target_update()
```

关于与环境交互过程在上面已经介绍过了，这里重点看 第 10 行的 if 语句，当经验池的长度大于一个`batch_size` 时，就开始调用`replay()` 函数来更新网络 `model` 的网络参数，然后调用`target_update()` 函数把 `model` 网络参数复制给 `target_model` 网络。

### 4、网络参数更新

```python
def replay(self):
    for _ in range(10):
        states, actions, rewards, next_states, done = self.buffer.sample()
        # compute the target value for the sample tuple
        # target [batch_size, action_dim]
        # target represents the current fitting level
        target = self.target_model(states).numpy()
        next_q_value = tf.reduce_max(self.target_model(next_states), axis=1)
        target_q = rewards + (1 - done) * args.gamma * next_q_value
        target[range(args.batch_size), actions] = target_q

        with tf.GradientTape() as tape:
            q_pred = self.model(states)
            loss = tf.losses.mean_squared_error(target, q_pred)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model_optim.apply_gradients(zip(grads, self.model.trainable_weights))
```

这部分应该就是 DQN 的核心代码了，在`replay()` 函数中，我们循环更新更新当前网络十次，目的就是改变两个网络的更新频率，有利于网络收敛。

具体的更新部分：我们知道，DQN就是把Q-Learning中的Q表格换成了神经网络，两者之间有很多 相似之处，我们可以类比Q-Learning 的更新方式。对于Q表格形式，我们获取某一个状态的动作价值Q是直接通过下标得到的，那么在神经网络中就需要把状态输入神经网络，经过前向计算得到。
$$
\Delta w = \alpha (r + \gamma\;max_{a'}\; \hat{Q}(s', a', w) - \hat{Q}{(s, a, w)})\cdot \nabla_w\hat{Q}{(s, a, w)}
$$
第三行首先获取一个`batch_size`的数据，这个过程称为 `sample` 。第7行我们首先获取当前的动作价值，target 表示的是根据当前的网络参数计算得到的动作价值。然后第8行先获取当前网络参数下 的下一个状态的所有动作，然后使用`reduce_max()` 函数找出最大的动作价值。然后第9行和第10行利用下一个状态最大的动作价值来计算出 `target_q` ，也就是 $r + \gamma\;max_{a'}\; \hat{Q}(s', a', w)$ 部分，然后更新`target` 。注意上面我们计算target时一直在使用 `target_model` 网络，target网络只有在评估网络状态时才使用。

接着我们使用 `q_pred = self.model(states)` 网络获取当前 网络的状态，也就是 公式中的 $\hat{Q}{(s, a, w)}$ ，利用MSE函数计算其损失函数，最后更新 `model` 网络。

完整代码请参考[强化学习——DQN代码地址](https://github.com/NovemberChopin/RL_Tutorial/blob/master/code/DQN.py) 还请给个 `star` ，谢谢各位了

## 三、DQN 小结

虽然 DQN 提出的这两个解决方案不错，但是仍然还有问题没有解决，比如：

-  目标 Q 值（Q Target ）计算是否准确？全部通过 $max\;Q$ 来计算有没有问题？
-  Q 值代表动作价值，那么单纯的动作价值评估会不会不准确？

对应第一个问题的改进就是 Double DQN ，第二个问题的改进是 Dueling DQN。他们都属与DQN的改进版，我们下篇文章介绍。