# RL_Tutorial
这是楼主强化学习的笔记，也可以在[博客](https://blog.csdn.net/november_chopin/category_10080614.html?spm=1001.2014.3001.5482)查看。

#### 1、下载

```
git clone git@github.com:NovemberChopin/RL_Tutorial.git
```
推荐环境：
- tensorflow: 2.2.0
- tensorlayer: 2.2.3
- tensorflow-probability: 0.6.0

#### 2、教程

`Doc`目录为强化学习教程，博客上就是这部分内容，也算是一个强化学习简明教程吧，对于传统的强化学习部分有详细的介绍，以及公式推导。后面就是常用的强化学习算法，最新的算法比如 `SAC` 和 `PPO` 暂时没有包括在内，希望后续能补充上。

#### 3、代码

`code` 目录为一些算法的实现，一个算法一个文件，比较清晰。
代码结构应该是比较容易理解的，难懂的地方都做了注释。

代码使用 `tensorlayer` 这个框架写的，就是对 `tensorflow` 中的 `layer` 进行了一些包装。使得更加方便与易用。如果能看懂  `tensorflow` ，那么 `tensorlayer` 绝对不在话下，几乎没有学习成本，所以大可不必担心。另外 `tensorlayer` 为强化学习提供了一些API，使得编写强化学习算法更加方便。这是[官方文档](https://tensorlayercn.readthedocs.io/zh/latest/index.html)介绍：

>TensorLayer 是为研究人员和工程师设计的一款基于Google TensorFlow开发的深度学习与强化学习库。 它提供高级别的（Higher-Level）深度学习API，这样不仅可以加快研究人员的实验速度，也能够减少工程师在实际开发当中的重复工作。 TensorLayer非常易于修改和扩展，这使它可以同时用于机器学习的研究与应用。 此外，TensorLayer 提供了大量示例和教程来帮助初学者理解深度学习，并提供大量的官方例子程序方便开发者快速找到适合自己项目的例子。 更多细节请点击 [这里](https://github.com/tensorlayer/tensorlayer) 。

注意：代码默认都是以测试模式运行的，程序开始时会读取 `model` 文件夹下相应的模型参数。

- 如果在 `pyCharm` 中运行 `main` 函数或者如下命令行 运行是默认测试模式的。

```
>>python ./code/DQN.py
```

如果想要训练模式，可以把代码中参数`--train` 的默认值改为 `True`。

```python
parser.add_argument('--train', dest='train', default=False)
```



- 第二种方法是命令行运行时候传入参数`--train=True` ，如下

```
(tf2.X) D:\Algorithm\RL_Tutorial>python ./code/DQN.py --train=True
```

这样就是训练模式了。另外每次训练完后程序都会把参数自动保存。

#### 4、后记

由于水平有限，教程中难免会有错误，还请大家多多指出。
