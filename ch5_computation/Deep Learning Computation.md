# Deep Learning Computation

[TOC]

随着深度学习的发展，为了开发时代码的稳定性、简洁性，研究人员封装了许多易用的高级API，这些灵活的开源库使得人们可以快速开发出一个模型原型，不用进行一些底层的重复性的工作。

慢慢地，深度学习的库越来越粗糙抽象，现在研究人员已经从单个神经元的行为转变为从层的角度构思网络，在设计架构时考虑的是更粗糙的块 `block` 。

## 层和块

- 回顾之前的学习

  - 单一输出的线性模型：（1）接受一些输入，可以多维度（2）生成相应的 **标量** 输出（3）具有一组相关参数（权重 $\bf w$ 与偏置 $b$）
  - 具有多个输出的神经网络：（1）接受一组输入（2）生成相应的一组输出（3）由一组可调整的参数描述
  - 多层感知机 `MLP` ：整个模型及其组成层都是上述架构，（1）接受原始输入 `feature` （2）生成相应输出 `predict` （3）包含一些参数 `parameters` 。同时每个中间层的输入由上一层提供，输出作为下一层的输入，并且具有一组可调参数，通过下一层的反向传播梯度更新。

- 层 <= 块 <= 模型：事实证明，“比单个层大”但“比整个模型小”的组件更有价值。复杂的网络通过 `block` 能够有更加简洁易懂的表示，并且多个块也能组合成更大的块（通常通过递归实现）

- `block` 

  - 可以描述（1）单个层（2）多个层组成的组件（3）整个模型本身
  - **必须** 由类 `class` 表示，任何一个子类都 **必须** 定义一个将输入转换成输出的前向传播函数 `forward`，并且 **必须** 存储需要的参数（有些 `block` 不需要参数）。为了计算梯度，`block` **必须** 具有反向传播函数 `backward` ，在自定义块时，由于自动微分 `autograd` ，只需考虑 **前向传播** 和**必需的参数** 即可

- `nn.Sequential` 类

  - `torch` 提供的 `block` ，定义了一种特殊的 `Module` ，它维护了一个由 `Module` 组成的有序列表。一个调用例子如下：

    ```python
    # 使用nn.Sequetial实现MLP
    import torch
    from torch import nn
    from d2l import torch as d2l
    
    net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
    X = torch.rand(2,20)
    net(X)
    ```

    这里调用 `net(X)` 来实现前向传播，获得模型输出，实际上是 `net.__call__(X)` 的简写，它将列表的每个块连接在一起，每个块的输出作为下一个块的输入。

- 自定义 `block` 

  - 每个块必须提供的基本功能：
    - 将输入数据作为其前向传播函数的参数
    - 通过前向传播函数生成输出，输出形状可与输入形状不同
    - 计算输入关于输出的梯度，通过反向传播函数访问，一般是自动进行的
    - 存储和访问前向传播计算所需参数，如 $\bf w$ $b$ 
    - 根据需要初始化模型参数

  ```python
  class MLP(nn.Module):
      # 用模型声明参数层。这里声明两个全连接的层
      def __init__(self):
          super().__init__()
          self.hidden = nn.Linear(20,256) # hidden layer
          self.out = nn.Linear(256,10) # output layer
          
      # 定义模型的前向传播，及如何根据输入X返回模型所需输出
      def forward(self,X):
          # 这里使用ReLU的函数版本
          return self.out(F.relu(self.hidden(X)))
  ```

  前向传播 `forward` 函数以 `X` 作为输入，计算带有激活函数的隐藏表示，并输出未经规范的值。

  块的主要优点在于它的 **多功能性** ，可以子类化块以创建（1）层（2）整个模型（3）中等复杂度的各种组件。在后续的 `CNN` 章节中也会利用到块。

- 顺序块

  定义一个属于自己的 `Sequetial`

  我们需要把其他模块 `Module` 按序串联起来，需要定义两个 **关键函数** ：

  - 将块逐个追加到列表的函数
  - 前向传播函数，将输入按追加块的顺序传递给组成的 `block chain` 

  ```python
  class MySequential(nn.Module):
      def __init__(self,*args):
          super().__init__()
          for idx,module in enumerate(args):
              # module是Module子类的一个实例
              # 变量 _modules中，module的类型是 OrderedDict
              self._modules[str(idx)] = module
              
      def forward(self,X):
          # OrderedDict保证了按照成员的添加顺序来遍历
          for block in self._modules.values():
              X = block(X)
          return X
  ```

  `__init__` 将每个模块添加到有序字典中。
  之所以每个 `Module` 都有一个 `_modules` 属性，而不是自己定义一个列表，是因为在模块参数初始化的过程中，系统知道去 `_modules` 字典中查找需要初始化参数的子块

- 在前向传播函数中执行代码

  当需要更强的灵活性时，需要自定义块，如（1）在前向传播时执行Python的控制流，包括条件、循环等（2）执行任意的数学运算，而不是简单依赖预定义的网络层

  合并 *常数参数* ，它既不是上一层的输出，也不是可以更新的项。

  ```python
  class FixedHiddenMLP(nn.Module):
      def __init__(self):
          super().__init__()
          # 常量参数，不计算梯度，不会更新，训练期间保持不变
          self.rand_weight = torch.randn(20,20,requires_grad=False)
          self.linear = nn.Linear(20,20)
      
      def forward(self,X):
          X = self.linear(X)
          # 使用创建的常量参数以及relu和mm函数
          X = F.relu(torch.mm(X,self.rand_weight) + 1)
          # 复用全连接层，相当于两个全连接层共享参数
          X = self.linear(X)
          # Python控制流
          while X.abs().sum() > 1:
              X /= 2
          return X.sum()
  ```

  `self.rand_weight` 是一个常量参数，不会计算梯度。此外代码还实现了前向传播中执行Python控制流。

- 混搭组合块

  ```python
  class NestMLP(nn.Module):
      def __init__(self):
          super().__init__()
          self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU())
          self.linear = nn.Linear(32,16)
          
      def forward(self,X):
          return self.linear(self.net(X))
  chimera = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
  chimera(X)
  ```

- 效率

  Python的 **全局解释器锁** 

  在深度学习环境中，我们会担心速度极快的GPU要等到CPU运行Python代码后才能运行另一个作业



## 参数管理 Parameters

选择完模型架构后，就进入了训练阶段。训练的目标是为了得到性能好的模型，而所谓的“好”，关键就在于参数的值，需要通过训练使参数能够最小化损失，而这些参数也将被保存，用于复用和预测。

**代码较多，并且需要结合代码理解，所以笔记在 `Parameters` notebook里**



## 延后初始化 deferred_init

之前的参数初始化只是针对权重 $\bf w$ 和偏置项 $\bf b$ 的初始化，但是忽略了如下参数

- 输入维度
- 前一层的输出维度
- 模型要包含多少参数

代码能够在没有给出这些参数的情况下依然跑通，即使深度学习网络无法判断网络的输入维度是什么，这归功于 **参数初始化** ，指导数据第一次通过模型传递时，框架才能动态推断出每个层的大小。

延后初始化使框架能够根据输入维度，自动推断出参数矩阵的形状。



## 自定义层

在前文提到自定义块 `block` ，实际上，我们也可以自定义层，因为难免有的时候会遇到自己想要实现的功能在深度学习库中找不到合适的API。

- 不带参数的层

  ```python
  # 不带参数的层
  class CenteredLayer(nn.Module):
      def __init_(self):
          super().__init__()
          
      def forward(self,X):
          return X - X.mean()
  layer = CenteredLayer()
  layer(torch.FloatTensor([1,2,3,4,5]))
  net = nn.Sequential(nn.Linear(8,128),CenteredLayer())
  Y = net(torch.rand(8))
  Y.mean()
  # 得到的是接近于0的数，因为浮点数精度问题
  ```

- 带参数的层

  ```python
  class MyLinear(nn.Module):
      def __init__(self,in_units,units):
          super().__init__()
          self.weight = nn.Parameter(torch.randn(in_units,units))
          self.bias = nn.Parameter(torch.randn(units))
          
      def forward(self,X):
          linear = torch.matmul(X,self.weight.data) + self.bias.data
          return F.relu(linear)
      
  linear = MyLinear(5,3)
  linear.weight
  net = nn.Sequential(MyLinear(5,3),MyLinear(3,1))
  net(torch.rand(1,5))
  ```

  

## 读写文件

得到训练好的模型，实际上就是得到模型的参数配置。在训练过程中定期保存中间结果，在面对一些突发情况时，比如服务器突然断电，不会损失之前训练的结果，在问题解决后还能接着上一次中断的地方继续训练，尤其是训练耗时较长的模型时，保存中间结果更加重要。

本节主要介绍如何保存和加载权重向量与整个模型

- 加载和保存张量

  - 单个张量：`load` 加载，`save` 保存

  ```python
  x = torch.arange(4)
  # 存储
  torch.save(x,"x-file")
  # 加载
  x1 = torch.load("x-file")
  ```

  - 张量列表

  ```python
  # 存储和加载张量列表
  y = torch.zeros(4)
  torch.save([x,y],"x-files")
  x2,y2 = torch.load("x-files")
  ```

  - 张量字典

  ```python
  mydict = {"x":x,"y":y}
  torch.save(mydict,"mydict")
  mydict2 = torch.load("mydict")
  ```

- 加载和保存模型参数，同样使用 `save` 和 `load` ，但是与存取单独向量稍有不同

  深度学习框架提供了内置函数用于加载和保存整个模型。因为一个模型中有成百上千个参数分布在各层，若是使用存取单独张量的方法，会很麻烦。

  需要注意的是，这里保存的是模型参数，而不是整个模型，模型本身包含代码，很难序列化。因此为了恢复模型，在读取模型参数时，会先使用代码生成模型架构，再加载参数。

  以 `MLP` 为例

  ```python
  # 先定义一个MLP
  class MLP(nn.Module):
      def __init__(self):
          super().__init__()
          self.hidden = nn.Linear(20,256)
          self.output = nn.Linear(256,10)
      def forward(self,X):
          return self.output(F.relu(self.hidden(X))
  
  net = MLP()
  X = torch.randn(2,20)
  Y = net(X)
                             
  # 保存模型参数
  torch.save(net.state_dict(),"mlp-params")
                             
  # 先使用代码生成模型架构，再加载参数
  clone = MLP()
  clone.load_state_dict(torch.load("mlp-params"))
  clone.eval()
                             
  Y_clone = clone(X)
  # 检查两个模型是否一样
  Y_clone == Y
  ```



## GPU

本节主要讨论如何利用GPU的计算性能进行研究

- 如何使用单个GPU

  - 查看显卡信息

    ```sh
    # terminal
    nvidia-smi
    # notebook
    !nvidia-smi
    ```

    在Pytorch中，每一个数组都有设备 `device` ，通常叫做环境 `context` ，默认情况下，所有变量相关的计算会分配给CPU，但是训练神经网络模型时，我们通常希望参数在GPU上，当有多张GPU时，还可以显式地指定分配给哪张GPU，通过编号指定。

  - 指定设备

    ```python
    import torch
    from torch import nn
    torch.device('cpu'),torch.device('cuda'),torch.device('cuda:1')
    ```

    此外，有些时候一张GPU不一定能够放下所有参数及计算过程中产生的中间值，需要进行多卡并行，因此需要最大限度地减少GPU之间传输数据所需的时间。

  - 查询可用的GPU数量

    ```python
    torch.cuda.device_count()
    ```

    ```python
    def try_gpu(i=0):
        '''如果存在，返回gpu(i)；如果不存在，返回cpu()'''
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
       return torch.device('cpu')
    def try_all_gpus():
        '''返回所有可用的gpu，如果没gpu，返回cpu'''
        devices = [torch.device(f'cuda:{i}') 
                   for i in range(torch.cuda.device_count())]
        return devices if devices else [torch.device('cpu')]
    
    # 调用
    try_gpu(),try_gpu(10),try_all_gpus()
    ```

- 张量与GPU

  - 查询张量所在设备

    ```python
    x = torch.tensor([1,2,3])
    x.device
    ```

    需要注意的是，无论何时对多个项进行操作（如加减乘除），他们必须位于同一设备上，否则框架将不知道在哪里存储结果，甚至不知道在哪里运算

  - 存储在GPU上

    ```python
    # 创建时指定设备
    X = torch.ones(2,3,device=try_gpu())
    Y = torch.zeros(2,3,device=try_gou(1)
    X,Y
    ```

- 复制

  假如张量 `X` 在GPU0上，而张量 `Y` 在GPU1上，此时需要进行 `X+Y` 的操作，不能简单地直接将 `X` 加上 `Y` ，这会导致异常，两个张量位于不同的设备上，模型不知道在哪里计算、在哪里存储。

  此时，我们需要将 `X` 移动到 `Y` 所在的设备上，再执行加法运算

  ```python
  Z = X.cuda(1)
  X,Z
  ```

    此时 `X` 仍在GPU0上不变，GPU1上会出现 `X` 的副本 `Z` ，可以进行 `Z+Y` 的操作

- why GPU

  GPU的并行计算能力很强，因此计算效率远远高于CPU，但是设备之间（GPU，CPU或其他）传输数据比计算慢得多，这也让并行化变得困难，需要先等待数据被接收，才能进行后续操作。所以，拷贝数据需谨慎。

  根据经验，多个小操作耗费的时间比一个大操作长，一次执行几个操作比代码中散布的许多单个操作要好得多。

  当打印张量或Numpy数组时，若是数据不在内存中，框架会首先将它复制到内存，会导致额外的传输开销，并且受制于全局解释器，一切都要等Python完成。

- 神经网络与GPU

  网络可以指定设备

  ```python
  net = nn.Sequential(nn.Linear(10,1))
  net = net.to(device=try_gpu())
  ```

  当输入为GPU上的张量时，模型会在该设备上进行计算，结果也会存储在这台设备上

  ```python
  # 验证
  X = torch.randn(10)
  Y = net(X)
  net[0].weight.data.device == X.device,Y.device == X.device
  ```

- 记录GPU上模型的中间结果时，最好用日志 `log` 记录。

  一个典型错误是：计算GPU上每个小批量的损失，并在命令行中将其报告给用户 / 将其记录在Numpy的 `ndarray` 中，这会触发全局解释器锁，使所有GPU阻塞。*不经意地移动数据可能会显著降低性能*

  