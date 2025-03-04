# Preliminaries

[TOC]

## numpy——ndarray

#### 介绍

在深度学习中，数据需要经过（1）获取（2）存储（3）处理。数据常见为n维数组，常用的存储方式是以张量 `Tensor` 形式。深度学习中叫做 `Tensor`，`Numpy` 库中叫做 `ndarray` ，`Tensor` 与 `ndarray` 相比，优势在于：

- GPU支持加速计算，计算效率高，`ndarray` 仅支持CPU计算
- `Tensor` 支持自动微分，更适合深度学习，因为优化时涉及到梯度

#### 基本使用方法

导入 `torch` 库

```python
import torch
```

- 创建

  ```python
  x = torch.arange(12)
  ```

- 查看形状

  ```python
  x.shape
  ```

- 查看元素总个数

  ```python
  x.numel()
  ```

- 更改张量形状，但是元素总个数不变

  ```python
  X = x.reshape((3,4))
  ```

  可以使用 `-1` 自动计算维度

  ```
  X = x.reshape(-1,4)
  ```

  程序能够自动计算出行数为3

- 创建全0张量

  ```python
  x = torch.zeros(2,3,4)
  ```

  三维矩阵，2 * 3 * 4

- 创建全1张量

  ```python
  x = torch.ones(2,3,4)
  ```

- 随机初始化，`randn` 会从 $\mu=0,\sigma=1$ 的正态分布中随机抽取值

  ```python
  x = torch.randn(2,3,4)
  ```

- 列表 —> 张量

  ```python
  x = torch.tensor([[1,2,3],[4,5,6]])
  ```

  `x` 作为张量，形状为 `(2,3)`

  外层列表相当于轴0，内层列表相当于轴1。

#### 运算符

- 按元素运算：要求是张量形状一样

  - `+` `-` `*` `/` `**` 

  ```
  x = torch.tensor([1,2,3])
  y = torch.tensor([4,5,6])
  x + y,x - y,x * y,x / y,x ** y
  ```

  - `exp` ：$e^x$

  ```python
  x = torch.tensor([1,2,3])
  torch.exp(x)
  ```

  - `concatenate` ：将张量按照指定的轴进行拼接

  ```python
  x = torch.arange(12,dtype=torch.float32).reshape(3,4)
  y = torch.zeros(3,4)
  torch.cat((x,y),dim=0),torch.cat((x,y),dim=1)
  ```

  `dim=0` 的结果是`[[0,1,2,3],[4,5,6,7],[8,9,10,11],[0,0,0,0],[0,0,0,0],[0,0,0,0]]`

  `dim=1` 的结果是 `[0,1,2,3,0,0,0,0],[4,5,6,7,0,0,0,0],[8,9,10,11,0,0,0,0]`

  - 用逻辑运算符 `(==,!=,>,<)` 构建张量，要求进行运算的张量形状相同

  ```python
  x == y
  ```

  得到的元素值为 `True` or `False` ，形状与 `X` `Y` 相同

  - 对张量中所有元素求和

  ```python
  x.sum()
  ```

  - 广播机制 `broadcasting` ：在某些时候，即使形状不同也可以调用广播机制来执行按元素操作

    工作方式：（1）复制元素来扩展数组，达到相同形状（2）对生成的数组执行按元素操作前提条件：两个张量进行广播时，从最后一个维度开始向前比较，需要满足任一条件（1）两个维度大小相等（2）其中一个维度为1

    满足：

  ```python
  x = torch.arange(3).reshape(3, 1) 
  y = torch.arange(2).reshape(1, 2)
  a + b
  ```

  ​	不满足：因为最后一个维度相同，为1，但是第一个维度不同且不存在为1的

  ```
  x = torch.arange(3).reshape(3,1) 
  y = torch.arange(2).reshape(2,1)
  a + b
  ```

#### 索引与切片

张量元素可通过索引访问，第一个元素index=0，最后一个index=-1

切片是指可以通过指定索引范围访问部分（连续）元素，**左开右闭**

- 索引访问

  ```python
  x[-1],x[0],x[3]
  ```

- 切片

  ```
  x[0:-1]
  ```

- 修改指定位置的元素

  ```python
  x[1,2] = 9 # (1,2)处
  x[0:2,:] = 9 # 第1行~第2行的所有列
  ```

#### 节省内存

因为运行一些操作可能会为新结果分配内存，`y = y + x` 将指向新分配的内存张量 `y`

- （题外话）

  ```python
  x = torch.arange(12)
  y = x.clone() # 分配新内存给x的副本y
  ```

- 查看前后张量是否分配到同一块内存

  ```python
  before = id(y)
  y = y + x
  id(y) == before
  ```

  这样做的弊端在于：（1）每次进行运算都需要分配新的内存，对于有着超大参数的模型，内存占用很大，并且大部分是不必要的内存占用（2）当进行完运算后，张量的引用指向新的内存位置，但是其他引用仍然指向旧地址，会引发冲突

- 执行原地操作：切片法将新结果赋值给原张量

  ```python
  # 法1
  y[:] = y + x
  # 法2
  y += x
  ```

  可以使用上一个方法验证前后 `y` 的地址是否变化

#### 转化为其他 `Python` 对象

- `numpy` 的 `ndarray`

   `ndarray` 与 `tensor` 共享同一个底层内存，二者的相互转化很容易实现，并且是就地操作，改变其中一个张量的同时也更改另一个张量

  ```python
  x = torch.arange(2)
  a = x.numpy()
  b = torch.tensor(a)
  # 查看类型
  type(a),type(b)
  ```



## pandas

导入 `pandas` 库

- 读取数据集

  ```python
  data = pd.read_csv(file_path)
  ```

  得到的 `file` 将是 `Dataframe` 类型的数据

- 处理缺失值

  ```python
  inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
  inputs = inputs.fillna(inputs.mean()) # 用平均值填充缺失值
  ```

- 转化张量

  ```python
  torch.tensor(data.values)
  ```



## Linear Algebra

- 标量 `scaler`

  ```python
  torch.tensor(2)
  ```

- 向量 `vector`

  ```python
  torch.tensor(2,2)
  ```

- 长度、维度、形状

  ```python
  len(x) # 长度
  x.shape # 形状
  # 张量的维度是指张量具有的轴数，而向量的维度是指向量的长度，相当于张量某个轴的维度
  ```

- 矩阵

  ```python
  x = torch.arange(20).reshape(4,5)
  ```

  先创建 `[0,..,19]` 的矩阵，在规定形状为 `4*5`

  - 矩阵转置

    ```
    x.T
    ```

- 张量

  向量是标量的推广，矩阵是向量的推广，张量表示有任意多个轴的n维数组

  ```python
  torch.arange(60).reshape(3,4,5)
  ```

- 通过求和降维：对张量的指定轴进行求和来降维，因为求和后只剩下一个元素

  ```python
  x = torch.arange(20).reshape(5,4)
  x_sum_axis_0 = x.sum(axis=0)
  ```

   `x_sum_axis_0` 是按行相加，最后形状为 `4`

  同理，若是 `axis=1` ，那么按列相加，最后形状是 `5`

  *求平均值*

  ```python
  # 法1
  x.sum()/x.numel()
  # 法2
  x.mean()
  ```

  所以，求和的应用有

  - 沿指定轴降维
  - 求平均值

  当然，用 `x.mean(axis=m)` 的方法也可以实现降维

- 非降维求和：在求和时保持总和或轴数不变

  - 按行相加的和仍占一个轴，所以即使 `5*4` 的张量进行沿轴0求和，`keepdims=True` 仍可以让结果保持两个轴，即形状为 `5*1` ，此时可以用广播机制计算 `x/sum_x`，

  ```python
  x.sum(axis=0,keepdims=True)
  ```

  - 沿指定轴计算累计总和：形状不变，按指定轴进行和的累加

  ```python
  x.cumsum(axis=0)
  ```

- 点积 $x^Ty$

  前提：`x` 的列数等于 `y` 的行数

  含义：夹角的余弦 / 加权平均

  ```python
  torch.dot(x,y)
  # 等价于
  torch.sum(x * y)
  ```

- 矩阵—向量积

  前提：`X` 的行数等于 `y` 的行数，`y` 是列向量，`X` 是矩阵

  ```python
  torch.mv(X,y)
  ```

- 矩阵—矩阵乘法

  对于 $A_{n\times k}*B_{k\times m}$ 矩阵乘法实际上相当于进行了 `m` 次向量积，将结果拼接在一起形成了 $n\times m$ 的矩阵

  ```python
  A = torch.arange(20,dtype=torch=float32).reshape(5,4)
  B = torch.ones(4,5)
  torch.mm(A,B)
  ```

  `A` 在创建时要指定数据类型为 `float32` ，这是深度学习中常用的数据类型，若没有指定，`torch.arange` 会自动创建 `long` 类型的张量，而 `torch.ones` 创建的是 `float32` 的张量，矩阵乘法时数据类型不一致，会报错。

- 范数：表示一个向量有多大，这里的大小不涉及维度，而是分量大小

  - $L_2$ 范数定义： $\|\mathbf{x}\|_2=\sqrt{\sum_{i=1}^{n}x_{i}^{2}}$ ，也可以直接写成 $\|\mathbf{x}\|$ 
  - $L_1$ 范数定义：$\|\mathbf{x}\|_{1}=\sum_{i=1}^{n}|x_i|$ ，表示向量元素绝对值之和
  - $L_2$ 范数与 $L_1$ 范数是 $L_p$ 范数的特例：$\|\mathbf{x}\|_p=(\sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}x_{ij}^2})^{\frac{1}{p}}$  
  - $Frobenius$ 范数定义：$\|\mathbf{x}\|_F=\sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}x_{ij}^2}$ ，表示矩阵元素平方和的平方根，$L_2$ 范数是向量的元素和平方根，那么 $Frobenius$ 范数可以看成矩阵的 $L_2$ 范数
  - 目标函数通常表示为范数，深度学习中常用 $L_2$ 范数与 $L_1$ 范数
  - 计算 $L_2$ 范数 

  ```python
  x = torch.tensor([3,4])
  torch.norm(x) # 向量的L2范数
  ```

  ​	得到结果 `5` 

   - 计算 $L_1$ 范数

  ```python
  torch.abs(x).sum()
  ```

    - 计算 $Frobenius$ 范数

  ```python
  x = torch.ones(2,3)
  torch.norm(x) # 矩阵的L2范数
  ```



## Calculus (微积分)

在微分学中，最重要的是优化，在深度学习中这个概念也同样重要。

深度学习的模型效果好，意味着（1）拟合观测数据的效果好（2）在从未见过的数据上表现良好，也即：

- 优化：用模型拟合观测数据

- 泛化："生成"比纯拟合训练数据更有效的模型，因为它知道如何处理全新的数据

- *曲线可视化（Jupyter中）*

  ```python
  def use_svg_display():  #@save
      """使用svg格式在Jupyter中显示绘图"""
      backend_inline.set_matplotlib_formats('svg')
  def set_figsize(figsize=(3.5, 2.5)):  #@save
      """设置matplotlib的图表大小"""
      use_svg_display()
      d2l.plt.rcParams['figure.figsize'] = figsize
  def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
      """设置matplotlib的轴"""
      axes.set_xlabel(xlabel)
      axes.set_ylabel(ylabel)
      axes.set_xscale(xscale)
      axes.set_yscale(yscale)
      axes.set_xlim(xlim)
      axes.set_ylim(ylim)
      if legend:
          axes.legend(legend)
      axes.grid()
  def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
           ylim=None, xscale='linear', yscale='linear',
           fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
      """绘制数据点"""
      if legend is None:
          legend = []
  
      set_figsize(figsize)
      axes = axes if axes else d2l.plt.gca()
  
      # 如果X有一个轴，输出True
      def has_one_axis(X):
          return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                  and not hasattr(X[0], "__len__"))
  
      if has_one_axis(X):
          X = [X]
      if Y is None:
          X, Y = [[]] * len(X), X
      elif has_one_axis(Y):
          Y = [Y]
      if len(X) != len(Y):
          X = X * len(Y)
      axes.cla()
      for x, y, fmt in zip(X, Y, fmts):
          if len(x):
              axes.plot(x, y, fmt)
          else:
              axes.plot(y, fmt)
      set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
      
  # usage
  x = np.arange(0, 3, 0.1)
  plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
  ```

- 偏导数、梯度、链式法则

## `autograd` 工具

- 为变量的梯度指定新的内存，防止每次计算都新分配空间

  ```python
  x = torch.arange(12,requires_grad=True)
  x.grad # 访问x的梯度值
  ```

- 计算 `y` 对 `x` 的梯度

  ```python
  # 这里不可以直接写 y=x*x，因为计算的是向量乘法，y是个向量，每个元素是x每个分量的平方，而是一个值，不可以进行y.backward()
  y = torch.dot(x,x)
  y.backward() # only for scaler outputs
  x.grad
  ```

  此时 `x.grad` 的结果是 `2x` 

  验证方法：

  ```python
  x.grad == 2 * x
  # 不可以写
  # 正常得到的向量应该都为true
  ```

- 清除当前梯度值：用于计算新的梯度前，防止对后续计算产生干扰

  ```python
  x.grad.zero_()
  ```

- 一般来说，不能直接用 `backward()` 计算张量梯度，但是在深度学习中，进行梯度优化的数据却常常是一个矩阵，此时需要明白计算梯度是为了优化损失函数，而不是计算微分矩阵，可以将转化为计算偏导数之和，也就是对一批数据，先求和再计算梯度，等效于手算张量梯度的结果。

  ```python
  y = x * x # 得到的是张量而非标量
  y.sum().backward()
  x.grad()
  x.grad == 2 * x
  ```

  【“等效”的解释：

  ```python
  z = x.sum()
  z.backward()
  x.grad
  ```

  得到的结果是 `[1,1,1,...]` ，所以 `m*[1,1,1,...] = [m,m,m,...]`

  】

- 分离计算：将有些变量从计算图中移出。

  例如，`z=y+x`，其中 `y=f(x)` ，在计算 `z` 对 `x` 的导数时，出于某种原因希望将 `y` 视为常数，也就是要将 `y` 移出计算图中，这就需要 *分离变量* 。

  将 `y` 分离出一个值相同的变量 `u` ， `z=u+x` ，不一样的是，`u` 丢弃了计算图中如何计算 `y` 的任何信息，在 `z` 计算对 `x` 的导数时，梯度不会从 `u` 向后流经到 `x` 

  ```python
  y = x * x
  u = y.detach()
  z = u * x # 使用的是y分离的变量u
  z.sum().backward()
  x.grad == u 
  # 正常来说，结果应该是[True,True,...]
  ```

- `Python` 控制流的梯度计算

  - `Python` 控制流包含（1）条件，如 `if-else` （2）循环，如 `while` （3）任意函数调用 `custom_f()` 

  如果构建函数的计算图需要经过 `Python` 控制流，仍然可以进行梯度计算。

  ```python
  def f(a):
      b = a * 2
      while b.norm()<1000:
          b *= 2
      if b.sum()>2:
          c = b
      else:
          c = 100 * b
      return c
          	
  a = torch.randn(size=(),requires_grad=True)
  d = f(a)
  d.backward()
  ```

  `f` 对 `a` 是分段线性的，即对于任何 `a` ，总存在常量标量 `k` ， `f=a*k` ，所以验证梯度应写为

  ```python
  a.grad == d/a # d对a分段线性，不同段的常量值不同，所以不能用一个固定的常数值来验证
  ```



## Probability

机器学习就是做出预测，对于输入的一组数据，根据各个特征分量值，通过一系列的计算（利用学习的参数），得到一个向量，分量是对应类别的概率，选择最高概率作为预测结果

#### 重要定理

- 大数定律

  可视化，以投骰子为例

  ```python
  counts = multinomial.Multinomial(10, fair_probs).sample((500,))
  cum_counts = counts.cumsum(dim=0)
  estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
  
  d2l.set_figsize((6, 4.5))
  for i in range(6):
      d2l.plt.plot(estimates[:, i].numpy(),
                   label=("P(die=" + str(i + 1) + ")")) # 前面提到的可视化方法
  d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
  d2l.plt.gca().set_xlabel('Groups of experiments')
  d2l.plt.gca().set_ylabel('Estimated probability')
  d2l.plt.legend();
  ```

- 中心极限定理

#### 重要概念

- 概率论公理：（1）概率非负（2）概率和为1（3）对于互斥事件 $A_1,A_2,A_3$ ，$P(A_1UA_2UA_3)=P(A_1)+P(A_2)+P(A_3)$
- 随机变量
- 当有多个随机变量时：联合概率、条件概率、贝叶斯定理、边际概率、独立性 
- **Jupyter中的应用很清晰，可以多看看**
- 期望与方差

#### 代码实现（以投骰子为例）

- 抽样：从概率分布中抽取样本，有需要新导入的库。*（抽一个样本）*

  ```python
  import torch
  from torch.distributions import multinomial
  
  fair_probs = torch.ones(6) /6
  multinomial.Multinomial(1,fair_probs).sample() # 传入一个概率向量，从中抽取一个样本
  # 返回的结果是与概率向量等长的向量，分量代表该事件出现次数，分量和=抽取次数
  ```

- 抽多个样本

  ```python
  multinomial.Multinomial(10,fair_probs).sample()
  # 分量和=抽取次数
  ```

  若使用python的循环，速度将会很慢，但是使用深度学习框架。可以同时抽取

  若是想得到每个事件出现概率，可以

  ```python
  counts = multinomial.Multinomial(100,fair_probs).sample()
  counts /= 100
  ```

  

## lookup api

- 查看模块中可以调用的类与函数

  ```python
  dir(torch.distributions)
  ```

- 查找特定函数或类的用法

  ```python
  help(torch.ones
  ```

   

