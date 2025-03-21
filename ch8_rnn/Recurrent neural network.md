# Recurrent neural network

[TOC]

加上本章需要面对的 *序列信息*，目前为止一共碰到过以下三种类型的数据：

- 表格数据：常用 `MLP` 解决，行是样本，列是特征

- 图像数据：图像的空间结构包含了许多隐藏信息，使用 `CNN` 处理空间信息

- 序列数据：与前两个不同的是：前两种数据都假设数据独立同分布，然而序列数据之间是互相依赖的，是 *有顺序的*，一个文本打乱了顺序之后可能就变成毫无意义的文字了，因此需要针对此类数据设计特定的模型，也就是本章需要学习的 `recurrent neural network` ，它用于处理序列信息，通过引入状态变量存储过去和当前的输入，从而确定当前的输出。大部分的序列信息都是文本数据。

  ​	

## 序列模型 `sequence`

**序列数据的特征**

- 先见之明往往比事后诸葛亮要困难

  - 外推法：对超出一直观测范围进行预测，即数据是 `out of domain`

  - 内插法：在现有观测值之间进行估测，即数据是 `in domain` 

​	`OOD` 预测比 `ID` 预测更困难

- 时间性：放假时候看剧的人更多，圣诞节的时候看圣诞电影的人更多
- 连续性：不同搭配会有不同效果，尽管组成的字是一样的，比如“狗咬人”和“人咬狗”，人与人之间的互动也是连续的，比如帖子评论区的互动
- 强相关性：后面发生的事情会受到前面事情的影响，有时这种影响很大，比如地震和余震

处理序列数据需要新的统计工具和深度神经网络架构。

**主要任务**

已知截至 $t-1$ 时间内的数据 $x_{1},x_{2},...,x_{t-1}$ ，需要根据过去时间的数据来预测下一时间的数据，即 $x_t$ 。

与以前不同的是，在预测完 $x_t$ 后，当预测 $x_{t+1}$ 时，$x_t$ 会作为过去时刻数据出现在输入中，并且随着时间推移，输入的数据量也会增加，因此需要一个近似的方法来简化计算。

- 自回归 `autoregressive models` ：取距离最近一段时间，窗口长度为 $\tau$ ，只需要将 $x_{t-1},x_{t-2},...,x_{t-\tau}$ 作为输入即可。在预测 $x_{t+1}$ 时输入数据变为 $x_t,x_{t-1},...,x_{t+1-\tau}$ ，此处的 $x_t$ 正是之前预测的结果，自回归模型的一个特点就是自己对自己执行回归。

  - 优点：当 $t>\tau$ 时，输入数据量固定，能够像之前那样训练一个深度神经网络

- 隐变量自回归 `latent autoregressive models` ：保留对过去的一些总结 $h_t$ ，结合一些过去数据，来预测当前状态。预测 $x_t$ 和总结 $h_t$ 是同时更新的。
  $$
  \bf x_t=P(x_t|h_t) \\
  \bf h_t=g(h_{t-1},x_{t-1})
  $$
  

  由于 $h_t$ 从未被观测到，因此也叫 *隐变量自回归* 。

  <img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250313205723255.png" alt="image-20250313205723255" style="zoom:70%;" />

​	上述两种方法并非直接获取训练数据，而是生成训练数据。一个经典的方法是：*使用历史观测来预测下一个未来预测* ，常见的假设是 *虽然特定值 $x_t$ 可能会改变，但是序列本身的时间动力学不会变，而且也不可能通过现有数据来预测新的动力学，新的动力学一定受新的数据的影响* 。静止不变的动力学在统计上被叫做 *静止的 `stationary`* 。生成方式（序列值的预估方式）如下：
$$
\bf P(x_1,...,x_{\Tau})=\prod_{t=1}^{\Tau}P(x_t|x_{t-1},...,x_1)
$$
​	上述考虑对连续的对象有效，如股票价格预测，若是处理离散数据，如单词，需要使用分类器而非回归。

**马尔可夫模型** 

在自回归模型的近似中，选择了长度为 $\tau$ 的时间窗口，使用 $x_{t-1},..,x_{\tau}$ 而非 $x_{t-1},...,x_1$ ，只要近似是精确的，就称它满足 *马尔科夫条件 `Markov condition`* 。

若是 $\tau=1$ ，就会得到一个一阶马尔可夫模型：
$$
P(x_1,...,x_{\Tau})=\prod_{t=1}^{\Tau}P(x_{t}|x_{t-1}),\space P(x_1|x_0)=P(x_1)
$$
当处理对象是离散值时，可以使用动态规划沿着马尔科夫链精确计算结果，如书中写的 $P(x_{t+1}|x_{t-1})$ 的求解，就是通过中间值 $x_{t}$ 的遍历来计算的。

**因果关系** 

原则上，将 $P(x_1,...,x_{t-1})$ 顺序打乱也没关系，因为根据条件概率公式 $P(x_t|x_1,...,x_{t-1})=\frac{P(x_1,...,x_{t-1},x_t)}{P(x_1,...,x_{t-1})}$ ，$x_1,...,x_{t-1},x_t$ 的顺序对最终概率值并没有影响，然而在逻辑上是不合理的，时间数据存在一个自然的方向——前进，未来不能影响过去，即使改变 $x_t$ ，也不会影响过去数据的分布，所以解释 $P(x_t|x_{t-1})$ 比解释 $P(x_{t-1}|x_t)$ 更容易。

**训练** 

具体代码见 `notebook`



## 文本预处理 `text preprocessing`

**常见预处理步骤**

- 加载文本

- 将文本切分成词元 `token`  (如 `word` or `char` )。

  **`token` 是文本的基本单元。** 

- 根据词元构建词表，将拆分后的词元映射成数字索引

  - 语料：将输入文本去重后，得到它们的唯一词元，对这些词元进行统计，统计的结果就叫 *语料 `corpus`* 

  - 词表：字典，根据语料的统计结果，按照 `token` 的出现频率将其映射为数字（用索引进行映射），每个 `token` 都是唯一的，也都被唯一标识。

    ​	需要注意的是，为了节省空间和提高搜索效率，出现频率较低的 `token` 将会被移除。	

    ​	特殊规则：

    		- 语料库中不存在或被删除的 `token` 会映射到特定的未知词元 `<unk>`
    		- 填充词元：`<pad>` 
    		- 序列开始词元： `<bos>` 
    		- 序列结束词元： `<eos>`

- 根据词表，将文本序列转化为数字索引序列（编码 `encoding`）

  模型需要的输入是数字，但是文本是字符串，因此需要进行转化



具体代码见 `notebook` 



## 语言模型与数据集 `Sequence model`

**语言模型** 

- 目标：预测文本序列 $\bf x_1,x_2,...,x_T$ 的联合概率， $\bf x_t$ 叫做 *文本序列在时间步 $t$ 出的观测或标签*。文本序列的联合概率计算公式如下：
  $$
  \bf P(x_1,x_2,..,x_T)
  $$
  理想状态下，模型能够基于模型本身生成文本。生成文本不仅是生成语法合理的内容，语义上也要有意义，需要模型理解文本。

**学习语言模型** 

假设文本序列长度是 $4$ ，比如 ${deep,learning,is,fun}$ ，那么联合概率的计算如下：
$$
P(x_1,x_2,x_3,x_4)=P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)P(x_4|x_1,x_2,x_3)
$$
为了计算联合概率，需要计算每个单词出现的概率，这些概率本质上就是语言模型的 *参数* 。

若是以单词为单位进行词元化，那么可以使用单词出现频率作为单词概率。
$$
P(x_1)=\frac{n(x_1)}{n}
$$
其中 $n(x_1)$ 是文本中 $x_1$ 出现的频率，*理论上* $P(x_1)$ 是指任何以单词 $x_1$ 为首的句子的概率，但是这里可以用出现频率来近似。 $n$ 是文本单词总数。

对于联合概率 $P(x_1,x_2)$ ，估计这类单词的概率比较困难，因为出现的频率会低很多，尤其是对于不常见的组合，当计算三个以上单词的联合概率时，概率接近于0，因为数据集是有限的1，但是文本组合很多，并不是每种组合都能在数据集中出现。

常见的策略是 **拉普拉斯平滑 `Laplace smoothing`** 

- 具体方法：在所有计数中添加一个小常量，用 $n$ 表示单词总数，用 $m$ 表示唯一单词的数量，计算公司如下：

$$
P(x_1)=\frac{n(x_1)+\epsilon_1/m}{n}
$$

$$
P(x_2|x_1)=\frac{n(x_1,x_2)+\epsilon_2P(x_2)}{n(x_1)+\epsilon_2}
$$

$$
P(x_3|x_1,x_2)=\frac{n(x_1,x_2,x_3)+\epsilon_3P(x_3)}{n(x_1,x_2)+\epsilon_3}
$$

其中，$\epsilon_1$ $\epsilon_2$ $\epsilon_3$ 是超参数，当 $\epsilon_1=0$ 时，不使用平滑，当 $\epsilon_1=+\infty$ 时，$P(x_1)$ 接近于均匀分布 $\frac{1}{m}$ 

- 局限性：（1）忽略了单词的意思（2）需要存储所有计数，空间开销大（3）长单词序列大部分是没出现的组合，若是模型只是简单地统计先前看到的单词序列频率，那么实际上无法预测可能的情况，表现不佳

**马尔可夫模型与 $n$ 元语法** 

回顾马尔可夫

- 一阶马尔可夫模型：
- $P(x_{t+1}|x_t,...,x_1)=P(x_{t+1}|x_t)$ 

可以看到与上面提到的联合概率分布计算的公式比较相似，将它应用于语言模型：

- 一元语法 `unigram` ：$P(x_1,x_2,x_3,x_4)=P(x_1)P(x_2)P(x_3)P(x_4)$ ，单词之间相互独立
- 二元语法 `bigram` ：$P(x_1,x_2,x_3,x_4)=P(x_1)P(x_2|x_1)P(x_3|x_2)P(x_4|x_3)$ ，只和前一个单词有关系
- 三元语法 `trigram` ：$P(x_1,x_2,x_3,x_4)=P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)P(x_4|x_2,x_3)$ ，和前两个单词有关系

$n$ 元语法截断相关性，是一种实用的语言模型。

**自然语言统计** 

- 停用词 `stop words` ：虽然出现频率很高，但是没什么学习的意义，比如 `the` `a` `I` 等等。可以在计算联合概率时将它们过滤掉，在模型中仍然会使用它们

- 奇普夫定律：词频以一种明确的方式迅速衰减，大致遵循双对数坐标图上的一条直线，第 $i$ 个最常用单词的频率 $n_i$ 满足
  $$
  n_i\sim\frac{1}{i^a}
  $$
   等价于
  $$
  \log n_i=-\alpha\log i+c￥
  $$

- 

​	$\alpha$ 是刻画分布的指数，$c$ 是常数，*这也说明了光靠计数统计和平滑来建模语言序列是不可行的，因为它会高估尾部单词的频率。*

​	关于一元语法、二元语法、三元语法的分析见 `notebook` ，实验表明 *奇普夫定律* 不仅适用于一元语法，也适用于其他 $n$ 元语法。

**读取长序列数据** 

- 问题：对于长序列，在数据集中很少出现或者不出现

对于长序列文本，因为模型处理能力有限，因此需要划分成相同时间步的子序列（分批），每次处理小批量子序列。

问题在于，如何选择初始偏移量进行划分，如果只选择一个偏移量，那么用于训练网络的子序列的覆盖范围将是有限的，因此可以选择随机偏移量来划分序列，以同时获得 *随机性* 和 *覆盖性* 。

- 随机采样 `random sampling` 策略

  每个样本是在原始序列上任意捕获的子序列，来自相邻的、随机的小批量中的子序列在原始序列上不一定相邻。

  输入 $\bf x$ 是子序列，输出 $y$ 是该子序列下一词元

- 顺序分区 `sequential patitioning` 策略

**顺序分区** 

保证两个相邻小批量的子序列在原始序列中也是相邻的，保留了拆分子序列的顺序



## Recurrent neural network 

**背景** 

在之前的语言模型中，如 $n$ 元语法，设置了时间步长 $T$，单词 $x_t$ 只与它之前的 $T-1$ 个单词有关，即 $x_t$ 的概率计算公式为 $P(x_t|x_1,x_2,..,x_{t-1})$ ，若是希望 $t-(T-1)$ 之前的单词也能对 $x_t$ 产生影响，需要扩大时间步，这会导致参数量呈指数增加，因为词表 $\mathcal{V}$ 需要存储 $|\mathcal{V}|^n$ 个数字。

与其对统计概率建模，还不如使用隐变量模型

**隐变量模型** 
$$
P(x_t|x_1,...,x_{t-1})\approx P(x_t|h_{t-1})
$$
其中 $h_{t-1}$ 是 **隐状态** ，也叫做 **隐变量 `hidden variable`** ，存储了从开始到时间步 $t-1$ 的所有信息，
$$
h_t=f(x_t,h_{t-1})
$$
$h_t$ 能够存储从开始到时间步 $t$ 的所有值，并不是一个近似值，虽然这样会使模型的计算和存储开销增大。

- 隐藏层：从输入到输出路径上，被隐藏的步骤节点
- 隐藏变量（隐状态）：在给定步骤所做的事情的输入，并且这些输入只能由先前数据计算得到

**无隐状态的神经网络** 

`MLP`
$$
\mathbf H=\phi(\bf XW_{xh}+b_h)
$$

$$
\bf O=HW_{hq}+b_q
$$

$\bf H$ 是隐藏层，先进行矩阵乘法，再通过广播机制加上偏置项

**有隐状态的循环神经网络** 

在计算当前状态的输出时，需要考虑上一个时间步的隐藏变量 $\bf H_{t-1}$ ，所有隐变量的形状都应该是 $n\times h$ 
$$
\bf H_t=\phi(X_t W_{xh}+H_{t-1}W_{hh}+b_h)
$$
$\bf H_t$ 保留了序列从开始到时间步 $t$ 的所有信息，因此这样的隐藏变量又叫 *隐状态* ，又因为计算隐状态的公式是一样的，所以是 *循环* 的。——基于循环计算的隐状态神经网络叫做 **循环神经网络 `recurrent neural network`** ，计算 $\bf H_t$ 的层叫做 *循环层* 。
$$
\bf O_t=H_tW_{hq}+b_q
$$
`RNN` 的参数包括：

（1）隐藏层权重 $\bf W_{xh}\in\mathbb{R}^{d\times h}$     $\bf W_{hh}\in\mathbb{R}^{h\times h}$     $\bf b_h\in\mathbb{R}^{1\times h}$

（2）输出层权重 $\bf W_{hq}\in\mathbb{R}^{h\times q}$     $\bf b_{q}\in\mathbb{R}^{1\times q}$

即使随着时间增加，参数量也不会增多

<img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250318202128091.png" alt="image-20250318202128091" style="zoom:70%;" />

看中间的状态，将输入 $\bf X_t$ 与复制的上一层隐状态 $\bf H_{t-1}$ 连接起来，送入激活函数，输出当前时间步的隐状态 $\bf H_t$ 。

这里将 $\bf X_{t}W_{xh}+H_{t-1}W_{hh}$ 的计算等价为 $\bf cat(X_{t},H_{t-1})\cdot cat(W_{xh}+W_{hh})$ ，验证见 `notebook` 

**基于 `RNN`  的字符级语言模型** 

> 语言模型的目标：通过过去和当前词元预测下一个词元

本节是使用神经网络来进行语言建模，用 `RNN` 来构建语言模型，*字符级* 是指token的单位是 `char` 而非 `word` ，输出前先用 `softmax` 进行归一化，使用 `cross-entropy` 作为损失函数。

**困惑度** 

度量语言模型的质量，用于评估 `RNN` 

- 背景

  - 方法1：使用计算似然概率，会遇到长序列不常出现的问题，那么评估长文本出现概率必然比短文本出现概率小得多

  - 方法2：信息论的 **熵** ，衡量公式如下
    $$
    \frac{1}{n}\sum_{t=1}^{n}-\log P(x_t|x_{t-1},...,x_1)
    $$
    $P$ 由语言模型给出，$x_t$ 是实际观察到的词元

  - 方法3：困惑度 `Perplexity`

​		更常用
$$
exp(-\frac{1}{n}\sum_{t=1}{n}\log P(x_t|x_{t-1},...,x_1))
$$
​		理解：“下一个词元的实际选择数的调和平均数”。

​		理想情况下，模型总是完美估计，$P=1$ ，困惑度为 $e^0=1$ 。

​		最坏情况下，模型一个也没预测正确，$P=0$ ，那么困惑度 $e^{-({-\infty})}=e^{+\infty}=-\infty$ 

​		基线情况下，模型的预测是词表的所有可用词元上的均匀分布，困惑度等于词表中唯一词元的数	量，在没有压缩序列的情况下，这是能做到的最好的编码方式，也就是说，任何实际模型都必须超越基	线效果。困惑度的度量方式提供了一个重要上限。

**梯度裁剪** 

处理长度为 $T$ 的序列时，在反向传播计算 $T$ 个时间步上的梯度时会产生长度为 $\mathcal{O}(T)$ 的矩阵乘法链，若是 $T$ 很大，会导致数值不稳定，比如梯度消失或梯度爆炸。

在 `RNN` 训练过程中，需要使用额外的方式保证训练的稳定性

- 方法：梯度裁剪

- 原理

  $f$ 在常数 $L$ 下利普希茨连续是指，对任意 $x$ $y$ ，都有：
  $$
  |f(x)-f(y)|\leq L\|x-y\|
  $$
  因此若是在更新过程中计算得到参数的梯度为 $g$ ，那么新的参数为 $x-\eta g$ ，计算公式如下：
  $$
  |f(x)-f(x-\eta g)|\leq L\eta\|g\|
  $$
  说明每次更新的幅度不会超过 $L\eta\|g\|$ ，**好处** 是限制了事情变糟的程度，尤其是当前进方向是错误的时，**坏处** 是限制了取得进展的速度，即限制了收敛速度。

- 若是梯度 $g$ 很大，可能无法收敛，容易震荡，可以通过降低学习率 $\eta$ 来解决，问题在于出现大梯度的次数真的有那么多吗？若是不经常出现，那么降低学习率便没有什么意义，还会让收敛变慢。

  **梯度裁剪** 就是将梯度 $g$ 投影回给定半径 $\theta$ ，如下
  $$
  g\leftarrow \min(1,\frac{\theta}{\|g\|})g
  $$
  通过这样做，（1）$g$ 的范数永远不会超过 $\theta$ （2）更新后的梯度与原梯度的方向保持一致（3）限制任何小批量数据对参数向量的影响，一定程度上保证了模型的稳定性，防止梯度爆炸

  具体实现见 `notebook` 



## 通过时间反向传播

仅使用反向传播在具有隐状态的序列模型，为了计算效率，在通过时间反向传播时会缓存中间值。

详细分析序列模型的梯度计算方式

**公式** 

见原文

**梯度计算** 

- 完全计算

以下两种方法均是截断，截断是为了计算方便和训练稳定性

- 随机截断

  在 $\tau$ 步后截断，通过一个随机变量 $\xi_t$ 实现，其分布如下：
  $$
  P(\xi_t=0)=1-\pi_t \\
  P(\xi_t=\pi_t^{-1})=\pi_t
  $$
  $\xi_t$ 的期望为 $1$ ，当 $\xi_t=0$ 时，递归计算终止在这个时间步。

- 常规截断

  

比较三种策略

<img src="C:\Users\xinling\AppData\Roaming\Typora\typora-user-images\image-20250319203630787.png" alt="image-20250319203630787" style="zoom:50%;" />

从上到下依次是： （1）随机截断：文本会被划分成不同长度的片段

​				（2）常规截断：文本被划分成相同长度的子序列

​				（3）完全计算：通过时间的完全反向传播，但是会产生在计算上不可行的表达式

虽然随机截断在理论上具有很强的吸引力，但是实际表现不一定比常规截断要好：（1）在对过去若干个时间步进行截断后，观测结果足以捕获实际的依赖关系（2）增加的方差抵消了随时间步数增多而增多的梯度的精确（3）我们真正想要的是短范围内交互的模型，是截断的通过时间反向传播方法具备的的轻度正则化效果
