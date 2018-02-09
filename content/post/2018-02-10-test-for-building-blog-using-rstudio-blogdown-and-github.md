---
title: Test for Building Blog using Rstudio Blogdown and Github
date: '2018-02-10'
slug: test-for-building-blog-using-rstudio-blogdown-and-github
categories:
  - Tech
tags:
  - Blogdown
  - Github
---

> **前言**：最近正好Lab里面有可以调动的GPU资源，我就练习用``R``语言利用``Keras``接口，调用``Tensorflow``的深度学习库，构建RNN model, 发现其实很多参数的调整并不很了解，不知如何下手去调，借此机会把深度学习相关的概念都整理下。

---

#  一些基本概念

### Contents
1. [符号计算](#符号计算)
2. [张量](#张量)
3. [data_format](#data_format)
4. [函数式模型](#函数式模型)
5. [Batch](#Batch)
6. [Epochs](#Epochs)
7. [训练误差与测试误差](#训练误差与测试误差)
8. [卷积](#卷积)
9. [卷积高级概念](#卷积高级概念)
10. [softmax](#softmax)
11. [loss function](#lossfunction)
12. [activation function](#activationfunction)

### [Reference](#ref)
---

<h3 id="符号计算">1. 符号计算</h3>

Keras的底层库使用Theano或TensorFlow，这两个库也称为Keras的后端。无论是Theano还是TensorFlow，都是一个“符号式”的库。

因此，这也使得Keras的编程与传统的Python代码有所差别。笼统的说，符号主义的计算首先定义各种变量，然后建立一个“计算图”，计算图规定了各个变量之间的计算关系。建立好的计算图需要编译以确定其内部细节，然而，此时的计算图还是一个“空壳子”，里面没有任何实际的数据，只有当你把需要运算的输入放进去后，才能在整个模型中形成数据流，从而形成输出值。

就像用管道搭建供水系统，当你在拼水管的时候，里面是没有水的。只有所有的管子都接完了，才能送水。

Keras的模型搭建形式就是这种方法，在你搭建Keras模型完毕后，你的模型就是一个空壳子，只有实际生成可调用的函数后（K.function），输入数据，才会形成真正的数据流。

使用计算图的语言，如Theano，以难以调试而闻名，当Keras的Debug进入Theano这个层次时，往往也令人头痛。没有经验的开发者很难直观的感受到计算图到底在干些什么。尽管很让人头痛，但大多数的深度学习框架使用的都是符号计算这一套方法，因为符号计算能够提供关键的计算优化、自动求导等功能。

<h3 id="张量">2. 张量</h3>

张量，或tensor，是本文档会经常出现的一个词汇，在此稍作解释。使用这个词汇的目的是为了表述统一，张量可以看作是向量、矩阵的自然推广，我们用张量来表示广泛的数据类型。规模最小的张量是0阶张量，即标量，也就是一个数。当我们把一些数有序的排列起来，就形成了1阶张量，也就是一个向量，如果我们继续把一组向量有序的排列起来，就形成了2阶张量，也就是一个矩阵，把矩阵摞起来，就是3阶张量，我们可以称为一个立方体，具有3个颜色通道的彩色图片就是一个这样的立方体，把立方体摞起来，好吧这次我们真的没有给它起别名了，就叫4阶张量了，不要去试图想像4阶张量是什么样子，它就是个数学上的概念。

**张量的阶数有时候也称为维度，或者轴，轴这个词翻译自英文axis。** 譬如一个矩阵``[[1,2],[3,4]]``，是一个2阶张量，有两个维度或轴，沿着第0个轴（为了与python的计数方式一致，本文档维度和轴从0算起）你看到的是``[1,2]，[3,4]``两个向量，沿着第1个轴你看到的是``[1,3]，[2,4]``两个向量。

要理解“沿着某个轴”是什么意思，不妨试着运行一下下面的代码：

<h3 id="data_format">3. data_format</h3>

这是一个无可奈何的问题，在如何表示一组彩色图片的问题上，Theano和TensorFlow发生了分歧，'th'模式，也即Theano模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式（100,3,16,32），Caffe采取的也是这种方式。第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数。后面两个就是高和宽了。这种theano风格的数据组织方法，称为“channels_first”，即通道维靠前。

而TensorFlow，的表达形式是（100,16,32,3），即把通道维放在了最后，这种数据组织方式称为“channels_last”。

Keras默认的数据组织形式在~/.keras/keras.json中规定，可查看该文件的``image_data_format``一项查看，也可在代码中通过``K.image_data_format()``函数返回，请在网络的训练和测试中保持维度顺序一致。

<h3 id="函数式模型">4. 函数式模型</h3>

在Keras 0.x中，模型其实有两种，
- 一种叫``**Sequential**``，称为序贯模型，也就是单输入单输出，一条路通到底，层与层之间只有相邻关系，跨层连接统统没有。这种模型编译速度快，操作上也比较简单。
- 第二种模型称为**``Graph``**，即图模型，这个模型支持多输入多输出，层与层之间想怎么连怎么连，但是编译速度慢。可以看到，Sequential其实是Graph的一个特殊情况。

在Keras1和Keras2中，图模型被移除，而增加了了**``functional model API``**，这个东西，更加强调了Sequential是特殊情况这一点。一般的模型就称为Model，然后如果你要用简单的Sequential，OK，那还有一个快捷方式Sequential。

由于**``functional model API``**在使用时利用的是“函数式编程”的风格，我们这里将其译为函数式模型。总而言之，只要这个东西接收一个或一些张量作为输入，然后输出的也是一个或一些张量，那不管它是什么鬼，统统都称作“模型”。

<h3 id="Batch">5. Batch</h3>

Batch，中文为批，一个batch由若干条数据构成。batch是进行网络优化的基本单位，网络参数的每一轮优化需要使用一个batch。batch中的样本是被并行处理的。与单个样本相比，一个batch的数据能更好的模拟数据集的分布，batch越大则对输入数据分布模拟的越好，反应在网络训练上，则体现为能让网络训练的方向“更加正确”。

深度学习的优化算法，说白了就是梯度下降。每次的参数更新有两种方式。

- 第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为**Batch gradient descent，批梯度下降**。

- 另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为**随机梯度下降，stochastic gradient descent**。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。

- 为了克服两种方法的缺点，现在一般采用的是一种折中手段，**mini-batch gradient decent，小批的梯度下降**，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。

基本上现在的梯度下降都是基于mini-batch的，所以Keras的模块中经常会出现``batch_size``，就是指这个。

顺便说一句，Keras中用的优化器SGD是**stochastic gradient descent**的缩写，但不代表是一个样本就更新一回，还是基于mini-batch的。

<h3 id="Epochs">6. Epochs</h3>

训练的时候一般采用stochastic gradient descent（SGD），一次迭代选取一个batch进行update。epochs被定义为向前和向后传播中所有批次的单次训练迭代。这意味着1个周期是整个输入数据的单次向前和向后传递。简单说，epochs指的就是训练过程中数据将被“轮”多少次，就这样。

>举个例子
>训练集有1000个样本，batchsize=10，那么： 
>训练完整个样本集需要： 
>100次iteration，1次epoch。
>具体的计算公式为： 
>one epoch = numbers of iterations = N = 训练样本的数量/batch_size

>注：
>在LSTM中我们还会遇到一个``seq_length``,其实 
>``batch_size = num_steps * seq_length``

<h3 id="训练误差与测试误差">7. 训练误差与测试误差</h3>


一个Keras的模型有两个模式：训练模式和测试模型。一些正则机制，如Dropout，L1/L2正则项在测试模式下将不被启用。

另外，**训练误差**是训练数据每个batch的误差的平均。在训练过程中，每个epoch起始时的batch的误差要大一些，而后面的batch的误差要小一些。另一方面，每个epoch结束时计算的**测试误差**是由模型在epoch结束时的状态决定的，这时候的网络将产生较小的误差。

> Tips: 可以通过定义回调函数将每个epoch的训练误差和测试误差并作图，如果训练误差曲线和测试误差曲线之间有很大的空隙，说明你的模型可能有过拟合的问题。


<h3 id="卷积">8. 卷积</h3>

粗略来讲，什么是卷积呢？

你可以把卷积想象成一种**混合信息的手段**。想象一下装满信息的两个桶，我们把它们倒入一个桶中并且通过某种规则搅拌搅拌。也就是说卷积是一种混合两种信息的流程。

卷积也可以形式化地描述，事实上，它就是一种数学运算，跟减加乘除没有本质的区别。虽然这种运算本身很复杂，但它非常有助于简化更复杂的表达式。在物理和工程上，卷积被广泛地用于化简等式——等会儿简单地形式化描述卷积之后——我们将把这些领域的思想和深度学习联系起来，以加深对卷积的理解。但现在我们先从实用的角度理解卷积。

##### 如何对图像应用卷积

当我们在图像上应用卷积时，我们在两个维度上执行卷积——水平和竖直方向。我们混合两桶信息：第一桶是输入的图像，由三个矩阵构成——RGB 三通道，其中每个元素都是 0 到 255 之间的一个整数。第二个桶是卷积核（kernel），单个浮点数矩阵。可以将卷积核的大小和模式想象成一个搅拌图像的方法。卷积核的输出是一幅修改后的图像，在深度学习中经常被称作 feature map。对每个颜色通道都有一个 feature map。

![iamge](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2bf8abb86.jpg)

这是怎么做到的呢，我们现在演示一下如何通过卷积来混合这两种信息。一种方法是从输入图片中取出一个与卷积核大小相同的区块——这里假设图片为 100×100，卷积核大小为 3×3，那么我们取出的区块大小就是 3×3——然后对每对相同位置的元素执行乘法后求和（不同于矩阵乘法，却类似向量内积，这里是两个相同大小的矩阵的「点乘」）。乘积的和就生成了 feature map 中的一个像素。当一个像素计算完毕后，移动一个像素取下一个区块执行相同的运算。当无法再移动取得新区块的时候对 feature map 的计算就结束了。这个流程可以用如下的动画演示：

![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2c23ee990.gif)

你可能注意到这里有个正规化因子 m，这里 m 的值为 kernel 的大小 9；这是为了保证输入图像和 feature map 的亮度相同\

##### 为什么机器学习中图像卷积有用

图像中可能含有很多我们不关心的噪音。一个好例子是我和 Jannek Thomas 在 Burda Bootcamp 做的项目。Burda Bootcamp 是一个让学生像黑客马拉松一样在非常短的时间内创造技术风暴的实验室。与 9 名同事一起，我们在 2 个月内做了 11 个产品出来。 其中之一是针对时尚图像用深度编码器做的搜索引擎：你上传一幅时尚服饰的图片，编码器自动找出款式类似的服饰。

如果你想要区分衣服的式样，那么衣服的颜色就不那么重要了；另外像商标之类的细节也不那么重要。最重要的可能是衣服的外形。一般来讲，女装衬衫的形状与衬衣、夹克和裤子的外观非常不同。如果我们过滤掉这些多余的噪音，那我们的算法就不会因颜色、商标之类的细节分心了。我们可以通过卷积轻松地实现这项处理。

我的同事Jannek Thomas通过索贝尔边缘检测滤波器（与上上一幅图类似）去掉了图像中除了边缘之外的所有信息——这也是为什么卷积应用经常被称作滤波而卷积核经常被称作滤波器（更准确的定义在下面）的原因。由边缘检测滤波器生成的 feature map 对区分衣服类型非常有用，因为只有外形信息被保留下来。

![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2c52a09f2.jpg)
*彩图的左上角是搜索 query，其他是搜索结果，你会发现自动编码器真的只关注衣服的外形，而不是颜色。*

再进一步：有许多不同的核可以产生多种 feature map，比如锐化图像（强调细节），或者模糊图像（减少细节），并且每个 feature map 都可能帮助算法做出决策（一些细节，比如衣服上有 3 个纽扣而不是两个，可能可以区分一些服饰）。

使用这种手段——读入输入、变换输入、然后把feature map 喂给某个算法——被称为特征工程。特征工程非常难，很少有资料帮你上手。造成的结果是，很少有人能熟练地在多个领域应用特征工程。特征工程是——纯手工——也是 Kaggle 比赛中最重要的技能。

**特征工程这么难的原因是，对每种数据每种问题，有用的特征都是不同的：图像类任务的特征可能对时序类任务不起作用；即使两个任务都是图像类的，也很难找出相同的有效特征，因为视待识别的物体的不同，有用的特征也不同。这非常依赖经验。**

所以特征工程对新手来讲特别困难。不过对图像而言，是否可以利用卷积核自动找出某个任务中最适合的特征？

##### 进入卷积神经网络

卷积神经网络就是干这个的。不同于刚才使用固定数字的卷积核，我们赋予参数给这些核，参数将在数据上得到训练。随着卷积神经网络的训练，这些卷积核为了得到有用信息，在图像或 feature map 上的过滤工作会变得越来越好。这个过程是自动的，称作特征学习。特征学习自动适配新的任务：我们只需在新数据上训练一下自动找出新的过滤器就行了。这是卷积神经网络如此强大的原因——不需要繁重的特征工程了！

通常卷积神经网络并不学习单一的核，而是同时学习多层级的多个核。比如一个 32x16x16 的核用到 256×256 的图像上去会产生 32 个 241×241（latex.png）的 feature map。所以自动地得到了 32 个有用的新特征。这些特征可以作为下个核的输入。一旦学习到了多级特征，我们简单地将它们传给一个全连接的简单的神经网络，由它完成分类。这就是在概念上理解卷积神经网络所需的全部知识了（池化也是个重要的主题，但还是在另一篇博客中讲吧）。

<h3 id="卷积高级概念">9. 卷积高级概念</h3>

我们将认识到刚才对卷积的讲解是粗浅的，并且这里有更优雅的解释。通过深入理解，我们可以理解卷积的本质并将其应用到许多不同的数据上去。万事开头难，第一步是理解卷积原理。

#### 卷积定理
要理解卷积，不得不提convolution theorem, 它将时域和空域上的复杂卷积对应到了频域中的元素间简单的乘积。这个定理非常强大，在许多科学领域中得到了广泛应用。卷积定理也是快速傅里叶变换算法被称为 20 世纪最重要的算法之一的一个原因。

$$h(x) = f \otimes g=\int_{-\infty}^{\infty}f(x-u)g(u)du = \mathcal{F}^{-1}\left(\sqrt{2\pi}\mathcal{F}|f|\mathcal{F}|g| \right)$$

$$feature map = input\otimes kernel = \sum_{y=0}^{columns}\left(\sum_{x=0}^{rows}input(x-a, y-b)kernel(x,y) \right)=\mathcal{F}^{-1}\left(\sqrt{2\pi}\mathcal{F}|input|\mathcal{F}|kernel|\right)$$

第一个等式是一维连续域上两个连续函数的卷积；第二个等式是二维离散域（图像）上的卷积.

> $\otimes$指的是卷积，$\mathcal{F}$指的是傅立叶变换, $\mathcal{F}^{-1}$指的是傅立叶逆变换，$\sqrt{2\pi}$是一个正规化常量。这里的**离散**指的是数据由有限个变量构成“像素”；一维指的是一维的“时间”，图像则是二维的，视频则是三维的。

#### 快速傅立叶变换
快速傅里叶变换是一种将时域和空域中的数据转换到频域上去的算法。**傅里叶变换用一些正弦和余弦波的和来表示原函数**。必须注意的是，傅里叶变换一般涉及到复数，也就是说一个实数被变换为一个具有实部和虚部的复数。通常虚部只在一部分领域有用，比如将频域变换回到时域和空域上；而在这篇博客里会被忽略掉。你可以在下面看到一个信号（一个以时间为参数的有周期的函数通常称为信号）是如何被傅里叶变换的：

![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2d1e16eba.gif)

*红色是时域，蓝色为频域*

##### 傅立叶域上的图像
![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2d9ce3610.jpg)

我们如何想象图片的频率呢？想象一张只有两种模式的纸片，现在把纸片竖起来顺着线条的方向看过去，就会看到一个一个的亮点。这些以一定间隔分割黑白部分的波就代表着频率。在频域中，低频率更接近中央而高频率更接近边缘。频域中高强度（亮度、白色）的位置代表着原始图像亮度改变的方向。这一点在接下来这张图与其对数傅里叶变换（对傅里叶变换的实部取对数，这样可以减小像素亮度的差别，便于观察更广的亮度区域）中特别明显：

![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2dbe22977.jpg)

#### 频率过滤与卷积
为什么卷积经常被描述为过滤，为什么卷积核经常被称为过滤器呢？通过下一个例子可以解释：
![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2dd9ad2e3.jpg)

如果我们对图像执行傅里叶变换，并且乘以一个圆形（背景填充黑色，也就是 0），我们可以过滤掉所有的高频值（它们会成为 0，因为填充是 0）。注意过滤后的图像依然有条纹模式，但图像质量下降了很多——这就是 jpeg 压缩算法的工作原理（虽然有些不同但用了类似的变换），我们变换图形，然后只保留部分频率，最后将其逆变换为二维图片；压缩率就是黑色背景与圆圈的比率。

我们现在将圆圈想象为一个卷积核，然后就有了完整的卷积过程——就像在卷积神经网络中看到的那样。要稳定快速地执行傅里叶变换还需要许多技巧，但这就是基本理念了。

现在我们已经理解了卷积定理和傅里叶变换，我们可以将这些理念应用到其他科学领域，以加强我们对深度学习中的卷积的理解。

#### 流体力学的启发
流体力学为空气和水创建了大量的微分方程模型，傅里叶变换不但简化了卷积，也简化了微分，或者说任何利用了微分方程的领域。有时候得到解析解的唯一方法就是对微分方程左右同时执行傅里叶变换。在这个过程中，我们常常将解写成两个函数卷积的形式，以得到更简单的表达。这是在一个维度上的应用，还有在两个维度上的应用，比如天文学。

**扩散**

你可以混合两种液体（牛奶和咖啡），只要施加一个外力（汤勺搅拌）——这被称为对流，而且是个很快的过程。你也可以耐心等待两种液体自然混合——这被称为扩散，通常是很慢的过程。

想象一下，一个鱼缸被一块板子隔开，两边各有不同浓度的盐水。抽掉板子后，两边的盐水会逐步混合为同一个浓度。浓度差越大，这个过程越剧烈。

现在想象一下，一个鱼缸被 256×256 个板子分割为 256×256 个部分（这个数字似乎不对），每个部分都有不同浓度的盐水。如果你去掉所有的挡板，浓度类似的小块间将不会有多少扩散，但浓度差异大的区块间有巨大的扩散。这些小块就是像素点，而浓度就是像素的亮度。浓度的扩散就是像素亮度的扩散。

这说明，扩散现象与卷积有相似点——初始状态下不同浓度的液体，或不同强度的像素。为了完成下一步的解释，我们还需要理解传播子。

**理解传播子**

传播子就是密度函数，表示流体微粒应该往哪个方向传播。问题是神经网络中没有这样的概率函数，只有一个卷积核——我们要如何统一这两种概念呢？

我们可以通过正规化来讲卷积核转化为概率密度函数。这有点像计算输出值的 softmax。下面就是对第一个例子中的卷积核执行的 softmax 结果：

![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2e0d93931.jpg)
现在我们就可以从扩散的角度来理解图像上的卷积了。我们可以把卷积理解为两个扩散流程。首先，当像素亮度改变时（黑色到白色等）会发生扩散；然后某个区域的扩散满足卷积核对应的概率分布。这意味着卷积核正在处理的区域中的像素点必须按照这些概率来扩散。

在上面那个边缘检测器中，几乎所有临近边缘的信息都会聚集到边缘上（这在流体扩散中是不可能的，但这里的解释在数学上是成立的）。比如说所有低于 0.0001 的像素都非常可能流动到中间并累加起来。与周围像素区别最大的区域会成为强度的集中地，因为扩散最剧烈。反过来说，强度最集中的地方说明与周围对比最强烈，这也就是物体的边缘所在，这解释了为什么这个核是一个边缘检测器。

所以我们就得到了物理解释：卷积是信息的扩散。我们可以直接把这种解释运用到其他核上去，有时候我们需要先执行一个 softmax 正规化才能解释，但一般来讲核中的数字已经足够说明它想要干什么。比如说，你是否能推断下面这个核的的意图？
![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2e270317e.jpg)

#### 量子力学的启发

传播子是量子力学中的重要概念。在量子力学中，一个微粒可能处于一种叠加态，此时它有两个或两个以上属性使其无法确定位于观测世界中的具体位置。比如，一个微粒可能同时存在于两个不同的位置。

但是如果你测量微粒的状态——比如说现在微粒在哪里——它就只能存在于一个具体位置了。换句话说，你通过观测破坏了微粒的叠加态。传播子就描述了微粒出现位置的概率分布。比如说在测量后一个微粒可能——根据传播子的概率函数——30% 在 A，70% 在 B。
通过量子纠缠，几个粒子就可以同时储存上百或上百万个状态——这就是量子计算机的威力。

如果我们将这种解释用于深度学习，我们可以把图片想象为位于叠加态，于是在每个 3*3 的区块中，每个像素同时出现在 9 个位置。一旦我们应用了卷积，我们就执行了一次观测，然后每个像素就坍缩到满足概率分布的单个位置上了，并且得到的单个像素是所有像素的平均值。为了使这种解释成立，必须保证卷积是随机过程。这意味着，同一个图片同一个卷积核会产生不同的结果。这种解释没有显式地把谁比作谁，但可能启发你如何把卷积用成随机过程，或如何发明量子计算机上的卷积网络算法。量子算法能够在线性时间内计算出卷积核描述的所有可能的状态组合。

#### 概率论的启发

卷积与互相关紧密相连。互相关是一种衡量小段信息（几秒钟的音乐片段）与大段信息（整首音乐）之间相似度的一种手段（youtube 使用了类似的技术检测侵权视频）。

$$convoluted\ x = f \otimes g= \int_{-\infty}^{\infty}f(x-u)g(u)du=\mathcal{F}^{-1}\left(\sqrt{2\pi}\mathcal{F}|f|\mathcal{F}|g| \right)$$

$$cross-correlated \ x=f*g=\int_{-\infty}^{\infty}f(x-u)g(u)^{*}du=\mathcal{F}^{-1}\left(\sqrt{2\pi}\mathcal{F}|f|(\mathcal{F}|g|)^{*} \right)$$

$$f(x)*g(x) = f^{*}(-x)\otimes g(x)$$

虽然互相关的公式看起来很难，但通过如下手段我们可以马上看到它与深度学习的联系。在图片搜索中，我们简单地将 query 图片上下颠倒作为核然后通过卷积进行互相关检验，结果会得到一张有一个或多个亮点的图片，亮点所在的位置就是人脸所在的位置。

![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2e624d8e9.jpg)

这个例子也展示了通过补零来使傅里叶变换稳定的一种技巧，许多版本的傅里叶变换都使用了这种技巧。另外还有使用了其他 padding 技巧：比如平铺核，分治等等。我不会展开讲，关于傅里叶变换的文献太多了，里面的技巧特别多——特别是对图像来讲。

在更底层，卷积网络第一层不会执行互相关校验，因为第一层执行的是边缘检测。后面的层得到的都是更抽象的特征，就有可能执行互相关了。可以想象这些亮点像素会传递给检测人脸的单元（Google Brain 项目的网络结构中有一些单元专门识别人脸、猫等等；也许用的是互相关？）

#### 统计学的启发

统计模型和机器学习模型的区别是什么？统计模型只关心很少的、可以解释的变量。它们的目的经常是回答问题：药品 A 比药品 B 好吗？

机器学习模型是专注于预测效果的：对于年龄 X 的人群，药品 A 比 B 的治愈率高 17%，对年龄 Y 则是 23%。

机器学习模型通常比统计模型更擅长预测，但它们不是那么可信。统计模型更擅长得到准确可信的结果：就算药品 A 比 B 好 17%，我们也不知道这是不是偶然，我们需要统计模型来判断。

对时序数据，有两种重要的模型：**weighted moving average** 和 **autoregressive** 模型，后者可归入 ARIMA model (autoregressive integrated moving average model)。比起 LSTM，ARIMA 很弱。但在低维度数据（1-5 维）上，ARIMA 非常健壮。虽然它们有点难以解释，但 ARIMA 绝不是像深度学习算法那样的黑盒子。如果你需要一个可信的模型，这是个巨大的优势。

> 注：我们可以将这些统计模型写成卷积的形式，然后深度学习中的卷积就可以解释为产生局部 ARIMA 特征的函数了。这两种形式并不完全重合，使用需谨慎。

$$autoregressed\ x=C(kernel) + white noise \otimes kernel$$

$$weighted\ moving\ averaged \ x=input \otimes kernel $$

$C$是一个以核为参数的函数，$white\ nose$是正则化的均值为0，方差为1的互不相关的数据。

当我们预处理数据的时候，经常将数据处理为类似 white noise 的形式：将数据移动到均值为 0，将方差调整为 1。我们很少去除数据的相关性，因为计算复杂度高。但是在概念上是很简单的，我们旋转坐标轴以重合数据的特征向量：

![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2f05b0c4c.jpg)

现在如果我们将 C 作为 bias，我们就会觉得这与卷积神经网络很像。所以:

- **卷积层的输出可被解释为白噪音数据经过 autoregressive model 的输出。**

- **weighted moving average 的解释更简单：就是输入数据与某个固定的核的卷积。**

看看文末的高斯平滑核就会明白这个解释。高斯平滑核可以被看做每个像素与其邻居的平均，或者说每个像素被其邻居平均（边缘模糊）。

虽然单个核无法同时创建 autoregressive 和 weighted moving average 特征，但我们可以使用多个核来产生不同的特征。

#### 总结
这篇博客中我们知道了卷积是什么、为什么在深度学习中这么有用。图片区块的解释很容易理解和计算，但有其理论局限性。我们通过学习傅里叶变换知道傅里叶变换后的时域上有很多关于物体朝向的信息。通过强大的卷积定理我们理解了卷积是一种在像素间的信息流动。之后我们拓展了量子力学中传播子的概念，得到了一个确定过程中的随机解释。我们展示了互相关与卷积的相似性，并且卷积网络的性能可能是基于 feature map 间的互相关程度的，互相关程度是通过卷积校验的。最后我们将卷积与两种统计模型关联了起来。

个人来讲，我觉得写这篇博客很有趣。曾经很长一段时间我都觉得本科的数学和统计课是浪费时间，因为它们太不实用了（哪怕是应用数学）。但之后——就像突然中大奖一样——这些知识都相互串起来了并且带了新的理解。我觉得这是个绝妙的例子，启示我们应该耐心地学习所有的大学课程——哪怕它们一开始看起来没有用。

![image](https://www.jiqizhixin.com/data/upload/ueditor/20170328/58da2f67133d9.png)
*上文高斯平滑核问题的答案*

<h3 id="softmax">10. Softmax </h3>
softmax函数将K维的实数向量压缩成另一个k维的实数向量，其中向量中的每个元素取值都介于(0，1)之间。softmax主要用来做多分类问题，是logistic回归模型在多分类问题上的推广，softmax公式:

$$\sigma(z)_{j}=\frac{e^{z_j}}{\sum_{k=1}^{K}e^{z_k}}$$

> 当$k=2$时，转换为逻辑回归形式

softmax一般作为神经网络最后一层，作为输出层进行多分类，softmax的输出的每个值都是>=0，并且其总和为1，所以可以认为其为概率分布。

![image](http://upload-images.jianshu.io/upload_images/2316091-6b9dd389de4a8473.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*softmax示意图*

![image](http://upload-images.jianshu.io/upload_images/2316091-53011bceedc5c6b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*softmax输出层示意图*


```python
%pylab inline
Populating the interactive namespace from numpy and matplotlib

# load packages
from IPython.display import SVG
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.optimizers import SGD, Adam
from keras.utils.visualize_util import model_to_dot
from keras.utils import np_utils
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# Using Tensorflow backend
#设置随机数种子,保证实验可重复
import numpy as np
np.random.seed(0)
#设置线程
THREADS_NUM = 20
tf.ConfigProto(intra_op_parallelism_threads=THREADS_NUM)

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
print('原数据结构：')
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#数据变换
#分为10个类别
nb_classes = 10

x_train_1 = X_train.reshape(60000, 784)
#x_train_1 /= 255
#x_train_1 = x_train_1.astype('float32')
y_train_1 = np_utils.to_categorical(Y_train, nb_classes)
print('变换后的数据结构：')
print(x_train_1.shape, y_train_1.shape)

x_test_1 = X_test.reshape(10000, 784)
y_test_1 = np_utils.to_categorical(Y_test, nb_classes)
print(x_test_1.shape, y_test_1.shape)

```
原数据结构：
```
((60000, 28, 28), (60000,))
((10000, 28, 28), (10000,))
```
变换后的数据结构：
```
((60000, 784), (60000, 10))
((10000, 784), (10000, 10))
```
```Python
# 构建一个softmax模型
# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

model = Sequential()
model.add(Dense(nb_classes, input_shape=(784,)))#全连接，输入784维度, 输出10维度，需要和输入输出对应
model.add(Activation('softmax'))

sgd = SGD(lr=0.005)
#binary_crossentropy，就是交叉熵函数
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#model 概要
model.summary()
```
```Python
_________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
=========================================================================================
dense_1 (Dense)                  (None, 10)            7850        dense_input_1[0][0]
_________________________________________________________________________________________
activation_1 (Activation)        (None, 10)            0           dense_1[0][0]
=========================================================================================
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0
_________________________________________________________________________________________
```
```Python
SVG(model_to_dot(model).create(prog='dot', format='svg'))
```
```Python
from keras.callbacks import Callback, TensorBoard
import tensorflow as tf

#构建一个记录的loss的回调函数
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 构建一个自定义的TensorBoard类，专门用来记录batch中的数据变化
class BatchTensorBoard(TensorBoard):
    def __init__(self,log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False):
        super(BatchTensorBoard, self).__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.batch = 0
        self.batch_queue = set()

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_end(self,batch,logs=None):
        logs = logs or {}

        self.batch = self.batch + 1

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = float(value)
            summary_value.tag = "batch_" + name
            if (name,self.batch) in self.batch_queue:
                continue
            self.writer.add_summary(summary, self.batch)
            self.batch_queue.add((name,self.batch))
        self.writer.flush()
```
```Python
tensorboard = TensorBoard(log_dir='/home/tensorflow/log/softmax/epoch')
my_tensorboard = BatchTensorBoard(log_dir='/home/tensorflow/log/softmax/batch')

model.fit(x_train_1, y_train_1,
          nb_epoch=20,
          verbose=0,
          batch_size=100,
          callbacks=[tensorboard, my_tensorboard])
```

<h3 id="lossfunction">11. loss function</h3>
损失函数(loss function)，是一种将一个事件（在一个样本空间中的一个元素）映射到一个表达与其事件相关的经济成本或机会成本的实数上的一种函数；在统计学中损失函数是一种衡量损失和错误（这种损失与“错误的”估计有关，如费用或者设备的损失）程度的函数。

**交叉熵(Cross-entropy)**就是神经网络中常用的损失函数。交叉熵性质：
- 非负性
- 当真实输出$a$与期望输出 $y$接近的时候，代价函数接近于0.(比如$y=0,  a\sim 0; y=1, a\sim 1$时，代价函数都接近 $0$)

![image](https://image-1251363007.file.myqcloud.com/2017/02/23/3fc7890d012d45b683e9163a84e9275a.jpeg)

一个比较简单的理解就是使得预测值Yi和真实值Y'对接近，即两者的乘积越大，coss-entropy越小。

交叉熵和准确度变化图像可以看 TensorBoard 。

<h3 id="activationfunction">12. Activation function</h3>
日常 coding 中，我们会很自然的使用一些激活函数，比如：sigmoid、ReLU等等。不过好像忘了问自己一(n[Math Processing Error])件事：

- 为什么需要激活函数？
- 激活函数都有哪些？都长什么样？有哪些优缺点？
- 怎么选用激活函数？

#### 为什么需要激活函数？

激活函数通常有如下特性，这也是我们使用激活函数的原因:
- **非线性**，当激活函数是线性的时候，一个两层的神经网络就可以逼近基本上所有的函数了。但如果激活函数是恒等激活函数的时候，即`$f(x)=x$`，就不满足这个性质了，而且如果MLP使用的是恒等激活函数，那么其实整个网络跟单层神经网络是等价的。
- **可微性**，当优化方法是基于梯度的时候，这个性质是必须的
- **单调性**，当激活函数是单调的时候，单层网络能够保证是凸函数
- $f(x)\approx x$：当激活函数满足这个性质的时候，如果参数的初始化是random的很小的值，那么神经网络的训练将会很高效；如果不满足这个性质，那么需要很用心的去设置初始值。
- **输出值的范围**：当激活函数输出值是有限的时候，基于梯度的优化方法会更加稳定，因为特征的表示受有限值的影响更显著；当激活函数的输出是无限的时候，模型的输出会更加高效，不过在这种情况下，一般需要更小的learning rate

#### 激活函数

##### Sigmoid

![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-sigmoid_relu.png)

Sigmoid也称为S形函数，取值范围为(0, 1)。Sigmoid将一个实数映射到(0，1)的区间，可以用来做二分类。Sigmoid在特征相差比较复杂或者相差不大是特别大时效果比较好。Sigmoid是常用的非线性激活函数，他的数学形式如下：
$$
f(x)=\frac{1}{1+e^{-x}}
$$
正如前一节提到的，它能够把输入的连续实值“压缩”到0和1之间。如果是，非常大的负数，那么输出就是0，如果是非常大的正数，输出就是1. Sigmoid被使用很多，不过近几年，用它的人越来越少，主要是因为一些**缺点**：
- Sigmoids saturate and kill gradients: sigmoid 有一个非常致命的缺点，当输入非常大或者非常小的时候（saturation），这些神经元的梯度是接近于0的，从图中可以看出梯度的趋势。所以，你需要尤其注意参数的初始值来尽量避免saturation的情况。如果你的初始值很大的话，大部分神经元可能都会处在saturation的状态而把gradient kill掉，这会导致网络变的很难学习。
- Sigmoid的output不是0均值，这是不可取的，因为这会导致后一层的神经元将得到上一层输出的非0均值的信号作为输入。产生的一个结果就是：如果数据进入神经元的时候是正的, e.g. $x>0$, elementwise in $f = w^T+b$, 那么$w$计算出的梯度也会始终是正的。

##### tanh

tanh是上图中的右图，可以看出，tanh跟sigmoid还是很相似的，实际上，tanh是sigmoid的变形：
$$
tanh(x)=2*sigmoid(2x)-1
$$
与sigmoid不同的是，tanh是0均值的，因此，实际应用中，tanh会比sigmoid更好

##### ReLU

![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-relu.png)

近年来，ReLU变得越来越受欢迎，它的数学表达式如下：
$$
f(x)=max(0, x)
$$
很显然，从图左可以看出，输入信号<0时，输出都是0，输入大于零的情况下，输出等于输入。$w$是二维的情况下，使用ReLU之后的效果图如下：

![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-relu-perf.png)

**ReLU的优点**

- [Krizhevsky et al.](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)发现使用 ReLU 得到的SGD的收敛速度会比 sigmoid/tanh 快很多(看右图)。有人说这是因为它是linear，而且 non-saturating
- 相比于 sigmoid/tanh，ReLU 只需要一个阈值就可以得到激活值，而不用去算一大堆复杂的运算。

**ReLU的缺点**
ReLU 的缺点就是训练的时候很”脆弱”，很容易就”die”，也就是，一个非常大的梯度流过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了。实际操作中，如果你的learning rate 很大，那么很有可能你网络中的40%的神经元都”dead”了。 
当然，如果你设置了一个合适的较小的learning rate，这个问题发生的情况其实也不会太频繁。

**Leaky-ReLU**

leaky-ReLU就是来解决“dying ReLU”的问题的，与ReLU不同的是：
$$
f(x)=ax, \ (x<0)
$$

$$
f(x)=x, \ (x>=0)
$$

在这里，$a$是一个很小的常数，这样，即修正了数据分布，又保留了一些负轴的值，使得负轴信息不会全部丢失。

![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-leaky.png)

关于Leaky ReLU 的效果，众说纷纭，没有清晰的定论。有些人做了实验发现 Leaky ReLU 表现的很好；有些实验则证明并不是这样。

**P-ReLU**

Parametric ReLU: 对于Leaky ReLU中的 $a$, 通常是通过先验知识人工赋值的。然而可以观察到，损失函数对α的导数我们是可以求得的，可不可以将它作为一个参数进行训练呢？ 

Kaiming He的论文《Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》指出，不仅可以训练，而且效果更好。原文说使用了Parametric ReLU后，最终效果比不用提高了1.03%.

**R-ReLU**

Random ReLU是leaky ReLU的random版本，也就是 $a$ 是random的，它首次在kaggle的NDSB比赛中被提出的。

核心思想就是，在训练过程中， $a$ 是从高斯缝补 $U(l,u)$中随机出来的，然后再测试过程中进行修正。数学表示如下：

![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-rrelu.png)

在测试阶段，把训练过程中所有的 $a_{ij}$ 取个平均值。NDSB冠军的 $a$ 是从 $U(3,8)$ 中随机出来的，那么在测试阶段，激活函数就是：
$$
y_{ij} = \frac{x_{ij}}{\frac{l+u}{2}}
$$
看看cifar-100中的实验结果:

![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-per.png)

**Maxout**

![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-mm.png)

Maxout出现在ICML2013上，作者Goodfellow将maxout和dropout结合后，号称在MNIST, CIFAR-10, CIFAR-100, SVHN这4个数据上都取得了start-of-art的识别率。Maxout 公式如下： 
$$
f_i(x)=max_{j\in [1,k]}z_{ij}
$$
假设 $w$ 是2维，那么有：
$$
f(x)=max(w_1^Tx+b_1, w_2^Tx+b_2)
$$
可以注意到，ReLU和Leaky ReLU都是它的一个变形，比如，$w_1, \ b_1=0$ 的时候，就是ReLU.

Maxout的拟合能力是非常强的，它可以拟合任意的的凸函数。作者从数学的角度上也证明了这个结论，即只需2个maxout节点就可以拟合任意的凸函数了（相减），前提是”隐隐含层”节点的个数可以任意多。

![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-maxout2.png)

所以，Maxout 具有 ReLU 的**优点**（如：计算简单，不会 saturation），同时又没有 ReLU 的一些**缺点**, （如：容易 go die）。不过呢，还是有一些缺点的嘛：就是把参数double了。

**其他激活函数**

![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-ac1.png)
![image](http://7pn4yt.com1.z0.glb.clouddn.com/blog-ac2.png)

 #### 如何选用激活函数?

注意事项：

1. 如果你使用 ReLU，那么一定要小心设置 learning rate，而且要注意不要让你的网络出现很多 “dead” 神经元，如果这个问题不好解决，那么可以试试 Leaky ReLU、PReLU 或者 Maxout.

2. 最好不要用 sigmoid，你可以试试 tanh，不过可以预期它的效果会比不上 ReLU 和 Maxout.

3. 通常来说，很少会把各种激活函数串起来在一个网络中使用的。

<h3 id="ref">Reference</h3>
1. [深度学习入门必须理解这25个概念](https://cloud.tencent.com/community/article/924905)
2. [Keras中文文档](http://keras-cn.readthedocs.io/en/latest/)
3. [深度学习的一些基本概念](http://www.jianshu.com/p/6f6e9903029a)
4. [如何开启深度学习之旅？这三大类125篇论文为你导航](https://www.jiqizhixin.com/articles/2017-03-06-2)
5. [Machine Learning, Andrew Ng](http://openclassroom.stanford.edu/MainFolder/)