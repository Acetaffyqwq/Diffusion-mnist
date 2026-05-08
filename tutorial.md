encode 过程是一个不断加噪声的过程，直到最后变成完全噪声。

$$
\begin{align}
x_{t}=\sqrt{ 1-\beta_{t} }x_{t-1}+\sqrt{ \beta_{t} }N(0,1) \\
q(x_{t}|x_{t-1})=N(x_{t};\sqrt{ 1-\beta_{t} }x_{t-1}, \beta_{t}I)
\end{align}
$$

这样保证方差一直是 $1$ （最前面已经归一化），并且均值是逐步下降的，最后几乎变成 $N(0,1)$.

这是「正向传播」，如果能学习一个「降噪」神经网络，$p_{\theta}(x_{t-1}|(x_{t},t))$，那么从 $N(0,1)$ 出发回退就可以得到一张样本中的图片。

## 重参数化

想要快速得到一个 $x_{t}$ 样本，但是不能一项一项递推，令 $\alpha=1-\beta$.

$$
\begin{align}
x_{t}=\sqrt{ \alpha_{t} }x_{t-1}+\sqrt{1- \alpha_{t} }\epsilon \\
=\sqrt{ \alpha_{t}\alpha_{t-1} }x_{t-2}+(\sqrt{\alpha_{t}(1-\alpha_{t-1}) }+\sqrt{ 1-\alpha_{t} })\epsilon \\
=\sqrt{ \alpha_{t}\alpha_{t-1} }x_{t-2}+\sqrt{ 1-\alpha_{t}\alpha_{t-1} }\epsilon
\end{align}
$$

最后一步是因为是两个正态分布加起来，均值还是 $0$ ，因此只要算一下方差（也就是平方和）即可。

归纳得到，$\overline \alpha_{t}=\prod_{i=1}^t \alpha_{i}$.

$$
x_{t}= \sqrt{ \overline \alpha_{t} }x_{0}+\sqrt{ 1-\overline a_{t} } \epsilon
$$

直接重参数，取一个 $\epsilon$，就可以生成一组 $x_{t}$.

## 猜想的做法

既然想要训练一个还原的神经网络，那么直接的想法就是给参数 $(t,x_{t})$ 就得到上一层加了什么噪声。
那就按照前面的重参数方法一步直接生成一个 $x_{t}$，然后预测其上一步噪声 $\epsilon_{t}$（也应该是一个 $N(0,1)$ 中的采样结果）。然后拿这个噪声去 MSE 一下真实的，就可以一步一步还原了。

具体怎么还原一步呢：

$$
\begin{align}
q(x_{t-1}|(x_{t},x_{0}))= \frac{q(x_{t}|(x_{t-1},x_{0}))q(x_{t-1}|x_{0})}{q(x_{t}|x_{0})}
\end{align}
$$

右边三个都是正态分布，容易证明正态分布之间乘除还是正态分布，所以左边可以表示成：

$$
q(x_{t-1}|(x_{t},x_{0})) = N(x_{t-1};\mu(x_{t},x_{0}),\sigma^2(x_{t},x_{0}))
$$

具体解出 $\mu(x_{t},x_{0}),\beta(x_{t},x_{0})$.

$$
\begin{align}
q(x_{t-1}|(x_{t},x_{0}))=\frac{q(x_{t}|(x_{t-1},x_{0}))q(x_{t-1}|x_{0})}{q(x_{t}|x_{0})} \\
\propto \exp \left( -\frac{1}{2} \left(  \frac{(x_{t}-\sqrt{ \alpha_{t} }x_{t-1})^2}{\beta_{t}} + \frac{(x_{t-1}-\sqrt{ \overline \alpha_{t-1} }x_{0})^2}{1-\overline \alpha_{t-1}} -\frac{(x_{t}-\sqrt{ \overline \alpha_{t}x_{0} })^2}{1-\overline \alpha_{t}} \right) \right) \\
=\exp\left(  -\frac{1}{2} \left(  {\color{red} \left(  \frac{\alpha_{t}}{\beta_{t}}+\frac{1}{1-\overline a_{t-1}} \right)} x^2_{t-1}+ {\color{cyan} \left( -\frac{2\sqrt{ \alpha_{t} }x_{t}}{\beta_{t}}- \frac{2\sqrt{ \overline \alpha_{t-1} }x_{0}}{1-\overline \alpha_{t-1}} \right)}x_{t-1} \right) + C(x_{t},x_{0}) \right)
\end{align}
$$

因为要解的是 $x_{t-1}$ 的分布，所以别的和 $x_{t},x_{0}$ 有关的东西可以看成常数。
于是得到：

$$
\begin{align}
\sigma^2(x_{t},x_{0}) =  1/ \left( \frac{\alpha_{t}}{\beta_{t}}+\frac{1}{1-\overline a_{t-1}} \right)= \frac{1-\overline a_{t-1}}{1-\overline a_{t}} \beta_{t}=\sigma^2_{t} \\
\mu(x_{t},x_{0})=\frac{ \frac{\sqrt{ \alpha_{t} }x_{t}}{\beta_{t}}+\frac{\sqrt{ \overline \alpha_{t-1} }x_{0}}{1-\overline \alpha_{t-1}} }{\frac{\alpha_{t}}{\beta_{t}}+\frac{1}{1-\overline a_{t-1}}} \\
=\frac{1}{1-\overline \alpha_{t}} ( (1-\overline \alpha_{t-1})\sqrt{ \alpha_{t} }x_{t}+\sqrt{ \overline \alpha_{t-1} }\beta_{t} x_{0}  )
\end{align}
$$

但是我们实际上在逆推时不能用 $x_{0}$ ，所以用 $x_{t}=\sqrt{ \overline \alpha_{t} }x_{0}+\sqrt{ 1-\overline a_{t} }\epsilon$，所以可以用 $x_{t}$ 来表示 $x_{0}$。

$$
\begin{align}
\mu(x_{t},\epsilon)=\frac{1}{1-\overline \alpha_{t}}  \left( (1-\overline \alpha_{t-1})\sqrt{ \alpha_{t} }x_{t}+\frac{\sqrt{ \overline \alpha_{t-1} }\beta_{t}}{\sqrt{ \overline \alpha_{t} }}(x_{t}-\sqrt{ 1-\overline a_{t} }\epsilon)   \right) \\
=\frac{1}{\sqrt{ \alpha_{t} }}\left( x_{t}- \frac{1-\alpha_{t}}{\sqrt{ 1-\overline \alpha_{t} }}\epsilon \right)
\end{align}
$$

这里看似丢失了 $x_{0}$ 的信息，但实际上是把 $x_{0}$ 变成了 $\epsilon$ 这个未知数，而 $\epsilon$ 是一个符合 $N(0,1)$ 的未知数，我们可以用一个神经网络去推理这个 $\epsilon$ 是多少（也就是推理是用那个未知数来生成的）。

$$
\mu_{\theta}(x_{t})=\frac{1}{\sqrt{ \alpha_{t} }}\left( x_{t}- \frac{1-\alpha_{t}}{\sqrt{ 1-\overline \alpha_{t} }}\epsilon_{\theta}(x_{t},t) \right)
$$

于是就有了一个大概做法：
- 训练：随一个 $\epsilon,t$，然后生成 $x_{t},t$，预测 $\epsilon_{\theta}(x_{t},t)$，找一个损失函数（比如和真实 $\epsilon$ 的 MSE）。
- 生成：从 $t=T$ 出发，每次 $q_{\theta}(x_{t-1}|x_{t})=N(x_{t-1};\mu_{\theta}(x_{t}),\sigma^2_{t})$.

那么具体是哪个损失函数呢？根据 VAE 的类似理论，损失应当是每一步 $p_{\theta}(x_{t-1}|x_{t}),p(x_{t-1}|(x_{t},x_{0}))$ 的 KL 散度。而这两个分布的方差相同，因此 KL 散度就是均值的差的平方，也就是 $MSE(\epsilon,\epsilon_{\theta}(x_{t},t))$.

## Unet

具体怎么用一个神经网络来预测 $\epsilon_{\theta}$ 呢？  
要分析 $x_t$ 整个图像的信息，这就要求必须要用基于二维卷积的特征提取，然后还要把 $t$ 的信息放进去，可以用一个 embbeding 实现。

这就需要一个 Unet 网络结构，分为 encode（提取特征），连接层（加入 $t$）， decode（从特征生成 $\epsilon$）三个过程。  
encode 结构采用「预激活风格」，也就是层次结构是：

```
norm -> relu -> conv -> norm -> relu -> conv -> pool
```

也就是先激活，再卷积，这样较为平滑的激活数据卷积能更好提取特征。

连接层主要是对于 encode 的结果进行一个整合，再加入 $t$，那么前面的整合很简单，可以看成是 encode 的一层去掉 pool，这样就可以把提取的特征彼此关联。

decode 需要用压缩过的特征来扩展出一张图片，但是会发现从凝练的特征（例如有一个耳朵）想要变成一张图片，仅仅用一个卷积核是很难实现的，因此想到在 encode 过程中，把每一层的结果记录下来，作为「浅层结构」，在知道前面图片的情况下，特征就可以去完善图片了，因此把浅层结构和深层特征并在一起，然后进行一个 encode 去掉 pool 的过程。