# 支持向量机(SVM)



## 基本型

在二分类问题中，通过划分超平面来将不同类的样本分开，超平面通过如下线性方程描述
$$
\omega^Tx+b=0
\tag{1}
$$
若超平面能将样本正确分开，则有
$$
\begin{cases}
\omega^Tx_i+b\geq+1 & y_i=+1\\
\omega^Tx_i+b\leq-1 & y_i=-1\\
\end{cases}
$$
+1、-1表示超平面与最近的样本之间有距离，距离可根据$\omega,b$的成倍变化而变化，最近的样本称为**支持向量**



问题的关键使找到使间隔最大的超平面，即
$$
\mathop{max}_{\omega,b}\frac{2}{||\omega||}\\
s.t.y_i(\omega^Tx_i+b)\geq1,\quad i=1,2,...,m
\tag{2}
$$
即
$$
\mathop{min}_{\omega,b}\frac{||\omega||}{2}\\
s.t.y_i(\omega^Tx_i+b)\geq1,\quad i=1,2,...,m
\tag{2}
$$
上式为SVM的**基本型**



## 对偶问题

(2)式为关于$\omega$的凸二次规划问题，可用**拉格朗日乘子法**解决

(2)式的拉格朗日函数为
$$
L(\omega,b,\alpha)=\frac{1}{2}||\omega||^2+\sum^{m}_{i=1}\alpha_i(1-y_i(\omega_ix_i+b))
\tag{3}
$$
**原问题**可改写为
$$
\mathop{min}_{\omega,b}\mathop{max}_{\alpha_i}L(\omega,b,\alpha)\\
s.t.\alpha_i\geq0,\quad i=1,2,...,m
\tag{4}
$$
原问题的**对偶问题**为
$$
\mathop{max}_{\alpha_i}\mathop{min}_{\omega,b}L(\omega,b,\alpha)\\
s.t.\alpha_i\geq0,\quad i=1,2,...,m
\tag{5}
$$
令$L(\omega,b,\alpha)$对$\omega$和$b$的偏导为零可得
$$
\omega=\sum^m_{i=1}\alpha_iy_ix_i\\
0=\sum^m_{i=1}\alpha_iy_i
\tag{6}
$$
将(5)式中的$\omega$和$b$消去，得到
$$
\mathop{max}_{\alpha_i} \sum^m_{i=1}\alpha_i - \frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_jx^T_ix^j\\
s.t.\sum^m_{i=1}\alpha_iy_i=0\\
\alpha_i\geq0, \quad i=1,2,...,m
\tag{7}
$$
(7)式是关于$\alpha$的二次规划问题，可使用**二次规划算法**求解，但是问题的规模正比于样本数，**训练开销大**



## SMO

对于求解SVM的对偶问题(7)式

问题满足**KKT条件**
$$
\begin{cases}
\alpha_i\geq0 & 对偶问题可行条件\\
y_i(\omega_ix_i+b)-1\geq0 & 原问题可行条件\\
\alpha_i(y_i(\omega_ix_i+b)-1)=0 & 互补松弛条件
\end{cases}
\tag{8}
$$


**SMO**是基于**贪心思想**的**迭代算法**，基本思路是

​	1.将所有$\alpha$初始化为0

​	2.挑选最偏离KKT条件的$\alpha_i$：

​		把 $y_i(\omega_ix_i+b)-1$从小到大排序，从前往后选取第一个$y_i(\omega_ix_i+b)-1<0$的$\alpha_i$

​		若没有可选的$\alpha$，则迭代终止

​	3.选择使$\alpha_j$改变最大的$\alpha_j$：

​		固定除$\alpha_i$和$\alpha_j$之外的其他$\alpha$，令$c=-\sum_{k\neq i,j}\alpha_ky_k=\alpha_iy_i+\alpha_jy_j$，带入(1)中，得到仅关于$\alpha_j$的函数		Q($\alpha_j$)，对其求导为0，得到$\alpha_{j,new}$，$\alpha_{j,new}-\alpha_j$即$\alpha_j$的变化量

​	4.$\alpha_{i,new}$的值可由$\alpha_{i,new}y_i+\alpha_{j,new}y_j=-\sum_{k\neq i,j}\alpha_ky_k=\alpha_iy_i+\alpha_jy_j$得到

​	5.判断迭代后的$\alpha_i\alpha_j$是否符合$\alpha\geq0$ ；若不符合，则返回重选$\alpha_j$

​	6.计算$\omega_{new},b_{new}$：

​		$\omega_{new}=\sum^m_{i=1}\alpha_iy_ix_i$

​		$b_{new}=\frac{1}{m}\sum^m b_i$，$b_i$为支持向量对应的截距，即$\alpha_i\neq0$的$x_i$，$b_i=y_i-\omega_{new}x_i$

​	7.若所有$\alpha$均符合KKT条件，或者达到迭代次数，则停止迭代



## 核函数



在现实分类任务中，很难找到线性超平面，使得能够正确划分所有样本

可将原空间**映射**到更**高维度的空间**，使得存在一个高维的超平面，能够正确划分所有样本

$\phi(x)$表示将样本$x$映射后的向量，模型可表示为
$$
\mathop{min}_{\omega,b}\frac{||\omega||}{2}\\
s.t.y_i(\omega^T\phi(x_i)+b)\geq1,\quad i=1,2,...,m
\tag{9}
$$
由于映射空间维数可能很高，向量内积$\phi(x_i)^T\phi(x_j)$的计算通常比较困难

为了避免这个障碍，可设想这样一个函数
$$
\kappa(x_i,x_j)=<\phi(x_i),\phi(x_j)>=\phi(x_i)^T\phi(x_j)
\tag{10}
$$
$\kappa(.,.)$称为**核函数**，为**半正定**的对称矩阵



## 软间隔



为了解决**过拟合**问题，引入“软间隔”，允许在一些样本上分类错误，原问题改为
$$
\mathop{min}_{\omega,b}\frac{1}{2}||\omega||^2+C\sum^m_{i=1}l(1-y_i(\omega^Tx_i+b))
\tag{11}
$$
其中C>0，$l(x)$为**损失函数



### 损失函数



#### 0/1损失函数

$l_{0/1}(x)=
\begin{cases}
1, & x>0;\\
0, & otherwise.
\end{cases}$



#### *hinge*损失函数

$l_{hinge}(x)=max(0,x)$



#### 指数损失函数

$l_{exp}(x)=exp(x)$



#### 对率损失函数

$l_{log}=log(1+exp(x))$

与逻辑回归相似，性能相当

但是逻辑回归可以输出有概率意义，SVM不具有概率意义；逻辑回归可直接用于多分类任务，SVM不能



#### 松弛变量

引入松弛变量$\xi$，原问题改为
$$
\mathop{min}_{\omega,b}\frac{1}{2}||\omega||^2+C\sum^m_{i=1}\xi_i\\
s.t.y_i(\omega^Tx_i+b)\geq1-\xi_i\\
\xi_i\geq0,\quad i=1,2,...,m
\tag{13}
$$
对应的**对偶问题**为
$$
\mathop{max}_{\alpha_i} \sum^m_{i=1}\alpha_i - \frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_jx^T_ix^j\\
s.t.\sum^m_{i=1}\alpha_iy_i=0\\
0\leq \alpha_i\leq C, \quad i=1,2,...,m
\tag{14}
$$
可见，软间隔与硬间隔的**唯一差距**是$\alpha$有上界C，**计算方法**与硬间隔**一致**



## 支持向量回归(SVR)

 

假设我们能容忍$f(x)$与$y$之间最多有$\epsilon$的偏差，相当于以$f(x)$为中心，构建了一个宽度为$2\epsilon$的间隔带，若样本落入带中，则认为是预测正确的
$$
\mathop{min}_{\omega,b}\frac{1}{2}||\omega||^2+C\sum^m_{i=1}l(f(x_i)-y_i)
\tag{15}
$$
可见SVR**类似于**软间隔

