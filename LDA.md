# LDA



线性判别分析(Linear Discriminant Analysis)是经典的线性学习方法，亦称“Fisher判别方法”.

## 原理

将数据投影到直线$\omega$上，则两类样本的中心在直线上的投影分别为$\omega^T\mu_0$和$\omega^T\mu_1$；若将所有样本点都投影到直线上，则两类样本的协方差分别为$\omega^T \Sigma_0 \omega$和$\omega^T \Sigma_1 \omega$.

要想使同类样例的投影尽可能接近，须让同类样例投影点的协方差尽可能小，即$\omega^T \Sigma_0 \omega+\omega^T \Sigma_1 \omega$尽可能小

要想使异类样例的投影尽可能远离，须让异类样例投影点的中心距离尽可能大，即$||\omega^T \mu_0-\omega\mu_1||^2_2$尽可能大

同时考虑以上两点，则**最大化目标**:
$$
J=\frac{||\omega^T \mu_0-\omega\mu_1||^2_2}{\omega^T \Sigma_0 \omega+\omega^T \Sigma_1 \omega}\\
=\frac{\omega^T(\mu_0-\mu_1)(\mu_0-\mu_1)^T\omega}{\omega^T(\Sigma_0+\Sigma_1)\omega}
$$
定义“类内散度矩阵” 
$$
S_\omega=\Sigma_0+\Sigma_1\\
=\sum_{x\in X_0}(x-\mu_0)(x-\mu_0)^T + \sum_{x\in X_1}(x-\mu_1)(x-\mu_1)^T
$$
定义“类间散度矩阵”
$$
S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T
$$
则最大化目标**重写为**
$$
J=\frac{\omega^T S_b\omega}{\omega^T S_\omega\omega}
$$
因为分子分母都是关于$\omega$ 的二次项，且仅考虑$\omega$的方向，故**等价于**
$$
min -\omega^TS_0\omega\\
s.t. \omega^TS_\omega\omega=1\\
$$
根据拉格朗日乘子法，**等价于**
$$
S_b\omega=\lambda S_\omega \omega
$$
注意到$S_b\omega$的方向恒为$\mu_0-\mu_1$，不妨令
$$
S_b\omega=\alpha(\mu_0-\mu_1)
$$
可得
$$
\omega=S^{-1}_\omega(\mu_0-\mu_1)
$$





















