# SMO

对于求解SVM的对偶问题
$$
max \sum^m_{i=1}\alpha_i - \frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_jx^T_ix^j\\
s.t.\sum^m_{i=1}\alpha_iy_i=0\\
\alpha_i\geq0
\tag{1}
$$
问题满足**KKT条件**
$$
\begin{cases}
\alpha_i\geq0 & 对偶问题可行条件\\
y_i(\omega_ix_i+b)-1\geq0 & 原问题可行条件\\
\alpha_i(y_i(\omega_ix_i+b)-1)=0 & 互补松弛条件
\end{cases}
\tag{2}
$$
显然是关于$\alpha$的二次规划问题，该问题的规模正比于样本数，训练开销大



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

