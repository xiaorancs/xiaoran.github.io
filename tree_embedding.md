# 树模型集成学习
集成学习主要有两个思想，分别是bagging和boosting。树模型的集成模型都是使用树作为基模型，最常用的cart树，常见的集成模型有RandomForest、GBDT、Xgboost、Lightgbm、Catboost。

# 概要介绍
## RandomForest
随机森林(Random Forest,RF)是Bagging的一个扩展变体。RF在以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入了随机属性选择。既然模型叫做随机森林，森林我们可以理解为是多棵树的集合就是森林，随机主要有两个点进行有放回的采样，
1. 每次建树特征个数随机选择
2. 每次建树样本个数随机选择

随机森林中基学习器的多样性不仅来自样本扰动，还来自属性扰动，这就使得最终集成得泛化性能可通过个体学习器之间差异度得增加而进一步提升。使得模型更加鲁棒。

## GBDT
GBDT使用的是加法模型和前向分布算法，而AdaBoost算法是前向分布加法算法的特例，前向分布算法学习的是加法模型，当基函数为基本分类器时，该加法模型等价于Adaboost的最终分类器。
GBDT也是迭代，使用了前向分布算法，但是弱学习器限定了只能使用CART回归树模型，同时迭代思路和Adaboost也有所不同。在GBDT的迭代中，假设我们前一轮迭代得到的强学习器是, 损失函数是, 我们本轮迭代的目标是找到一个CART回归树模型的弱学习器，让本轮的损失函数最小。也就是说，本轮迭代找到决策树，要让样本的损失尽量变得更小。GBDT本轮迭代只需拟合当前模型的残差。

## Xgboost
Xgboost是gbdt的改进或者说是梯度提升树的一种，Xgb可以说是工程上的最佳实践模型，简单的说xgb=gbdt+二阶梯度信息+随机特征和样本选择+特征百分位值加速+空值特征自动划分。还有必要的正则项和最优特征选择时的并行计算等。

## Lightgbm
首先，GBDT是一个非常流行的机器学习算法，另外基于GBDT实现的XGBoost也被广泛使用。但是当面对高纬度和大数据量时，其效率和可扩展性很难满足要求。主要的原因是对于每个特征，我们需要浏览所有的数据去计算每个可能分裂点的信息增益，真是非常耗时的。基于此，提出了两大技术：Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB).

##　catboost
CatBoost = Category + Boosting. 
2017年7月21日，俄罗斯Yandex开源CatBoost，亮点是在模型中可直接使用Categorical特征并减少了tuning的参数。


# 核心公式
1. gbdt的前向分布公式
   $$f_m(x)=f_{m-1}(x)+\beta_m b(x;\gamma_m)$$

2. gbdt的第m轮的扶梯度公式
    $$-\left[
        \frac{\partial L(y,f(x_i))}{\partial f(x_i)}
    \right]_{f(x)=f_{m-1}(x)}$$

3. gbdt格式化损失函数
    $$L(y,f_m(x))=L(y,f_{m-1}(x)+\beta_m b(x;\gamma_m))$$

4. 泰勒展开式   
若函数f（x）在包含x0的某个闭区间[a,b]上具有n阶导数，且在开区间（a,b）上具有（n+1）阶导数，则对闭区间[a,b]上任意一点x，成立下式：
   $$f(x)=f(x_0)+f'(x_0)(x-x_0)+\frac{f''(x0)}{2!}(x-x_0)^2+ ... + \frac{f^{(n)}(x_0)}{n!}(x-x_0)^n+R_n(x)$$
   $$f(x+\Delta x)=f(x)+f'(x)\Delta x + \frac{1}{2!}f''(x)\Delta x^2+...+\frac{1}{n!}f^{(n)}(x)\Delta x^n+R_n(x)$$
    其中，$R_n(x)$是$(x-x_0)^n的高阶无穷小.$

5. xgboost的目标公式(t轮迭代)
   $$obj^{(t)}=\sum_{i=1}^{n}l(y_i,\hat{y}_i^t)+\sum_{i=1}^{t}\Omega(f_i)$$
   $$=\sum_{i=1}^{n}l(y,\hat y_{i}^{(t-1)}+f_t(x_i))+\Omega(f_t)+constant$$

6. xgboost损失函数的泰勒二阶展开
    $$l^{(t)} \eqsim \sum_{i=1}^{n}[l(y_i,\hat y ^{(t-1)})+g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)]+\Omega(f_t)$$
    其中，$l(y_i,\hat y ^{(t-1)})$是常数，$g_i=\partial_{\hat{y}^{(t-1)}}l(y_i, \hat{y}^{(t-1)})$, $h_i=\partial_{\hat{y}^{(t-1)}}^2l(y_i, \hat{y}^{(t-1)})$. 常数对目标函数的优化不相关，于是可以将目标函数转化为如下:
    $$l^{(t)} = \sum_{i=1}^{n}[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)]+\Omega(f_t)$$
    $$=\sum_{i=1}^{n}[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)]+\lambda T+\frac{1}{2}\sum_{j=1}^{T}\omega_j^2$$
    $$=\sum_{j=1}^{T}[(\sum_{i \in I_j}g_i) \omega_j + \frac{1}{2}(\sum_{i \in I_j}h_i) \omega_j^2] + \lambda T + \frac{1}{2}\sum_{i=1}^{T} \omega_j^2$$
    $$=\sum_{i=1}^{n}[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)]+\lambda T+\frac{1}{2}\sum_{j=1}^{T}\omega_j^2$$
    $$=\sum_{j=1}^{T}[(\sum_{i \in I_j}g_i) \omega_j + \frac{1}{2}(\sum_{i \in I_j}h_i+\lambda) \omega_j^2] + \lambda T$$
    求上式最小化的参数，对$\omega$求导数并另其等于0，得到下式:
    $$\frac{\partial l^{(t)}}{\partial \omega_j}=0$$
    $$\sum_{i \in I_j}+(\sum_{i \in I_j}h_i + \lambda) \omega_j=0$$
    $$\omega_j^*=-\frac{\sum_{i \in I_j}g_i}{\sum_{i \in I_j}h_i + \lambda}$$
    
    将上式带入损失函数，得到最小损失：
    $$\hat{l}^{(t)}(q)=-\frac{1}{2}\sum_{j=1}^{T}\frac{(\sum_{i \in I_j}g_i)^2}{\sum_{i \in I_j}h_i+ \lambda}+\gamma T \tag{1}$$

    根据公式(1)可以作为特征分裂的指标.计算公式如下(这个值越大越好):

    $$L_{split}=\frac{1}{2}
    \left[
        \frac{（\sum_{i \in I_L}g_i)^2}{\sum_{i \in I_L}h_i+\lambda} + 
        \frac{（\sum_{i \in I_R}g_i)^2}{\sum_{i \in I_R}h_i+\lambda} - 
        \frac{（\sum_{i \in I}g_i)^2}{\sum_{i \in I}h_i+\lambda}
        \right ] - \lambda$$


# 算法十问
1. 随机森林为什么能够更鲁棒？

2. RF分类和回归问题如何预测y值？

3. 相同数据量，训练RF和gbdt谁可以更快？谁对异常值不敏感？

4. 解释一个什么是gb，什么是dt，即为什么叫做gbdt？

5. gbdt为什么用负梯度代表残差？

6. gbdt是如何选择特征？

7. gbdt应用在多分类问题？

8. RF和GBDT的区别？

9. Xgboost相对gbdt做了哪些改进？

10. xgb如何在计算特征时加速的？

11. xgb为什么使用二阶梯度信息，为什么不使用三阶或者更高梯度信息？

12. lgb相对xgb做了哪些改进？

13. 比较一下catboost、lgb和xgb？

14. 如果将所有数据复制一倍放入训练数据集，RF和GBDT分别有什么表现？

15. gbdt如何防止过拟合？由于gbdt是前向加法模型，前面的树往往起到决定性的作用，如何改进这个问题？


# 面试真题


