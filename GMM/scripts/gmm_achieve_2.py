import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mul_nor_1
from numpy.random import multivariate_normal as mul_nor_2

from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
 
 
# 绘制椭圆参考代码，https://github.com/SJinping/Gaussian-ellipse/blob/master/gaussian_%20ellipse.py
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
 
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
 
    if ax is None:
        ax = plt.gca()
 
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
 
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
 
    ax.add_artist(ellip)
    return ellip
 
 
def plot(data, mu, covariance, class_label):
    plt.scatter(data[:, 0], data[:, 1], c=class_label)
    n_components = len(mu)
    for j in range(n_components):
        plot_cov_ellipse(covariance[j], mu[j])
        pass
    plt.show()




# 初始化模型参数
# shape 是表示样本规模的二元组，(样本数, 特征数)
# K 表示模型个数
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    print("Parameters initialized.")
    print("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


def phi(Y, mu_k, cov_k):
    norm = mul_nor_1(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)
# E 步：计算每个模型对样本的响应度
# Y 为样本矩阵，每个样本一行，只有一个特征时为列向量
# mu 为均值多维数组，每行表示一个样本各个特征的均值
# cov 为协方差矩阵的数组，alpha 对应的每个高斯模型的权重数组
 
def getExpectation(Y, mu, cov, alpha):
    # 样本数
    N = Y.shape[0]
    # 模型数
    K = alpha.shape[0]

    # 为避免使用单个高斯模型或样本，导致返回结果的类型不一致
    # 因此要求样本数和模型个数必须大于1
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    # 响应度矩阵，行对应样本，列对应响应度
    gamma = np.mat(np.zeros((N, K)))

    # 计算各模型中所有样本出现的概率，行对应样本，列对应模型
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    # 计算每个模型对每个样本的响应度
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma



# M 步：迭代模型参数
# Y 为样本矩阵，gamma 为响应度矩阵
def maximize(Y, gamma):
    # 样本数和特征数
    N, D = Y.shape
    # 模型数
    K = gamma.shape[1]

    #初始化参数值
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    # 更新每个模型的参数
    for k in range(K):
        # 第 k 个模型对所有样本的响应度之和
        Nk = np.sum(gamma[:, k])
        # 更新 mu
        # 对每个特征求均值
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        # 更新 cov
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        # 更新 alpha
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha

# 高斯混合模型 EM 算法
# 给定样本矩阵 Y，计算模型参数
# K 为模型个数
# times 为迭代次数
# 返回每个高斯模型的参数数组和对应的权值数组

def GMM_EM(Y, K, times):    
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    print("{sep} Result {sep}".format(sep="-" * 20))
    print("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


