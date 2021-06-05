import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
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
 

class GaussianMixtureModel:
 
    def __init__(self, n_components, maxIter=1e4, eps=1e-9):
        self.n_components = n_components
        self.class_prior = np.ones(n_components) * 1 / n_components
        self.mu = None
        self.covariance = None
        self.W = None
        self.pdfs = None
        self.eps = eps
        self.maxIter = maxIter
 
    def __initParameters(self, X):
        '''
        初始化模型参数mu,sigma,class_prior
        :param X:
        :return:
        '''
        m, n = X.shape
        self.W = np.random.random((m, self.n_components))
        self.mu = np.random.random((self.n_components, n))
        minCol = np.min(X, axis=0)
        maxCol = np.max(X, axis=0)
        self.mu = minCol + self.mu * (maxCol - minCol)
        self.covariance = np.zeros((self.n_components, n, n))
        dist = np.tile(np.sum(X * X, axis=1).reshape((m, 1)), (1, self.n_components)) + np.tile(
            np.sum(self.mu * self.mu, axis=1).T,
            (m, 1)) - 2 * np.dot(X, self.mu.T)
        self.pdfs = np.zeros((m, self.n_components))
        labels = np.argmin(dist, axis=1)
        for i in range(self.n_components):
            clusterX = X[labels == i, :]
            self.class_prior[i] = clusterX.shape[0] / m
            self.covariance[i, :, :] = np.cov(clusterX.T)
 
    def train(self, X):
        '''
        EM算法得到模型参数，迭代停止条件为：1迭代轮数达到上限   2似然函数的变化极其微小，小于某个阈值
        :param X:
        :return:
        '''
        self.__initParameters(X)
        num = 0
        preLogLikelihood = self.__logLikelihood(X)
        while num < self.maxIter:
            self.__expectation(X)
            self.__maximize(X)
            # plot(X, self.mu, self.covariance,y)
            num += 1
            logLikelihood = self.__logLikelihood(X)
            if abs(logLikelihood - preLogLikelihood) < self.eps:
                break
            preLogLikelihood = logLikelihood
        plot(X, self.mu, self.covariance,y)
 
    # 根据当前的各个组分先验概率、均值向量和协方差矩阵计算对数似然函数值
    def __logLikelihood(self, X):
        for j in range(self.n_components):
            a = multivariate_normal.pdf(X, self.mu[j], self.covariance[j])
            # print(a)
            self.pdfs[:, j] = self.class_prior[j] * multivariate_normal.pdf(X, self.mu[j], self.covariance[j])
        return np.mean(np.log(np.sum(self.pdfs, axis=1)))
 
    # EM算法的E步，计算样本x_i来自第k个高斯分布的概率
    def __expectation(self, X):
        '''
        对于样本x_i来自第k个高斯分布的概率
        :return:
        '''
        for j in range(self.n_components):
            self.pdfs[:, j] = self.class_prior[j] * multivariate_normal.pdf(X, self.mu[j], self.covariance[j])
            self.W = self.pdfs / np.sum(self.pdfs, axis=1).reshape(-1, 1)
 
    def __maximize(self, X):
        '''
        N_k表示所有数据点属于第k类的概率之和
        更新类别先验，类的期望中心和协方差
        :return:
        '''
        m, n = X.shape
        self.class_prior = np.sum(self.W, axis=0) / np.sum(self.W)
        for j in range(self.n_components):
            self.mu[j] = np.average(X, axis=0, weights=self.W[:, j])
            cov = 0
            for i in range(m):
                tmp = (X[i, :] - self.mu[j, :]).reshape(-1, 1)
                cov += self.W[i, j] * np.dot(tmp, tmp.T)
            self.covariance[j, :, :] = cov / np.sum(self.W[:, j])
 
 
# 用三个不同的高斯分布生成三个聚类作为GMM算法的数据
num1, mu1, covar1 = 400, [0.5, 0.5], np.array([[1, 0.5], [0.5, 3]])
X1 = np.random.multivariate_normal(mu1, covar1, num1)
# 第二簇的数据
num2, mu2, covar2 = 600, [5.5, 2.5], np.array([[2, 1], [1, 2]])
X2 = np.random.multivariate_normal(mu2, covar2, num2)
# 第三簇的数据
num3, mu3, covar3 = 1000, [1, 7], np.array([[6, 2], [2, 1]])
X3 = np.random.multivariate_normal(mu3, covar3, num3)
# 合并在一起
Mydata = np.vstack((X1, X2, X3))
print(Mydata.shape)
# 计算聚类结果的对数似然函数值
y = np.hstack((np.zeros(len(X1)), np.ones(len(X2)), 2 * np.ones(len(X3))))
print(len(y))
myGMM = GaussianMixtureModel(3)
 
myGMM.train(Mydata)