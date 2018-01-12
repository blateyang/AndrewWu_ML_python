# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

def exeTime(func):
    """ 耗时计算装饰器
    """
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        return back, time.time() - t0
    return newFunc

def Load_data(path):
    '''
    加载数据
    :param path: 路径
    :return: X,y
    '''
    df = np.loadtxt(path, delimiter=',') # 原数据文件中各列是以','隔开的
    lines = df.shape[0]
    X = df[:, :-1]
    y = df[:, -1]
    X = np.reshape(X, (lines, X.size/lines))
    y = np.reshape(y, (lines, 1))
    return X, y

# Compute Cost
def computeCost(X, y, theta):
    '''
    计算代价函数
    :param X: 数据集
    :param y: 标签向量
    :param theta: 权值矩阵
    :return: 代价函数
    '''
    return 1.0/(2*y.shape[0])*np.sum((X.dot(theta)-y)**2)

# Gradient Descent
@exeTime
def gradientDescent(X, y, theta, alpha, num_iters):
    '''
    梯度下降
    :param X: 数据集
    :param y: 标签向量
    :param theta: 权值矩阵
    :param alpha: 学习率
    :param num_iters: 迭代次数
    :return: theta,J_history
    '''
    dLen = y.shape[0]
    J_history = np.zeros((num_iters, 1))
    for iter in xrange(num_iters):
        # save the theta from last iteration
        tempTheta = np.copy(theta)
        for j in xrange(X.shape[1]):
            theta[j] = tempTheta[j] - alpha / dLen * np.sum((X.dot(tempTheta)-y) * np.reshape(X[:, j], (dLen, 1)))
            # X[:, j]的维度是(97,)，需要reshape成(97,1)才能与前面进行相乘
        # # update for theta0
        # theta[0] = tempTheta[0] - alpha / dLen * np.sum(X.dot(tempTheta) - y)
        # # update for theta1
        # # 下式'*'左边的shape是(3,1)，X[:, 1]的shape是(3,)，如果不reshape,相乘结果会变成(3,3)而不是(3,1)
        # temp2 = np.sum((X.dot(tempTheta) - y) * np.reshape(X[:, 1], (dLen, 1)))
        # theta[1] = tempTheta[1] - alpha / dLen * temp2

        J_history[iter] = computeCost(X, y, theta)
        # print theta
        # print J_history[iter]
        # pass
    return theta, J_history

def normalize(X):
    '''
    归一化
    :param X: 数据集
    :return: 归一化后的数据集
    '''
    norm_X = np.zeros(X.shape)
    for j in xrange(X.shape[1]):
        min_feat_value = np.min(X[:, j], axis=0)
        max_feat_value = np.max(X[:, j], axis=0)
        if min_feat_value != max_feat_value:
            norm_X[:, j] = (X[:, j]-min_feat_value)/(max_feat_value-min_feat_value)
    return norm_X

def standardize(X):
    '''
    标准化
    :param X: 数据集
    :return: 标准化后的数据集
    '''
    std_X = np.zeros(X.shape)
    for j in xrange(X.shape[1]):
        feature = X[:, j]
        meanValue = feature.mean(axis=0)
        stdValue = feature.std(axis=0)
        if stdValue != 0:
            std_X[:, j] = (feature-meanValue)/stdValue
        return std_X

# df = np.loadtxt('C:\Users\Guangjun\Documents\AndrewWu_ML_python\ex1-linear regression\ex1data1.txt', delimiter=',')
# x = df[:, 0]
# y = df[:, -1]
X, y = Load_data('C:\Users\Guangjun\Documents\AndrewWu_ML_python\ex1-linear regression\ex1data1.txt')
# x = np.array([1, 2, 3])
# y = np.array([2, 4.2, 5.8])
dLen = X.shape[0]

x0 = np.ones((dLen, 1), dtype='float64')
x_extend = np.concatenate((x0, X), axis=1) # x_extend.shape = (97, 2)
norm_X = normalize(x_extend)
numFeat = norm_X.size/dLen
theta = np.zeros((numFeat, 1))
iterations = 1500
alpha = 0.01
result, execute_time = gradientDescent(norm_X, y, theta, alpha, iterations)
theta, Loss = result
print 'theta:', theta
print 'Last Loss:', Loss[-1]
print 'execute time:', execute_time

# plt.scatter(X, y)
# plt.xlabel('Profit in $10,000s')
# plt.ylabel('Population of City in 10,000s')
# xDim = np.linspace(0, 25, 251)
# yPred = theta[0, 0] + theta[1, 0] * xDim
# plt.plot(xDim, yPred, color='red')
# plt.show()