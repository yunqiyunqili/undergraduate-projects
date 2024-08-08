import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import linspace


def getDataSet():
    dataSet = [
        [0.697, 0.460, '是'],
        [0.774, 0.376, '是'],
        [0.634, 0.264, '是'],
        [0.608, 0.318, '是'],
        [0.556, 0.215, '是'],
        [0.403, 0.237, '是'],
        [0.481, 0.149, '是'],
        [0.437, 0.211, '是'],
        [0.666, 0.091, '否'],
        [0.243, 0.267, '否'],
        [0.245, 0.057, '否'],
        [0.343, 0.099, '否'],
        [0.639, 0.161, '否'],
        [0.657, 0.198, '否'],
        [0.360, 0.370, '否'],
        [0.593, 0.042, '否'],
        [0.719, 0.103, '否']
    ]

    for i in range(len(dataSet)):  # '是'换为1，'否'换为-1。
        if dataSet[i][-1] == '是':
            dataSet[i][-1] = 1
        else:
            dataSet[i][-1] = -1
    return np.array(dataSet)

def calErr(dataSet, feature, threshVal, inequal, D):
    """
    计算数据带权值的错误率。
    :param dataSet:     [密度，含糖量，好瓜]
    :param feature:     [密度，含糖量]
    :param threshVal:
    :param inequal:     'lt' or 'gt. (大于或小于）
    :param D:           数据的权重。错误分类的数据权重会大。
    :return:            错误率。
    """
    DFlatten = D.flatten()#变一维
    errCnt = 0
    i = 0
    if inequal == 'lt':
        for data in dataSet:
            if (data[feature] <= threshVal and data[-1] == -1) or \
               (data[feature] > threshVal and data[-1] == 1):
                errCnt += 1*DFlatten[i]
            i += 1
    else:
        for data in dataSet:
            if (data[feature] >= threshVal and data[-1] == -1) or \
               (data[feature] < threshVal and data[-1] == 1):
                 errCnt += 1*DFlatten[i]
            i += 1
    return errCnt

#建立树桩
def buildStump(dataSet, D):
    """
    通过带权重的数据，建立错误率最小的决策树桩。
    :param dataSet:  数据集
    :param D:     权重
    :return:    包含建立好的决策树桩的信息,如feature，threshVal，inequal，err。
    """
    m, n = dataSet.shape
    bestErr = np.inf  #正无穷大
    bestStump = {}    #存放给定权重向量D时所得到的最佳决策树的相关信息
    numSteps = 16.0  #每个特征迭代的步数
    for i in range(n-1):   #遍历所有特征
        rangeMin = dataSet[:,i].min()  #
        rangeMax = dataSet[:,i].max()  #
        stepSize = (rangeMax - rangeMin) / numSteps  #通过计算最大值和最小值来计算步长
        for j in range(m):  #遍历列上的每个值
            threVal = rangeMin + float(j)*stepSize #设定阈值
            for inequal in ['it', 'gt']:
                err = calErr(dataSet, i,threVal, inequal, D)  # 错误率
                if err < bestErr:           # 如果错误更低，保存划分信息
                    bestErr = err
                    bestStump['feature'] = i
                    bestStump['threshVal'] = threVal
                    bestStump['inequal'] = inequal
                    bestStump['err'] = err
    return bestStump

def predict(data, bestStump):
    if bestStump['inequal'] == 'lt':
        if data[bestStump['feature']] <= bestStump['threshVal']:
            return 1
        else:
            return -1
    else:
        if data[bestStump['feature']] >= bestStump['threshVal']:
            return 1
        else:
            return -1


def AdaBoost(dataSet, T):
    """
    每学到一个学习器，根据其错误率确定两件事。
    1.确定该学习器在总学习器中的权重。正确率越高，权重越大。
    2.调整训练样本的权重。被该学习器误分类的数据提高权重，正确的降低权重，
      目的是在下一轮中重点关注被误分的数据，以得到更好的效果。
    :param dataSet:  数据集。
    :param T:        迭代次数，即训练多少个分类器
    :return:         字典，包含了T个分类器。
    """
    m, n = dataSet.shape
    D = np.ones((1, m)) / m
    classLabel = dataSet[:, -1].reshape(1, -1)  # 摊平变成一行
    G = {}
    for t in range(T):
        stump = buildStump(dataSet, D)  # 最好的树
        err = stump['err']
        alpha = np.log((1 - err) / err) / 2  # 权重
        pre = np.zeros((1, m))  # array [[a,b,c]]
        for i in range(m):
            pre[0][i] = predict(dataSet[i], stump)
        a = np.exp(-alpha * classLabel * pre)  # classLabel array[[]]-1*m;pre---array[[]]--1*m
        D = D * a / np.dot(D, a.T)  # 变成一个数，dot内积

        G[t] = {}
        G[t]['alpha'] = alpha
        G[t]['stump'] = stump
    return G

#通过Adaboost得到的总的分类器得到分类结果。
def adaPredict(data, G):
    """
    通过Adaboost得到的总的分类器来进行分类。
    :param data:    待分类数据。
    :param G:       字典，包含了多个决策树桩
    :return:        预测值
    """
    score = 0
    for key in G.keys():
        pre = predict(data, G[key]['stump'])
        score += G[key]['alpha']*pre
    flag = 0
    if score >0:
        flag = 1
    else:
        flag = -1
    return flag

#准确率
def calcAcc(dataSet, G):
    rightCnt =0
    for data in dataSet:
        pre = adaPredict(data, G)
        if pre == data[-1]:
            rightCnt +=1
    return rightCnt / float(len(dataSet))


def plotData(data, clf):
    X1, X2 = [], []
    Y1, Y2 = [], []
    datas = data
    labels = data[:, 2]
    for data, label in zip(datas, labels):
        if label > 0:
            X1.append(data[0])
            Y1.append(data[1])
        else:
            X2.append(data[0])
            Y2.append(data[1])

    x = linspace(0, 0.8, 100)
    y = linspace(0, 0.6, 100)

    for key in clf.keys():

        z = [clf[key]['stump']['threshVal']] * 100
        if clf[key]['stump']['feature'] == 0:
            plt.plot(z, y)
        else:
            plt.plot(x, z)

    plt.scatter(X1, Y1, marker='+', label='好瓜', color='r')
    plt.scatter(X2, Y2, marker='_', label='坏瓜', color='green')

    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.xlim(0, 0.8)
    plt.ylim(0, 1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.legend(loc='upper left')
    plt.show()

def main():
    dataSet = getDataSet()
    for t in [3, 5, 11]:   # 学习器的数量
        G = AdaBoost(dataSet, t)
        print('集成学习器（字典）：',f"G{t} = {G}")
        print('准确率=',calcAcc(dataSet, G))
        #绘图函数
        plotData(dataSet,G)
if __name__ == '__main__':
    main()
