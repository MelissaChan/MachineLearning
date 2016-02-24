# __author__ = MelissaChan
# -*- coding: utf-8 -*-
# 16-2-24 下午4:50

from numpy import *

# 加载数据
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 设置x0为0,数据每行前个值为x1,x2,第三个值为标签
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升算法
def gradAscent(dataMatIn,classLables):# 前者为每列一个特征的二位数组
    # 将数据转化为矩阵
    dataMatrix = mat(dataMatIn)
    lableMat = mat(classLables).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001   # 步长
    maxCycles = 500 # 最大迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid((dataMatrix*weights))
        error = (lableMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 随机梯度上升算法
def stocGradAscent0(dataMatrix,classLables):
    # modified, but apparently i don't know why
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLables[i] - h
        # ValueError: operands could not be broadcast together with shapes (3,) (0,)
        weights = weights + alpha  * error * dataMatrix[i]
    return weights

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLables,numIter=150):# 增加第三个参数：迭代次数
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    # alpha = 0.01
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # 每次都更新a，减少数据波动或者高频波动
            # 常数项使得每次更新都有影响
            # ？？避免参数严格下降
            alpha = 4 / (1.0 + i + j) + 0.01
            # 选取随机样本来更新回归系数
            randomIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randomIndex]*weights))
            error = classLables[randomIndex] - h
            weights = weights + alpha  * error * dataMatrix[randomIndex]
            del dataIndex[randomIndex]
    return weights

# 做图
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

# 测试
dataArr, lablesMat = loadDataSet()
# weights =  gradAscent(dataArr, lablesMat)
# weights = stocGradAscent0(dataArr,lablesMat)
weights = stocGradAscent1(dataArr,lablesMat)
# plotBestFit(weights.getA())
# plotBestFit(weights)
plotBestFit(weights)











