# __author__ = MelissaChan
# -*- coding: utf-8 -*-
# 16-2-3 下午5:47

from numpy import *
import operator

# k-近邻算法
# 参数：输入、训练集、标签、邻近数目
# 分类器
def classify(inX,dataSet,lables,k):
    # 1.距离计算
    dataSetSize = dataSet.shape[0]                  # 训练集大小
    diffMat = tile(inX,(dataSetSize,1)) - dataSet   # 获得数据之差
    sqDiffMat = diffMat**2                          # 平方
    sqDistance = sqDiffMat.sum(axis=1)              # 求和
    distances = sqDistance**0.5                     # 开方
    sortedDistIndicies = distances.argsort()        # 距离排序


    # 2.选择距离最小的k个点
    classCount = {}                                                     # 设置计数器
    for i in range(k):                                                  # 循环k次，迭代到第i个
        voteILable = lables[sortedDistIndicies[i]]                      # 记录前k个的标签
        classCount[voteILable] = classCount.get(voteILable,0) + 1       # 查找&记录


    # 3.排序
    sortedClassCount = sorted(classCount.iteritems(),    # 为标签出现次数排序
        key = operator.itemgetter(1), reverse = True)

    # 4.返回出现频率最高的标签
    return sortedClassCount[0][0]


# 建立训练集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group, lables

# 格式化输入
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# 编写时测试
group,lables = createDataSet()
print classify([1,1.1],group,lables,3)
print "--------------------"
datingDataMat,datingLables = file2matrix("datingTestSet2.txt")
print datingDataMat
print "--------------------"
print datingLables[0:20]
print "--------------------"
print autoNorm(datingDataMat)
print "--------------------"

# 散点图
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLables),15.0*array(datingLables))
plt.show()

# 完全版测试 约会系统
def datingClassText():
    hoRio = 0.10 # 训练测试比
    datingDataMat,datingLables = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTextVecs = int(m*hoRio)
    errorCount = 0.0
    for i in range(numTextVecs):
        classifyResult = classify(normMat[i,:],normMat[numTextVecs:m,:],datingLables[numTextVecs:m],3)
        #print "classify result: %d, real reult: %d"%(classifyResult,datingLables[i])
        if(classifyResult != datingLables[i]):
            errorCount += 1.0
    print "error rate : %f" %(errorCount / float(numTextVecs))

datingClassText()

# 约会网站预测函数
def classifyPerson():
    resultList = ["完全不对眼","一般般喜欢","爱死了！"]
    game = float(raw_input("用来玩游戏的时间比例是？"))
    miles = float(raw_input("每年获得的飞行常旅客公里数？"))
    iceCream = float(raw_input("每天吃的冰淇淋公升数量？"))
    datingDataMat,datingLables = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([game,miles,iceCream])
    classifyResult = classify((inArr - minVals) / ranges,normMat,datingLables,3)
    print resultList[classifyResult - 1]

classifyPerson()

# 测试 手写数字识别系统
from os import listdir
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print "classify result: %d, real answer: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "error rate: %f" % (errorCount/float(mTest))

handwritingClassTest()
