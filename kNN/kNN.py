# __author__ = MelissaChan
# -*- coding: utf-8 -*-
# 16-2-3 下午5:47

from numpy import *
import operator

# k-近邻算法
# 参数：输入、训练集、标签、邻近数目
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

# 测试
group,lables = createDataSet()
print classify([1,1.1],group,lables,3)
print "--------------------"
datingDataMat,datingLables = file2matrix("datingTestSet2.txt")
print datingDataMat
print "--------------------"
print datingLables[0:20]

