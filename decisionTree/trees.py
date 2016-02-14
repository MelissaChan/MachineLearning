# __author__ = MelissaChan
# -*- coding: utf-8 -*-
# 16-2-10 下午2:46

# 计算香农熵
from math import log
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    lableCount = {}
    for featVec in dataSet:
        currentLable = featVec[-1]
        if currentLable not in lableCount.keys():
            lableCount[currentLable] = 0
        lableCount[currentLable] += 1
    shannonEnt = 0.0
    for key in lableCount:
        prob = float(lableCount[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

# 划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 寻找出现最多的分类
import  operator
def majorCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount: classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 寻找最优划分方式
def chooseBestFeatherToSplit(dataSet):
    numFeathers = len(dataSet[0]) - 1           # 特征数
    baseEntropy = calcShannonEnt(dataSet)       # 初始熵作为基本熵
    bestInfoGain = 0.0; bestFeather = -1        # 定义熵减和最优特征序号
    for i in range(numFeathers):                        # 遍历特征值
        featList = [example[i] for example in dataSet]  # 复制特征值list
        uniqueVals = set(featList)                      # 去重
        newEntropy = 0.0                                # 每个特征值的熵
        for value in uniqueVals:                        # 遍历集合
            subDataSet = splitDataSet(dataSet,i,value)  # 循环划分数据集
            prob = len(subDataSet)/float(len(dataSet))  # 概率
            newEntropy += prob * calcShannonEnt(subDataSet) # 计算特征值的熵
        infoGain = baseEntropy - newEntropy             # 熵减
        if(infoGain > bestInfoGain):                    # 最大熵减
            bestInfoGain = infoGain
            bestFeather = i
    return bestFeather

# 建树
def createTree(dataSet,lables):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征停止划分
    if len(dataSet[0]) == 1:
        return majorCnt(classList)
    bestFeat = chooseBestFeatherToSplit(dataSet)
    besatFeatLable = lables[bestFeat]
    myTree = {besatFeatLable:{}}
    del(lables[bestFeat])
    featValues = [example[bestFeat] for example in  dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = lables[:]
        myTree[besatFeatLable][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLables)
    return myTree

# 分类器
def classify(inputTree,featLables,testVec):
    firstStr = inputTree.keys()[0]
    secDict = inputTree[firstStr]
    featIndex = featLables.index(firstStr)
    for key in secDict.keys():
        if testVec[featIndex] == key:
            if type(secDict[key]).__name__=='dict':
                classLable = classify(secDict[key],featLables,testVec)
            else: classLable = secDict[key]
    return  classLable


# 建立测试数据
def createDataSet():
    dataSet = [[1,1,"yes"],[1,1,"yes"],[1,0,"no"],[0,1,"no"],[0,1,"no"]]
    lables = ["no surfacing","flippers"]
    return dataSet,lables

# 序列化决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# 测试
import treePlotter
data,lables = createDataSet()
myTree = treePlotter.retrieveTree(0)
storeTree(myTree,"/home/melissa/桌面/temp.txt")
print classify(myTree,lables,[1,0])
print classify(myTree,lables,[1,1])

fr = open("/home/melissa/桌面/lenses.txt")
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLables = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLables)
print lensesTree
treePlotter.createPlot2(lensesTree)