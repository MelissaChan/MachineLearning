# __author__ = MelissaChan
# -*- coding: utf-8 -*-
# 16-2-14 下午4:37

# 词表转换向量
# 创建词汇表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
# 转换为向量
def setOfWords2Vec(vocablist,inputset):
    returnVec = [0] * len(vocablist)
    for word in inputset:
        if word in  vocablist:
            returnVec[vocablist.index(word)] = 1 # 每个单词只出现一次
        else:print "the word: %s is not in my vocabulary!" %word
    return returnVec
def bagOfWord2Vec(vocablist,inputset):
    returnVec = [0] * len(vocablist) # 每个单词出现多次
    for word in inputset:
        returnVec[vocablist.index(word)] += 1
    return returnVec

# 测试数据集
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','please','help'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmatian','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','stupid','dog','food']]
    classVec = [0,1,0,1,0,1]
    return  postingList,classVec

# 训练函数
from numpy import *
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0num = ones(numWords); p1num = ones(numWords) # 为避免为乘数为0,初始化次数为1,分母为2
    p0denom = 2.0; p1denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1num += trainMatrix[i]
            p1denom += sum(trainMatrix[i])
        else:
            p0num += trainMatrix[i]
            p0denom += sum(trainMatrix[i])
    p1vec = log(p1num/p1denom)  # 自然对数避免下溢出
    p0vec = log(p0num/p0denom)
    return p0vec,p1vec,pAbusive

# 分类函数
def classify(vec2Classify,p0vec,p1vec,pClass1):
    p1 = sum(vec2Classify * p1vec) + log(pClass1)
    p0 = sum(vec2Classify * p0vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:return 0

# 测试封装
postlist,classvec = loadDataSet()
myVocabList = createVocabList(postlist)
# print myVocabList
trainMat = []
for postinDoc in postlist:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
p0v,p1v,pAb = trainNB0(array(trainMat),array(classvec))

testEntry = ['love','my','dalmatian']
testEntry2 = ['stupid','my','my']
thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
thisDoc2 = array(setOfWords2Vec(myVocabList,testEntry2))
# print testEntry2,'classified as: ',classify(thisDoc2,p0v,p1v,pAb)
# print testEntry,'classified as: ',classify(thisDoc,p0v,p1v,pAb)
# print p0v
# print p1v

# 垃圾邮件过滤器
# 文本解析
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
# 过滤器
def spamTest():
    # 导入并解析文本
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('/home/melissa/桌面/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('/home/melissa/桌面/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 随机构建训练集
    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    # 对测试集进行分类
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWord2Vec(vocabList, docList[docIndex])
        if classify(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)

for i in range(10):
    spamTest()