# __author__ = MelissaChan
# -*- coding: utf-8 -*-
# 16-2-3 下午5:47

from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group, lables

group,lables = createDataSet()
print group
print lables

