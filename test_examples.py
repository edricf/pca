from find_split import find_split
from entropy import entropy
import random
import numpy as np

parameters = {'num_classes':2, 'max_depth':3, 'min_leaf_num':4, 'min_entropy':5,
             'debug':True}

# test example 1-D, 2-class, linearly seperable
x=np.ones((10,1))
y=np.ones((10,1))
x[4:]=2
y[4:]=2
x[0]=3
print(entropy(y,2))
print(find_split(x,y,parameters))

# 10D, #100, 2-class, random
x=np.random.rand(100,10)
y=np.ones((100,1))
y[:50]=2
print(entropy(y,2))
print(find_split(x,y,parameters))

# XOR data
x=np.random.rand(100,2)
x[:,0]=1
x[:50,0]=0
x[:,1]=0
x[24:74,1]=1
y=np.ones((100,1))
y[25:50]=2
y[75:100]=2
print(entropy(y,2))
print(find_split(x,y,parameters))