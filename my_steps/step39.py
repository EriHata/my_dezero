if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable
import my_dezero.functions as F

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.sum(x)
#y.backward()
print(y)
#print(x.grad)

x = np.array(([[1,2,3],[4,5,6]]))
y = np.sum(x, axis=0)
print(y)
print(x.shape, '->', y.shape)

x = np.array([[1,2,3], [4,5,6]])
y = np.sum(x, keepdims=True)
print(y)
print(y.shape)  # 2次元のまま

# Dezeroのsum関数でも2つの引数を指定できるように修正
x = Variable(np.array([[1,2,3], [4,5,6]]))
y = x.sum(axis=0)
print(y)

x = Variable(np.random.randn(2,3,4,5))
y = x.sum(keepdims=True)
print(y.shape)