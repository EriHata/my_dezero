if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable
import my_dezero.functions as F

x = Variable(np.array([[1,2,3], [4,5,6]]))
print(id(x))
y = F.get_item(x, 1)
print(y)

y.backward()
print(x.grad)
print(y.grad)  # Noneなのでy.shapeの形で自動補完される


# 同じ要素を複数回抜き出す
x = Variable(np.array([[1,2,3], [4,5,6]]))
print(id(x))
indices = np.array([0,0,1])
y = F.get_item(x, indices)
print(y)

Variable.__getitem__ = F.get_item

print('-------------')
y = x[1]
print(y)
print(y.shape)
y = x[:, 2]
print(y)
print(y.shape)


from my_dezero.models import MLP
print('MLP model')
model = MLP((10, 3))
x = np.array([[0.2, -0.4], [-0.7, 2.4], [3.0, 0.5]])
y = model(x)
print(y) # 一番要素の値が高いインデックスが、モデルの分類したクラスになる


print('------softmax-------')
from my_dezero import Variable, as_variable, as_array
import my_dezero.functions as F

def softmax1d(x):
	c = as_variable(as_array(np.max(x)))  # over flow対策
	x = as_variable(x)
	y = F.exp(x-c)
	sum_y = F.sum(y)
	return y / sum_y

model = MLP((10, 3))
x = np.array([[0.2, -0.4]])
y = model(x)
p = softmax1d(y.data)
print(y)
print(p)

print('-----cross_entropy-----')
import my_dezero.functions as F
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.sotfmax_cross_entropy_simple(y, t)
lrint(loss)