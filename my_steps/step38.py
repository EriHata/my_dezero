if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from my_dezero import Variable
import my_dezero.functions as F

# reshapeについて
x = np.array([[1,2,3], [4,5,6]])
print(x.shape)
y = np.reshape(x, (6,))
# y = x.reshape((6,)) 
# どちらでも結果は一緒
print(y.shape)

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

# Variableクラスにreshapeメソッドを追加し
# shapeを可変長引数で取得することでこの書き方が可能になった
x = Variable(np.array([[1,2,3], [4,5,6]]))
y = x.reshape((2,3))
print(y.shape)
z = x.reshape(2,3)
print(z.shape)


# transposeについて
print('----------------------------')
x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.transpose(x)
print(y)
y.backward()
print(x.grad)

x = Variable(np.random.rand(2,3))
print(x)
print(type(x))
y = x.transpose()
print(y)
print(type(y))
z = x.T  # ここで関数を呼んでいるのではなく、関数の結果をインスタンス変数として格納し、呼び出している
print(ｚ)
print(type(z))
