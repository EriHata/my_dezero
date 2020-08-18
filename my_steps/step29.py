if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable

def f(x):
	y = x**4 - 2*x**2
	return y

def dx2(x):
	return 12*x**2 - 4

x = Variable(np.array(2.0))
iters = 10

print("newton's method")

for i in range(iters):
	print(i, x.data)

	y = f(x)
	#print('y:', type(y))
	x.cleargrad()
	y.backward()

	#print('x.grad',x.grad)
	#print(type(x.grad))
	#print('dx2 ', type(dx2(x)))
	x.data -= x.grad/dx2(x.data) # data必要？なぜ？


# dataないと<class 'my_dezero.core_simple.Variable'> is not supportedのエラー
# ちょっと謎のまま　沼に嵌りそうなので一旦中止

# 勾配降下法だと

print('gradient descent')

x = Variable(np.array(2.0))
lr = 0.01  # learning rate
iters = 100 # 勾配を求める→移動するを何回繰り返すかの回数

for i in range(iters):
	print(i, x)

	y = f(x)

	# 前のbackwardで求めた勾配が残っていると加算されていってしまうからNoneで初期化
	x.cleargrad()
	y.backward()

	x.data -= lr*x.grad