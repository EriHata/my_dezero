if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import x

def rosenblock(x0,x1):
	y = 100*(x1-x0**2)**2 + (x0-1)**2
	return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

#y = rosenblock(x0,x1)
#y.backward()
#print(x0.grad, x1.grad)

lr = 0.001  # learning rate
iters = 1000000 # 勾配を求める→移動するを何回繰り返すかの回数

for i in range(iters):
	print(x0,x1)

	y = rosenblock(x0,x1)

	# 前のbackwardで求めた勾配が残っていると加算されていってしまうからNoneで初期化
	x0.cleargrad()
	x1.cleargrad()
	y.backward()

	x0.data -= lr*x0.grad
	x1.data -= lr*x1.grad

