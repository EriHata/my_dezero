if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import my_dezero.functions as F
from my_dezero.core import Variable, Parameter 

x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))
y = x * p
z = p * x

print(isinstance(p, Parameter))
print(isinstance(x, Parameter))
print(isinstance(y, Parameter))
print(isinstance(z, Parameter))

print(isinstance(y, Variable))
print(isinstance(z, Variable))

import my_dezero.layers as L
layer = L.Layer()

# ここでparameterを追加しているはずなんだけどうまくいっていない
layer.p1 = Parameter(np.array(1))
layer.p2 = Parameter(np.array(2))
layer.p3 = Variable(np.array(3))
layer.p4 = 'test'
print(type(layer.p1))
print(layer.p1)
print(layer.p2)
print(layer.p3)

print(layer._params)
print('---------------')

for name in layer._params:
	print(name, layer.__dict__[name])


print('---------------')
# toydataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2*np.pi*x) + np.random.rand(100, 1)

l1 = L.Linear(10)  # 出力サイズの指定
l2 = L.Linear(1)

def predict(x):
	y = l1(x)
	y = F.sigmoid_simple(y)
	y = l2(y)
	return y
lr = 0.2
iters = 10000

for i in range(iters):
	# 推論
	y_pred = predict(x)
	# lossを計算
	loss = F.mean_square_error(y, y_pred)

	# parameterのgradをリセットしてbackprapagation
	l1.cleargrads()
	l2.cleargrads()
	loss.backward()

	# parameterのインスタンスに勾配をもとに修正した重みを格納
	for l in [l1, l2]:
		for p in l.params():
			p.data -= lr * p.grad.data

	# lossの確認用
	if i % 1000 == 0:
		print(loss)

