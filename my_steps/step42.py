if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable
import my_dezero.functions as F
import matplotlib.pyplot as plt

# toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2*x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

# parameter
W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
	y = F.matmul(x,W) + b
	return y

# x0, x1はベクトルを想定
def mean_square_error(x0, x1):
	diff = x0 - x1
	return F.sum(diff**2) / len(diff)

lr = 0.1
iters = 100  # 損失関数をもとにしてパラメータを更新する回数

# 勾配降下法で最適なパラメータを求める
for i in range(iters):
	# 推論
	y_pred = predict(x)
	# 推論結果と真の値のMSEを求める
	loss = mean_square_error(y, y_pred)

	# 前のiterの勾配を削除
	W.cleargrad()
	b.cleargrad()

	# 勾配を求める
	loss.backward()

	# 勾配降下法でparameterを更新
	W.data -= lr*W.grad.data
	b.data -= lr*b.grad.data

	# 確認用 位置違くない？
	print('W : ', W.data, ' b : ', b.data, ' loss : ', loss.data)

plt.scatter(x.data, y.data)
plt.plot(x.data, predict(x).data, color='red')
plt.show()
