if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable
import my_dezero.functions as F
import matplotlib.pyplot as plt

# toy dataset
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)

# 重みの初期化
I, H, O = 1, 10, 1  # input, Hidden, output
W1 = Variable(0.01 * np.random.randn(I,H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H,O))
b2 = Variable(np.zeros(0))

# NNの推論関数
def predict(x):
	y = F.linear(x, W1, b1)
	y = F.sigmoid(y)
	y = F.linear(y, W2, b2)
	return y

lr = 0.2
iters = 10000

# NNの学習
for i in range(iters):
	# 推論
	y_pred = predict(x)
	# 損失関数求める
	loss = F.mean_square_error(y, y_pred)

	# 前のiterの勾配を消す
	W1.cleargrad()
	b1.cleargrad()
	W2.cleargrad()
	b2.cleargrad()
	# lossから各parameterの勾配を求める
	loss.backward()

	# 各parameterを更新する
	W1.data -= lr * W1.grad.data
	b1.data -= lr * b1.grad.data
	W2.data -= lr * W2.grad.data
	b2.data -= lr * b2.grad.data

	if i % 1000 == 0:  # 1000かいごとに結果を出力
		print(loss)

# 予測モデルの可視化
plt.scatter(x, y, s=10)
t = np.arange(0, 1, .01)[:, np.newaxis]  # ２次元にしている
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()
