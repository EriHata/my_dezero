if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable, Model
import my_dezero.layers as L
import my_dezero.functions as F 

# データセットの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2*np.pi*x) + np.random.randn(100, 1)

# ハイパラ設定
lr = 0.2
max_iter = 10000
hidden_size = 10

# モデル定義
class TwoLayerNet(Model):
	def __init__(self, hidden_size, out_size):
		super().__init__()
		self.l1 = L.Linear(hidden_size)
		self.l2 = L.Linear(out_size)

	def forward(self, x):
		y = self.l1(x)
		#y = F.sigmoid_simple(y)
		y = F.sigmoid(self.l1(x))
		y = self.l2(y)
		return y
# スカラーを返すmodel
model = TwoLayerNet(hidden_size, 1)

# 学習の開始
for i in range(max_iter):
	# 推論とlossの計算
	#print(i)
	y_pred = model(x)
	#print(i)
	loss = F.mean_square_error(y, y_pred)

	# 前回の勾配をリセットしてlossからbackpropagationして勾配を求める
	model.cleargrads()
	loss.backward()

	# 勾配をもとにしてparameterの更新
	for p in model.params():
		p.data -= p.grad.data*lr

	if i % 1000 == 0:
		print(loss)
print(loss)

# modelの可視化
model.plot(x)