if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable
from my_dezero import optimizers
import my_dezero.functions as F
from my_dezero.models import MLP

# データセットの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2*np.pi*x) + np.random.randn(100, 1)

# ハイパラ設定
lr = 0.2
max_iter = 10000
hidden_size = 10

# modelインスタンス生成
model = MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)

# 学習の開始
for i in range(max_iter):
	# 推論とlossの計算
	y_pred = model(x)
	loss = F.mean_square_error(y, y_pred)

	# 前回の勾配をリセットしてlossからbackpropagationして勾配を求める
	model.cleargrads()
	loss.backward()
	print(i)

	# 勾配をもとにしてparameterの更新　これだけで済んでしまう
	optimizer.update()

	if i % 1000 == 0:
		print(loss)
print(loss)
