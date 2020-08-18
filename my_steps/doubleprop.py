
# ヘッセ行列とベクトルの結果だけがほしいとき
# ヘッセ行列を直接求めると計算時間がかかるので
# double propagationを使って計算する
# ∇^2f(x)v = ∇(v^T∇f(x))
# ∇(x)はgradf(x)

if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable
import my_dezero.functions as F

# 変数が2つ
x = Variable(np.array([1.0, 2.0]))
v = Variable(np.array([4.0, 5.0]))

# 多変量関数を作成
def f(x):
	t = x ** 2
	y = F.sum(t)
	return t

# ヘッセ行列とベクトルの結果を求めるためにdouble propagationを使う
y = f(x)
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = F.matmul(v, gx)
z.backward()
print(x.grad)


