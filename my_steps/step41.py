if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable
import my_dezero.functions as F


x = Variable(np.random.randn(2,3))
W = Variable(np.random.randn(3,4))
# 実装した行列積の関数
y = F.matmul(x, W)
# 逆伝播が正常に動くか確認
y.backward()
print(x)
print(x.grad)
print(x.grad.shape)
print('--------------')
print(y)
print(W.grad)
print(W.grad.shape)