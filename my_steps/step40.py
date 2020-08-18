if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable

x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))
y = x0 + x1

# broadcastできているか確認
print(y)

# 逆伝播してx1の勾配がもとのx1と同じshapeになっているか確認
y.backward()
print(x1.grad)



