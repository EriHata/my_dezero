if '__file__' in globals():
	import sys,os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as numpy
from my_dezero import Variable

from my_dezero.utils import get_dot_grap


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1

# 変数に名前をつける
x0.name='x0'
x1.name='x1'
y.name='y'

txt = get_dot_grap(y, verbose=False)
print(txt)

# dotファイルとして保存
with open('sample.dot', 'w') as o:
	o.write(txt)