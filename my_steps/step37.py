if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from my_dezero import Variable
import my_dezero.functions as F

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.sin(x)
print(y)

c = Variable(np.array([[10,20,30], [40,50,60]]))
y = x + c
print(y)

x = Variable(np.array([[1,2,3], [4,5,6]]))
c = Variable(np.array([[10,20,30], [40,50,60]]))
t = x + c
y = F.sum(t)

y.backward(retain_grad=True)
print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)