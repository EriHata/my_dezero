if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_dezero import Variable

# Test functions for optimization
def sphere(x,y):
	z = x**2 + y**2
	return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x,y)
z.backward()
print(x.grad, y.grad)

def matyas(x,y):
	z = 0.26*(x**2 + y**2) - 0.48*x*y
	return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x,y)
z.backward()
print(x.grad, y.grad)

def goldstein(x,y):
	z = (1 + (x+y+1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) *\
	(30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
	return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x,y)
z.backward()
print(x.grad, y.grad)


# numerical gradient　で勾配確認
eps = 1e-4

x1 = Variable(np.array(1.0+eps))
x2 = Variable(np.array(1.0-eps))
y = Variable(np.array(1.0))
x = Variable(np.array(1.0))
y1 = Variable(np.array(1.0+eps))
y2 = Variable(np.array(1.0-eps))

x_grad = (goldstein(x1,y) - goldstein(x2,y)) / (2*eps)
y_grad = (goldstein(x,y1) - goldstein(x,y2)) / (2*eps)

print(x_grad, y_grad)