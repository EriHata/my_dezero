if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import matplotlib.pyplot as plt
from my_dezero import Variable
from my_dezero.utils import plot_dot_graph
import my_dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 2

for i in range(iters):
	gx = x.grad
	x.cleargrad()
	gx.backward(create_graph=True)

#Drawing a calcuration graph
gx = x.grad
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')
