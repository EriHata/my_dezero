import numpy as np

x = np.array([1,2,3])

def reshape(*shape):
	if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
		shape = shape[0]
	print(shape)

reshape((1,2))
reshape([1,2])
reshape(1,2)

