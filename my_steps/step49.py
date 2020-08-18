if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Datasetクラスを継承して前処理ができるようにする
import numpy as np
import my_dezero

train_set = my_dezero.datasets.Spiral(train=True)
print(train_set[0])
print(len(train_set))