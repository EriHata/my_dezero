if '__file__' in globals():
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# データセットが読み込めるかの確認
import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt

x, t = dezero.datasets.get_spiral(train=True)
print(x)
print(x.shape)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

# データセット確認OK