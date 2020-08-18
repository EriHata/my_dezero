import my_dezero.functions as F
import my_dezero.layers as L
from my_dezero import Layer
from my_dezero import utils

class Model(Layer):
	def plot(self, *inputs, to_file='model.png'):
		y = self.forward(*inputs)
		return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
	def __init__(self, fc_output_sizes, activation=F.sigmoid_simple):
		super().__init__()
		# 活性化関数
		self.activation = activation
		self.layers = []

		# layerを作る
		# fc_output_sizeは全結合層の出力サイズのlist or tuple
		for i, out_size in enumerate(fc_output_sizes):
			layer = L.Linear(out_size)
			setattr(self, 'l'+ str(i), layer)  # 層の数がわからないのでself.〇〇みたいに設定できないからsetattrを使う
			self.layers.append(layer)

	def forward(self, x):
		for l in self.layers[:-1]:  # 最終層の手前までforで回す
			x = self.activation(l(x))
		return self.layers[-1](x)  # 最終層の出力結果を返す

