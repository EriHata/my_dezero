import numpy as np
from my_dezero import utils
from my_dezero.core import Function, as_variable
from my_dezero import cuda

# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)
		y = xp.sin(x)
		return y

	def backward(self, gy):
		x, = self.inputs
		gx = gy*cos(x)
		return gx

def sin(x):
	return Sin()(x)

class Cos(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)
		y = xp.cos(x)
		return y

	def backward(self, gy):
		x, = self.inputs
		gx = gy * -sin(x)
		return gx

def cos(x):
	return Cos()(x)

class Tanh(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)
		y = xp.tanh(x)
		return y

	def backward(self, gy):
		y = self.outputs[0]()
		gx = gy * (1 - y**2)
		return gx

def tanh(x):
	return Tanh()(x)

class Reshape(Function):
	def __init__(self, shape):
		self.shape = shape  # 変形するshape

	def forward(self, x):
		self.x_shape = x.shape  # 入力xのshapeを記憶しておく。backwardのときに使えるように
		y = x.reshape(self.shape)
		return y

	def backward(self, gy):
		return reshape(gy, self.x_shape)

class Exp(Function):
	def forward(self, x):
		# print(__class__.__name__)
		xp = cuda.get_array_module(x)
		y = xp.exp(x)
		# print(type(y))
		return y

	def backward(self, gy):
		y = self.outputs[0]()  # weakref
		gx = gy * y
		return gx

def exp(x):
	return Exp()(x)

class Log(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)
		y = xp.log(x)
		return y

	def backward(self, gy):
		x, = self.inputs
		gx = gy / x
		return gx

def log(x):
	return Log()(x)

# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================

def reshape(x, shape):
	if x.shape == shape:
		return as_variable(x)
	return Reshape(shape)(x)

class Transpose(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)
		y = xp.transpose(x)
		return y

	def backward(self, gy):
		gx = transpose(gy)
		return gx

def transpose(x):
	return Transpose()(x)

class GetItem(Function):
	def __init__(self, slices):
		self.slices = slices

	def forward(self, x):
		y = x[self.slices]
		return y

	def backward(self, gy):
		x, = self.inputs
		f = GetItemGrad(self.slices, x.shape)
		return f(gy)

def get_item(x, slices):
	return GetItem(slices)(x)

class GetItemGrad(Function):
	def __init__(self, slices, in_shape):
		self.slices = slices
		self.in_shape = in_shape

	def forward(self, gy):
		xp = cuda.get_array_module(gy)
		gx = xp.zeros(self.in_shape, dtype=gy.dtype)
		if xp is np:
			xp.add.at(gx, self.slices, gy)
		else:
			xp.scatter_add(gx, self.slices, gy)
		return gx

	def backward(self, ggx):
		return get_item(ggx, self.slices)

# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
	def __init__(self, axis, keepdims):
		self.axis = axis
		self.keepdims = keepdims

	def forward(self, x):
		self.x_shape = x.shape  # 元々のshape 逆伝播する際にgxのshapeとして利用する
		y = x.sum(axis=self.axis, keepdims=self.keepdims)  # 組み込み関数があるのでnp.sumじゃなくても良い
		return y

	def backward(self, gy):
		gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
		gx = broadcast_to(gy, self.x_shape)
		return gx

def sum(x, axis=None, keepdims=False):
	return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
	def __init__(self, shape):
		self.shape = shape  # 変更したいshape

	def forward(self, x):
		self.x_shape = x.shape  # もとの入力xのshape
		xp = cuda.get_array_module(x)
		y = xp.broadcast_to(x, self.shape)
		return y

	def backward(self, gy):
		gx = sum_to(gy, self.x_shape)
		return gx

def broadcast_to(x, shape):
	if x.shape == shape:
		return as_variable(x)
	return BroadcastTo(shape)(x)

class SumTo(Function):
	def __init__(self, shape):
		self.shape = shape  # 変更したいshape

	def forward(self, x):
		self.x_shape = x.shape  # もとの入力xのshape
		y = utils.sum_to(x, self.shape)
		return y

	def backward(self, gy):
		gx = broadcast_to(gy, self.x_shape)
		return gx

def sum_to(x, shape):
	if x.shape == shape:
		return as_variable(x)
	return SumTo(shape)(x)

class MatMul(Function):
	def forward(self, x, W):
		y = x.dot(W)
		return y

	def backward(self, gy):
		x, W = self.inputs
		gx = matmul(gy, W.T)
		gW = matmul(x.T, gy)
		return gx, gW

def matmul(x, W):
	return MatMul()(x, W)


def linear_simple(x, W, b=None):
	x, W = as_variable(x), as_variable(W)
	t = matmul(x, W)
	if b is None:
		return t

	y = t + b
	t.data = None  # 逆伝播の際に使わないから消去
	return y

def matmul(x, W):
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x, W, b):
    	# print(__class__.__name__)
    	y = x.dot(W)
    	if b is not None:
    		y += b
    	# print(type(y))
    	return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)

# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================

# クラスを作った関数だとcore.pyでTypeError: <class 'my_dezero.core.Variable'> is not supportedになってしまう
class Sigmoid_Simple(Function):
	def forward(self, x):
		# print(__class__.__name__)
		y = 1 / (1+exp(-x))
		# print(type(y))
		return y.data

	def backward(self, gy):
		y = self.outputs[0]()  # weakref
		gx = gy * y * (1 - y)
		return gy

def sigmoid_simple(x):
	return Sigmoid_Simple()(x)

#def sigmoid_simple(x):
#    x = as_variable(x)
#    y = 1 / (1 + exp(-x))
#    return y

class Sigmoid(Function):
    def forward(self, x):
    	xp = cuda.get_array_module(x)
    	y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
    	return y

    def backward(self, gy):
    	y = self.outputs[0]()  # weakref
    	gx = gy * y * (1 - y)
    	return gx


def sigmoid(x):
	return Sigmoid()(x)


def softmax_simple(x, axis=1):
	x = as_variable(x)
	y = exp(x)
	sum_y = sum(y, axis=axis, keepdims=True)
	return y / sum_y

class Softmax(Function):
	def __init__(self, axis=1):
		self.axis = axis  # forwardとbackwardで使うから

	def forward(self, x):
		xp = cuda.get_array_module(x)
		y = x - x.max(axis=self.axis, keepdims=True)  # ここでaxis使うから __init__で定義しておく
		c = (np.max(x))  # over flow対策
		y = exp(x-c)
		sum_y = sum(y)
		y = y / sum_y
		return y.data

	def backward(self, gy):
		y = self.outputs[0]()
		gx = y * gy
		sumdx = gx.sum(axis=self.axis, keepdims=True)
		gx -= y * sumdx
		return gx

def softmax(x):
	return Softmax()(x)


# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================
class MeanSquaredError(Function):
	def forward(self, x0, x1):
		diff = x0 - x1
		y = (diff**2).sum() / len(diff)
		return y

	def backward(self, gy):
		x0, x1 = self.inputs
		diff = x0 - x1
		gy = broadcast_to(gy, diff.shape)
		gx0 = gy * diff * (2./len(diff))
		gx1 = -gx0
		return gx0, gx1

def mean_square_error(x0, x1):
	return MeanSquaredError()(x0, x1)

def sotfmax_cross_entropy_simple(x, t):  # x 予測ラベル、t 教師ラベル
	# Functionクラスを継承しない関数の場合、最初に入力変数をvariableにする
	# tはone-hot vectorではなく正解レベルを並べたN次元ベクトル
	x, t = as_variable(x), as_variable(t)
	N = x.shape[0]  # データ数
	p = softmax(x)
	p = clip(p, 1e-15, 1.0)  # log(0)を防ぐため
	log_p = log(p).data  # Variableで帰ってくるのでndarrayにしておかないとlog_p[]でアクセスするときにTypeError: 'Variable' object is not subscriptableとなる
	tlog_p = log_p[np.arange(N), t.data]
	y = -1 * sum(tlog_p) / N
	return y

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

# =============================================================================
# accuracy / dropout / batch_norm / embed_id
# =============================================================================

def accuracy(y, t):
	y, t = as_variable(y), as_variable(t)
	pred = y.data.argmax(axis=1).reshape(t.shape)
	result = (pred == t.data)
	acc = result.mean()
	return Valiable(as_array(acc))

# =============================================================================
# max / min / clip
# =============================================================================

class Max(Function):
	def __init__(self, axis=None, keepdims=False):
		self.axis = axis
		self.keepdims = keepdims

	def forward(self, x):
		y = x.max(axis=self.axis, keepdims=self.keepdims)  # 組み込み関数を使う
		return y

	# backwardの処理わかっていない
	def backward(self, gy):
		x = self.inputs[0]
		y = self.outputs[0]()  # weakref

		shape = utils.max_backward_shape(x, self.axis)
		gy = reshape(gy, shape)
		y = reshapw(y, shape)
		cond = (x.data == y.data)
		gy = broadcast_to(gy, cond.shape)
		return gy * cond

# Maxクラスを継承して、forwardをオーバーライドする
# backwardは同じ処理になるのでそのまま使う
class Min(Max):
	def forward(self, x):
		y = x.min(axis=self.axis, keepdims=self.keepdims)
		return y

def max(x, axis=None, keepdims=False):
	return Max(axis, keepdims)(x)

def min(x, axis=None, keepdims=False):
	return Min(axis, keepdims)(x)


class Clip(Function):
	def __init__(self, x_min, x_max):
		self.x_min = x_min
		self.x_max = x_max

	def forward(self, x):
		xp = cuda.get_array_module(x)
		y = xp.clip(x, self.x_min, self.x_max)
		return y

	def backward(self, gy):
	    x, = self.inputs
	    mask = (x.data >= self.x_min) * (x.data <= self.x_max)
	    return gx

def clip(x, x_min, x_max):
	return Clip(x_min, x_max)(x)
