import weakref
import numpy as np
import contextlib
import my_dezero

# ========================================================
# Config
# ========================================================
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


# ========================================================
# Variable / Function
# ========================================================
try:
    import cupy
    array_types = (np.ndarray, cp.ndarray)
except ImportError:
    array_types = (np.ndarray)

class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return my_dezero.functions.transpose(self)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = my_dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))
            # Variableインスタンスにすることで逆伝播の計算でも計算グラフが作られる

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakrefVariabel

    def reshape(self, *shape): # ＊〇〇で可変長引数を取れる
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return my_dezero.functions.reshape(self, shape)

    def transpose(self):
        return my_dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return my_dezero.functions.sum(self, axis, keepdims)


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        # print('--------------')
        # print('Function')
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        #print(len(ys))
        #print(type(ys[0]))
        #print(ys[0].shape)
        #print(ys[0])
        outputs = [Variable(as_array(y)) for y in ys]

        # 逆伝播を可能にするかどうかを決める
        # 可能にする場合は計算途中が必要になるのでself.inputs, self.outputsで参照してメモリ上に残しておく
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]


    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Parameter(Variable):
    pass


# ========================================================
# 四則演算 / 演算子のoverload
# ========================================================
class Add(Function):
    def forward(self, x0, x1):
        # print(__class__.__name__)
        self.x0_shape, self.x1_shape = x0.shape, x1.shape  # broadcastで逆伝播する用に記憶しておく
        y = x0 + x1
        # print(type(y))
        return y

    def backward(self, gy):
        gy0, gy1 = gy, gy
        if self.x0_shape != self.x1_shape:  # 足し算の時点でbroadcastが発生していれば
            gy0 = my_dezero.functions.sum_to(gy0, self.x0_shape)
            gy1 = my_dezero.functions.sum_to(gy1, self.x1_shape)
        return gy0, gy1


def add(x0, x1):
    x1 = as_array(x1, my_dezero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        # print(__class__.__name__)
        y = x0 * x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs  # gyがVariableインスタンスなのでx0,x1もVariableインスタンスのままでいい
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1, my_dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        # print(__class__.__name__)
        # print(type(-x))
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        # print(__class__.__name__)
        self.x0_shape, self.x1_shape = x0.shape, x1.shape  # broadcastで逆伝播する用に記憶しておく
        y = x0 - x1
        return y

    def backward(self, gy):
        gy0, gy1 = gy, gy
        if self.x0_shape != self.x1_shape:  # 足し算の時点でbroadcastが発生していれば
            gy0 = my_dezero.functions.sum_to(gy0, self.x0_shape)
            gy1 = my_dezero.functions.sum_to(gy1, self.x1_shape)
        return gy0, -gy1



def sub(x0, x1):
    x1 = as_array(x1, my_dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, my_dezero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        # print(__class__.__name__)
        y = x0/x1
        # print(type(y))
        return y

    def backward(self, gy):
        x0,x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy*(-x0/(x1**2))
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1, my_dezero.cuda.get_array_module(x0.data))
    return Div()(x0 , x1)


def rdiv(x0, x1):
    x1 = as_array(x1, my_dezero.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c*x**(c-1)*gy
        return gx


def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
