import os
import numpy as np
from my_dezero import Variable, as_variable
from my_dezero import cuda



def _dot_var(v, verbose=False):
	dot_var = '{}[label="{}", color=orange, style=filled]\n'

	name = '' if v.name is None else v.name
	
	if verbose and v.data is not None:
		if v.name is not None:
			name += ': '
		name += str(v.shape) + ' ' + str(v.dtype)

	return dot_var.format(id(v), name)


def _dot_func(f):
	dot_func = '{}[label="{}", color=lightblue, style=filled, shape=box]\n'
	txt = dot_func.format(id(f), f.__class__.__name__)

	dot_edge = '{} -> {}\n'
	for x in f.inputs:
		txt += dot_edge.format(id(x), id(f))
	for y in f.outputs:
		txt += dot_edge.format(id(f), id(y()))  # y is weakref

	return txt


# 使用例
#x0 = Variable(np.array(1.0))
#x1 = Variable(np.array(1.0))
#y = x0+x1

#txt = _dot_func(y.creator)  # creatorにfunctionオブジェクトが格納されている」
#print(txt)


def get_dot_graph(output, verbose=True):
	txt = ''
	funcs = []
	seen_set = set()

	def add_func(f):
		if f not in seen_set:
			funcs.append(f)
			# funcs.sort(key=lambda x:xgeneration) fと各変数の繋がりさえわかればいいので世代は考えなくていい
			seen_set.add(f)

	add_func(output.creator)
	txt += _dot_var(output, verbose)

	while funcs:
		func = funcs.pop()
		# 関数ノードとedgeを書く
		txt += _dot_func(func)
		# 変数ノードを書く
		for x in func.inputs:
			txt += _dot_var(x, verbose)

			if x.creator is not None:
				add_func(x.creator)
	return 'digraph g{\n'+ txt +'}'

import os 
import subprocess


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
	dot_graph = get_dot_graph(output, verbose)
	# dot データをfileに保存
	tmp_dir = os.path.join(os.path.expanduser('~'), '.my_dezero')
	if not os.path.exists(tmp_dir):  # ~/.my_dezeroディレクトリがなかったら作成
		os.mkdir(tmp_dir)
	graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

	with open(graph_path, 'w') as f:
		f.write(dot_graph)

	# dotコマンドを呼ぶ
	extension = os.path.splitext(to_file)[1][1:]  # 拡張子(png, pdfなど)
	cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
	subprocess.run(cmd, shell=True)


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not hasattr(axis, 'len'):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def logsumexp(x, axis=1):
	xp =cuda.get_array_module(x)
	m = x.max(axis=axis, keepdims=True)
	y = x - m
	xp.exp(y, out=y)
	s = y.sum(axis=axis, keepdims=True)
	xp.log(s, out=s)
	m += s
	return m



def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y




