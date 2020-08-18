import contextlib

@contextlib.contextmanager
def config_test():
	print('start')  # pre processing
	try:
		yield
	finally:
		print('done')  # post processing


with config_test():
	print('process...')