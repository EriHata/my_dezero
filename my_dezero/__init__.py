is_simple_core = False

if is_simple_core:
	from my_dezero.core_simple import Variable
	from my_dezero.core_simple import Function
	from my_dezero.core_simple import using_config
	from my_dezero.core_simple import no_grad
	from my_dezero.core_simple import as_array
	from my_dezero.core_simple import as_variable
	from my_dezero.core_simple import setup_variable

else:
	from my_dezero.core import Variable
	from my_dezero.core import Function
	from my_dezero.core import Parameter
	from my_dezero.core import using_config
	from my_dezero.core import no_grad
	from my_dezero.core import as_array
	from my_dezero.core import as_variable
	from my_dezero.core import setup_variable
	from my_dezero.layers import Layer
	from my_dezero.models import Model
	from my_dezero.dataloaders import DataLoader

	import my_dezero.functions
	import my_dezero.layers
	import my_dezero.datasets

setup_variable()