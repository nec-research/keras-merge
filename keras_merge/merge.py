__all__ = ['merge']

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, InputLayer, Concatenate
from typing import List, Dict, Tuple, Union, Any
from keras.engine.keras_tensor import KerasTensor

def merge(
		A: Model,
		B: Model,
		inputs: Union[List[KerasTensor], Tuple[KerasTensor], Dict[str, KerasTensor], KerasTensor],
		outputs: Union[tuple, list, dict, KerasTensor],
		mapping: Union[tuple, list],
		**kwargs: Dict[str, Any]) -> Model:

	def clone(layer: Layer) -> Layer:
		assert isinstance(layer, Layer)
		return layer.__class__.from_config(layer.get_config())

	refs = {}
	def deref(nodes: Union[tuple, list, dict, KerasTensor]):
		if		isinstance(nodes, tuple):	return tuple(deref(n) for n in nodes)
		elif	isinstance(nodes, list):	return list (deref(n) for n in nodes)
		elif	isinstance(nodes, dict):	return {k: deref(v) for k, v in nodes.items()}
		elif	isinstance(nodes, KerasTensor):
			x = refs.get(nodes.ref())
			assert x is not None, f'missing ref: {nodes}'
			return x
		assert False, f'unsupported type: {nodes}'

	def clone_model(M: Model) -> None:
		assert isinstance(M, Model)
		for l in M.layers:
			if l.output.ref() not in refs:
				node	= l.inbound_nodes[0]; assert node.layer is l
				cl		= clone(l)
				oref	= l.output.ref()

				if isinstance(l, InputLayer):
					refs[oref]	= cl.output
				else:
					cinputs		= deref(node.keras_inputs)
					refs[oref]	= cl(cinputs) if isinstance(l, Concatenate) else cl(*cinputs)

	clone_model(A)

	assert isinstance(mapping, (list, tuple))
	for src, dst in mapping:
		assert isinstance(src, KerasTensor)
		assert isinstance(dst, KerasTensor)
		refs[dst.ref()] = refs.get(src.ref())

	clone_model(B)

	new_inputs  = deref(inputs)
	new_outputs = deref(outputs)

	assert isinstance(kwargs, dict)
	return Model(inputs=new_inputs, outputs=new_outputs, **kwargs)