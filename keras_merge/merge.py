__all__ = ['merge']

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, InputLayer, Concatenate
from typing import Optional, List, Dict, Tuple, Union, Any
from keras.engine.keras_tensor import KerasTensor
from keras.utils.object_identity import Reference

def merge(
		A: Model,
		B: Model,
		inputs: Union[List[KerasTensor], Tuple[KerasTensor], Dict[str, KerasTensor], KerasTensor],
		outputs: Union[tuple, list, dict, KerasTensor],
		mapping: Union[tuple, list],
		**kwargs: Optional[Dict[str, Any]]) -> Model:
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
					refs[oref] = cl.output
				else:
					cinputs = deref(node.keras_inputs)
					if isinstance(l, Concatenate):	refs[oref] = cl( cinputs)
					else:							refs[oref] = cl(*cinputs)

	clone_model(A)

	assert isinstance(mapping, (list, tuple))
	for k, v in mapping:
		assert isinstance(k, KerasTensor)
		assert isinstance(v, KerasTensor)
		refs[k.ref()] = refs.get(v.ref())

	clone_model(B)

	new_inputs  = deref(inputs)
	new_outputs = deref(outputs)

	assert kwargs is None or isinstance(kwargs, dict)
	return Model(inputs=new_inputs, outputs=new_outputs, **kwargs)