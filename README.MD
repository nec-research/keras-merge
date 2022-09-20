# Keras Merge

Keras Merge allows to merge two Keras model instances without having access to
their generating functions.

## Release Notes
<table>
<tr><th>Version</th><th>Comment</th></tr>

<tr><td>v1.0</td><td>
Initial Release
</td></tr>

</table>

## Usage

```python
import keras_merge as km

from tensorflow.keras import Model, Input
import numpy as np
from keras.utils import plot_model

def model(mul: bool) -> Model:
    x = Input(shape=(2, 3))
    y = Input(shape=(2, 3))
    if mul: z = x * y
    else:   z = x + y
    return Model(inputs=[x, y], outputs=z)

A = model(True)
B = model(False)
C = km.merge(A, B,				# models
	[*A.inputs, B.inputs[0]],	# inputs
	B.outputs,					# outputs
	[							# mapping [(src->dst), ...]
		(A.outputs[0], B.inputs[1])
	]
)

a = np.random.rand(1, 2, 3)
b = np.random.rand(1, 2, 3)
c = np.random.rand(1, 2, 3)

d = A([a, b])
e = B([c, d])
f = C([a, b, c])

print("A+B:    ", np.reshape(e, -1))
print("Merged: ", np.reshape(f, -1))

plot_model(A, to_file='A.png')
plot_model(B, to_file='B.png')
plot_model(C, to_file='C.png')
```

## How to build

```bash
pip3 install illyrian
illyrian keras_merge.json
pip3 install keras_merge-1.0-py3-none-any.whl
```