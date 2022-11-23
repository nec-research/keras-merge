import keras_merge as km
import keras
from keras import Model, Input
import numpy as np
from keras.utils import plot_model

def model():
	x = Input(shape=(3, 3, 3))
	y = keras.layers.Conv2D(3, 3, padding="same")(x)
	return Model(inputs=x, outputs=y)

A = model()
B = model()

C = km.merge(A, B, A.inputs, B.outputs, [(A.outputs[0], B.inputs[0])])

a0 = np.random.rand(1, 3, 3, 3)

a1 = A(a0)
a2 = B(a1)
b2 = C(a0)

print("A+B:    ", np.reshape(a2, -1))
print("Merged: ", np.reshape(b2, -1))

plot_model(A, to_file='A.png')
plot_model(B, to_file='B.png')
plot_model(C, to_file='C.png')