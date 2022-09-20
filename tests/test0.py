import keras_merge as km
from tensorflow.keras import Model, Input
import numpy as np
from keras.utils import plot_model

def model(mul):
    x = Input(shape=(2, 3))
    y = Input(shape=(2, 3))
    if mul: z = x * y
    else:   z = x + y
    return Model(inputs=[x, y], outputs=z)

A = model(True)
B = model(False)

C = km.merge(A, B, [*A.inputs, B.inputs[0]], B.outputs, [(B.inputs[1], A.outputs[0])])

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