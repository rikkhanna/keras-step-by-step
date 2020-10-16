# Building models with keras functional API

A "layer" is a simple input-output transformation (such as the scaling & center-cropping transformations above). For instance, here's a linear projection layer that maps its inputs to a 16-dimensional feature space:

dense = keras.layers.Dense(units=16)

A "model" is a directed acyclic graph of layers. You can think of a model as a "bigger layer" that encompasses multiple sublayers and that can be trained via exposure to data.

# The most common and most powerful way to build Keras models is the Functional API.

A "layer" is a simple input-output transformation (such as the scaling & center-cropping transformations above). For instance, here's a linear projection layer that maps its inputs to a 16-dimensional feature space:

- dense = keras.layers.Dense(units=16)

A "model" is a directed acyclic graph of layers. You can think of a model as a "bigger layer" that encompasses multiple sublayers and that can be trained via exposure to data.

The most common and most powerful way to build Keras models is the Functional API. To build models with the Functional API, you start by specifying the shape (and optionally the dtype) of your inputs. If any dimension of your input can vary, you can specify it as None. For instance, an input for 200x200 RGB image would have shape (200, 200, 3), but an input for RGB images of any size would have shape (None, None, 3).

# Let's say we expect our inputs to be RGB images of arbitrary size

inputs = keras.Input(shape=(None, None, 3))
