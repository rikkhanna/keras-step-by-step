# Building models with keras functional API

A "layer" is a simple input-output transformation (such as the scaling & center-cropping transformations above). For instance, here's a linear projection layer that maps its inputs to a 16-dimensional feature space:

dense = keras.layers.Dense(units=16)

A "model" is a directed acyclic graph of layers. You can think of a model as a "bigger layer" that encompasses multiple sublayers and that can be trained via exposure to data.

# The most common and most powerful way to build Keras models is the Functional API.
