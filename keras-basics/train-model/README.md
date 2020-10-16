# Training models with fit()

At this point, you know:

- How to prepare your data (e.g. as a NumPy array or a tf.data.Dataset object)
- How to build a model that will process your data

The next step is to train your model on your data. The Model class features a built-in training loop, the fit() method. It accepts Dataset objects, Python generators that yield batches of data, or NumPy arrays.

Before you can call fit(), you need to specify an optimizer and a loss function. This is the compile() step:

- model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
  loss=keras.losses.CategoricalCrossentropy())

Loss and optimizer can be specified via their string identifiers

- model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

Once your model is compiled, you can start "fitting" the model to the data.

Besides the data, you have to specify two key parameters: the batch_size and the number of epochs (iterations on the data).
Here our data will get sliced on batches of 32 samples, and the model will iterate 10 times over the data during training.

- model.fit(numpy_array_of_samples, numpy_array_of_labels, batch_size=32, epochs=10)
