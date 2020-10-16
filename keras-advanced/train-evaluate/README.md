# Training and evaluation with built in methods

- Training
  This guide covers training, evaluation, and prediction (inference) models when using built-in APIs for training & validation (such as model.fit(), model.evaluate(), model.predict()).

When passing data to the built-in training loops of a model, you should either use NumPy arrays (if your data is small and fits in memory) or tf.data Dataset objects.

inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# The compile() method: specifying a loss, metrics, and an optimizer

- To train a model with fit(), you need to specify a loss function, an optimizer, and optionally, some metrics to monitor.

- You pass these to the model as arguments to the compile() method:

model.compile(
optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
loss=keras.losses.SparseCategoricalCrossentropy(),
metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

- The metrics argument should be a list -- your model can have any number of metrics.

- same can also be passed in string format
  model.compile(
  optimizer="rmsprop",
  loss="sparse_categorical_crossentropy",
  metrics=["sparse_categorical_accuracy"],
  )
