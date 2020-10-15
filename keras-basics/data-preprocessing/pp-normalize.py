from tensorflow.keras.layers.experimental.preprocessing import Normalization
import numpy as np
# Example image data, with values in the [0, 255] range
training_data = np.random.randint(
    0, 256, size=(64, 200, 200, 3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))
