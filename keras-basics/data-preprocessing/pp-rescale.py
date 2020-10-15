# Both the Rescaling layer and the CenterCrop layer are stateless, so it isn't necessary to call adapt() in this case.
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
import numpy as np

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(
    0, 256, size=(64, 200, 200, 3)).astype("float32")

cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)

output_data = scaler(cropper(training_data))
print("shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))
