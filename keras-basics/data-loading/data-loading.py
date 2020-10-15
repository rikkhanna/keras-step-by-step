import numpy as np
import tensorflow as tf
from tensorflow import keras

# obtaining a labeled dataset from image files on disk

# The label of a sample is the rank of its folder in alphanumeric order.
# label 0 will be elonmusk and 1 will be mark and 2 will be stevejobs

dataset = keras.preprocessing.image_dataset_from_directory(
    './images', batch_size=64, image_size=(200, 200))

# For demonstration, iterate over the batches yielded by the dataset.
for data, labels in dataset:
    print(data.shape)  # (64,)
    print(data.dtype)  # string
    print(labels.shape)  # (64,)
    print(labels.dtype)  # int32
