# Data Loading and preprocessing

Neural networks don't process raw data, like text files, encoded JPEG image files, or CSV files. They process vectorized & standardized representations.

- Text files need to be read into string tensors, then split into words. Finally, the words need to be indexed & turned into integer tensors.
- Images need to be read and decoded into integer tensors, then converted to floating point and normalized to small values (usually between 0 and 1).
- CSV data needs to be parsed, with numerical features converted to floating point tensors and categorical features indexed and converted to integer tensors. Then each feature typically needs to be normalized to zero-mean and unit-variance.

# Data Loading

Keras models accepts three types of inputs:

- Numpy array, good option if your data fits in memory.
- Tensorflow Datasets objects: suitable for datasets that do not fit in memory and that are streamed from disk or distributed filesystem
- Python generators that yields batches of data

Before you start training a model, you will need to make your data available as one of these formats. If you have a large dataset and you are training on GPU(s), consider using Dataset objects, since they will take care of performance-critical details, such as:

Asynchronously preprocessing your data on CPU while your GPU is busy, and buffering it into a queue.
Prefetching data on GPU memory so it's immediately available when the GPU has finished processing the previous batch, so you can reach full GPU utilization.
Keras features a range of utilities to help you turn raw data on disk into a Dataset:

tf.keras.preprocessing.image_dataset_from_directory turns image files sorted into class-specific folders into a labeled dataset of image tensors.
tf.keras.preprocessing.text_dataset_from_directory does the same for text files.
In addition, the TensorFlow tf.data includes other similar utilities, such as tf.data.experimental.make_csv_dataset to load structured data from CSV files.
