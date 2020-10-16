# Load the data: the Cats vs Dogs dataset

Raw data download
First, let's download the 786M ZIP archive of the raw data:

Link: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip

you will have a PetImages folder which contain two subfolders, Cat and Dog. Each subfolder contains image files for each category.

# Filter out corrupted images

- When working with lots of real-world image data, corrupted images are a common occurence. Let's filter out badly-encoded images that do not feature the string "JFIF" in their header.

# Generate a Dataset

# Visualize the data

# Using image data augmentation

- When you don't have a large image dataset, it's a good practice to artificially introduce sample diversity by applying random yet realistic transformations to the training images, such as random horizontal flipping or small random rotations. This helps expose the model to different aspects of the training data while slowing down overfitting.

# Standardizing the data

- Our image are already in a standard size (180x180), as they are being yielded as contiguous float32 batches by our dataset. However, their RGB channel values are in the [0, 255] range. This is not ideal for a neural network; in general you should seek to make your input values small. Here, we will standardize values to be in the [0, 1] by using a Rescaling layer at the start of our model.

# Two options to preprocess the data

- There are two ways you could be using the data_augmentation preprocessor:

- Option 1: Make it part of the model, like this:

inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.experimental.preprocessing.Rescaling(1./255)(x)
... # Rest of the model

- With this option, your data augmentation will happen on device, synchronously with the rest of the model execution, meaning that it will benefit from GPU acceleration.

- Note that data augmentation is inactive at test time, so the input samples will only be augmented during fit(), not when calling evaluate() or predict().

- If you're training on GPU, this is the better option.

# Option 2: apply it to the dataset, so as to obtain a dataset that yields batches of augmented images, like this:

augmented_train_ds = train_ds.map(
lambda x, y: (data_augmentation(x, training=True), y))

- With this option, your data augmentation will happen on CPU, asynchronously, and will be buffered before going into the model.

- If you're training on CPU, this is the better option, since it makes data augmentation asynchronous and non-blocking.

# Configure the dataset for performance

- Let's make sure to use buffered prefetching so we can yield data from disk without having I/O becoming blocking:

# Build a model

- We'll build a small version of the Xception network. We haven't particularly tried to optimize the architecture; if you want to do a systematic search for the best model configuration, consider using Keras Tuner.

- Note that:

  - We start the model with the data_augmentation preprocessor, followed by a Rescaling layer.
  - We include a Dropout layer before the final classification layer.
