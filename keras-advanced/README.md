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
