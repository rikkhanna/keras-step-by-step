# Image Classification from scratch

# This example shows how to do image classification from scratch,
# starting from JPEG image files on disk, without leveraging pre-trained weights or
# a pre-made Keras Application model. We demonstrate the workflow on
# the Kaggle Cats vs Dogs binary classification dataset.

# We use the image_dataset_from_directory utility to generate the datasets,
# and we use Keras image preprocessing layers for image standardization and data augmentation.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# download raw images and store them in images folder

# Filter out the corrupted images

# When working with lots of real-world image data, corrupted images are a common occurence.
# Let's filter out badly-encoded images that do not feature the string "JFIF" in their header.
import os


def filterOutCorruptedImages(foldersList, mainImageFolder):
    num_skipped = 0
    for folder_name in foldersList:
        PetImages = os.path.join('images', mainImageFolder)
        folder_path = os.path.join(PetImages, folder_name)
        for fname in os.listdir(folder_path):
            fPath = os.path.join(folder_path, fname)
            try:
                fobj = open(fPath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fPath)
    return num_skipped


# print("Deleted %d images" %
#       filterOutCorruptedImages(("Cat", 'Dog'), 'PetImages'))


# Generate a Dataset using filtered images

def train_dataset(batchSize, imgSize, MainImageFolder):
    train_dset = tf.keras.preprocessing.image_dataset_from_directory(
        MainImageFolder,
        validation_split=0.2,
        subset='training',
        seed=1337,
        image_size=imgSize,
        batch_size=batchSize
    )
    return train_dset


def validation_dataset(batchSize, imgSize, MainImageFolder):
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        MainImageFolder,
        validation_split=0.2,
        subset='validation',
        seed=1337,
        image_size=imgSize,
        batch_size=batchSize
    )
    return val_dataset


train_ds = train_dataset(32, (180, 180), "images/PetImages")
validation_ds = validation_dataset(32, (180, 180), "images/PetImages")

# After training the dataset, we can Visualize the images


def visualizeDataset(train_ds, numImages):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(numImages):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.xlabel(int(labels[i]))
        plt.show()


def data_augmentation():
    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ])
    return data_augmentation


def imageDataAugmentation(train_ds, numImages):

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(numImages):
            augmented_images = data_augmentation()(images)
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(augmented_images[i].numpy().astype('uint8'))
            plt.xlabel(int(labels[i]))
        plt.show()


# imageDataAugmentation(train_ds, 9)


# Configure the dataset for performance
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = validation_ds.prefetch(buffer_size=32)

# Standardize the data for neural network

# using option 2 coz we are training on CPU
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation()(x, training=True), y))


# build Model
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    # x = data_augmentation(inputs)

    # we'll be passing augmented data

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


image_size = (180, 180)
model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
