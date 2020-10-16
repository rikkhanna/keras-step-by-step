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


def imageDataAugmentation(train_ds, numImages):
    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ])
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(numImages):
            augmented_images = data_augmentation(images)
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(augmented_images[i].numpy().astype('uint8'))
            plt.xlabel(int(labels[i]))
        plt.show()


imageDataAugmentation(train_ds, 9)

# Standardize the data for neural network
