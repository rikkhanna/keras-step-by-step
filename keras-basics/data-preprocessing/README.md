# Data Preprocessing with keras

Once your data is in the form of string/int/float NumpPy arrays, or a Dataset object (or Python generator) that yields batches of string/int/float tensors, it is time to preprocess the data. This can mean:

- Tokenization of string data, followed by token indexing.
- Feature normalization.
- Rescaling the data to small values (in general, input values to a neural network should be close to zero -- typically we expect either data with zero-mean and unit-variance, or data in the [0, 1] range.

# Ideal machine learning model is end to end

In general, you should seek to do data preprocessing as part of your model as much as possible, not via an external data preprocessing pipeline. That's because external data preprocessing makes your models less portable when it's time to use them in production. Consider a model that processes text: it uses a specific tokenization algorithm and a specific vocabulary index. When you want to ship your model to a mobile app or a JavaScript app, you will need to recreate the exact same preprocessing setup in the target language. This can get very tricky: any small discrepancy between the original pipeline and the one you recreate has the potential to completely invalidate your model, or at least severely degrade its performance.

It would be much easier to be able to simply export an end-to-end model that already includes preprocessing. The ideal model should expect as input something as close as possible to raw data: an image model should expect RGB pixel values in the [0, 255] range, and a text model should accept strings of utf-8 characters. That way, the consumer of the exported model doesn't have to know about the preprocessing pipeline.

# Using keras preprocessing layers

In Keras, you do in-model data preprocessing via preprocessing layers. This includes:

- Vectorizing raw strings of text via the TextVectorization layer
- Feature normalization via the Normalization layer
- Image rescaling, cropping, or image data augmentation

The key advantage of using Keras preprocessing layers is that they can be included directly into your model, either during training or after training, which makes your models portable.
