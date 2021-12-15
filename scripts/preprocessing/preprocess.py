import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser(
    description='Specify the path from where the datasets should be loaded and where the preprocessed datasets should be stored')
parser.add_argument('-input', type=str, help="Path to dataset folders")
parser.add_argument('-output', type=str,
                    help="Path to where the preprocessed datasets should be stored")

args = parser.parse_args()


def preprocess(ds):
    """
    Preparing our data for our model.
      Args:
        - ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: the dataset we want to preprocess

      Returns:
        - ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: preprocessed dataset
    """
    ds = ds.map(lambda image, label: (
        tf.cast(image, tf.float32), tf.cast(label, tf.float32)))

    # perfornm -1, 1 min max normalization
    ds = ds.map(lambda image, label: ((image)/255, label))

    ds = ds.map(lambda image, label: (
        image+tf.random.normal(shape=tf.shape(image)), label))

    ds = ds.map(lambda image, label: (tf.clip_by_value(image, 0, 1), label))

    # cache
    ds = ds.cache()
    # shuffle, batch, prefetch our dataset
    ds = ds.shuffle(5000)
    ds = ds.batch(64, drop_remainder=True)
    ds = ds.prefetch(20)
    return ds


# loading our created raw data
train_ds = tf.data.experimental.load(
    args.input+"/train", element_spec=(tf.TensorSpec(shape=(28, 28, 1), dtype=tf.dtypes.uint8), tf.TensorSpec(shape=(28, 28, 1), dtype=tf.dtypes.uint8)))
valid_ds = tf.data.experimental.load(
    args.input+"/valid", element_spec=(tf.TensorSpec(shape=(28, 28, 1), dtype=tf.dtypes.uint8), tf.TensorSpec(shape=(28, 28, 1), dtype=tf.dtypes.uint8)))
test_ds = tf.data.experimental.load(
    args.input+"/test", element_spec=(tf.TensorSpec(shape=(28, 28, 1), dtype=tf.dtypes.uint8), tf.TensorSpec(shape=(28, 28, 1), dtype=tf.dtypes.uint8)))

# performing preprocessing steps
train_ds = preprocess(train_ds)
valid_ds = preprocess(valid_ds)
test_ds = preprocess(test_ds)

# saving our preprocessed data
tf.data.experimental.save(train_ds, args.output+"/train")
tf.data.experimental.save(valid_ds, args.output+"/valid")
tf.data.experimental.save(test_ds, args.output+"/test")
