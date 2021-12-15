from re import split
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Specify the path where the created datasets should be stored')
parser.add_argument('-output', type=str,
                    help="Path to where the created datasets should be stored")

args = parser.parse_args()


train_ds_unsplit, test_ds = tfds.load(
    name="mnist", split=["train", "test"], as_supervised=True)

train_ds = train_ds_unsplit.take(50000)
valid_ds = train_ds_unsplit.skip(50000)


def extract_images(ds):
    images = []
    for img, _ in ds:
        images.append(img)
    return tf.data.Dataset.from_tensor_slices((images, images))

#train_ds = np.concatenate([x for x, _ in train_ds], axis=0)


train_ds = extract_images(train_ds)
valid_ds = extract_images(valid_ds)
test_ds = extract_images(test_ds)

# locally saving the created datasets
tf.data.experimental.save(train_ds, args.output+"/train")
tf.data.experimental.save(valid_ds, args.output+"/valid")
tf.data.experimental.save(test_ds, args.output+"/test")
