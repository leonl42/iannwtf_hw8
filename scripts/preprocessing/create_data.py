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


train_ds,valid_ds, test_ds = tfds.load(
    name="mnist", split=["train[0%:80%]","train[80%:100%]", "test"], as_supervised=True)

def extract_images(ds):
    images = []
    targets = []
    for img, t in ds:
        images.append(img)
        targets.append(t)
    return tf.data.Dataset.from_tensor_slices((images, images,targets))

train_ds = extract_images(train_ds)
valid_ds = extract_images(valid_ds)
test_ds = extract_images(test_ds)

# locally saving the created datasets
tf.data.experimental.save(train_ds, args.output+"/train")
tf.data.experimental.save(valid_ds, args.output+"/valid")
tf.data.experimental.save(test_ds, args.output+"/test")
