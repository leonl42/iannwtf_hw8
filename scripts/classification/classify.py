import argparse
from autoencoder import Autoencoder
from util import train_step, test, visualize
import tensorflow as tf
import tkinter

parser = argparse.ArgumentParser(
    description='Specify the path where the preprocessed datasets are stored')
parser.add_argument('-input', type=str,
                    help="Path to where the preprocessed datasets are stored")

args = parser.parse_args()


def classify(model, optimizer, num_epochs, train_ds, valid_ds):
    """
    Trains and tests our predefined model.
        Args:
            - model <tensorflow.keras.Model>: our untrained model
            - optimizer <keras function>: optimizer for the model
            - num_epochs <int>: number of training epochs
            - train_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our training dataset
            - valid_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our validation set for testing and optimizing hyperparameters
        Returns:
            - results <list<list<float>>>: list with losses and accuracies
            - model <tensorflow.keras.Model>: our trained MLP model
    """

    tf.keras.backend.clear_session()

    # initialize the loss: categorical cross entropy
    cross_entropy_loss = tf.keras.losses.MeanSquaredError()

    # initialize lists for later visualization.
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    # testing on our valid_ds once before we begin
    valid_loss, valid_accuracy = test(
        model, valid_ds, cross_entropy_loss, False)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    # Testing on our train_ds once before we begin
    train_loss, _ = test(model, train_ds, cross_entropy_loss, False)
    train_losses.append(train_loss)

    # training our model for num_epochs epochs.
    for epoch in range(num_epochs):
        print(
            f'Epoch: {str(epoch+1)} starting with (validation set) accuracy {valid_accuracies[-1]} and loss {valid_losses[-1]}')

        # training (and calculating loss while training)
        epoch_loss_agg = []

        for input, target in train_ds:
            train_loss = train_step(
                model, input, target, cross_entropy_loss, optimizer, True)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        print(f'Epoch: {str(epoch+1)} train loss: {train_losses[-1]}')

        # testing our model in each epoch to track accuracy and loss on the validation set
        valid_loss, valid_accuracy = test(
            model, valid_ds, cross_entropy_loss, False)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

    results = [train_losses, valid_losses, valid_accuracies]
    return results, model


# loading our created raw data
train_ds = tf.data.experimental.load(
    args.input+"/train", element_spec=(tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32), tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32)))
valid_ds = tf.data.experimental.load(
    args.input+"/valid", element_spec=(tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32), tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32)))
test_ds = tf.data.experimental.load(
    args.input+"/test", element_spec=(tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32), tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32)))


learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

models = [Autoencoder([28, 28])]

train_losses = []
valid_losses = []
valid_accuracies = []


def _from_rgb(rgb):
    """
    translates an rgb tuple of integers to hex
        Args:
            - rgb: <tuple> tuple of rgb values
        Returns:
            - rgb: <string> rgb value translated to hex
    """
    return "#%02x%02x%02x" % rgb


with tf.device('/device:gpu:0'):
    # training the model
    for model in models:
        results, trained_model = classify(
            model, optimizer, 20, train_ds, valid_ds)
        trained_model.summary()

        # saving results for visualization
        train_losses.append(results[0])
        valid_losses.append(results[1])
        valid_accuracies.append(results[2])

    # testing the trained model
    # (this code snippet should only be inserted when one decided on all hyperparameters)
    _, test_accuracy = test(
        trained_model, test_ds, tf.keras.losses.BinaryCrossentropy(), False)
    print("Accuracy (test set):", test_accuracy)

    # visualizing losses and accuracy
    #visualize(train_losses, valid_losses, valid_accuracies)


root = tkinter.Tk()
root.geometry("1200x600")
canvas = tkinter.Canvas(root)
canvas.pack()

f, l = next(iter(train_ds.take(1)))
img1 = next(iter(f))
img2 = next(iter(l))

pred = models[0](f)
img1 = next(iter(pred))

size = [5, 5]
offset = [0, 0]
for image in [img1, img2]:

    for count_x, x in enumerate(image):
        for count_y, y in enumerate(x):
            g = int(y[0]*100)
            canvas.create_rectangle(count_x*size[0]+offset[0], count_y*size[1]+offset[1], count_x*size[0]+size[0]+offset[0], count_y *
                                    size[1]+size[1]+offset[1], fill=_from_rgb((g, g, g)), outline=_from_rgb((g, g, g)))

    offset = [offset[0]+150, offset[1]]


root.mainloop()
