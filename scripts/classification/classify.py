import argparse
from autoencoder import AutoencoderConv
from util import train_step, test, visualize_stat, visualize_latent_space,visualize_interpolation
import tensorflow as tf


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
    loss = tf.keras.losses.BinaryCrossentropy()

    # initialize lists for later visualization.
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    # testing on our valid_ds once before we begin
    valid_loss, valid_accuracy = test(model, valid_ds, loss, is_training=False,visual =True)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    # Testing on our train_ds once before we begin
    train_loss, _ = test(model, train_ds, loss, is_training=False, visual=False)
    train_losses.append(train_loss)

    # training our model for num_epochs epochs.
    for epoch in range(num_epochs):
        print(
            f'Epoch: {str(epoch+1)} starting with (validation set) accuracy {valid_accuracies[-1]} and loss {valid_losses[-1]}')

        # training (and calculating loss while training)
        epoch_loss_agg = []

        #visualizing test images each 5 epochs
        visual = False
        if epoch%5==0:
            visual = True

        for input, target_image, _ in train_ds:
            train_loss = train_step(model, input, target_image, loss, optimizer, is_training=True)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        print(f'Epoch: {str(epoch+1)} train loss: {train_losses[-1]}')

        # testing our model in each epoch to track accuracy and loss on the validation set
        valid_loss, valid_accuracy = test(model, valid_ds, loss,is_training= False, visual=visual)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

    results = [train_losses, valid_losses, valid_accuracies]
    return results, model

# loading our created raw data
train_ds = tf.data.experimental.load(
    args.input+"/train", element_spec=(tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32),tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32), tf.TensorSpec(shape=(64), dtype=tf.int64)))
valid_ds = tf.data.experimental.load(
    args.input+"/valid", element_spec=(tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32),tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32), tf.TensorSpec(shape=(64), dtype=tf.int64)))
test_ds = tf.data.experimental.load(
    args.input+"/test", element_spec=(tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32),tf.TensorSpec(shape=(64, 28, 28, 1), dtype=tf.dtypes.float32), tf.TensorSpec(shape=(64), dtype=tf.int64)))


learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

models = [AutoencoderConv()]

train_losses = []
valid_losses = []
valid_accuracies = []

with tf.device('/device:gpu:0'):
    # training the model
    for model in models:
        results, trained_model = classify(model, optimizer, 10, train_ds, valid_ds)
        trained_model.summary()

        # saving results for visualization
        train_losses.append(results[0])
        valid_losses.append(results[1])
        valid_accuracies.append(results[2])

    # testing the trained model
    # (this code snippet should only be inserted when one decided on all hyperparameters)
    #_, test_accuracy = test(trained_model, test_ds, tf.keras.losses.BinaryCrossentropy(), False)
    #print("Accuracy (test set):", test_accuracy)

    # visualizing losses and accuracy
    #visualize_stat(train_losses, valid_losses, valid_accuracies)
    #visualize_latent_space(trained_model,test_ds.take(16))
    visualize_interpolation(trained_model,test_ds.take(2))
