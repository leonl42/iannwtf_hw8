import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def train_step(model, input, target, loss_function, optimizer, is_training):
    """
    Performs a forward and backward pass for  one dataponit of our training set
      Args:
        - model <tensorflow.keras.Model>: our created MLP model
        - input <tensorflow.tensor>: our input
        - target <tensorflow.tensor>: our target
        - loss_funcion <keras function>: function we used for calculating our loss
        - optimizer <keras function>: our optimizer used for backpropagation

      Returns:
        - loss <float>: our calculated loss for the datapoint
      """

    with tf.GradientTape() as tape:

        # forward step
        prediction = model(input)

        # calculating loss
        loss = loss_function(target, prediction)

        # calculaing the gradients
        gradients = tape.gradient(loss, model.trainable_variables)

    # updating weights and biases
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def my_tf_round(x, decimals=0):
  # source: https://stackoverflow.com/questions/46688610/tf-round-to-a-specified-precision
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def test(model, test_data, loss_function, is_training):
    """
    Test our MLP, by going through our testing dataset,
    performing a forward pass and calculating loss and accuracy
      Args:
        - model <tensorflow.keras.Model>: our created MLP model
        - test_data <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our preprocessed test dataset
        - loss_funcion <keras function>: function we used for calculating our loss

      Returns:
          - loss <float>: our mean loss for this epoch
          - accuracy <float>: our mean accuracy for this epoch
    """

    # initializing lists for accuracys and loss
    accuracy_aggregator = []
    loss_aggregator = []

    for (input, target) in test_data:

        # forward step
        prediction = model(input)

        # calculating loss
        loss = loss_function(target, prediction)

        # add loss and accuracy to the lists
        loss_aggregator.append(loss.numpy())

        for t, p in zip(target, prediction):
            accuracy_aggregator.append(
                tf.cast(my_tf_round(t, 1) == my_tf_round(p, 1), tf.float32))

    # calculate the mean of the loss and accuracy (for this epoch)
    loss = tf.reduce_mean(loss_aggregator)
    accuracy = tf.reduce_mean(accuracy_aggregator)

    return loss, accuracy


def visualize(train_losses, valid_losses, valid_accuracies):
    """
    Displays the losses and accuracies from the different models in a plot-grid.
    Args:
      train_losses <list<list<float>>>: mean training losses per epoch
      valid_losses <list<list<float>>>: mean  losses per epoch
      valid_accuracies <list<list<float>>>: mean accuracies (testing dataset) per epoch
    """

    titles = ["LSTM:1-layer", "LSTM:2-layer"]
    fig, axs = plt.subplots(2, 2)
    #fig.set_size_inches(13, 6)
    # making a grid with subplots
    for j in range(2):
        axs[0, j].plot(train_losses[j])
        axs[0, j].plot(valid_losses[j])
        axs[1, j].plot(valid_accuracies[j])
        last_accuracy = valid_accuracies[j][-1].numpy()
        axs[1, j].sharex(axs[0, j])
        axs[0, j].set_title(titles[j])
        axs[1, j].set_title("Last Accuracy: "+str(round(last_accuracy, 4)))

    fig.legend([" Train_ds loss", " Valid_ds loss",
               " Valid_ds accuracy"], loc="lower right")
    plt.xlabel("Training epoch")
    fig.tight_layout()
    plt.show()
