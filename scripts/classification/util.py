import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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


def test(model, test_data, loss_function, is_training,visual=True):
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
    for (input, target_image, _) in test_data:
        # forward step
        prediction = model(input)
        # calculating loss
        loss = loss_function(target_image, prediction)

        # add loss and accuracy to the lists
        loss_aggregator.append(loss.numpy())

        for t, p in zip(target_image, prediction):
            accuracy_aggregator.append(
                tf.cast(my_tf_round(t, 1) == my_tf_round(p, 1), tf.float32))

        if visual:
            visualize_sample_img(target_image ,prediction)
            visual = False

    # calculate the mean of the loss and accuracy (for this epoch)
    loss = tf.reduce_mean(loss_aggregator)
    accuracy = tf.reduce_mean(accuracy_aggregator)

    return loss, accuracy


def visualize_sample_img(target,prediction):
    """
    """
    fig, axs = plt.subplots(4, 2)
    for i in range(4):
        axs[i,0].imshow(target[i], cmap="Greys_r")
        axs[i,1].imshow(prediction[i], cmap="Greys_r")

    plt.show()


def visualize_stat(train_losses,valid_losses,valid_accuracies):
    """
    Displays the losses and accuracies from the different models in a plot-grid.
    Args:
      train_losses <list<list<float>>>: mean training losses per epoch
      valid_losses <list<list<float>>>: mean  losses per epoch
      valid_accuracies <list<list<float>>>: mean accuracies (testing dataset) per epoch
    """

    titles = [""]
    fig, axs = plt.subplots(2, 1)
    #fig.set_size_inches(13, 6)
    # making a grid with subplots
    for j in range(1):
        axs[0].plot(train_losses[j])
        axs[0].plot(valid_losses[j])
        axs[1].plot(valid_accuracies[j])
        last_accuracy = valid_accuracies[-1][-1].numpy()
        axs[1].sharex(axs[0])
        axs[0].set_title(titles[j])
        axs[1].set_title("Last Accuracy: "+str(round(last_accuracy,4)))

    fig.legend([" Train_ds loss"," Valid_ds loss"," Valid_ds accuracy"],loc="lower right")
    plt.xlabel("Training epoch")
    fig.tight_layout()
    plt.show()

def visualize_latent_space(model,x_data):

    # targets
    t = []
    for _,_,target in x_data:
        t.append(target)

    x_encoded = model.encoder.predict(x_data)

    # Compute t-SNE embedding of latent space
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(x_encoded)

    # Plot images according to t-sne embedding
    fig, ax = plt.subplots()
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t)
    plt.legend(handles=scatter.legend_elements()[0],labels=["0","1","2","3","4","5","6","7","8","9"],title="number")
    plt.show()

def interpolate_points(p1, p2, n_steps=10):
	# interpolate ratios between the points
	ratios = np.linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = []
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	return np.asarray(vectors)

def visualize_interpolation(model,data):
    pass
    #encoded_img= []
    #for i, (input,_,_) in enumerate(data):
    #    encoded_img.append(model.encoder.predict(input))

    #interpolated_imgs=interpolate_points(encoded_img[0][0].flatten(),encoded_img[1][0].flatten())

    #fig, axs = plt.subplots(1, 2)
    #for i in range(2):
    #    decoded_img = model.decoder(interpolated_imgs[i].reshape(64,10))
    #    axs[i].imshow(decoded_img[0], cmap="Greys_r")
    #plt.show()
