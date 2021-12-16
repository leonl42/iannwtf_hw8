import tensorflow as tf
from math import ceil, floor, sqrt


class Encoder(tf.keras.Model):

    def __init__(self, should_output_features):
        super(Encoder, self).__init__()
        self._l1 = tf.keras.layers.Conv2D(
            filters=20, kernel_size=4, strides=2, padding="same", activation="relu")
        self._l2 = tf.keras.layers.Conv2D(
            filters=30, kernel_size=4, strides=2, padding="same", activation="relu")
        self._l3 = tf.keras.layers.Conv2D(
            filters=40, kernel_size=2, strides=1, padding="same", activation="relu")

        self._l4 = tf.keras.layers.GlobalAveragePooling2D()
        self._l5 = tf.keras.layers.Dense(
            should_output_features, activation="softmax")

    def get_layers(self):
        return [self._l1, self._l2, self._l3, self._l4, self._l5]

    def call(self, x, training):
        x = self._l1(x, training=training)
        x = self._l2(x, training=training)
        x = self._l3(x, training=training)
        x = self._l4(x, training=training)
        x = self._l5(x, training=training)

        return x


class Decoder(tf.keras.Model):

    def __init__(self, should_input_features):
        super(Decoder, self).__init__()
        self._should_input_features = should_input_features
        self._l1 = tf.keras.layers.Dense(
            should_input_features, activation="relu")
        self._l2 = tf.keras.layers.Conv2DTranspose(
            filters=10, kernel_size=4, strides=2, padding="same", activation="relu")
        self._l3 = tf.keras.layers.Conv2DTranspose(
            filters=15, kernel_size=4, strides=2, padding="same", activation="relu")
        self._l4 = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=2, strides=1, padding="same", activation="relu")
        # self._l5 = tf.keras.layers.Conv2DTranspose(
        #    filters=1, kernel_size=1, strides=1, padding="same", activation="sigmoid")

    def call(self, x, training):
        x = self._l1(x, training=training)
        x = tf.reshape(x, (64, int(sqrt(self._should_input_features)),
                       int(sqrt(self._should_input_features)), 1))
        x = self._l2(x, training=training)
        x = self._l3(x, training=training)
        x = self._l4(x, training=training)
        #x = self._l5(x, training=training)

        return x


class Autoencoder(tf.keras.Model):

    def __init__(self, input_dimensions):
        super(Autoencoder, self).__init__()
        self._encoder = Encoder(20)

        # get values of encoder
        kernels = []
        paddings = []
        strides = []
        for layer in self._encoder.get_layers():
            if(isinstance(layer, tf.keras.layers.Conv2D)):
                kernels.append(list(layer.kernel_size))
                paddings.append(layer.padding)
                strides.append(list(layer.strides))
            elif(isinstance(layer, tf.keras.layers.AveragePooling2D) or isinstance(layer, tf.keras.layers.MaxPool2D)):
                kernels.append(list(layer.pool_size))
                paddings.append(layer.padding)
                if layer.strides is None:
                    strides.append(list(layer.pool_size))
                else:
                    strides.append(list(layer.strides))

        sizes = [input_dimensions]
        for kernel, padding, stride in zip(kernels, paddings, strides):

            # compute the sizes for each layer
            if padding == "valid":
                new_size = []
                for dim_kernel, dim_stride, dim_size in zip(kernel, stride, sizes[-1]):
                    new_size.append(ceil((dim_size-dim_kernel+1)/dim_stride))
                sizes.append(new_size)
            elif padding == "same":
                new_size = []
                for dim_stride, dim_size in zip(stride, sizes[-1]):
                    new_size.append(floor(dim_size/dim_stride))
                sizes.append(new_size)

        # calculate the flattened size from the output layer
        flatten_size = 1
        for dim in sizes[-1]:
            flatten_size *= dim
        self._decoder = Decoder(flatten_size)

    def call(self, x, training):
        x = self._encoder(x, training=training)
        x = self._decoder(x, training=training)

        return x
