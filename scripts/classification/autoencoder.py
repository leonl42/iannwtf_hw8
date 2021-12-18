import tensorflow as tf

class EncoderConv(tf.keras.Model):

    def __init__(self, latent_dim):
        super(EncoderConv, self).__init__()
        self._conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation="relu")
        self._conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2, padding="same", activation="relu")
        self._flatten = tf.keras.layers.Flatten()
        self._embedding = tf.keras.layers.Dense(latent_dim, activation="relu")

    def call(self, x, training):
        x = self._conv1(x, training=training)
        x = self._conv2(x, training=training)
        x = self._flatten(x, training=training)
        x = self._embedding(x, training=training)
        return x


class DecoderConv(tf.keras.Model):

    def __init__(self, should_input_features):
        super(DecoderConv, self).__init__()

        self._dense = tf.keras.layers.Dense(should_input_features, activation="relu")
        self._convt1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding="same", activation="relu")
        self._convt2 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same", activation="relu")
        self._out = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=2, strides=1, padding="same", activation="sigmoid")

    def call(self, x, training):
        x = self._dense(x, training=training)
        x = tf.reshape(x, (64,7,7,8))
        x = self._convt1(x, training=training)
        x = self._convt2(x, training=training)
        x = self._out(x, training=training)

        return x

class AutoencoderConv(tf.keras.Model):

    def __init__(self):
        super(AutoencoderConv, self).__init__()
        self.encoder = EncoderConv(latent_dim=10)
        self.decoder = DecoderConv(7*7*8)

    def call(self, x, training):
        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        return x
