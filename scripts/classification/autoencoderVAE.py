import tensorflow as tf

#Quelle:https://keras.io/examples/generative/vae/
#könnte weiterhelfen https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/
class EncoderVAE(tf.keras.Model):

    def __init__(self, latent_dim):
        super(EncoderVAE, self).__init__()
        self._conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation="relu")
        self._conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2, padding="same", activation="relu")
        self._flatten = tf.keras.layers.Flatten()
        self._embedding = tf.keras.layers.Dense(latent_dim, activation="relu")
        self._mean = tf.keras.layers.Dense(latent_dim,name="mean")
        self._log_var = tf.keras.layers.Dense(latent_dim,name="logvar")


    def call(self, x, training):
        x = self._conv1(x, training=training)
        x = self._conv2(x, training=training)
        x = self._flatten(x, training=training)
        x = self._embedding(x, training=training)
        mean = self._mean(x)
        log_var = self._log_var(x)
        x = self.sampling([mean,log_var])
        return mean,log_var,x

    def sampling(self,inputs):
        mean,log_var = input_dimensions
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

class DecoderVAE(tf.keras.Model):
    def __init__(self, should_input_features):
        super(DecoderVAE, self).__init__()

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


class AutoencoderVAE(tf.keras.Model):

    def __init__(self):
        super(AutoencoderVAE, self).__init__()
        self.encoder = EncoderVAE(latent_dim=10)
        self.decoder = DecoderVAE(7*7*8)

    def call(self, x, training):
        mean,log_var,x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        return mean,log_var,x
