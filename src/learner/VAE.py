import tensorflow as tf
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, encoder, decoder, name="autoencoder", learning_rate=0.005, cross_entropy_loss=False,
                 kl_loss_weight=1.0, from_logits=True, **kwargs):
        assert isinstance(encoder, tf.keras.Model)
        assert isinstance(decoder, tf.keras.Model)
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.sampling = Sampling()
        self.decoder = decoder
        self.klLossWt = kl_loss_weight
        if cross_entropy_loss:
            self.reconstructLoss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
        else:
            self.reconstructLoss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.totalLoss = 0.0

    @property
    def metrics(self):
        return self.totalLoss

    def set_kl_loss_weight(self, weight):
        self.klLossWt = weight

    def loss(self, inputs, outputs, z_mean, z_log_var):
        # reconstruction loss + KL divergence (regularization loss)
        return self.decoderLoss(inputs, outputs) + self.encoderLoss(z_mean, z_log_var) * self.klLossWt

    def encoderLoss(self, z_mean, z_log_var):
        # KL divergence (regularization loss)
        return -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

    def decoderLoss(self, inputs, outputs):
        # reconstruction loss
        return tf.reduce_mean(self.reconstructLoss(inputs, outputs))

    def call(self, inputs, **kwargs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstructed = self.decoder(z)
        return z_mean, z_log_var, reconstructed

    def predict(self, inputs, **kwargs):
        return self.call(inputs, **kwargs)

    def encode(self, inputs, **kwargs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling.call((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def decode(self, z):
        return self.decoder(z)

    def reconstruct(self, inputs, **kwargs):
        return self.call(inputs, **kwargs)[2].numpy()

    def generate(self, z, **kwargs):
        reconstructed = self.decoder(z)
        return reconstructed.numpy()

    def fit(self, x=None, epochs=1, **kwargs):
        self.totalLoss = 0.
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, y = self.call(x)
                loss = self.loss(x, y, z_mean, z_log_var)

            grads = tape.gradient(loss, self.trainable_weights)
            self.totalLoss += loss
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self.totalLoss / epochs
