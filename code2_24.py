import tensorflow as tf


class L1L4Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l1=0.01, l4=0.01):
        self.l1 = l1
        self.l4 = l4

    def __call__(self, weights):
        sq = tf.math.square(weights)
        return self.l1 * tf.math.reduce_sum(tf.math.abs(weights)) + \
               self.l4 * tf.math.reduce_sum(tf.math.square(sq))

    def get_config(self):
        return {"l1": self.l1, "l4": self.l4}

layer = tf.keras.layers.Dense(2, input_shape=(5,), kernel_regularizer=L1L4Regularizer(),
                              kernel_initializer="ones")
input = tf.ones(shape=(1, 5))
output = layer(input)
print(layer.losses)
