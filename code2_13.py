import tensorflow as tf
from tensorflow.keras.layers import Layer


class LassoLossLayer(Layer):
    def __init__(self, features, neurons):
        super().__init__()
        self.wt = tf.Variable(tf.random.normal((features, neurons), dtype=tf.float32),
                              trainable=True)
        self.bias = tf.Variable(tf.zeros((neurons,), dtype=tf.float32),
                                trainable=True)
        self.meanMetric = tf.keras.metrics.Mean()

    def call(self, inputs):
        # LASSO regularization loss
        self.add_loss(tf.reduce_sum(tf.abs(self.wt)))
        self.add_loss(tf.reduce_sum(tf.abs(self.biias)))
        # metric to calculate mean of inputs
        self.add_metric(self.meanMetric(inputs))
        return tf.matmul(inputs, self.wt) + self.bias

