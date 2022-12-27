import tensorflow as tf
from tensorflow.keras.layers import Layer


class CustomDenseLayer(Layer):
    def __init__(self, neurons):
        super().__init__()
        self.neurons = neurons

    def build(self, input_shape):
        # input_shape[-1] is the number of features for this layer
        self.wt = tf.Variable(tf.random.normal((input_shape[-1], self.neurons), dtype=tf.float32),
                                   trainable=True)
        self.bias = tf.Variable(tf.zeros((self.neurons,), dtype=tf.float32),
                                trainable=True)
        self.upperBound = tf.constant(0.9, dtype=tf.float32, shape=(input_shape[-1],))

    def call(self, inputs):
        return tf.matmul(tf.minimum(self.upperBound, inputs), self.wt) + self.bias


layer = CustomDenseLayer(5)
print(layer.weights)
print(layer.trainable_weights)

input = tf.random_normal_initializer(mean=0.5)(shape=(2, 5), dtype=tf.float32)
layer(inputs=input)
print(layer.weights)
print(layer.trainable_weights)
