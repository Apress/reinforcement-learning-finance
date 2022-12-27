import tensorflow as tf

class MyActivation(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x):
        return tf.where(x < self.threshold, 0, x)

layer = tf.keras.layers.Dense(1, activation=MyActivation(0.5), input_shape=(2,))
input = tf.constant([[0, 0]], dtype=tf.float32)
output = layer(input)
print(output)
