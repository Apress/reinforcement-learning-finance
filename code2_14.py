import tensorflow as tf
from tensorflow.keras import Model


class CustomSequentialModel(Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        # layers
        self.layer2 = tf.keras.layers.Softmax()
        self.layer1 = tf.keras.layers.Dense(10, activation=tf.keras.activations.relu)

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        return self.layer2(x)


output = model(tf.random.normal((2, 10), dtype=tf.float32))
print(output)
print(tf.reduce_sum(output, axis=1))

