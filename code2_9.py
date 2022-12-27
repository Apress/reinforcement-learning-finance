import tensorflow as tf
import numpy as np


class ExampleModule(tf.Module):
    def __init__(self, name=None):
        super(ExampleModule, self).__init__(name=name)
        self.weights = tf.Variable(np.random.random(5), name="weights")
        self.const = tf.Variable(np.array([1.0]), dtype=tf.float64,
                                 trainable=False, name="constant")

    def __call__(self, x, *args, **kwargs):
        return tf.matmul(x, self.weights[:, tf.newaxis]) + self.const[tf.newaxis, :]


em = ExampleModule()
x = tf.constant(np.ones((1, 5)), dtype=tf.float64)
print(em(x))
