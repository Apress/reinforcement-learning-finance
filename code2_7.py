import tensorflow as tf
import numpy as np

tensor = tf.constant(np.ones((3, 3), dtype=np.int32))

print(tensor)

print(tf.reduce_sum(tensor))

print(tf.reduce_sum(tensor, axis=1))

@tf.function
def sigmoid_activation(inputs, weights, bias):
    x = tf.matmul(inputs, weights) + bias
    return tf.divide(1.0, 1 + tf.exp(-x))

inputs = tf.constant(np.ones((1, 3), dtype=np.float64))
weights = tf.Variable(np.random.random((3, 1)))

bias = tf.ones((1, 3), dtype=tf.float64)
print(sigmoid_activation(inputs, weights, bias))

import timeit

tf.config.experimental_run_functions_eagerly(False)
t1 = timeit.timeit(lambda: sigmoid_activation(inputs, weights, tf.constant(np.random.random((1, 3)))), number=1000)
print(t1)

