import tensorflow as tf
input = tf.random.normal((1, 5), dtype=tf.float32)
(input < 0).numpy().sum()
layer = tf.keras.layers.Dense(10, activation="relu", input_shape=(5,))
output = layer(input)
assert (output < 0).numpy().sum() == 0

layer2 = tf.keras.layers.Dense(10, activation=tf.keras.activations.relu, input_shape=(5,))
output2 = layer2(input)
assert (output2 < 0).numpy().sum() == 0