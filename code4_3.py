import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(10, input_shape=(None, 5)))
model.add(tf.keras.layers.Dense(6))
print(model.summary())