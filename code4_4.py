import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(10, input_shape=(None, 5), return_sequences=True))
model.add(tf.keras.layers.Dense(6))
print(model.summary())

input = tf.constant(tf .ones((2, 4, 5)))
output = model(input)
print(output.shape)