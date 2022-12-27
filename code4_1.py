import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()
lyr = tf.keras.layers.SimpleRNN(1, return_sequences=True)
model.add(lyr)
input_shape = (None, 4, 4)
model.build(input_shape)
print(model.summary())
model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["mse"])