from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=(20, 20, 1)))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(5, (4, 4), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
print(model.summary())