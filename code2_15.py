import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# generate data
x = np.linspace(0, 5, 400, dtype=np.float32)  # 400 points spaced from 0 to 5
x = tf.constant(x)
y = 4 * x + 2.5 + tf.random.truncated_normal((400,), dtype=tf.float32)
sns.scatterplot(x.numpy(), y.numpy())
plt.ylabel("y = 4x + 2.5 + noise")
plt.xlabel("x")
plt.show()

# create test and training data
x_train, y_train = x[0:350], y[0:350]
x_test, y_test = x[350:], y[350:]

# create the model
seq_model = tf.keras.Sequential()
seq_model.add(tf.keras.layers.Dense(5, input_shape=(1,)))
seq_model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.relu))
seq_model.add(tf.keras.layers.Dense(1))
print(seq_model.summary())


# Custom loss function with optional regularlization
class Loss(tf.keras.losses.Loss):
    def __init__(self, beta, weights):
        super().__init__()
        self.weights = weights
        self.beta = beta

    def call(self, y_true, y_pred):
        reg_loss = 0
        for i in range(len(self.weights)):
            reg_loss += tf.reduce_mean(tf.square(self.weights[i]))
        return tf.reduce_mean(tf.square(y_pred - y_true)) + self.beta * reg_loss


my_loss = Loss(0, seq_model.get_weights())

# compile the model
seq_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=my_loss,
                  metrics=[tf.keras.metrics.MeanSquaredError()])

# fit the model to training data
history = seq_model.fit(x_train, y_train, batch_size=10, epochs=10)

# plot the history
plt.plot(history.history["mean_squared_error"], label="mean_squared_error")
plt.ylabel("Mean Square Error")
plt.xlabel("Epoch")
plt.show()

# predict unseen test data
y_pred = seq_model.predict(x_test)
plt.plot(x_test, y_test, '.', label="Test Data")
plt.plot(x_test, 4 * x_test + 2.5, label="Underlying Data")
plt.plot(x_test, y_pred.squeeze(), label="Predicted Values")
plt.legend()
plt.show()