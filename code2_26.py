import tensorflow as tf
import numpy as np
import datetime
import shutil

# load tensorboard extension
%load_ext tensorboard

# specify base logs dir
base_log_dir = "logs\\fit\\"

# clear previous logs
try:
	shutil.rmtree(base_log_dir)
except OSError as e:
	pass

# create some data
nfeature = 10
nsample = 100
nsampletest = 20
X = 1 + np.random.random((nsample, nfeature))
y = 2*X.sum(axis=1) + 4

Xtest = 1 + np.random.random((nsampletest, nfeature))
ytest = 2*Xtest.sum(axis=1) + 4

nnet = tf.keras.models.Sequential()
nnet.add(tf.keras.layers.Dense(4, input_shape=(nfeature,)))
nnet.add(tf.keras.layers.Dense(10, activation="relu"))
nnet.add(tf.keras.layers.Dropout(0.2))
nnet.add(tf.keras.layers.Dense(1))

nnet.compile(optimizer="adam", loss="MSE", metrics=[tf.keras.metrics.MeanAbsoluteError()])

# specify log directory
log_dir = base_log_dir + datetime.datetime.now().strftime("run%Y%m%d_%H%M%S")

# create TensorBoard callback
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# provide the callback to fit method
nnet.fit(X, y, epochs=10, callbacks=[tb_callback])