import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):
	def call(self, y_true, y_pred):
		return tf.reduce_mean(tf.abs(y_true - y_pred))