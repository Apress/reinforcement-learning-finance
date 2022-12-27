import tensorflow as tf

metric = tf.keras.metrics.KLDivergence()
metric.update_state([[1, 0], [0, 1]], [[0.3, 0.7], [0.5, 0.5]])
print(metric.result().numpy())