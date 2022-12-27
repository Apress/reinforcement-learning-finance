import tensorflow as tf
X = tf.constant(tf.random.normal((5, 4)))
W = tf.Variable(tf.ones((4, 6), dtype=tf.float32)) # watched by default
b = tf.constant(tf.ones(6, dtype=tf.float32))  # not watched by default
with tf.GradientTape() as tape:
	tape.watch(b)
	y = tf.matmul(X, W) + b
vars = [W, b]
grads = tape.gradient(y, vars)
for i, grad in enumerate(grads):
	print(f"Variable shape: {vars[i].shape}, gradient shape: {grad.shape}")