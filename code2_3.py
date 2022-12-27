import tensorflow as tf

tensor = tf.constant ([[1, 2], [2, 2]], dtype=tf.float32)

print(tensor [1:, :])

print(tf.slice (tensor, begin=[0,1], size=[2, 1]))

print(tf.gather_nd(tensor, indices =[[0, 1], [1, 0]]))
