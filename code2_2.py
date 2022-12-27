import tensorflow as tf

ar = tf.constant ([[1, 2], [2, 2]], dtype=tf.float32)

print(ar)

# elementwise multiplication
print(ar * ar)

# matrix multiplication C = tf.matmult(A, B) => cij = sum k (aik * bkj)
print(tf.matmul(ar, tf.transpose(ar)))

# generic way of matrix multiplication
print(tf.einsum("ij,kj->ik", ar, ar))

# cross product
print(tf.einsum("ij,kl->ijkl", ar, ar))