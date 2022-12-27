import tensorflow as tf

tensor = tf.sparse.SparseTensor(indices =[[1,0], [2,2]], values=[1, 2],
                                dense_shape=[3, 4])
print(tensor)

print(tf.sparse.to_dense(tensor))