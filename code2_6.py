import tensorflow as tf

tensor = tf.constant ([[1, 2], [3, 4]])
variable = tf.Variable(tensor)
print( variable )

# return the index of highest element
print(tf.math.argmax(variable))

print(tf.convert_to_tensor( variable ))

print(variable.assign([[1,2], [1, 1]]))