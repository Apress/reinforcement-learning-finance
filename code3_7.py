import tensorflow as tf
convLayer = tf.keras.layers.Conv2D(10, (4, 4), strides=(2, 2), kernel_initializer="ones",
							       bias_initializer="ones", input_shape=(8,8,3))
deconvLayer = tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2,2),
  	                     kernel_initializer=tf.keras.initializers.Constant(1.0/(49*4)),
						 bias_initializer="ones")
input = tf.constant(tf.ones((1, 8, 8, 3), dtype=tf.float32))
out1 = convLayer(input)
out2 = deconvLayer(out1)
assert out2.shape == input.shape