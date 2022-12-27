import tensorflow as tf

tensor = tf.constant ([[ list (range(3)) ],
                       [ list (range(1, 4)) ],
                       [ list (range(2, 5)) ]], dtype=tf.float32)
print(tensor)