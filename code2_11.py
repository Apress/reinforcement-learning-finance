import tensorflow as tf
from code2_10 import SimpleModel

path = r"C:\temp\simplemodel"
model = SimpleModel()
checkpoint = tf.train.Checkpoint(model)
checkpoint.write(path)


model2 = SimpleModel()
model_orig = tf.train.Checkpoint(model2)
model_orig.restore(path)