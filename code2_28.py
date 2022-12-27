import tensorflow as tf
dset = tf.data.Dataset.range(5)
batches = dset.batch(2)
for batch in batches:
    print(batch)