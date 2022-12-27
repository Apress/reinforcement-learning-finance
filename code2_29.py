import tensorflow as tf
dset2 = tf.data.Dataset.range(10)
shard = dset2.shard(num_shards=3, index=0)
for element in shard:
    print(element)