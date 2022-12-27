import numpy as np
import tensorflow as tf
import pandas as pd


def create_dataset():
    # from list and numpy array
    lst = [4] * 4
    ar = np.ones(1, dtype=np.int32) * 4
    dset1 = tf.data.Dataset.from_tensor_slices(lst)
    dset2 = tf.data.Dataset.from_tensor_slices(ar)
    for e1, e2 in zip(dset1, dset2):
        assert e1 == e2

    # from csv file
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["r1", "r2", "r3"]})
    filename = r"C:\prog\cygwin\home\samit_000\RLPy\data\book\test.csv"
    df.to_csv(filename, index=False)
    dataset = tf.data.TextLineDataset([filename])
    for row in dataset:
        print(row)


if __name__ == "__main__":
    create_dataset()