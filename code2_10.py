import tensorflow as tf


class InferInputSizeModule(tf.Module):
    def __init__(self, noutput, name=None):
        super().__init__(name=name)
        self.weights = None
        self.noutput = noutput
        self.bias = tf.Variable(tf.zeros([noutput]), name="bias")

    def __call__(self, x, *args, **kwargs):
        if self.weights is None:
            self.weights = tf.Variable(tf.random.normal([x.shape[-1], self.noutput]))

        output = tf.matmul(x, self.weights) + self.bias
        return tf.nn.sigmoid(output)


class SimpleModel(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.layer1 = InferInputSizeModule(noutput=4)
        self.layer2 = InferInputSizeModule(noutput=1)

    def __call__(self, x, *args, **kwargs):
        x = self.layer1(x)
        return self.layer2(x)

model = SimpleModel()
print(model(tf.ones((1, 10))))
