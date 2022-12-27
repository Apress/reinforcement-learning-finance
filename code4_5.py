import tensorflow as tf


class CustomRNN(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.nunit = units
        self.state_size = units
        self.prev2Output = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.xWt = self.add_weight(shape=(input_shape[-1], self.nunit),
                                   initializer=tf.keras.initializers.RandomNormal(),
                                   name="xWt")
        self.h1Wt = self.add_weight(shape=(self.nunit, self.nunit),
                                    initializer=tf.keras.initializers.RandomNormal(),
                                    name="h1")
        self.h2Wt = self.add_weight(shape=(self.nunit, self.nunit),
                                    initializer=tf.keras.initializers.RandomNormal(),
                                    name="h2")
        self.built = True

    def call(self, inputs, states):
        prevOutput = states[0]
        output = tf.matmul(inputs, self.xWt) + tf.matmul(prevOutput, self.h1Wt)
        if self.prev2Output is not None:
            output += tf.matmul(self.prev2Output, self.h2Wt)
        self.prev2Output = prevOutput
        return output, [output]

cell = CustomRNN(5)
layer = tf.keras.layers.RNN(cell)
input = tf.ones((2, 6, 5))
y = layer(input)
print(y)
print(y.shape)