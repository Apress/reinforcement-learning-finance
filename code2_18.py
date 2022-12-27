import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class ClassifyCluster(object):
    def __init__(self):
        self.meansX = [-5, -2.5, 7, 12]
        self.meansY = [5, -3, -4, 6]
        self.stddevs = [1, 1.5, 2.7, 2]
        self.nCluster = 4
        self.nTraining = 4000
        self.nTesting = 80
        self.nFeature = 5
        self.nnet = self.buildModel()

    def trainModel(self):
        # generate training data: 4 clusters with 1000 points each
        pts = self.nTraining // self.nCluster
        randvals = np.random.standard_normal((pts, 2, self.nCluster)).astype(np.float32)
        x = []
        y = []

        for i in range(4):
            x.append(self.meansX[i] + self.stddevs[i] * randvals[:, 0, i])
            y.append(self.meansY[i] + self.stddevs[i] * randvals[:, 1, i])

        labels = np.repeat(np.arange(self.nCluster, dtype=np.int32), randvals.shape[0])
        points_order = np.array(range(len(labels)), dtype=np.int32)
        np.random.shuffle(points_order)

        x_col = np.concatenate(x)
        y_col = np.concatenate(y)

        sns.scatterplot(x=x_col, y=y_col, hue=labels)
        plt.show()

        xy_col = np.multiply(x_col, y_col)
        x2_col = np.multiply(x_col, x_col)
        y2_col = np.multiply(y_col, y_col)

        xy_data = np.concatenate((x_col[:, np.newaxis], y_col[:, np.newaxis], xy_col[:, np.newaxis],
                                  x2_col[:, np.newaxis], y2_col[:, np.newaxis]), axis=1)
        xy_data_tf = tf.constant(xy_data[points_order, :])
        labels_tf = tf.constant(labels[points_order, np.newaxis])
        history = self.nnet.fit(xy_data_tf, labels_tf, batch_size=20, epochs=15)
        plt.plot(history.history["loss"])
        plt.xticks(range(len(history.history["loss"])))
        plt.xlabel("Epochs")
        plt.ylabel("Categorical Crossentropy Loss")
        plt.grid()
        plt.show()

        # find the accuracy on test data
        result = self.nnet.predict(xy_data)
        predicted_class = np.argmax(result, axis=1)
        accuracy = (predicted_class == labels).sum() / float(labels.shape[0])
        print(f"Model accuracy on training data = {accuracy}")

    def buildModel(self):
        # build the neural network model and train
        nnet = tf.keras.models.Sequential()
        nnet.add(tf.keras.layers.Dense(5, input_shape=(self.nFeature,)))
        nnet.add(tf.keras.layers.Dense(15))
        nnet.add(tf.keras.layers.Dense(4))
        nnet.add(tf.keras.layers.Dense(4, activation="sigmoid"))
        nnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy())
        return nnet

    def testModel(self):
        # generate 80 points of testing data
        randvals = np.random.standard_normal((self.nTesting, 2)).astype(np.float32)
        test_labels = np.random.choice(self.nCluster, self.nTesting)
        xy_test = np.ndarray((self.nTesting, self.nFeature), dtype=np.float32)
        for i, label in enumerate(test_labels):
            xy_test[i, 0] = self.meansX[label] + self.stddevs[label] * randvals[i, 0]
            xy_test[i, 1] = self.meansY[label] + self.stddevs[label] * randvals[i, 1]
            xy_test[i, 2] = xy_test[i, 0] * xy_test[i, 1]
            xy_test[i, 3] = xy_test[i, 0] * xy_test[i, 0]
            xy_test[i, 4] = xy_test[i, 1] * xy_test[i, 1]

        result = self.nnet.predict(xy_test)
        predicted_class = np.argmax(result, axis=1)
        accuracy = (predicted_class == test_labels).sum() / float(test_labels.shape[0])
        print(f"Model accuracy on testing data = {accuracy}")


if __name__ == '__main__':
    classify = ClassifyCluster()
    classify.trainModel()
    classify.testModel()

