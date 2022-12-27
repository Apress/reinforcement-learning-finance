import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import os
import seaborn as sns

logging.basicConfig(level=logging.DEBUG)


class FashionMNistCNNClassify(object):
    LOGGER = logging.getLogger("FashionMNistCNNClassify")

    def __init__(self, datadir: str, batchsize: int = 10, epochs: int = 20) -> None:
        (trainx, trainy), (testx, testy) = tf.keras.datasets.fashion_mnist.load_data()
        self.classes = ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
                        "Bag", "Boot"]
        self.nClass = len(self.classes)
        trainx = trainx/255.0
        testx = testx/255.0
        self.trainingData = (trainx, trainy)
        self.testingData = (testx, testy)
        self.inputDir = datadir
        self.batchSize = batchsize
        self.nEpoch = epochs
        self.nnet = self.model()

    def checkpointModel(self, nnet):
        checkpointFile = os.path.join(self.inputDir, "checkpoint_fmnist_cnn_wt")
        if not os.path.exists(checkpointFile):
            nnet.predict(np.ones((20, 28, 28, 1), dtype=np.float32))
            tf.keras.models.save_model(nnet, checkpointFile, overwrite=False)
        else:
            nnet = tf.keras.models.load_model(checkpointFile)
        return nnet

    def model(self):
        nnet = tf.keras.models.Sequential()
        nnet.add(tf.keras.layers.Conv2D(filters=100, kernel_size=(2, 2), padding="same", input_shape=(28, 28, 1)))
        nnet.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        nnet.add(tf.keras.layers.Conv2D(filters=60, kernel_size=(2, 2), padding="same", activation="relu"))
        nnet.add(tf.keras.layers.Flatten())
        nnet.add(tf.keras.layers.Dense(50, activation="relu"))
        nnet.add(tf.keras.layers.Dense(10))
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()
        nnet.compile(optimizer=self.optimizer,
                     loss=self.loss,
                     metrics=[self.metric])
        nnet = self.checkpointModel(nnet)
        return nnet

    def plotConfusionMatrix(self, labels: np.ndarray, predictions: np.ndarray) -> None:
        predictedLabels = np.argmax(predictions, axis=1)
        fig, ax = plt.subplots()
        cm = np.zeros((self.nClass, self.nClass), dtype=np.int32)
        for i in range(labels.shape[0]):
            cm[labels[i], predictedLabels[i]] += 1
        sns.heatmap(cm, annot=True, fmt="d", linewidths=0.25, ax=ax)
        ax.set_xticks(range(1+self.nClass))
        ax.set_yticks(range(1+self.nClass))
        ax.set_xticklabels(["0"] + self.classes, rotation=20)
        ax.set_yticklabels(["0"] + self.classes, rotation=20)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.show()

    def plotConvergenceHistory(self, history, metricName):
        plt.plot(history.epoch, history.history[metricName])
        plt.xlabel("Epoch")
        plt.ylabel(metricName)
        plt.grid(True)
        plt.legend()
        plt.show()

    def testModel(self):
        for X, y in [self.trainingData, self.testingData]:
            predictClass = self.nnet.predict(X[..., np.newaxis])
            self.plotConfusionMatrix(y, predictClass)

    def trainModel(self):
        history = self.nnet.fit(self.trainingData[0][..., np.newaxis], self.trainingData[1],
                                    batch_size=self.batchSize, epochs=self.nEpoch)
        self.plotConvergenceHistory(history, self.metric._name)
        self.plotConvergenceHistory(history, "loss")
        return history


if __name__ == "__main__":
    dname = r"C:\prog\cygwin\home\samit_000\RLPy\data\book"
    fmnist = FashionMNistCNNClassify(dname, batchsize=100, epochs=40)
    fmnist.trainModel()
    fmnist.testModel()