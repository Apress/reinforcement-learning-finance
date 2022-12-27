import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import os
import seaborn as sns

logging.basicConfig(level=logging.DEBUG)


class History(object):
    def __init__(self):
        self.history = {}
        self.epoch = None


class FashionMNistClassify(object):
    LOGGER = logging.getLogger("FashionMNistClassify")

    def __init__(self, datadir: str, batchsize: int = 10, epochs: int = 20, useGradTape: bool = True) -> None:
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
        self.useGradientTape = useGradTape
        self.nnet = self.model()

    def checkpointModel(self, nnet):
        checkpointFile = os.path.join(self.inputDir, "checkpoint_fmnist_wt")
        if not os.path.exists(checkpointFile):
            nnet.predict(np.ones((20, 28, 28), dtype=np.float32))
            tf.keras.models.save_model(nnet, checkpointFile, overwrite=False)
        else:
            nnet = tf.keras.models.load_model(checkpointFile)
        return nnet

    def model(self):
        nnet = tf.keras.models.Sequential()
        nnet.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        nnet.add(tf.keras.layers.Dense(80, activation="relu"))
        nnet.add(tf.keras.layers.Dense(20, activation="relu"))
        nnet.add(tf.keras.layers.Dense(10))
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
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
            predictClass = self.nnet.predict(X)
            self.plotConfusionMatrix(y, predictClass)

    def gradTapeTraining(self):
        trainDataset = tf.data.Dataset.from_tensor_slices(self.trainingData)
        trainDataset = trainDataset.batch(self.batchSize)
        totalLoss = np.zeros(self.nEpoch, dtype=np.float32)
        count = 0
        for X, y in trainDataset:
            for epoch in range(self.nEpoch):
                with tf.GradientTape() as tape:
                    predictedY = self.nnet(X)
                    loss = self.loss(y, predictedY)

                grads = tape.gradient(loss, self.nnet.trainable_weights)
                self.LOGGER.info("Epoch %d, loss %f", epoch, loss)
                totalLoss[epoch] += loss
                self.optimizer.apply_gradients(zip(grads, self.nnet.trainable_weights))
            count += 1
        totalLoss = totalLoss / count
        history = History()
        history.history["loss"] = totalLoss
        history.history[self.metric._name] = np.zeros(self.nEpoch)
        history.epoch = np.arange(self.nEpoch)
        return history

    def trainModel(self):
        if self.useGradientTape:
            history = self.gradTapeTraining()
        else:
            history = self.nnet.fit(self.trainingData[0], self.trainingData[1],
                                    batch_size=self.batchSize, epochs=self.nEpoch)
        self.plotConvergenceHistory(history, self.metric._name)
        self.plotConvergenceHistory(history, "loss")
        return history


if __name__ == "__main__":
    dname = r"C:\prog\cygwin\home\samit_000\RLPy\data\book"
    fmnist = FashionMNistClassify(dname, batchsize=10000, epochs=60)
    fmnist.trainModel()
    fmnist.testModel()