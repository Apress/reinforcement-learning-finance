import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from typing import List

logging.basicConfig(level=logging.DEBUG)


class AutoMPG(object):
    "Neural network model for regression. Predict automobile MPG"
    LOGGER = logging.getLogger("AutoMPG")

    def __init__(self, datadir: str, filename: str = "auto-mpg.data", trainingData: float = 0.8,
                 batchSize: int = 10, epochs: int = 20):
        self.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year",
                        "origin", "car name"]
        df = pd.read_csv(os.path.join(datadir, filename), header=None, names=self.columns, sep="\s+")
        self.inputDir = datadir
        self.batchSize = batchSize
        self.nEpoch = epochs
        df = self._preprocessData(df)
        self.resultCol = "mpg"
        self.categoricalCols = ["cylinders", "model year", "origin", "car name"]
        exclude = set(self.categoricalCols)
        self.numericCols = [c for c in df.columns if c not in exclude]
        self.featureCols = self.numericCols + self.categoricalCols
        self.featureCols.remove(self.resultCol)
        self.normalizeCols = {}
        self.categoricalMap = {}
        ntrain = int(trainingData * df.shape[0])
        trainIndex = np.random.choice(df.shape[0], ntrain, replace=False)
        testIndex = np.array([i for i in range(df.shape[0]) if i not in set(trainIndex)])
        self.trainDf = df.loc[trainIndex, :].reset_index(drop=True)
        self.testDf = df.loc[testIndex, :].reset_index(drop=True)
        self._normalizeNumericCols(self.trainDf)
        self.trainDf = self._applyNormalization(self.trainDf)
        self.testDf = self._applyNormalization(self.testDf)
        corr1 = self._correlWithOutput(self.trainDf)
        cov = self._correlWithinInputs(self.trainDf)
        self._processCategoricalCols(df)
        self.trainDf = self._applyCategoricalMapping(self.trainDf)
        self.testDf = self._applyCategoricalMapping(self.testDf)
        self.metrics = [tf.keras.metrics.MeanAbsoluteError()]
        self.featureCols.remove("car name")
        self.categoricalCols.remove("car name")

    def _processCategoricalCols(self, df: pd.DataFrame) -> None:
        """
        Process categorical columns by creating a mapping
        :param df: training dataframe
        :rtype: None
        """
        for col in self.categoricalCols:
            unique = np.sort(df.loc[:, col].unique())
            self.categoricalMap[col] = {u:i for i,u in enumerate(unique)}

    def _correlWithOutput(self, df: pd.DataFrame) -> np.ndarray:
        output = df.loc[:, self.resultCol].values
        mu = output.mean()
        sd = output.std()
        x = (output - mu)/sd
        ncols = copy.copy(self.numericCols)
        ncols.remove(self.resultCol)
        correl = np.zeros(len(ncols), dtype=np.float32)
        for i, col in enumerate(ncols):
            x1 = df.loc[:, col].values
            correl[i] = np.sum(2 * x * x1)/df.shape[0]
        plotdf = pd.DataFrame({"Feature": ncols, "Correlation": correl})
        sns.barplot(x="Feature", y="Correlation", data=plotdf)
        plt.show()

        mean, sd = self.normalizeCols[self.resultCol]
        mpg = df.loc[:, self.resultCol].values * sd + mean
        for col in self.categoricalCols:
            sns.histplot(data=df, x=mpg, hue=col)
            plt.show()
        return correl

    def _correlWithinInputs(self, df: pd.DataFrame) -> np.ndarray:
        ncols = copy.copy(self.numericCols)
        ncols.remove(self.resultCol)
        cov = np.cov(df.loc[:, ncols].values.T)
        mask = np.triu(np.ones_like(cov, dtype=bool))
        fig, ax = plt.subplots()
        sns.heatmap(cov, mask=mask, annot=True, linewidths=0.25, ax=ax)
        ax.set_xticks(0.5 + np.arange(cov.shape[0]+1))
        ax.set_yticks(0.5 + np.arange(cov.shape[0]+1))
        ax.set_xticklabels(ncols + ["_"], rotation=20)
        ax.set_yticklabels(ncols + ["_"], rotation=20)
        plt.show()
        return cov

    def _preprocessData(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "horsepower"] = df.loc[:, "horsepower"].replace("?", 0).astype(np.float32)
        func = lambda x: x.lower().split(" ", 3)[0]
        df.loc[:, "car name"] = df.loc[:, "car name"].map(func)
        return df

    def _normalizeNumericCols(self, trainingDf: pd.DataFrame) -> None:
        """
        Calclate normalizing params for numeric columns
        :param trainingDf:
        :return: None
        """
        for col in self.numericCols:
            mean = trainingDf.loc[:, col].mean()
            sd = trainingDf.loc[:, col].std()
            self.normalizeCols[col] = (mean, 2*sd)

    def _applyNormalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization as col = (x-mean)/(2*sd)
        :param df:
        :return: df
        """
        for col in self.numericCols:
            mean, sd2 = self.normalizeCols[col]
            df.loc[:, col] = (df.loc[:, col].values - mean) / sd2
        return df

    def _applyCategoricalMapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mapping to convert categorical columns to integers
        :rtype: pd.DataFrame with mapped categorical columns
        """
        for col in self.categoricalCols:
            df.loc[:, col] = df.loc[:, col].map(self.categoricalMap[col])
        return df

    def testRegularizers(self, regularizers: List[tf.keras.regularizers.Regularizer], names: List[str]) -> None:
        histDict = {}
        for regularizer, name in zip(regularizers, names):
            self.nnet = self.model(regularizer=regularizer)
            history = self.trainModel()
            histDict[name] = history
            self.testModel(name)

        for metric in self.metrics:
            self.plotConvergenceHistory(histDict, metric._name)
        self.plotConvergenceHistory(histDict, "loss")

    def model(self, regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
        nfeature = len(self.featureCols)
        nnet = tf.keras.models.Sequential()
        nnet.add(tf.keras.layers.Dense(12, activation="sigmoid", input_shape=(nfeature,), kernel_regularizer=regularizer))
        nnet.add(tf.keras.layers.Dense(3, activation="relu", kernel_regularizer=regularizer))
        nnet.add(tf.keras.layers.Dense(1, kernel_regularizer=regularizer))
        nnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                     loss=tf.keras.losses.MeanSquaredError(),
                     metrics=self.metrics)
        nnet = self.checkpointModel(nnet)
        return nnet

    def checkpointModel(self, nnet):
        checkpointFile = os.path.join(self.inputDir, "checkpoint_autompg_wt")
        if not os.path.exists(checkpointFile):
            nfeature = len(self.featureCols)
            nnet.predict(np.ones((20, nfeature), dtype=np.float32))
            tf.keras.models.save_model(nnet, checkpointFile, overwrite=False)
        else:
            nnet = tf.keras.models.load_model(checkpointFile)
        return nnet

    def trainModel(self, trainDf: pd.DataFrame = None) -> tf.keras.callbacks.History:
        if trainDf is None:
            trainDf = self.trainDf
        X = trainDf.loc[:, self.featureCols].values
        y = trainDf.loc[:, self.resultCol].values
        history = self.nnet.fit(X, y, batch_size=self.batchSize, epochs=self.nEpoch)
        return history

    def testModel(self, title: str) -> None:
        loss = tf.keras.losses.MeanSquaredError()
        for df in [self.trainDf, self.testDf]:
            features = df.loc[:, self.featureCols].values
            actVals = df.loc[:, self.resultCol].values
            predictVals = self.nnet.predict(features)
            lossval = loss(actVals[:, np.newaxis], predictVals)
            self.LOGGER.info("Loss for regularizer %s, number of data points %d: %f", title, df.shape[0], lossval)
            self.plotActualVsPredicted(actVals, predictVals.squeeze(), title=title)

    def plotActualVsPredicted(self, actualVals: np.ndarray, predictedVals: np.ndarray, title: str = None) -> None:
        mean, sd = self.normalizeCols[self.resultCol]
        y = actualVals * sd + mean
        x = predictedVals * sd + mean
        plt.scatter(x, y, c="red")
        p1 = max(max(x), max(y))
        p2 = min(min(x), min(y))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        if title:
            plt.title(title)
        plt.show()

    def plotConvergenceHistory(self, histDict: dict, metricName: str) -> None:
        for name, history in histDict.items():
            plt.plot(history.epoch, history.history[metricName], label=name)
        plt.xlabel("Epoch")
        plt.ylabel(metricName)
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    mpg = AutoMPG(r"C:\prog\cygwin\home\samit_000\RLPy\data\book", batchSize=1)
    regularizers = [None, tf.keras.regularizers.L1L2(l1=0.1, l2=0), tf.keras.regularizers.L1L2(l1=0, l2=0.1)]
    names = ["None", "L1", "L2"]
    mpg.testRegularizers(regularizers, names)