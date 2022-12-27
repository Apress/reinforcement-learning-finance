import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.DEBUG)


class ReturnPredictor(object):
    def __init__(self, dirname, trainTestSplit=0.7, nunit=15, ntimestep=10, batchSize=10, nepoch=40):
        filename = os.path.join(dirname, "SPY.csv")
        self.inputDir = dirname
        self.logger = logging.getLogger(self.__class__.__name__)
        df = pd.read_csv(filename)
        self.nUnit = nunit
        self.nTimestep = ntimestep
        self.dateCol = "Date"
        self.priceCol = "Adj Close"
        self.volCol = "Volume"
        self.volatilityCol = "VolatRatio"
        self.volumeCol = "VolumeRatio"
        self.momentumCol = "Momentum"
        self.returnCol = "Last1MoReturn"
        self.resultCol = "Fwd1MoReturn"
        self.daysInMonth = 21
        df = self.featureEngineer(df)
        ntrain = int(trainTestSplit * df.shape[0])
        self.trainDf = df.loc[14*self.daysInMonth+1:ntrain, :].reset_index(drop=True)
        self.testDf = df.loc[ntrain:, :].reset_index(drop=True)
        self.featureCols = [self.returnCol, self.momentumCol, self.volatilityCol, self.volumeCol]
        #self.plotData(self.trainDf)
        #self.plotData(self.testDf)
        self.nnet = self.model()
        self.rnnTrainData = self.prepareDataForRNN(self.trainDf)
        self.rnnTestData = self.prepareDataForRNN(self.testDf)
        self.batchSize = batchSize
        self.nEpoch = nepoch
        self.cellType = None

    def featureEngineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, self.dateCol] = pd.to_datetime(df.loc[:, self.dateCol])
        # 1 Month lagged returns
        returns = np.zeros(df.shape[0], dtype=np.float32)
        nrow = df.shape[0]
        returns[self.daysInMonth+1:] = np.divide(df.loc[self.daysInMonth:nrow-2, self.priceCol].values,
                                                 df.loc[0:nrow-self.daysInMonth-2, self.priceCol].values) - 1
        df.loc[:, self.returnCol] = returns
        # momentum factor
        momentum = np.zeros(df.shape[0], dtype=np.float32)
        returns3Mo = np.divide(df.loc[3*self.daysInMonth:nrow-2, self.priceCol].values,
                               df.loc[0:nrow-3*self.daysInMonth-2, self.priceCol].values) - 1
        num = returns[3*self.daysInMonth+1:]
        momentum[3*self.daysInMonth+1:] = np.divide(num, np.abs(num) + np.abs(returns3Mo))
        df.loc[:, self.momentumCol] = momentum

        # volatility factor
        df.loc[:, self.volatilityCol] = 0
        volatility = np.zeros(nrow, dtype=np.float32)
        rtns = returns[self.daysInMonth+1:2*self.daysInMonth+1]
        sumval = np.sum(rtns)
        sumsq = np.sum(rtns * rtns)
        for i in range(2*self.daysInMonth+1, nrow):
            mean = sumval / self.daysInMonth
            volatility[i] = np.sqrt(sumsq / self.daysInMonth - mean*mean)
            sumval += returns[i] - returns[i-self.daysInMonth]
            sumsq += returns[i] * returns[i] - returns[i-self.daysInMonth] * returns[i-self.daysInMonth]
        oneyr = 12 * self.daysInMonth
        df.loc[:, self.volatilityCol] = 0.0
        for i in range(oneyr+2*self.daysInMonth+1, nrow):
            df.loc[i, self.volatilityCol] = volatility[i] / np.mean(volatility[i-oneyr:i])

        # volume factor
        df.loc[:, self.volumeCol] = 0
        volume = df.loc[:, self.volCol].values
        for i in range(self.daysInMonth, nrow-1):
            df.loc[i+1, self.volumeCol] = volume[i] / np.mean(volume[i-self.daysInMonth:i])

        # result column
        df.loc[:, self.resultCol] = 0.0
        df.loc[0:nrow-self.daysInMonth-1, self.resultCol] = df.loc[self.daysInMonth:, self.returnCol].values
        return df

    def prepareDataForRNN(self, df):
        nfeat = len(self.featureCols)
        data = np.zeros((df.shape[0]-self.nTimestep, self.nTimestep, nfeat), dtype=np.float32)
        results = np.zeros((df.shape[0]-self.nTimestep, self.nTimestep), dtype=np.float32)
        raw_data = df[self.featureCols].values
        raw_results = df.loc[:, self.resultCol].values
        for i in range(0, data.shape[0]):
            data[i, :, :] = raw_data[i:i+self.nTimestep, :]
            results[i, :] = raw_results[i:i+self.nTimestep]
        return data, results

    def plotData(self, df: pd.DataFrame) -> None:
        df = df.set_index(keys=[self.dateCol])
        fig, axs = plt.subplots(nrows=len(self.featureCols)+1, ncols=1, figsize=(12, 16))
        axs[0].plot(df.index.values, df.loc[:, self.priceCol].values)
        axs[0].set_ylabel("Price")
        for i, col in enumerate(self.featureCols):
            axs[i+1].plot(df.index.values, df.loc[:, col].values)
            axs[i+1].set_ylabel(col)
        plt.show()

        boxplot = df[self.featureCols]
        sns.boxplot(data=boxplot)
        plt.show()

    def checkpointModel(self, nnet):
        checkpointFile = os.path.join(self.inputDir, "checkpoint_spricernn_%s_wt" % self.cellType)
        if not os.path.exists(checkpointFile):
            nnet.predict(np.ones((20, self.nTimestep, len(self.featureCols)), dtype=np.float32))
            tf.keras.models.save_model(nnet, checkpointFile, overwrite=False)
        else:
            nnet = tf.keras.models.load_model(checkpointFile)
        return nnet

    def model(self):
        nnet = tf.keras.Sequential()
        nfeat = len(self.featureCols)
        self.cellType = "LSTM"
        nnet.add(tf.keras.layers.LSTM(self.nUnit, input_shape=(None, nfeat)))
        #nnet.add(tf.keras.layers.GRU(self.nUnit, input_shape=(None, nfeat)))
        #nnet.add(tf.keras.layers.SimpleRNN(self.nUnit, input_shape=(None, nfeat)))
        nnet.add(tf.keras.layers.Dense(5, activation="relu"))
        nnet.add(tf.keras.layers.Dense(1))

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        nnet.compile(optimizer=self.optimizer,
                     loss=self.loss)
        nnet = self.checkpointModel(nnet)
        return nnet

    def plotConvergenceHistory(self, history, metricName):
        plt.plot(history.epoch, history.history[metricName])
        plt.xlabel("Epoch")
        plt.ylabel(metricName)
        plt.grid(True)
        #plt.legend()
        plt.show()

    def trainModel(self):
        history = self.nnet.fit(self.rnnTrainData[0], self.rnnTrainData[1],
                                batch_size=self.batchSize, epochs=self.nEpoch)
        self.plotConvergenceHistory(history, "loss")
        return history

    def testModel(self):
        mse = tf.keras.losses.MeanSquaredError()
        cnt = 0
        for X, y in [self.rnnTrainData, self.rnnTestData]:
            predict = self.nnet.predict(X)
            loss = mse(y[:, -1], predict[:, 0]).numpy()
            self.logger.info("final loss = %f", loss)
            # baseline model prediction that uses last month's return as prediction for 1 month return
            loss = mse(y[:, -1], X[:, -1, 0]).numpy()
            self.logger.info("baseline loss = %f", loss)
            # plot predicted vs actual vs baseline
            self.plotPredictedReturn(y, predict[:, 0], cnt == 0)
            cnt += 1

    def plotPredictedReturn(self, yActual: np.ndarray, yPred: np.ndarray, isTrain: bool) -> None:
        pxActual = np.zeros(yActual.shape[0], dtype=np.float32)
        pxPred = np.zeros(yActual.shape[0], dtype=np.float32)

        dts = [None] * yActual.shape[0]
        df = self.trainDf
        if not isTrain:
            df = self.testDf

        for i in range(pxActual.shape[0]):
            px = df.loc[i+self.nTimestep, self.priceCol]
            pxActual[i] = px*(1.0 + yActual[i, -1])
            pxPred[i] = px*(1.0 + yPred[i])
            dts[i] = df.loc[i+self.nTimestep, self.dateCol]
        plt.plot(dts, pxActual, label="Actual")
        plt.plot(dts, pxPred, "--", label="Predicted")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        title = "Training Data" if isTrain else "Testing Data"
        plt.title(title)
        plt.legend()
        plt.show()

        plt.plot(dts[-252*2:], pxActual[-252*2:], label="Actual")
        plt.plot(dts[-252*2:], pxPred[-252*2:], "--", label="Predicted")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        title = "Training Data" if isTrain else "Testing Data"
        plt.title(title)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    sp500file = r"C:\prog\cygwin\home\samit_000\RLPy\data\book"
    rpred = ReturnPredictor(sp500file, nepoch=80)
    rpred.trainModel()
    rpred.testModel()