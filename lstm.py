import numpy as np
import pandas as pd
import os.path
import statsmodels.tsa.stattools
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.regression.linear_model as lm
import tensorflow as tf
from tensorflow.keras import layers, models
import itertools
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator


class ColumnConfig(object):
    def __init__(self):
        self.CLOSE_PRICE = 'Adj Close'
        self.VOLUME = 'Volume'
        self.DATE = 'Date'
        # Date is index


class TransformedRiskMeasure(object):
    def __init__(self, name):
        self.name = name

    def calculateEMA(self, data_arr, ema=10):
        ema_arr = np.zeros(data_arr.shape[0])

        # for elements [0, 1, 2, ... ema-1] fill 1 element, 2 element, ... averages
        for i in range(ema):
            ema_arr[i] = np.mean(data_arr[0:(i+1)])

        ema_arr[ema] = np.mean(data_arr[0:ema])
        for i in range(ema, data_arr.shape[0]):
            ema_arr[i] = ((ema-1)*ema_arr[i-1] + data_arr[i])/float(ema)
            if np.isnan(ema_arr[i]):
                ema_arr[i] = np.mean(data_arr[i-ema:i])

        return ema_arr


class MktModel(object):
    DAYS_IN_WEEK = 5

    def __init__(self, dr):
        mkt_file = os.path.join(dr, "SP500.csv")
        self.df = pd.read_csv(mkt_file)
        self.df.loc[:, "Date"] = pd.to_datetime(self.df.loc[:, "Date"])
        self.confVal = 0.95
        self.df = self.calculateVars()

    def calculateVars(self):
        df = self.df
        px = df.loc[:, "Adj Close"].values
        rows = df.shape[0]
        ret = np.log(np.divide(px[self.DAYS_IN_WEEK-1:-1], px[0:rows-self.DAYS_IN_WEEK]))
        df.loc[:, "MktReturn"] = 0.0
        df.loc[self.DAYS_IN_WEEK:, "MktReturn"] = ret
        # volatility of returns
        df.loc[:, "MktVolatility"] = 0.0
        mvolat = np.zeros(df.shape[0], dtype=np.float64)
        mvol = np.zeros(df.shape[0], dtype=np.float64)
        avgVol = np.mean(df.Volume.values[0:int(rows*0.7)])
        for i in range(self.DAYS_IN_WEEK, df.shape[0]):
            mvolat[i] = np.std(df.loc[i - self.DAYS_IN_WEEK:i - 1, "MktReturn"])
            mvol[i] = np.sum(df.loc[i - self.DAYS_IN_WEEK:i - 1, "Volume"].values) / avgVol
        df.loc[:, "MktVolatility"] = mvolat
        df.loc[:, "MktVolume"] = mvol
        return df

    def buildModel(self, fname=None):
        df = self.df
        ret = df.loc[self.DAYS_IN_WEEK:, "MktReturn"].values
        # build a AR model
        pacf, confint = statsmodels.tsa.stattools.pacf(ret, alpha=0.05)
        # plot pacf, confint
        fig, ax = plt.subplots()
        #fig.suptitle("AR Model Identification")
        y_err = np.subtract(confint, np.reshape(np.repeat(pacf, 2), confint.shape))
        xpos = np.arange(len(pacf))
        ax.bar(xpos[1:], pacf[1:], yerr=y_err[1:, 1], alpha=0.5, ecolor="black", capsize=2)
        ax.set_title("AR Model Identification")
        ax.set(ylabel='PACF')
        ax.set(xlabel="Order")
        #ax.set_xticks(xpos)
        ax.yaxis.grid(True)
        #axs[1].set(ylabel="Conf Int")
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        self.df = df
        mod = self.buildOrder5Model(df)
        return mod, pacf, confint

    def buildOrder5Model(self, df):
        vals = df.loc[self.DAYS_IN_WEEK:, "MktReturn"].values
        laggedvals = [vals[0:-i*self.DAYS_IN_WEEK] for i in range(1, 6)]

        x_data = sm.add_constant(np.vstack([laggedvals[0][4*self.DAYS_IN_WEEK:],
                                            laggedvals[1][3*self.DAYS_IN_WEEK:],
                                            laggedvals[2][2*self.DAYS_IN_WEEK:],
                                            laggedvals[3][1*self.DAYS_IN_WEEK:],
                                            laggedvals[4],
                                            df.MktVolatility.values[5*self.DAYS_IN_WEEK:-self.DAYS_IN_WEEK],
                                            df.MktVolume.values[5*self.DAYS_IN_WEEK:-self.DAYS_IN_WEEK]]).T)
        lm_model = lm.OLS(vals[5*self.DAYS_IN_WEEK:], x_data)
        result = lm_model.fit()
        # check p values for significance
        print("R^2 = %f" % result.rsquared_adj)
        for pval in result.pvalues:
            if pval > (1 - self.confVal):
                print("Values are not significant at 95% significance level")
        self.df = df
        return result


class SectorModel(object):
    def __init__(self, dir_name, sector, mkt_df):
        sct_file = os.path.join(dir_name, "%s.csv"%sector)
        self.df = self.readData(sct_file, mkt_df)
        self.mktDf = mkt_df
        self.confVal = 0.95

    def readData(self, sct_file, mkt_df):
        df = pd.read_csv(sct_file)
        df.loc[:, "Date"] = pd.to_datetime(df.loc[:, "Date"])
        vals = df.loc[:, "Adj Close"].values
        ret = np.log(np.divide(vals[MktModel.DAYS_IN_WEEK-1:-1], vals[0:-MktModel.DAYS_IN_WEEK]))
        df.loc[:, "Return"] = 0.0
        df.loc[MktModel.DAYS_IN_WEEK:, "Return"] = ret

        config = ColumnConfig()
        ema_10 = TransformedRiskMeasure('PxEMA10')
        df.loc[:, ema_10.name] = ema_10.calculateEMA(df[config.CLOSE_PRICE].values, ema=10)
        ema_20 = TransformedRiskMeasure('PxEMA20')
        df.loc[:, ema_20.name] = ema_20.calculateEMA(df[config.CLOSE_PRICE].values, ema=20)
        df.loc[:, "ShortMLong"] = np.where(df.loc[:, ema_10.name].values > df.loc[:, ema_20.name].values, 1, 0)

        df.loc[:, "ActReturn"] = 0.0
        df.loc[0:df.shape[0]-MktModel.DAYS_IN_WEEK-1, "ActReturn"] = ret

        mkt_df.rename(columns={"Adj Close": "MktPx"}, inplace=True)
        df = pd.merge(df, mkt_df[["Date", "MktReturn", "MktPx", "MktVolume", "MktVolatility"]], on=["Date"], how="inner")
        return df

    def buildModel(self):
        df = self.df
        ret = df.loc[:, "Return"].values
        mktret = df.loc[:, "MktReturn"].values
        laggedret = ret[MktModel.DAYS_IN_WEEK:-MktModel.DAYS_IN_WEEK]
        df.loc[:, "LaggedReturn"] = 0.0
        df.loc[2*MktModel.DAYS_IN_WEEK:, "LaggedReturn"] = laggedret

        x_data = sm.add_constant(np.vstack([mktret[3*MktModel.DAYS_IN_WEEK:],
                                            laggedret[0:-MktModel.DAYS_IN_WEEK]]).T)
        lm_model = lm.OLS(ret[3*MktModel.DAYS_IN_WEEK:], x_data)
        result = lm_model.fit()
        # check p values for significance
        print("R^2 = %f" % result.rsquared_adj)
        for pval in result.pvalues:
            if pval > (1 - self.confVal):
                print("Values are not significant at 95% significance level")

        return result


class LSTMModel(object):
    def __init__(self, df, training_data_perc=0.70, validation_data_perc=0.05, symbol='',
                 return_sequences=True):
        self.symbol = symbol
        self.returnSequences = return_sequences
        self.nTimeSteps = 4
        rows = df.shape[0]
        trg_begin = 0
        trg_end = int(training_data_perc * rows)
        validation_begin = trg_end + 1
        validation_end = int((training_data_perc + validation_data_perc) * rows)
        self.df = df
        x_train, y_train = self.getTrainingData(df.loc[trg_begin:trg_end, :].reset_index(drop=True))
        x_valid, y_valid = self.getValidationData(df.loc[validation_begin:validation_end, :].reset_index(drop=True))
        self.lstm = self.buildLSTMModel(x_train, y_train, x_valid, y_valid)

    def getTrainingData(self, df):
        data_arr = df.loc[:, ["MktVolatility", "MktReturn", "MktVolume", "Return"]].values
        actret_arr = df.loc[:, "ActReturn"].values
        input_arr = np.zeros((data_arr.shape[0]-5*MktModel.DAYS_IN_WEEK, self.nTimeSteps, 4), dtype=np.float64)
        if self.returnSequences:
            output_arr = np.zeros((input_arr.shape[0], self.nTimeSteps))
        else:
            output_arr = np.zeros(input_arr.shape[0])
        debug_df = pd.DataFrame(data={"Date": df.Date})
        lcols = ["L%d"%i for i in range(self.nTimeSteps-1, -1, -1)]
        cols = list(itertools.product(lcols, ["MktVolatility", "MktReturn", "MktVolume", "Return"]))
        cols = [c[0]+c[1] for c in cols]
        cols2 = ["L%dActReturn"%i for i in range(self.nTimeSteps-1, -1, -1)]
        for cl1 in cols + cols2:
            debug_df.loc[:, cl1] = 0.0
        offset = 4*MktModel.DAYS_IN_WEEK
        for i in range(offset, data_arr.shape[0]-MktModel.DAYS_IN_WEEK):
            for j in range(self.nTimeSteps):
                input_arr[i-offset, j, :] = data_arr[i-(self.nTimeSteps-1-j)*MktModel.DAYS_IN_WEEK, :]
            debug_df.loc[i, cols] = input_arr[i-offset, :, :].flatten()
            if self.returnSequences:
                for j in range(self.nTimeSteps):
                    output_arr[i-offset, j] = actret_arr[i-(self.nTimeSteps-1-j)*MktModel.DAYS_IN_WEEK]
                debug_df.loc[i, cols2] = output_arr[i-offset, :]
            else:
                output_arr[i - offset] = actret_arr[i]
        df_final = pd.merge(df, debug_df, on=["Date"], how="left")
        return input_arr, output_arr

    def getValidationData(self, df):
        data_arr = df.loc[:, ["MktVolatility", "MktReturn", "MktVolume", "Return"]].values
        actret_arr = df.loc[:, "ActReturn"].values
        if data_arr.shape[0] <= 5*MktModel.DAYS_IN_WEEK:
            return None, None
        input_arr = np.zeros((data_arr.shape[0] - 5*MktModel.DAYS_IN_WEEK, self.nTimeSteps, 4), dtype=np.float64)
        if self.returnSequences:
            output_arr = np.zeros((input_arr.shape[0], self.nTimeSteps))
        else:
            output_arr = np.zeros(input_arr.shape[0])
        offset = 4*MktModel.DAYS_IN_WEEK
        for i in range(offset, data_arr.shape[0] - MktModel.DAYS_IN_WEEK):
            for j in range(self.nTimeSteps):
                input_arr[i - offset, j, :] = data_arr[i - (self.nTimeSteps-1-j)*MktModel.DAYS_IN_WEEK, :]
            if self.returnSequences:
                for j in range(self.nTimeSteps):
                    output_arr[i - offset, j] = actret_arr[i - (self.nTimeSteps-1-j)*MktModel.DAYS_IN_WEEK]
            else:
                output_arr[i - offset] = actret_arr[i]
        return input_arr, output_arr

    def buildLSTMModel(self, x_train, y_train, x_valid, y_valid):
        model = models.Sequential()
        lyr = layers.LSTM(8, return_sequences=self.returnSequences)
        #lyr = tf.keras.layers.SimpleRNN(8, return_sequences=self.returnSequences)
        model.add(lyr)
        model.add(layers.Dense(4))
        model.add(layers.Dense(1))
        input_shape = (None, self.nTimeSteps, 4)
        model.build(input_shape)
        model.summary()
        model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["mse"])
        if x_valid is not None:
            model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=5)
        else:
            model.fit(x_train, y_train, epochs=5)
        return model

    def predict(self, df, begin):
        lcols = ["L%d" % i for i in range(self.nTimeSteps - 1, -1, -1)]
        cols = list(itertools.product(lcols, ["MktVolatility", "MktReturn", "MktVolume", "Return"]))
        cols2 = ["L%dActReturn"%i for i in range(self.nTimeSteps-1, -1, -1)]
        if self.returnSequences:
            cols3 = ["L%dPrReturn"%i for i in range(self.nTimeSteps-1, -1, -1)]
        else:
            cols3 = ["L0PrReturn"]
        cols = [c[0] + c[1] for c in cols]
        data_arr = df.loc[begin:, ["MktVolatility", "MktReturn", "MktVolume", "Return"]].values
        actret_arr = df.loc[begin:, "ActReturn"].values
        results_df = pd.DataFrame(data={"Date": df.loc[begin:, "Date"]})
        for cl1 in cols+cols2+cols3:
            results_df.loc[:, cl1] = 0.0
        input_arr = np.zeros((1, self.nTimeSteps, 4), dtype=np.float64)
        output_arr = np.zeros((1, self.nTimeSteps), dtype=np.float64)
        for i in range(begin + 3*MktModel.DAYS_IN_WEEK, df.shape[0]-MktModel.DAYS_IN_WEEK):
            for j in range(self.nTimeSteps):
                input_arr[0, j, :] = data_arr[i - begin - (self.nTimeSteps-1-j)*MktModel.DAYS_IN_WEEK, :]
            results_df.loc[i, cols] = input_arr[0, :, :].flatten()
            for j in range(self.nTimeSteps):
                output_arr[0, j] = actret_arr[i - begin - (self.nTimeSteps-1-j)*MktModel.DAYS_IN_WEEK]
            results_df.loc[i, cols2] = output_arr[0, :]
            out1 = self.lstm.predict(input_arr)
            results_df.loc[i, cols3] = out1.flatten()
        results_df = pd.merge(results_df, df, on=["Date"], how="left")
        return results_df

    @staticmethod
    def plot(df, begin, secname, fname=None):
        fig, ax = plt.subplots(nrows=1, ncols=3)
        fig.set_size_inches((30, 7), forward=True)
        column = "Adj Close"
        ylabel = "Price"
        end = df.shape[0] - MktModel.DAYS_IN_WEEK
        dates = df.Date[begin:end+1].values
        majorLocator = YearLocator()  # every year
        minorLocator = MonthLocator()  # every month
        formatter = DateFormatter('%Y')

        ax[0].plot(dates, df.loc[begin:end, column].values)
        ax[0].set_ylabel(ylabel)
        ax[0].xaxis.set_major_locator(majorLocator)
        ax[0].xaxis.set_major_formatter(formatter)
        ax[0].xaxis.set_minor_locator(minorLocator)
        ax[0].format_xdata = DateFormatter('%Y-%m')
        ax[0].set_xlabel("Date")
        ax[0].grid(True)

        columns = ["L0PrReturn", "ActReturn"]
        ylabels = ["Pr. Return", "Ac. Return"]
        for i in range(1, 3):
            ax[i].bar(dates, df.loc[begin:end, columns[i-1]].values, alpha=0.5, ecolor="black")
            ax[i].set_ylabel(ylabels[i-1])
            ax[i].xaxis.set_major_locator(majorLocator)
            ax[i].xaxis.set_major_formatter(formatter)
            ax[i].xaxis.set_minor_locator(minorLocator)
            ax[i].format_xdata = DateFormatter('%Y-%m')
            ax[i].set_xlabel("Date")
            ax[i].grid(True)

        plt.title(secname)
        #plt.tight_layout()
        plt.show()
        plt.close(fig)


class RegressionModelPredictor(object):
    def __init__(self, dir_name, sector):
        mkt_coeff = os.path.join(dir_name, "mkt.csv")
        self.mktDf = pd.read_csv(mkt_coeff)
        sector_coeff = os.path.join(dir_name, "coeff.csv")
        self.sectorDf = pd.read_csv(sector_coeff)
        self.sectorDf = self.sectorDf.loc[self.sectorDf.Sector.eq(sector), :].reset_index(drop=True)

    def predict(self, df, begin):
        mkt_lags = len(self.mktDf.columns) - 3
        mkt_x = np.zeros(mkt_lags + 2, dtype=np.float64)
        mktret = df.MktReturn.values
        mktvolat = df.MktVolatility.values
        mktvolume = df.MktVolume.values
        secret = df.Return.values
        df.loc[:, "RegPrReturn"] = 0.0
        cols = ["const"] + ["L%d" % i for i in range(1, mkt_lags + 1)] + ["MktVolatility", "MktVolume"]
        mkt_coeff = self.mktDf.loc[0, cols].values
        sec_x = np.zeros(2, dtype=np.float64)
        sec_coeff = self.sectorDf.loc[0, ["const", "MktReturn", "LaggedReturn"]].values
        for i in range(begin + 3 * MktModel.DAYS_IN_WEEK, df.shape[0] - MktModel.DAYS_IN_WEEK):
            for j in range(mkt_lags):
                mkt_x[j] = mktret[i - j * MktModel.DAYS_IN_WEEK]
            mkt_x[mkt_lags] = mktvolat[i]
            mkt_x[mkt_lags+1] = mktvolume[i]
            pred_ret = mkt_coeff[0] + np.dot(mkt_coeff[1:], mkt_x)
            sec_x[0] = pred_ret
            sec_x[1] = secret[i]
            pred_sec_ret = sec_coeff[0] + np.dot(sec_coeff[1:], sec_x)
            df.loc[i, "RegPrReturn"] = pred_sec_ret
        df, rms_reg, rms_lstm = self.sqDiff(df, begin)
        return df, rms_reg, rms_lstm

    @staticmethod
    def plot(df, begin, fname=None, sec=''):
        fig, ax = plt.subplots(nrows=3, ncols=1)
        end = df.shape[0] - MktModel.DAYS_IN_WEEK
        dates = df.Date[begin:end + 1].values
        majorLocator = YearLocator()  # every year
        minorLocator = MonthLocator()  # every month
        formatter = DateFormatter('%Y')

        cols = ["RegPrReturn", "SqRegDiff", "SqLSTMDiff"]
        ylabels = ["Reg. Pr. Return", "Sq. Diff.", "Sq. Diff."]
        for i in range(3):
            ax[i].bar(dates, df.loc[begin:end, cols[i]].values, alpha=0.5, ecolor="black")
            ax[i].set_ylabel(ylabels[i])
            ax[i].xaxis.set_major_locator(majorLocator)
            ax[i].xaxis.set_major_formatter(formatter)
            ax[i].xaxis.set_minor_locator(minorLocator)
            ax[i].format_xdata = DateFormatter('%Y-%m')
            ax[i].set_xlabel("Date")
            ax[i].grid(True)
            #ax[i].title.set_text(sec)

        fig.suptitle(sec)
        plt.show()
        plt.close(fig)

    def sqDiff(self, df, begin):
        df.loc[:, "SqRegDiff"] = 0.0
        df.loc[:, "SqLSTMDiff"] = 0.0
        nr = df.shape[0]
        diff = np.subtract(df.loc[begin:, "RegPrReturn"].values, df.loc[begin:, "ActReturn"].values)
        df.loc[begin:, "SqRegDiff"] = np.multiply(diff, diff)
        diff = np.subtract(df.loc[begin:, "L0PrReturn"].values, df.loc[begin:, "ActReturn"].values)
        df.loc[begin:, "SqLSTMDiff"] = np.multiply(diff, diff)
        avg_rmsreg = np.sqrt(np.sum(df.loc[begin:nr - MktModel.DAYS_IN_WEEK, "SqRegDiff"].values) / (nr - begin - MktModel.DAYS_IN_WEEK))
        avg_rmslstm = np.sqrt(np.sum(df.loc[begin:nr - MktModel.DAYS_IN_WEEK, "SqLSTMDiff"].values) / (nr - begin - MktModel.DAYS_IN_WEEK))
        return df, avg_rmsreg, avg_rmslstm

    def trade(self, df, begin):
        sgs = ["RegSignal", "LSTMSignal"]
        cols = ["RegPrReturn", "L0PrReturn"]
        for signal, col in zip(sgs, cols):
            df.loc[:, signal] = 0
            skip = 0
            last_pos = 0
            for i in range(begin, df.shape[0] - MktModel.DAYS_IN_WEEK):
                if skip > i:
                    continue
                if last_pos == 0:
                    if df.loc[i, col] > 0:
                        last_pos = 1
                        df.loc[i, signal] = 1
                elif last_pos == 1:
                    if df.loc[i, col] < 0:
                        last_pos = 0
                        df.loc[i, signal] = -1
                else:
                    raise ValueError("Invalid value of last_pos: %d"%last_pos)
            if last_pos == 1:
                df.loc[df.shape[0] - MktModel.DAYS_IN_WEEK, signal] = -1
        return df


def regression(input_dir, output_dir):
    dir_name = input_dir
    model = MktModel(dir_name)
    pacf_file = os.path.join(output_dir, "mkt_pacf.png")
    vals = model.buildModel(pacf_file)
    mkt_params = vals[0].params
    sfilename = os.path.join(output_dir, "summary.txt")
    sfile = open(sfilename, "w")

    sfile.write(vals[0].summary().as_text())
    df1 = pd.DataFrame(data={"const": [mkt_params[0]], "L1": [mkt_params[1]], "L2": [mkt_params[2]],
                             "L3": [mkt_params[3]], "L4": [mkt_params[4]], "L5": [mkt_params[5]],
                             "MktVolatility": [mkt_params[6]], "MktVolume": [mkt_params[7]]})
    coeff_file = os.path.join(output_dir, "mkt.csv")
    df1.to_csv(coeff_file, index=False)
    """
    Communication services: XLC
    Consumer Discretionary: XLY
    Consumer Staples: XLP
    Energy: XLE
    Financials: XLF
    Healthcare: XLV
    Industrials: XLI
    Information Technology: XLK
    Materials: XLB
    Real Estate: XLRE
    Utilities: XLU
    """
    sectors = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
    results = pd.DataFrame(data={"Sector": sectors})
    results.loc[:, "const"] = 0.0
    results.loc[:, "MktReturn"] = 0.0
    results.loc[:, "LaggedReturn"] = 0.0
    for sec in sectors:
        smodel = SectorModel(dir_name, sec, model.df)
        res = smodel.buildModel()
        sfile.write("\n" + sec + "\n")
        sfile.write(res.summary().as_text())
        params = res.params
        row = results.Sector.eq(sec)
        results.loc[row, "const"] = params[0]
        results.loc[row, "MktReturn"] = params[1]
        results.loc[row, "LaggedReturn"] = params[2]
    coeff_file = os.path.join(output_dir, "coeff.csv")
    results.to_csv(coeff_file, index=False)
    sfile.close()


def runLSTM(input_dir, output_dir):
    dir_name = input_dir
    model = MktModel(dir_name)
    sectors = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
    return_seq = False
    for sec in sectors:
        smodel = SectorModel(dir_name, sec, model.df)
        avg_vol = np.mean(smodel.df.Volume.values[0:int(0.75 * smodel.df.shape[0])])
        lstm = LSTMModel(smodel.df, symbol=sec, return_sequences=return_seq)
        begin = int(0.75 * smodel.df.shape[0])
        result_df = lstm.predict(smodel.df, begin)
        result_df.to_csv(os.path.join(output_dir, "%s_lstmpredict.csv"%sec))
        plot_file = os.path.join(output_dir, "%s_plots.png" % sec)
        lstm.plot(result_df, 0, sec, plot_file)

def plotLSTMResults(input_dir, output_dir):
    dir_name = input_dir
    sectors = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
    rmsDf = pd.DataFrame(data={"Sector": sectors, "RMSReg": [0]*len(sectors), "RMSLSTM": [0]*len(sectors)})
    for sec in sectors:
        sec_file = os.path.join(dir_name, "%s.csv"%sec)
        df = pd.read_csv(sec_file)
        avg_vol = np.mean(df.Volume.values[0:int(0.75 * df.shape[0])])
        fl = os.path.join(output_dir, "%s_lstmpredict.csv"%sec)
        df = pd.read_csv(fl)
        plot_file = os.path.join(output_dir, "%s_plots.png"%sec)
        LSTMModel.plot(df, 0, sec, plot_file)
        rpred = RegressionModelPredictor(output_dir, sec)
        df, rms1, rms2 = rpred.predict(df, 0)
        print("Sector: %s, RMSReg %f, RMSLSTM %f" % (sec, rms1, rms2))
        rmsDf.loc[rmsDf.Sector.eq(sec), "RMSReg"] = rms1
        rmsDf.loc[rmsDf.Sector.eq(sec), "RMSLSTM"] = rms2
        df = rpred.trade(df, 0)
        reg_plot = os.path.join(output_dir, "reg_%s_plots.png"%sec)
        rpred.plot(df, 0, reg_plot, sec)

    rmsDf.to_csv(os.path.join(output_dir, "rmserr.csv"))
    print(rmsDf.to_latex(index=False))

if __name__ == "__main__":
    input_dir = r"C:\prog\cygwin\home\samit_000\value_momentum_new\value_momentum\data\sectors"
    output_dir = r"C:\prog\cygwin\home\samit_000\value_momentum_new\value_momentum\output\sector"
    regression(input_dir, output_dir)
    runLSTM(input_dir, output_dir)
    plotLSTMResults(input_dir, output_dir)
