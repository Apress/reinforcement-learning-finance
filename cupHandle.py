import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import os

import matplotlib.pyplot as plt
from matplotlib.dates import (YEARLY, DateFormatter,
                              YearLocator, MonthLocator, DayLocator)


class DatePlotter(object):
    def __init__(self):
        self.majorLocator = MonthLocator() #YearLocator()  # every year
        self.minorLocator = DayLocator()  # every month
        self.formatter = DateFormatter('%m/%d/%y') #DateFormatter('%Y')

    def plot(self, df, datecol, valcols, xlabel='date', ylabel=None, labels=None, round='Y'):
        if not labels:
            labels = valcols

        fig, ax = plt.subplots()
        for val,lb in zip(valcols, labels):
            ax.plot(datecol, val, data=df, label=lb)
        plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        # format the ticks
        ax.xaxis.set_major_locator(self.majorLocator)
        ax.xaxis.set_major_formatter(self.formatter)
        ax.xaxis.set_minor_locator(self.minorLocator)

        # round to nearest years.
        datemin = np.datetime64(df[datecol].values[0], round)
        nr = df.shape[0]-1
        datemax = np.datetime64(df[datecol].values[nr], round) + np.timedelta64(1, round)
        ax.set_xlim(datemin, datemax)

        # format the coords message box
        ax.format_xdata = DateFormatter('%Y-%m-%d')
        ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left')
        ax.grid(True)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()
        return plt


def plotData(price_data_dir, output_dir):
    df = pd.read_csv(os.path.join(output_dir, "ch_out.csv"))
    df.loc[:, "Begin"] = pd.to_datetime(df.loc[:, "Begin"])
    df.loc[:, "End"] = pd.to_datetime(df.loc[:, "End"])
    last_stock = None
    stock_df = None
    cnt = 0
    for rownum in range(df.shape[0]):
        stock = df.loc[rownum, "Stock"]
        begin = df.loc[rownum, "Begin"]
        end = df.loc[rownum, "End"]
        if stock != last_stock:
            stock_df = pd.read_csv(os.path.join(price_data_dir, "%s.csv" % stock))
            stock_df.loc[:, "Date"] = pd.to_datetime(stock_df.loc[:, "Date"])
            cnt = 0

        ibeg = stock_df.loc[stock_df.loc[:, "Date"].eq(begin), :].index[0]
        iend = stock_df.loc[stock_df.loc[:, "Date"].eq(end), :].index[0]

        dplt = DatePlotter()
        plt = dplt.plot(stock_df.loc[ibeg:iend, :], 'Date', ["Adj Close"], xlabel='Date',
                        ylabel='Price', labels=[stock], round='D')
        plt.title("Cup-and-Handle in %s" % stock)
        # plt.show()
        filename = os.path.join(output_dir, "%s_%d.png" % (stock, cnt))
        plt.savefig(filename)
        cnt = cnt + 1

def buildModel():
    model = models.Sequential()
    model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=(20, 20, 1)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(5, (4, 4), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="relu"))
    model.add(layers.Dense(2))
    model.summary()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model


def trainModel(model, df, training_rows):
    data = np.transpose(df.reset_index(drop=True).values)
    y_actual = np.array([int(c.startswith("t")) for c in df.columns], dtype=np.int)
    train_data = data[0:training_rows, :]
    train_data_final = np.zeros((training_rows, 20, 20, 1), dtype=np.float32)
    for i in range(training_rows):
        for j in range(20):
            pixel = int(train_data[i, j] * 20)
            if pixel == 20:
                pixel = 19
            train_data_final[i, j, pixel, 0] = 1
    train_output = y_actual[0:training_rows]
    model.fit(train_data_final, train_output, epochs=5)

    validation_data = data[training_rows:, :]
    validation_dt = np.zeros((validation_data.shape[0], 20, 20, 1), dtype=np.int)
    for i in range(training_rows, validation_dt.shape[0]):
        for j in range(20):
            pixel = int(train_data[i, j] * 20)
            if pixel == 20:
                pixel = 19
            validation_dt[i, j, pixel, 0] = 1
    validation_output = y_actual[training_rows:]
    model.evaluate(validation_dt, validation_output, verbose=2)
    #predictions = model(x_train[:1]).numpy()
    # this is a probabilistic model, add a softmax layer at the end
    new_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    return new_model


def rescaleXDimension(ar, xsize):
    if ar.shape[0] == xsize:
        return ar

    if ar.shape[0] > xsize:
        px = ar
        px2 = np.zeros(xsize, dtype=np.float64)
        px2[0] = px[0]
        px2[-1] = px[-1]
        delta = float(ar.shape[0])/xsize
        for i in range(1, xsize-1):
            k = int(i*delta)
            fac1 = i*delta - k
            fac2 = k + 1 - i*delta
            px2[i] = fac1 * px[k+1] + fac2 * px[k]

        return px2
    raise ValueError("df rows are less than required price array elements")


def identify(model, df_stock, ndays, stock, res_df):
    px_arr = df_stock.loc[:, "Adj Close"].values
    date_arr = df_stock.loc[:, "Date"].values
    days_identified = set(res_df.loc[res_df.loc[:, "Stock"].eq(stock), "Begin"])
    inp = np.zeros((1, 20, 20, 1), dtype=np.float32)
    for i in range(df_stock.shape[0] - ndays):
        if date_arr[i] in days_identified:
            continue
        inp[:, :, :, :] = 0
        px = px_arr[i:i+ndays]
        mn = px.min()
        mx = px.max()
        transform_px = np.divide(np.subtract(px, mn), mx-mn)
        transform = rescaleXDimension(transform_px, 20)
        for j in range(20):
            vl = int(transform[j] * 20)
            if vl == 20:
                vl = 19
            inp[0, j, vl, 0] = 1

        outval = model(inp).numpy()
        if outval[0, 1] >= 0.9:
            print("%s from %s - %s dates" % (stock, date_arr[i], date_arr[i+ndays-1]))
            res_df = res_df.append({"Stock":stock, "Begin": date_arr[i], "End": date_arr[i+ndays-1]},
                                   ignore_index=True)
    return res_df


def processData(stock_list, input_dir, price_data_dir, output_dir):
    res_df = pd.DataFrame(data={"Stock":[], "Begin":[], "End":[]})
    df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    df.drop(columns=["Day"], inplace=True)
    obs = len(df.columns)
    training_perc = 0.95
    train_rows = int(obs * training_perc)
    model = buildModel()
    model = trainModel(model, df, training_rows=train_rows)

    # predict
    period_begin = 40
    period_end = 70
    for stock in stock_list:
        df_stock = pd.read_csv(os.path.join(price_data_dir, "%s.csv"%stock))
        for period in range(period_begin, period_end):
            res_df = identify(model, df_stock, period, stock, res_df)
    res_df.to_csv(os.path.join(output_dir, "ch_out.csv"), index=False)


if __name__ == "__main__":
    input_dir = r"C:\prog\cygwin\home\samit_000\value_momentum_new\value_momentum\data"
    price_data_dir = r"C:\prog\cygwin\home\samit_000\value_momentum_new\value_momentum\data\price"
    output_dir = r"C:\prog\cygwin\home\samit_000\value_momentum_new\value_momentum\output\pattern"
    df = pd.read_table(os.path.join(input_dir, "dow.txt"), header=None)
    stocks = ["TRV", "IBM"] # df.loc[:, 0].values
    processData(stocks, input_dir, price_data_dir, output_dir)
    plotData(price_data_dir, output_dir)