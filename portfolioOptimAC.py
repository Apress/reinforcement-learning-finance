import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from src.learner.ActorCriticLearner import AdvantageActorCriticLearner
from src.lib.ActorCriticNetwork import ACNetwork
from src.lib.Emulator import StateAndRewardEmulator
from src.lib.Episode import Episode
from src.lib.Sample import Sample


class PortfolioEmulator(StateAndRewardEmulator):
    def __init__(self, dfs, var, covar, trx_cost, price_col, return_col, max_days, nstocks):
        self.dfs = dfs
        self.nStock = nstocks
        self.var = var
        self.covar = covar
        self.iVar = 0
        self.iCvar = nstocks
        self.iRet = nstocks + nstocks*(nstocks - 1) // 2
        self.iWeight = self.iRet + nstocks
        self.priceCol = price_col
        self.retCol = return_col
        self.trxCost = trx_cost
        self.maxDays = max_days
        self._begin = 0
        self._index = 0
        self._state = None

    def step(self, state, action):
        pass

    def setInitialState(self, state):
        self._state = state

    def setBeginIndex(self, value):
        self._begin = value
        self._index = value

    def getReward(self, state, action, index, begin):
        weights = state[self.iWeight:]
        begin_price = np.array([df.loc[begin, self.priceCol] for df in self.dfs])
        price = np.array([df.loc[index, self.priceCol] for df in self.dfs])
        if index - begin == self.maxDays:
            nshares = np.divide(weights, begin_price) * self.nStock
            pnl = (1 - self.trxCost) * np.sum(nshares * price)
        else:
            new_position = action
            pos_change = new_position - weights
            nshares = np.divide(pos_change, begin_price) * self.nStock
            price_change = np.array(
                [df.loc[index, self.priceCol] - df.loc[index - 1, self.priceCol] for df in self.dfs])
            pnl = np.sum(price_change * nshares) - self.trxCost * np.sum(nshares * price)

        return pnl

    def tfEnvStep(self, action: tf.Tensor) -> List[tf.Tensor]:
        self._index += 1
        index = self._index
        action = tf.squeeze(action)
        weights = self._state[self.iWeight:]
        price = [df.loc[index, self.priceCol] for df in self.dfs]
        done = False
        if index - self._begin == self.maxDays:
            pnl = (1 - self.trxCost) * tf.reduce_sum(weights * price)
            self._begin += 1
            self._index = self._begin
            done = True
        else:
            new_position = action
            pos_change = new_position - weights
            price_change = [df.loc[index, self.priceCol] - df.loc[index - 1, self.priceCol] for df in self.dfs]
            pnl = tf.reduce_sum(price_change * pos_change) - self.trxCost * tf.reduce_sum(pos_change * price)

        new_cvar = self.covar[index, :]
        new_var = self.var[index, :]
        new_ret = [df.loc[index, self.retCol] for df in self.dfs]
        next_state = tf.concat((new_var, new_cvar, new_ret, action), axis=0)
        self._state = next_state
        return [next_state, pnl, done]


class PortOptim(object):
    def __init__(self, stocks, inputdir, transaction_cost, training_data=0.75):
        self.stocks = stocks
        self.transactionCost = transaction_cost
        self.nStock = len(stocks)
        self.holdingPeriod = 21
        self.dfs = []
        self.priceCol = "Adj Close"
        self.dateCol = "Date"
        self.returnCol = "daily_return"

        for stock in stocks:
            filename = os.path.join(inputdir, "%s.csv" % stock)
            df = pd.read_csv(filename, parse_dates=[self.dateCol])
            df = self.calculateReturns(df)
            self.dfs.append(df)

        dates = self.dfs[0].loc[:, self.dateCol]
        self.nDate = dates.shape[0]
        for i in range(1, self.nStock):
            self.dfs[i] = pd.merge(dates, self.dfs[i], on=[self.dateCol], how="left")
        self.nTrain = int(training_data * self.dfs[0].shape[0])
        self.var, self.covar = self.calculateCovar()
        self.emulator = PortfolioEmulator(self.dfs, self.var, self.covar, self.transactionCost,
                                          self.priceCol, self.returnCol, self.holdingPeriod, self.nStock)
        self.acnet = self.createActorCritic()
        self.aclearner = AdvantageActorCriticLearner(self.acnet, discrete_actions=False)

    def calculateReturns(self, df: pd.DataFrame) -> pd.DataFrame:
        # 2 day return
        price = df.loc[:, self.priceCol].values
        df.loc[:, self.returnCol] = 0.0
        logPrice = np.log(price)
        logPriceDiff = logPrice[2:] - logPrice[0:-2]
        df.loc[3:, self.returnCol] = logPriceDiff[0:-1]
        return df

    def calculateCovar(self) -> Tuple[np.ndarray, np.ndarray]:
        dfs = self.dfs
        variances = np.zeros((self.nDate, self.nStock), dtype=np.float32)

        for index, df in enumerate(dfs):
            ret = df.loc[:, self.returnCol].values
            var = variances[:, index]
            sum_val = np.sum(ret[2:2+self.holdingPeriod])
            sumsq_val = np.sum(ret[2:2+self.holdingPeriod] * ret[2:2+self.holdingPeriod])
            mean_val = sum_val / self.holdingPeriod
            var[2+self.holdingPeriod-1] = sumsq_val / self.holdingPeriod - mean_val * mean_val
            for i in range(2+self.holdingPeriod, var.shape[0]):
                sum_val += ret[i] - ret[i-self.holdingPeriod]
                sumsq_val += ret[i] * ret[i] - ret[i-self.holdingPeriod] * ret[i-self.holdingPeriod]
                mean_val = sum_val / self.holdingPeriod
                var[i] = sumsq_val / self.holdingPeriod - mean_val * mean_val

        ncvar = self.nStock * (self.nStock - 1) // 2
        covar = np.zeros((self.nDate, ncvar), dtype=np.float32)
        count = 0

        for i1 in range(self.nStock):
            ret1 = self.dfs[i1].loc[:, self.returnCol].values
            for j in range(i1+1, self.nStock):
                ret2 = self.dfs[j].loc[:, self.returnCol].values
                cvar = covar[:, count]
                for i in range(2 + self.holdingPeriod, cvar.shape[0]):
                    begin = i - self.holdingPeriod
                    sum_val1 = np.sum(ret1[begin:begin + self.holdingPeriod])
                    sum_val2 = np.sum(ret2[begin:begin + self.holdingPeriod])
                    mean_val1 = sum_val1 / self.holdingPeriod
                    mean_val2 = sum_val2 / self.holdingPeriod
                    sumprod = np.sum((ret1[begin:i] - mean_val1) * (ret2[begin:i] - mean_val2))
                    cvar[i] = sumprod / (self.holdingPeriod * variances[i, i1] * variances[i, j])

                count += 1

        # calculate variance ratio
        variances[2+self.holdingPeriod+1:-1, :] = np.divide(variances[2+self.holdingPeriod+1:-1, :],
                                                            variances[2+self.holdingPeriod:-2, :])

        return variances, covar

    def createActorCritic(self):
        value_network = tf.keras.models.Sequential()
        # state: variance, cvar, ret, stock weights
        ninp = self.nStock + self.nStock*(self.nStock-1)//2 + self.nStock + self.nStock
        value_network.add(tf.keras.layers.Dense(4, input_shape=(ninp,)))
        value_network.add(tf.keras.layers.Dense(10, activation="relu"))
        value_network.add(tf.keras.layers.Dense(1))

        anet = tf.keras.models.Sequential()
        anet.add(tf.keras.layers.Dense(4, input_shape=(ninp,)))
        anet.add(tf.keras.layers.Dense(10, activation="relu"))
        anet.add(tf.keras.layers.Dense(self.nStock, activation="sigmoid"))
        anet.add(tf.keras.layers.Softmax())

        actor_optim = tf.keras.optimizers.Adam()
        critic_optim = tf.keras.optimizers.Adam()

        return ACNetwork(anet, value_network, self.emulator, 1.0, self.holdingPeriod, actor_optim, critic_optim)

    def randomAction(self):
        wts = np.random.random(self.nStock)
        return np.divide(wts, wts.sum())

    def getInitialWeights(self, day):
        wts = [1.0/df.loc[day, self.priceCol] for df in self.dfs]
        return np.divide(wts, np.sum(wts))

    def generateTrainingEpisodes(self):
        episodes = []
        samples = [None]
        begin = 0

        for i in range(2 + self.holdingPeriod, self.nTrain - self.holdingPeriod):
            curr_weights = self.getInitialWeights(begin)
            rets = [df.loc[i, self.returnCol] for df in self.dfs]
            state = np.concatenate((self.var[i, :], self.covar[i, :], rets, curr_weights))
            action = self.randomAction()
            reward = self.emulator.getReward(state, action, i, begin)
            if i - begin == self.holdingPeriod:
                begin = i
            samples[0] = Sample(state, action, reward, None)
            episode = Episode(samples)
            episodes.append(episode)

        return episodes

    def train(self):
        # create episodes for training
        episodes = self.generateTrainingEpisodes()
        self.emulator.setBeginIndex(2+self.holdingPeriod)
        self.aclearner.fit(episodes)

    def actorCriticPnl(self, day):
        pnl = -self.nStock
        wts = np.full(self.nStock, 1.0/self.nStock, dtype=np.float32)
        for i in range(day+1, day+self.holdingPeriod+1):
            new_cvar = self.covar[i-1, :]
            new_var = self.var[i-1, :]
            new_ret = [df.loc[i-1, self.returnCol] for df in self.dfs]
            state = np.concatenate((new_var, new_cvar, new_ret, wts))
            vals = self.aclearner.predict(state)
            abs_change = np.sum(np.abs(wts - vals[0]))
            if abs_change > 0.1:
                wts[:] = vals[0]
            pnl += self.emulator.getReward(state, wts, i, day)
        return pnl

    def buyAndHoldPnl(self, day):
        nstocks = [1.0 / df.loc[day, self.priceCol] for df in self.dfs]
        price = [df.loc[day+self.holdingPeriod, self.priceCol] for df in self.dfs]
        return -self.nStock + (1 - self.transactionCost) * np.sum(np.multiply(price, nstocks))

    def test(self):
        pnl_data = []
        pnl_bh_data = []
        pnl_diff = []
        dates = []
        self.emulator.setBeginIndex(self.nTrain)
        for i in range(self.nTrain, self.nDate-self.holdingPeriod-2, self.holdingPeriod):
            pnl = self.actorCriticPnl(i)
            pnl_bh = self.buyAndHoldPnl(i)
            pnl_diff.append(pnl - pnl_bh)
            pnl_data.append(pnl)
            pnl_bh_data.append(pnl_bh)
            dates.append(self.dfs[0].loc[i, self.dateCol])


        perf_df = pd.DataFrame(data={"Actor-Critic PNL": np.cumsum(pnl_data),
                                     "Buy-and-Hold PNL": np.cumsum(pnl_bh_data)},
                               index=np.array(dates))
        final_pnl = np.array([np.sum(pnl_data), np.sum(pnl_bh_data)])
        mean_pnl = np.array([np.mean(pnl_data), np.mean(pnl_bh_data)])
        sd_pnl = np.array([np.std(pnl_data), np.std(pnl_bh_data)])
        sr = np.sqrt(252.0/self.holdingPeriod) * mean_pnl/sd_pnl
        print("Actor-Critic: final PNL: %f, SR: %f" % (final_pnl[0], sr[0]))
        print("Buy-and-hold: final PNL: %f, SR: %f" % (final_pnl[1], sr[1]))
        sns.lineplot(data=perf_df)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    stocks = ["MSFT", "BA"]
    inputdir = r"C:\prog\cygwin\home\samit_000\RLPy\data\stocks"
    portopt = PortOptim(stocks, inputdir, 0.001)
    portopt.train()
    portopt.test()