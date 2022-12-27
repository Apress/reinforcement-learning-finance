import numpy as np
from enum import Enum, unique
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

@unique
class Actions(Enum):
    HOLD = 0
    EXERCISE = 1


class AmericanPutOption(object):

    def __init__(self, S0, volat, strike, maturity, rf, time_steps=2000, npaths = 10000):
        """
        S0: initital stock price
        volat: volatility of stock
        strike: strike price
        maturity: maturity of the option in years
        rf: risk free rate (assumed constant) annual rate
        time_steps: Number of time steps from 0 to maturity
        """
        self.nSamples = npaths
        self.S0 = S0
        self.volat = volat
        self.K = strike
        self.T = maturity
        self.rf = rf
        self.timeSteps = time_steps
        self.gamma = np.exp(-rf/float(time_steps))
        self.nPartSUp = time_steps*volat*np.sqrt(1.0/time_steps)
        fac = volat*np.sqrt(1.0/time_steps)
        self.probUp = (np.exp(rf/time_steps) - np.exp(-fac))/(np.exp(fac) - np.exp(-fac))

    def generatePath(self):
        path = [None] * (self.timeSteps + 1)
        state = (0, np.log(self.S0))
        path[0] = state
        t = 0
        incr = 1.0/self.timeSteps
        stockval = np.log(self.S0)
        incrS = self.volat*np.sqrt(1.0/self.timeSteps)
        for i in range(self.timeSteps):
            val = np.random.random()
            if val <= self.probUp:
                stockval += incrS
            else:
                stockval -= incrS
            t += incr
            path[i+1] = (t, stockval)
        return path

    def valueOnPath(self, path):
        val = max(0, self.K - np.exp(path[-1][1]))
        for i in range(len(path)-2, -1, -1):
            exercise_val = self.K - np.exp(path[i][1])
            val = max(self.gamma*val, exercise_val)
        return val

    def optionValue(self):
        value = 0.0
        for i in range(self.nSamples):
            path = self.generatePath()
            value += self.valueOnPath(path)
        return value/self.nSamples


if __name__ == "__main__":
    put_option = AmericanPutOption(20, 0.3, 21, 1, 0.005)
    logger.info("Put option price: %f", put_option.optionValue())


