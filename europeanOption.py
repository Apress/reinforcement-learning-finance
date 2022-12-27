import numpy as np
import logging
from scipy.stats import norm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("root")


class EuropeanOption(object):
    def __init__(self, s0, strike, maturity, rf, volatility, minsteps=20):
        """
        Initialize
        :param s0: Initial price of underlying asset
        :param strike: Strike price
        :param maturity: Maturity in years
        :param rf: Risk free rate (per annum)
        :param volatility: expressed per annum
        :param minsteps: Minimum number of time steps
        """
        self.s0 = s0
        self.strike = strike
        self.maturity = maturity
        self.rf = rf
        self.vol = volatility
        self.minSteps = minsteps

        self.deltaT = min(self.calculateDeltaT(), maturity/minsteps)
        self.df = np.exp(-rf * self.deltaT)
        self.sqrtTime = np.sqrt(self.deltaT)
        expected = np.exp((rf + volatility*volatility/2.0)*self.deltaT)
        self.up = np.exp(volatility * self.sqrtTime)
        self.down = np.exp(-volatility * self.sqrtTime)
        self.pUp = (expected - self.down)/(self.up - self.down)
        self.pDown = 1.0 - self.pUp
        self.ntime = int(np.ceil(maturity / self.deltaT))
        self.grid = np.zeros((2*self.ntime, self.ntime), dtype=np.float32)

    def evaluate(self):
        # values at time T
        grid = self.grid
        val = self.s0 * np.exp(-volatility * self.sqrtTime * self.ntime)
        for i in range(2*self.ntime):
            grid[i, -1] = max(val - self.strike, 0)
            val *= self.up

        for j in range(self.ntime-1, 0, -1):
            for i in range(self.ntime-j, self.ntime+j, 1):
                grid[i, j-1] = self.df * (self.pUp * grid[i+1, j] + self.pDown * grid[i-1, j])

        return grid[self.ntime, 0]

    def calculateDeltaT(self):
        val = self.vol / (self.rf + self.vol*self.vol/2.0)
        return val*val

    def blackScholes(self):
        d1 = (np.log(self.s0/self.strike) +
              (self.rf + self.vol*self.vol/2.0)*self.maturity)/(self.vol * np.sqrt(self.maturity))
        d2 = d1 - self.vol * np.sqrt(self.maturity)
        return self.s0 * norm.cdf(d1) - self.strike * np.exp(-self.rf * self.maturity) * norm.cdf(d2)


if __name__ == "__main__":
    price = 20.0
    strike = 22.0
    maturity = 2.0/12.0
    volatility = 0.2
    rf = 0.005
    eoption = EuropeanOption(price, strike, maturity, rf, volatility, minsteps=25)
    bsPrice = eoption.blackScholes()
    simPrice = eoption.evaluate()
    logger.info("Black Scholes price: %f, simulated price: %f", bsPrice, simPrice)