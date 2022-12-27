import numpy as np
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("root")


class EuropeanKnockInCallOption(object):
    def __init__(self, s0, strike, maturity, rf, volatility, barrier, minsteps=20):
        """
        Initialize
        :param s0: Initial price of underlying asset
        :param strike: Strike price
        :param maturity: Maturity in years
        :param rf: Risk free rate (per annum)
        :param volatility: expressed per annum
        :param barrier: Barrier for this knock-in option
        :param minsteps: Minimum number of time steps
        """
        self.s0 = s0
        self.strike = strike
        self.barrier = barrier
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
        self.price = None
        self.hitProb = self.calcBarrierHitProb()

    def calcBarrierHitProb(self):
        # calculate probability for t=0
        hitprob = np.zeros((2*self.ntime, self.ntime), dtype=np.float32)
        price = np.full(self.ntime*2, self.up, dtype=np.float32)
        price[0] = self.s0 * (self.down ** self.ntime)
        price = np.cumprod(price)
        self.price = price

        hitprob[:, -1] = np.where(price >= self.barrier, 1.0, 0.0)

        # for t = 1, 2, ... ntime-1
        for j in range(self.ntime-2, -1, -1):
            for i in range(self.ntime-j, self.ntime+j+1):
                if price[i] >= self.barrier:
                    hitprob[i, j] = 1.0
                else:
                    hitprob[i, j] = self.pUp * hitprob[i+1, j+1] + self.pDown * hitprob[i-1, j+1]
        return hitprob

    def evaluate(self):
        # values at time T
        grid = self.grid
        val = self.s0 * np.exp(-volatility * self.sqrtTime * self.ntime)
        for i in range(2*self.ntime):
            grid[i, -1] = self.hitProb[i, -1] * max(val - self.strike, 0)
            val *= self.up

        for j in range(self.ntime-1, 0, -1):
            for i in range(self.ntime-j, self.ntime+j, 1):
                val1 = 0
                if self.hitProb[i+1, j] > 0:
                    val1 = grid[i+1, j]/self.hitProb[i+1, j]
                val2 = 0
                if self.hitProb[i-1, j] > 0:
                    val2 = grid[i-1, j]/self.hitProb[i-1, j]
                grid[i, j-1] = self.df * self.hitProb[i, j-1] * (self.pUp * val1 + self.pDown * val2)

        return grid[self.ntime, 0]

    def calculateDeltaT(self):
        val = self.vol / (self.rf + self.vol*self.vol/2.0)
        return val*val

    def plotPrice(self):
        price = self.price
        time = np.full(self.ntime, self.deltaT, dtype=np.float32)
        time[0] = 0
        time = np.cumsum(time)
        x, y = np.meshgrid(price, time)
        fig = plt.figure()
        axs = fig.add_subplot(111, projection='3d')
        axs.plot_surface(x.T, y.T, self.grid)
        axs.set_xlabel('Stock Price')
        axs.set_ylabel('Time (Yrs)')
        axs.set_zlabel('Option Price')
        plt.show()

        fig, axs = plt.subplots(1, 1, constrained_layout=True)
        cs = axs.contourf(x.T, y.T, self.grid)
        fig.colorbar(cs, ax=axs, shrink=0.85)
        axs.set_title("European Barrier Knock-In Call Option")
        axs.set_ylabel("Time to Maturity (yrs)")
        axs.set_xlabel("Initial Stock Price")
        axs.locator_params(nbins=5)
        axs.clabel(cs, fmt="%1.1f", inline=True, fontsize=10, colors='w')
        plt.show()


if __name__ == "__main__":
    price = 20.0
    strike = 22.0
    maturity = 2.0/12.0
    barrier = 23.0
    volatility = 0.2
    rf = 0.005
    eoption = EuropeanKnockInCallOption(price, strike, maturity, rf, volatility, barrier, minsteps=25)
    simPrice = eoption.evaluate()
    logger.info("simulated price: %f", simPrice)
    eoption.plotPrice()