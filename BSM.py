import yfinance as yf
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd


# Get real financial data using yfinance
def fetch_data(tickers):
    ticker = yf.Ticker(tickers)
    option_dates = ticker.options
    option_data = ticker.option_chain(option_dates[0])
    data_hist = ticker.history(start = "2023-01-31", end = "2023-12-31")
    return option_data.calls, option_data.puts, data_hist

jpm_call, jpm_put, jpm_data = fetch_data("JPM")

jpm_call.to_csv("jpm_call.csv", index = False)
jpm_put.to_csv("jpm_put.csv", index = False)
jpm_data.to_csv("jpm_data.csv")


######################################################################
#jpm_call = pd.DataFrame(jpm_call)
#print(jpm_call.columns)
# jpm_call has 14 columns
# 'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
# 'change', 'percentChange', 'volume', 'openInterest',
# 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency'
######################################################################

# Historical Stock Price Change for JPM
#plt.plot(jpm_data["Close"])
#plt.show()

# BSM
class BSM:
    def __init__(self, S, K, T, r, sigma):
        self.S = S # Underlying price
        self.K = K # Option Strike price
        self.T = T # Time to maturity
        self.r = r # Risk-free rate
        self.sigma = sigma # Volatility

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + ((self.sigma**2) / 2) * self.T)) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - (self.sigma * np.sqrt(self.T))

    def call(self): # Let assume mean and standard deviation of d1 are 0 and 1
        return (self.S * si.norm.cdf(self.d1(), 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0))

    def put(self):
        return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0) - self.S * si.norm.cdf(-self.d1(), 0.0, 1.0))
bsm = BSM(S = 100, K = 100, T = 1, r = 0.05, sigma = 0.2)
print(bsm.call())
print(bsm.put())


# Calculate JPM historical volatility
def hist_vol(data, count):
    log_return = np.log(data["Close"] / data["Close"].shift(1))
    vol = np.sqrt(count) * log_return.std()
    return vol

print(hist_vol(jpm_data, len(jpm_data)))


# Greeks
class BSMGreeks(BSM):
    def delta_call(self):
        return si.norm.cdf(self.d1(), 0.0, 1.0)

    def delta_put(self):
        return -si.norm.cdf(-self.d2(), 0.0, 1.0)

    def gamma(self):
        return si.norm.pdf(self.d1(), 0.0, 1.0) / (self.S * self.sigma * np.sqrt(self.T))

    def theta_call(self):
        return (-self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * self.sigma / (2 * np.sqrt(self.T)) -
                self.r * self.K * np.exp(-self.r * self.T) * si.norm.pdf(self.d1(), 0.0, 1.0))

    def theta_put(self):
        return (-self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * self.sigma / (2 * np.sqrt(self.T)) +
                self.r * self.K * np.exp(-self.r * self.T) * si.norm.pdf(-self.d1(), 0.0, 1.0))

    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 0.1)

    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0)

greek = BSMGreeks(S = 100, K = 100, T = 1, r = 0.05, sigma = 0.2)
print(greek.rho_call())
print(greek.rho_put())


stock_price = np.linspace(80, 120, 100)
time_change = np.linspace(0.00001, 1, 100)
interest_change = np.linspace(0, 0.1, 100)
strike_change = np.linspace(80, 120, 100)

# Delta plot
## Underlying
deltas_call_S = [BSMGreeks(S = price, K = 100, T = 1, r = 0.05, sigma = 0.2).delta_call() for price in stock_price]
deltas_put_S = [BSMGreeks(S = price, K = 100, T = 1, r = 0.05, sigma = 0.2).delta_put() for price in stock_price]

plt.plot(stock_price, deltas_call_S, label = "Call", color = "blue")
plt.plot(stock_price, deltas_put_S, label = "Put", color = "red")
plt.title("Delta vs. Underlying")
plt.xlabel("Price")
plt.ylabel("Delta")
plt.legend()
plt.show()

## Strike
deltas_call_K = [BSMGreeks(S = 100, K = change, T = 1, r = 0.05, sigma = 0.2).delta_call() for change in strike_change]
deltas_put_K = [BSMGreeks(S = 100, K = change, T = 1, r = 0.05, sigma = 0.2).delta_put() for change in strike_change]

plt.plot(strike_change, deltas_call_K, label = "Call", color = "blue")
plt.plot(strike_change, deltas_put_K, label = "Put", color = "red")
plt.title("Delta vs. Strike")
plt.xlabel("Strike")
plt.ylabel("Delta")
plt.legend()
plt.show()

# Theta Plot
theta_call_T = [BSMGreeks(S = 100, K = 100, T = change, r = 0.05, sigma = 0.2).theta_call() for change in time_change]
theta_put_T = [BSMGreeks(S = 100, K = 100, T = change, r = 0.05, sigma = 0.2).theta_put() for change in time_change]

plt.plot(time_change, theta_call_T, label = "Call", color = "blue")
plt.plot(time_change, theta_put_T, label = "Put", color = "red")
plt.title("Theta vs. Time")
plt.xlabel("Time")
plt.ylabel("Theta")
plt.legend()
plt.show()

