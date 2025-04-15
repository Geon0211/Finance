import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import rankdata
from scipy.stats import norm
import matplotlib.pyplot as plt
from copulas.multivariate import GaussianMultivariate


# OHLCV is the data set of Bitcoin, Ethereum, Bitcoin Cash, and Litecoin from 12/20/2017 to 06/20/2018
data = pd.read_csv("OHLCV.csv")
need = ["BTC_close","ETH_close","BCH_close","LTC_close"]
close_data = data[need]

# Daily return
d_return = close_data.pct_change().dropna()

mean = d_return.mean().values
std = d_return.std().values
corr = d_return.corr().values

# plt.hist(d_return["BTC_close"], bins = 30)
# plt.show()
#
# plt.hist(d_return["ETH_close"], bins = 30)
# plt.show()
#
# plt.hist(d_return["BCH_close"], bins = 30)
# plt.show()
#
# plt.hist(d_return["LTC_close"], bins = 30)
# plt.show()


# Each cryptocurrency's distribution seems normally distributed but hard to define
# -> Transform to uniform marginal
uni = d_return.apply(lambda x: rankdata(x) / (len(x) + 1), axis = 0)

copula = GaussianMultivariate()
copula.fit(uni)
sample = copula.sample(100000)
cv = norm.ppf(sample) # Inverse CDF
w = [0.25, 0.25, 0.25, 0.25] # Equally weighted
port_return = cv @ w
var = -np.percentile(port_return, (1-0.95)*100)
es = -port_return[port_return <= -var].mean()
print("The VaR_{}% is {:.4f}".format(95, var))
print("The ES_{}% is {:.4f}".format(95, es))
