import math
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Value at Risk and Expected Shortfall
## Normal distribution with mean = 0 and std = 0.3 * (10000 / sqrt(250)) <- 250 trading dates
mu = 0
std = 0.3 * 10000 / math.sqrt(250)
alphas = [0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9999, 0.99999, 0.999999]
result_var_n = []
result_es_n = []
result_var_t = []
result_es_t = []
nu = 4
for i, a in enumerate(alphas):
    result_var_n.append(mu + std * st.norm.ppf(a))
    result_var_t.append(mu + std * st.t.ppf(a, nu))
    result_es_n.append(mu + std * st.norm.pdf(st.norm.ppf(a)) / (1-a))
    result_es_t.append(mu + std * (st.t.pdf(st.t.ppf(a, nu), nu) / (1-a)) * ((nu + st.t.ppf(a, nu) ** 2) / (nu-1)))

plt.plot(alphas, result_var_n, label = r"$VaR_{Normal}$")
plt.plot(alphas, result_var_t, label = r"$VaR_{t_4}$")
plt.plot(alphas, result_es_n, label = r"$ES_{Normal}$")
plt.plot(alphas, result_es_t, label = r"$ES_{t_4}$")
plt.legend()
plt.show()

r_n = []
r_t = []
for i in range(0,len(alphas)):
    r_n.append(result_es_n[i] / result_var_n[i])# Ratio ES to VaR
    r_t.append(result_es_t[i] / result_var_t[i]) # Ratio ES to VaR

plt.plot(alphas, r_n, label = r"$Ratio_{Normal}$")
plt.plot(alphas, r_t, label = r"$Ratio_{t_4}$")
plt.legend()
plt.show()


# Use real dataset (SP500 from Ecdat in R)
sp500 = pd.read_csv("x.csv")
balance = 1000000 # Amount I'm holding
r = np.cumprod(1 + sp500 / 100) # Get rid of percentage by dividing 100
loss = -r.iloc[1:].values + r.iloc[:-1].values
loss = loss.flatten() # Convert to 1-Dim
loss_sorted = np.sort(loss)

plt.hist(loss_sorted, bins = 200)
plt.show()

# get cdf and calculate VaR and ES
loss_cdf = np.arange(1, len(loss_sorted) + 1) / len(loss_sorted)
alpha = 0.95
index = np.argmax(loss_cdf >= alpha)
VaR = balance * loss_sorted[index]
ES = balance * loss_sorted[index:].mean()
print(VaR)
print(ES)






# Correlation
# Copula
# Collateralized Debt Obligation

