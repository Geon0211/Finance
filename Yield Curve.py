import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred

fred = Fred(api_key = "12c323c0527099c4da382e8f51000276")

# # Get Bond data
# def data_bond():
#     # US 2 year Treasury
#     trea_2 = yf.download("^IRX", start = "2010-01-01", end = "2025-01-01")
#     # US 10 year Treasury
#     trea_10 = yf.download("^TNX", start = "2010-01-01", end = "2025-01-01")
#
#     # Only Close price
#     trea_2 = trea_2["Close"]
#     trea_10 = trea_10["Close"]
#
#     return trea_2, trea_10
#
# # Run data_bond function
# trea_2, trea_10 = data_bond()
#
#
# y = trea_10.values - trea_2.values
# plt.plot(trea_2.index, y, color = "b", label = "Difference")
# plt.plot(trea_2, color = 'r', label = "2 year")
# plt.plot(trea_10, color = "g", label = "10 year")
# plt.xlabel("Date")
# plt.legend(loc="upper left")
# plt.show()

# Use FRED API
tickers = {"1M":"GS1M",
           "3M":"GS3M",
           "6M":"GS6M",
           "1Y":"GS1",
           "2Y":"GS2",
           "3Y":"GS3",
           "5Y":"GS5",
           "7Y":"GS7",
           "10Y":"GS10",
           "20Y":"GS20",
           "30Y":"GS30"}

start = "2020-01-01"
end = "2025-01-01"

df = pd.DataFrame()

for maturity, ticker in tickers.items():
    data = fred.get_series(ticker, start, end)
    df[maturity] = data

df.dropna(how="all", inplace = True)
print(df)

################# Yield Curve
yield_curve = {}
for maturity, ticker in tickers.items():
    data = fred.get_series_latest_release(ticker)
    if data is not None and not data.empty:
        yield_curve[maturity] = data.iloc[-1]

# print(data)
# The latest date is 2025-03-01
plt.plot(list(yield_curve.keys()), list(yield_curve.values()))
plt.show()
