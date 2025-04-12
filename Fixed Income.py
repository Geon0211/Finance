import yfinance as yf
import pandas as pd
from fredapi import Fred



# Calculate Bond Present Price
face = 1000
coupon = 0.04
interest = 0.05
time = 3
period = 1

def pv(face, coupon, interest, time, period):
    current = 0
    for t in range(1,time+1):
        if t != 3:
            x = (face*coupon) / (1+interest)**(t/period)
            current = current + x
        else:
            x = (face*coupon + face) / (1+interest)**(t/period)
            current = current + x
    return current
print("Current Price is ${:.2f}".format(pv(face, coupon, interest, time, period)))

cur = pv(face, coupon, interest, time, period)
# Yield to Maturity (YTM)
# YTM is the total return an investor can expect from a bond if held until it matures
def ytm_cal(face, coupon, interest, time, period):
    cur = pv(face, coupon, interest, time, period)
    ytm = ((face*coupon) + ((face - cur)/time)) / ((face + cur) / 2)
    return ytm
print("YTM is", ytm_cal(face,coupon, interest, time, period))

# Duration
# Duration is the sensitivity of change in interest rate

# Ex. Newly issued 3 year 4% annual-pay bond with a yield to maturity of 5%. Discounted at 5%.
def duration(coupon, T, m, r):
    duration = 0
    PV = pv(face, coupon, r, T, m)
    ytm = ytm_cal(face, coupon, r, T, m)
    for t in range(1,T+1):
        if t != T:
            dur = ((coupon * face) / (1 + ytm)**t) * (t / PV)
            duration += dur
        else:
            dur = ((coupon * face + face) / (1 + ytm)**t) * (t / PV)
            duration += dur

    return duration

duration = duration(coupon = 0.04, T = 3, m = 1, r = 0.05)
print("Duration when coupon {}, maturity {}, interest rate {}, is {}".format(0.04, 3, 0.05, duration))

# Convexity
# The relationship between bond prices and bond yields
# The curvature in the relationship between bond prices and interest rates
# It reflects the rate at which duration of a bond changes as interest rates change
# The second derivative of bond price to yield

def convexity(coupon, T, m, r, dy):
    ytm = ytm_cal(face,coupon, r, T, m)
    ytm_p = ytm + dy
    ytm_m = ytm - dy
    pv_p = pv(face, coupon, ytm_p, T, m)
    pv_m = pv(face, coupon, ytm_m, T, m)
    PV = pv(face, coupon, ytm, T, m)
    convexity = (pv_m + pv_p - 2*PV) / ((dy**2) * PV)
    return convexity

print(convexity(coupon = 0.04, T = 3, m = 1, r = 0.05, dy = 0.01))



