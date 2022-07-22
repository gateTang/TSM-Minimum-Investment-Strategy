import yfinance as yf
import numpy as np

def dailyReturn(stock):
    stockReturn = []
    for i in range(0,len(stock)-1):
        prevValue = stock.iat[i,0]
        currentValue = stock.iat[i+1,0]
        return1 = (currentValue/prevValue) - 1
        stockReturn.append(return1)
    stockReturn.insert(0, np.nan)
    stock['Daily Returns'] = stockReturn
    print(stock)
    return