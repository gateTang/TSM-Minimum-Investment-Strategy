import yfinance as yf
import numpy as np


def getTicker(stockList, dataList):
    for i in stockList:
        yfObj = yf.Ticker(i)
        data = yfObj.history(start='2000-01-01', end='2022-12-31')
        dataList.append(data)
    return dataList

def logReturnList(dataList, returnList):
    for i in dataList:
        data = i
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        returnList.append(returns)
    return returnList

def newTSMAlgo(returnList, performanceList, returns, period=1, short=False):
    if short:
        position1 = returnList[0].rolling(period).mean().map(
            lambda x: -1 if x <= 0 else 1)
        position2 = returnList[1].rolling(period).mean().map(
            lambda x: -1 if x <= 0 else 1)
        performanceList.append(position1, position2)
    else:
        position1 = returnList[0].rolling(period).mean().map(
            lambda x: 0 if x <= 0 else 1)
        position2 = returnList[1].rolling(period).mean().map(
            lambda x: 0 if x <= 0 else 1)
    performance1 = position1.shift(1) * returns
    performance2 = position2.shift(1) * returns
    return [performance1, performance2]