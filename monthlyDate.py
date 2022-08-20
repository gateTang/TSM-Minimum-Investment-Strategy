import pandas as pd
import datetime
import yfinance as yf

def monthlyIterator(top2Stocks):
    dataList = []
    for i in top2Stocks:
        yfObj = yf.Ticker(i)
        data = yfObj.history(start='2000-01-01', interval='1mo')
        #data = data['Close']
        data['Returns'] = data['Close'] / data['Close'].shift(1)
        newData = data[['Close','Returns']]
        dataList.append(newData)
    return dataList
    