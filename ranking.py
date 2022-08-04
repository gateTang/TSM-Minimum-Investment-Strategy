from multiprocessing.sharedctypes import Value
import pandas as pd
import datetime
import yfinance as yf

def lookBackRank(stockList, lookBackPeriod=30):
    totRetDict = {}
    topStocks = []
    topStats = []
    if lookBackPeriod == None:
        lookBackPeriod = 30
    for i in stockList:
        tod = datetime.datetime.now()
        print(lookBackPeriod)
        d = datetime.timedelta(days = lookBackPeriod)
        LBDate = tod - d
        adjDate = LBDate.strftime("%Y-%m-%d")

        yfObj = yf.Ticker(i)
        data = yfObj.history(start=adjDate)

        data['Expected Return'] = (data['Close'].shift(1)/data['Close'])-1
        totDailyRet = data['Expected Return'].cumsum()[-1]
        print(data)
        totRetDict.update({i:totDailyRet})

        topStocks  = sorted(totRetDict, key=totRetDict.get, reverse=True)[:3]
        top2Stocks  = sorted(totRetDict, key=totRetDict.get, reverse=True)[:2]
        
    return topStocks, top2Stocks