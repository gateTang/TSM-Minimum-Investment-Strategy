import pandas as pd
import datetime
import yfinance as yf

def tickerLoad(top2Stocks, lookBackPeriod=30):
    stockList =[]
    if lookBackPeriod == None:
        lookBackPeriod = 30
    for i in top2Stocks:
        tod = datetime.datetime.now()
        print(lookBackPeriod)
        d = datetime.timedelta(days = lookBackPeriod)
        LBDate = tod - d
        adjDate = LBDate.strftime("%Y-%m-%d")

        yfObj = yf.Ticker(i)
        data = yfObj.history(start=adjDate)

        data['Expected Return'] = (data['Close'].shift(1)/data['Close'])-1
        data = data.shift(-1)
        stockList.append(data)
        return stockList
def equityCurveDf(df, returnPercentage):
    percentage = list(returnPercentage.keys())
    data = list(returnPercentage.values())
    
    df = df.shift(-1)
    retPerList = list(df['Expected Return'])
    print("retPerList is: " + str(retPerList))

    for u in range(len(df)):
        percentage.append(percentage[u]*(1+retPerList[u])) 
    df['Equity Curve'] = percentage[0:len(df)]
    return df