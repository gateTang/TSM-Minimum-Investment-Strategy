import numpy as np
import pandas as pd

def SMA(data, period = 30, column = 'Close'):
    return data[column].rolling(window=period).mean()

def SMAstrategy (df):
    df = df.copy()
    buy = []
    sell = []
    flag = 0
    buy_price = 0
    
    for i in range (0, len(df)):
        if df['SMA'][i] > df['Close'][i] and flag == 0:
            buy.append(df['Close'][i])
            sell.append(np.nan)
            buy_price = df['Close'][i]
            flag = 1
        elif df['SMA'][i] < df['Close'][i] and flag == 1 and buy_price < df['Close'][i]:
            buy.append(np.nan)
            sell.append(df['Close'][i])
            buy_price = 0
            flag = 0
        else:
            buy.append(np.nan)
            sell.append(np.nan)
    return (buy, sell)