# Description: A practice on building an efficeint frontier for two and three assets.
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize

from sympy.solvers import solve
import sympy
from sympy import Symbol

import itertools

from dailyReturn import dailyReturn
from UI import stockAmt, portfolioSelect, stockUnzip
from expReturn import expectedReturn
from stats import stdDev, mean, standDev, correlCo
from portfolioStats import portfolioExpReturn, portfolioStdDev
from plotting import globalMin, sharpeRatio
from TSM import getTicker, logReturnList, newTSMAlgo

portfolio = []
heading = []
stockAmt = stockAmt()
stockList = []
#print(stockAmt)
for q in range(stockAmt):
    if len(portfolioSelect(portfolio, stockList)[0]) == stockAmt:
        break

for s in range(stockAmt):
    heading.append(s)

#Create dictionary for organizing many stocks
stock_dict = dict(zip(heading, portfolio))
#print(stock_dict)

combinations = list(itertools.combinations(range(stockAmt), 2))
stockListCombinations = list(itertools.combinations(stockList, 2))
stockCombinations = stockUnzip(stock_dict, combinations)

print(stockListCombinations)
#To identify what is the corresponding percentages for the ideal portfolio.
#for c in range(len(stockCombinations)):

expReturnList = []
for j in range (len(stockCombinations)):
        expAppendList1 = expectedReturn(stockCombinations[j][0])
        expAppendList2 = expectedReturn(stockCombinations[j][1])
        expReturnList.append((expAppendList1, expAppendList2))
print(expReturnList)

stdReturnList = []
for j in range (len(stockCombinations)):
        stdAppendList1 = stdDev(stockCombinations[j][0])
        stdAppendList2 = stdDev(stockCombinations[j][1])
        stdReturnList.append((stdAppendList1, stdAppendList2))

correlList = []
for o in range(len(stockCombinations)):
    correlation = correlCo([x for x in stockCombinations[o][0]['Daily Returns'].tolist() if str(x) != 'nan'], [x for x in stockCombinations[o][1]['Daily Returns'].tolist() if str(x) != 'nan'])
    correlList.append(correlation)
print(correlList)

portfolioExpList = []
for t in range(len(stockCombinations)):
    portfolioExpAppend = portfolioExpReturn(stockCombinations[t][0], stockCombinations[t][1])
    portfolioExpList.append(portfolioExpAppend)

portfolioStdDevList = []
for r in range(len(stockCombinations)):
    portfolioStdDevAppend = portfolioStdDev(stockCombinations[r][0], stockCombinations[r][1], correlation)
    portfolioStdDevList.append(portfolioStdDevAppend)
portfolioStdDevList

coorList = []
perCoorList = []
ratioList = []
percentageList = []
stockPercentileList = []

#txt = "Best combination: "+ str(sharpeDict[maxRatio]) +  " " + str(percentageDict[sharpeDict[maxRatio]]) + "\n Worst combination is: "+ str(sharpeDict[minRatio]) +  " " + str(percentageDict[sharpeDict[minRatio]])
for q in range(len(stockCombinations)):
    coor = globalMin(stockCombinations[q][0], stockCombinations[q][1], plt, correlation, stockListCombinations, stockList, q)[0]
    coor_dict = globalMin(stockCombinations[q][0], stockCombinations[q][1], plt, correlation, stockListCombinations, stockList, q)[1]
    percentile_dict = globalMin(stockCombinations[q][0], stockCombinations[q][1], plt, correlation, stockListCombinations, stockList, q)[2]
    coorList.append(coor)
    perCoorList.append(coor_dict)
    stockPercentileList.append(percentile_dict)
    ratioList = sharpeRatio(coorList[q][0], coorList[q][1], ratioList)
    percentageList.append(coorList[q][2])
maxRatio = max(ratioList)
minRatio = min(ratioList)
sharpeDict = dict(zip(ratioList, stockListCombinations))
percentageDict = dict(zip(stockListCombinations, percentageList))
print("Best combination is: "+ str(sharpeDict[maxRatio]) +  " " + str(percentageDict[sharpeDict[maxRatio]]))
print("Worst combination is: "+ str(sharpeDict[minRatio]) +  " " + str(percentageDict[sharpeDict[minRatio]]))

#plt.grid()
#plt.title('Minimum Variance Frontier ' + str(stockList))
#plt.xlabel('Standard Deviation/Risk (%)', fontsize=11)
#plt.ylabel('Daily Return (%)', fontsize=11)
#plt.plot(portfolioStdDev(stockCombinations[q][0], stockCombinations[q][1], correlation), portfolioExpReturn(stockCombinations[q][0], stockCombinations[q][1]))
solExp = globalMin(stockCombinations[q][0], stockCombinations[q][1], plt, correlation, stockListCombinations, stockList, q)[3]
percentages = str(solExp[0]) + (stockListCombinations[q][0])  + " | "+ str(100 - solExp[0]) + str(stockListCombinations[q][1])
#plt.plot(coor[0], coor[1], marker='o', label= 'Global Min | ' + percentages)
#plt.rcParams.update({'font.': 11})
#plt.annotate(percentages , (coor[0], coor[1]), fontsize=7.5)
#plt.legend(bbox_to_anchor=(1.05, 1.0),loc='upper left')
#plt.show()

#____________________________________________________________________________________

data = []

for item in range(len(perCoorList)):
    try:
        bestCoor = perCoorList[item][percentageDict[sharpeDict[maxRatio]]]
    except KeyError:
        pass
for h in range(len(stockPercentileList)):
    try:
        data = stockPercentileList[h][str(percentageDict[sharpeDict[maxRatio]])]
    except KeyError:
        pass
print(str(percentageDict[sharpeDict[maxRatio]]))
print(data)

#________________________________________________________________________________________

stk1 = str(data[0].values())[:-3:][14::]
stk2 = str(data[1].values())[:-3:][14::]

finalStockList = [stk1, stk2]

dataList = []
dataLog = getTicker(finalStockList, dataList)

returnList = []
returns = logReturnList(dataLog, returnList)[0]

performanceList = []
performance = newTSMAlgo(returnList, performanceList, returns, period=1, short=False)
years = (performance[0].index.max() - performance[0].index.min()).days / 365
perf_cum = [np.exp(performance[0].cumsum()), np.exp(performance[1].cumsum())]

percent1 = (int(str(data[0])[1:3])/100)
percent2 = (int(str(data[1])[1:3])/100)

tot = (perf_cum[0][-1]*percent1 + perf_cum[1][-1]*percent2) - 1
ann = (perf_cum[0][-1]*percent1 + perf_cum[1][-1]*percent2) ** (1 / years) - 1
vol = bestCoor[0]
#Continue to replicate the cell below.

rfr = 0.7/365
sharpe = (ann - rfr) / vol
print(f"1-day TSM Strategy yields:" +
      f"\n\t{tot*100:.2f}% total returns" + 
      f"\n\t{ann*100:.2f}% annual returns" +
      f"\n\t{sharpe:.2f} Sharpe Ratio")

stock_ret1 = np.exp(returnList[0].cumsum())
stock_ret2 = np.exp(returnList[1].cumsum())
b_tot = (stock_ret1[-1]*percent1 + stock_ret2[-1]*percent2) - 1
b_ann = (stock_ret1[-1]*percent1 + stock_ret2[-1]*percent2) ** (1 / years) - 1
b_vol = bestCoor[0] * np.sqrt(252)
b_sharpe = (b_ann - rfr) / b_vol
print(f"Baseline Buy-and-Hold Strategy yields:" + 
      f"\n\t{b_tot*100:.2f}% total returns" + 
      f"\n\t{b_ann*100:.2f}% annual returns" +
      f"\n\t{b_sharpe:.2f} Sharpe Ratio")

periods = [3, 5, 15, 30, 90]
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(4, 10)
minPlot = fig.add_subplot(gs[:2, 6:])
ax0 = fig.add_subplot(gs[:2, :6])
ax1 = fig.add_subplot(gs[2:, :5])
ax2 = fig.add_subplot(gs[2:, 5:])


for q in range(len(stockCombinations)):
    graphData = globalMin(stockCombinations[q][0], stockCombinations[q][1], minPlot, correlation, stockListCombinations, stockList, q)[4]
    minPlot.plot(graphData[1], graphData[2])
    minPlot.plot(graphData[5], graphData[6], marker='o', label= 'Global Min | ' + str(graphData[4]))
    minPlot.annotate(graphData[4] , graphData[3], fontsize=7.5)

minPlot.grid()
minPlot.set_title('Minimum Variance Frontier ' + graphData[0], fontsize=11)
minPlot.set_ylabel('Daily Returns (%)')
minPlot.set_xlabel('Standard Deviation/Risk (%)')

ax0.plot((np.exp(returns.cumsum()) - 1), label='B&H', linestyle='-')
perf_dict = {'tot_ret': {'buy_and_hold': (np.exp((returns[0].sum()*percent1 + returns[1].sum()*percent2) - 1))}}
perf_dict['ann_ret'] = {'buy_and_hold': b_ann}
perf_dict['sharpe'] = {'buy_and_hold': b_sharpe}
for p in periods:
    log_perf = newTSMAlgo(returnList, performanceList, returns, period=p, short=False)[0]*percent1 + newTSMAlgo(returnList, performanceList, returns, period=p, short=False)[1]*percent2
    perf = np.exp(log_perf.cumsum()) #Exponentialize the log returns.
    perf_dict['tot_ret'][p] = (perf[-1] - 1) # Adding each period to the nested dictionary.
    ann = (perf[-1] ** (1/years) - 1)
    perf_dict['ann_ret'][p] = ann
    vol = log_perf.std() * np.sqrt(252) #Volatility formula.
    perf_dict['sharpe'][p] = (ann - rfr) / vol #Sharpe Ratio formula.
    ax0.plot((perf - 1) * 100, label=f'{p}-Day Mean') #Plot first graph.
ax0.set_ylabel('Returns (%)')
ax0.set_xlabel('Date')
ax0.set_title('Cumulative Returns ' + str(percentageDict[sharpeDict[maxRatio]]))
ax0.grid()
ax0.legend()

print(perf_dict)
_ = [ax1.bar(i, v * 100) for i, v in enumerate(perf_dict['ann_ret'].values())]
ax1.set_xticks([i for i, k in enumerate(perf_dict['ann_ret'])])
ax1.set_xticklabels([f'{k}-Day Mean' 
    if type(k) is int else 'B&H' for 
    k in perf_dict['ann_ret'].keys()],
    rotation=45)
ax1.grid()
ax1.set_ylabel('Returns (%)')
ax1.set_xlabel('Strategy')
ax1.set_title('Annual Returns ' + str(percentageDict[sharpeDict[maxRatio]]))

_ = [ax2.bar(i, v) for i, v in enumerate(perf_dict['sharpe'].values())]
ax2.set_xticks([i for i, k in enumerate(perf_dict['sharpe'])])
ax2.set_xticklabels([f'{k}-Day Mean' 
    if type(k) is int else 'B&H' for 
    k in perf_dict['sharpe'].keys()],
    rotation=45)
ax2.grid()
ax2.set_ylabel('Sharpe Ratio')
ax2.set_xlabel('Strategy')
ax2.set_title('Sharpe Ratio ' + str(percentageDict[sharpeDict[maxRatio]]))
plt.tight_layout()
plt.show()