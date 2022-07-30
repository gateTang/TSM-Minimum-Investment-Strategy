# Description: A practice on building an efficeint frontier for two and three assets.
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize
from flask import Flask, request, jsonify, render_template, send_file

from sympy.solvers import solve
import sympy
from sympy import Symbol

import itertools

import io
import urllib, base64

from dailyReturn import dailyReturn
from UI import stockAmt, portfolioSelect, stockUnzip, newPortfolioSelect
from expReturn import expectedReturn
from stats import stdDev, mean, standDev, correlCo
from portfolioStats import portfolioExpReturn, portfolioStdDev
from plotting import globalMin, sharpeRatio
from TSM import getTicker, logReturnList, newTSMAlgo
from SMA import SMA, SMAstrategy
from ranking import lookBackRank

#--- FLASK WEBFRAME ------
app = Flask(__name__)

stockListed = []

@app.route('/', methods = ['POST', 'GET'])
def stockInput():
    resetState = 0
    topStocks = []
    lookBackPeriod = request.form.get('lookBackPeriod')
    
    if request.method == 'POST':
        stockNo = request.form.get('amount')
        #Requests for information in the <form> sent by home.html
        submitState = request.form.get('submitState')
        resetState = request.form.get('resetState')
        stockListed.append(stockNo)

        if None in stockListed:
            stockListed.remove(None)

        #print(stockListed) # for debugging
        #print(submitState)
        #print(resetState)
        if submitState =='1':
            confirmedList = stockListed
            submitState = 0
            #confirmedList.pop(-1)
            #print(confirmedList)
            #---D - Line 44 - 344
            portfolio = []
            heading = []
            #stockAmt = len(confirmedList)
            stockList = confirmedList
            #print(stockAmt)
            topStocks = lookBackRank(stockList, lookBackPeriod)
            print(topStocks)

            stockList = topStocks

            stockAmt = len(topStocks)
            for q in range(stockAmt):
                if len(newPortfolioSelect(portfolio, stockList)[0]) == stockAmt:
                    break

            heading = np.arange(0,stockAmt)
            # for s in range(stockAmt):
            #     heading.append(s)

            #Create dictionary for organizing many stocks
            stock_dict = dict(zip(heading, portfolio))
            #print(stock_dict)

            combinations = list(itertools.combinations(range(stockAmt), 2))
            #print(portfolio)
            stockListCombinations = list(itertools.combinations(stockList, 2))
            stockCombinations = stockUnzip(stock_dict, combinations)

            #print(stockListCombinations)
            #To identify what is the corresponding percentages for the ideal portfolio.
            #for c in range(len(stockCombinations)):

            expReturnList = []
            for j in range (len(stockCombinations)):
                    expAppendList1 = expectedReturn(stockCombinations[j][0])
                    expAppendList2 = expectedReturn(stockCombinations[j][1])
                    expReturnList.append((expAppendList1, expAppendList2))
            #print(expReturnList)

            stdReturnList = []
            for j in range (len(stockCombinations)):
                    stdAppendList1 = stdDev(stockCombinations[j][0])
                    stdAppendList2 = stdDev(stockCombinations[j][1])
                    stdReturnList.append((stdAppendList1, stdAppendList2))
            correlList = []
            firstList = []
            secondList = []
            correlList = []
            for o in range(len(stockCombinations)):
                combinationsDf = pd.concat([stockCombinations[o][0],stockCombinations[o][1]], axis=1)
                dropDf = combinationsDf.dropna(subset=['Daily Returns'])
                firstDf = dropDf.iloc[:,[0, 3]]
                print('FirstDF = '+str(firstDf))
                firstList = firstDf['Daily Returns'].tolist()
                secondDf = dropDf.iloc[:,[0, 3]]
                secondList = secondDf['Daily Returns'].tolist()
                #correlation = correlCo([x for x in stockCombinations[o][0]['Daily Returns'].tolist() if str(x) != 'nan'], [x for x in stockCombinations[o][1]['Daily Returns'].tolist() if str(x) != 'nan'])
                correlation = correlCo(firstList, secondList)
                correlList.append(correlation)
            #print(correlList)

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
            #print("Best combination is: "+ str(sharpeDict[maxRatio]) +  " " + str(percentageDict[sharpeDict[maxRatio]]))
            #print("Worst combination is: "+ str(sharpeDict[minRatio]) +  " " + str(percentageDict[sharpeDict[minRatio]]))

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
            #print(str(percentageDict[sharpeDict[maxRatio]]))
            #print(data)

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
            print('Years = ' + str(years))
            perf_cum = [np.exp(performance[0].cumsum()), np.exp(performance[1].cumsum())]

            percent1 = (int(str(data[0])[1:3])/100)
            percent2 = (int(str(data[1])[1:3])/100)

            tot = (perf_cum[0][-1]*percent1 + perf_cum[1][-1]*percent2) - 1
            ann = (perf_cum[0][-1]*percent1 + perf_cum[1][-1]*percent2) ** (1 / years) - 1
            vol = bestCoor[0]
            #Continue to replicate the cell below.

            rfr = 0.7/365
            sharpe = (ann - rfr) / vol
            # print(f"1-day TSM Strategy yields:" +
            #     f"\n\t{tot*100:.2f}% total returns" + 
            #     f"\n\t{ann*100:.2f}% annual returns" +
            #     f"\n\t{sharpe:.2f} Sharpe Ratio")

            stock_ret1 = np.exp(returnList[0].cumsum())
            stock_ret2 = np.exp(returnList[1].cumsum())
            b_tot = (stock_ret1[-1]*percent1 + stock_ret2[-1]*percent2) - 1
            b_ann = (stock_ret1[-1]*percent1 + stock_ret2[-1]*percent2) ** (1 / years) - 1
            b_vol = bestCoor[0]*100 * np.sqrt(252)
            b_sharpe = (b_ann - rfr) / b_vol
            print(f"\n\t{b_ann:.2f} b_ann" + 
                f"\n\t{b_vol:.2f}% b_vol" +
                f"\n\t{b_sharpe:.2f} b_sharpe")
            print(bestCoor)

            periods = [3, 5, 15, 30, 90, 180, 365]
            fig = plt.figure(figsize=(21, 12))
            gs = fig.add_gridspec(6, 10)
            minPlot = fig.add_subplot(gs[:2, :4])
            ax0 = fig.add_subplot(gs[:2, 4:])
            ax1 = fig.add_subplot(gs[2:4, :5])
            ax2 = fig.add_subplot(gs[2:4, 5:])
            ax3 = fig.add_subplot(gs[4:,:5])
            ax4 = fig.add_subplot(gs[4:, 5:])


            for q in range(len(stockCombinations)):
                graphData = globalMin(stockCombinations[q][0], stockCombinations[q][1], minPlot, correlation, stockListCombinations, stockList, q)[4]
                minPlot.plot(graphData[1], graphData[2])
                minPlot.plot(graphData[5], graphData[6], marker='o', label= 'Global Min | ' + str(graphData[4]))
                minPlot.annotate(graphData[4] , graphData[3], fontsize=7.5)

            minPlot.grid()
            minPlot.set_title('Fig 1. Minimum Variance Frontier ' + graphData[0], fontsize=11)
            minPlot.set_ylabel('Daily Returns (%)')
            minPlot.set_xlabel('Standard Deviation/Risk (%)')

            ax0.plot((np.exp(returns.cumsum()) - 1)*100, label='Simulation Lifespan', linestyle='-')
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
            ax0.set_title('Fig 2. Cumulative Returns | Best Pair: ' + str(percentageDict[sharpeDict[maxRatio]]))
            ax0.grid()
            ax0.legend()

            # print(perf_dict)
            _ = [ax1.bar(i, v * 100) for i, v in enumerate(perf_dict['ann_ret'].values())]
            ax1.set_xticks([i for i, k in enumerate(perf_dict['ann_ret'])])
            ax1.set_xticklabels([f'{k}-Day Mean' 
                if type(k) is int else 'Simulation Lifespan' for 
                k in perf_dict['ann_ret'].keys()],
                rotation=45)
            ax1.grid()
            ax1.set_ylabel('Returns (%)')
            ax1.set_xlabel('Strategy')
            ax1.set_title('Fig 3. Annual Returns | Best Pair: ' + str(percentageDict[sharpeDict[maxRatio]]))

            #del perf_dict['sharpe']['buy_and_hold']
            _ = [ax2.bar(i, v) for i, v in enumerate(perf_dict['sharpe'].values())]
            print('Perf Dict: '+str(perf_dict['sharpe']))
            ax2.set_xticks([i for i, k in enumerate(perf_dict['sharpe'])])
            ax2.set_xticklabels([f'{k}-Day Mean' 
                if type(k) is int else 'Simulation Lifespan' for 
                k in perf_dict['sharpe'].keys()],
                rotation=45)
            ax2.grid()
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_xlabel('Strategy')
            ax2.set_title('Fig 4. Sharpe Ratio | Best Pair: ' + str(percentageDict[sharpeDict[maxRatio]]))
            #plt.tight_layout()
            #plt.show()

            #------------------- FOR SMA STRATEGY --------------------------

            ann_retDict = perf_dict['ann_ret'].copy()
            sharpe_Dict = perf_dict['sharpe'].copy()

            del ann_retDict['buy_and_hold']
            del sharpe_Dict['buy_and_hold']

            #print(ann_retDict)
            #print(sharpe_Dict)

            combinedDict = {k: ann_retDict[k]*sharpe_Dict[k] for k in ann_retDict}
            maxValue = max(combinedDict, key=combinedDict.get)

            dataLog[0]['SMA120'] = SMA(dataLog[0].iloc[-500:], period=120)
            dataLog[0]['SMAMax'] = SMA(dataLog[0].iloc[-500:], period=maxValue)
            #dataLog[1]['SMA'] = SMA(dataLog[1])

            strat1 = SMAstrategy(dataLog[0])
            dataLog[0]['Buy'] = strat1[0]
            dataLog[0]['Sell'] = strat1[1]

            #ax2.figure(figsize=(16,8))
            ax3.set_title('Fig 5. SMA of '+ str(percentageDict[sharpeDict[maxRatio]].split(' | ')[0]))
            ax3.plot(dataLog[0].iloc[-500:]['Close'], label='Close Price')
            ax3.plot(dataLog[0].iloc[-500:]['SMA120'], label='SMA120')
            ax3.plot(dataLog[0].iloc[-500:]['SMAMax'], label='SMA'+str(maxValue))
            ax3.scatter(dataLog[0].index, dataLog[0]['Buy'], color='green', label='Buy Signal', marker='^')
            ax3.scatter(dataLog[0].index, dataLog[0]['Sell'], color='red', label='Sell Signal', marker='v')
            ax3.legend()
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Close Price in USD')


            #plt.plot(dataLog[0]['SMA'])
            #plt.plot(dataLog[0]['Close'])
            coorText = []
            idx20 = np.argwhere(np.diff(np.sign(dataLog[0]['SMAMax'].values - dataLog[0]['SMA120'].values))).flatten()
            ax3.scatter(dataLog[0]['SMAMax'].index[idx20], dataLog[0]['SMA120'].values[idx20], color='red')
            intersectCoor = pd.DataFrame(dataLog[0]['SMA120'].values[idx20], dataLog[0]['SMAMax'].index[idx20], ['Coordinates'])
            # print(intersectCoor.index)
            for l in range(len(intersectCoor)):
                coorIndex = str(intersectCoor.index[l])[0:10]
                coorY = str(dataLog[0]['SMA120'].values[idx20][l])[0:5]
                coorText.append('('+coorIndex + ',' + coorY + ')')
                ax3.annotate(coorText[l], (intersectCoor.index[l], dataLog[0]['SMA120'].values[idx20][l]), fontsize=9)

            #----------------

            combinedDict2 = {k: ann_retDict[k]*sharpe_Dict[k] for k in ann_retDict}
            maxValue2 = max(combinedDict2, key=combinedDict2.get)

            dataLog[1]['SMA120'] = SMA(dataLog[1].iloc[-500:], period=120)
            dataLog[1]['SMAMax'] = SMA(dataLog[1].iloc[-500:], period=maxValue2)

            strat2 = SMAstrategy(dataLog[1])
            dataLog[1]['Buy'] = strat2[0]
            dataLog[1]['Sell'] = strat2[1]

            ax4.set_title('Fig 6. SMA of '+ str(percentageDict[sharpeDict[maxRatio]].split(' | ')[1]))
            ax4.plot(dataLog[1].iloc[-500:]['Close'], label='Close Price')
            ax4.plot(dataLog[1].iloc[-500:]['SMA120'], label='SMA120')
            ax4.plot(dataLog[1].iloc[-500:]['SMAMax'], label='SMA'+str(maxValue))
            ax4.scatter(dataLog[1].dropna(subset=['SMA120']).index, dataLog[1].dropna(subset=['SMA120'])['Buy'], color='green', label='Buy Signal', marker='^')
            ax4.scatter(dataLog[1].dropna(subset=['SMA120']).index, dataLog[1].dropna(subset=['SMA120'])['Sell'], color='red', label='Sell Signal', marker='v')
            ax4.legend()
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Close Price in USD')
            ax4.set_label('Close Price in USD')


            #plt.plot(dataLog[0]['SMA'])
            #plt.plot(dataLog[0]['Close'])
            coorText = []
            idx20 = np.argwhere(np.diff(np.sign(dataLog[1]['SMAMax'].values - dataLog[1]['SMA120'].values))).flatten()
            ax4.scatter(dataLog[1]['SMAMax'].index[idx20], dataLog[1]['SMA120'].values[idx20], color='red')
            intersectCoor = pd.DataFrame(dataLog[1]['SMA120'].values[idx20], dataLog[1]['SMAMax'].index[idx20], ['Coordinates'])
            # print(intersectCoor.index)
            for l in range(len(intersectCoor)):
                coorIndex = str(intersectCoor.index[l])[0:10]
                coorY = str(dataLog[1]['SMA120'].values[idx20][l])[0:5]
                coorText.append('('+coorIndex + ',' + coorY + ')')
                ax4.annotate(coorText[l], (intersectCoor.index[l], dataLog[1]['SMA120'].values[idx20][l]), fontsize=9)

            plt.tight_layout()
            #plt.show()

            img = io.BytesIO()
            fig.savefig(img)
            img.seek(0)

            return send_file(img, mimetype='img/png', attachment_filename='Top3_' + graphData[0]+".jpg")
            #---D 
        if resetState == '1':
            del stockListed[:]
            resetState = '0'
    return render_template("home.html", stockListed=stockListed, lookBackPeriod=lookBackPeriod, topStocks=topStocks)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

# portfolio = []
# heading = []
# stockAmt = stockAmt()
# stockList = []
# #print(stockAmt)
# for q in range(stockAmt):
#     if len(portfolioSelect(portfolio, stockList)[0]) == stockAmt:
#         break

# for s in range(stockAmt):
#     heading.append(s)

# #Create dictionary for organizing many stocks
# stock_dict = dict(zip(heading, portfolio))
# #print(stock_dict)

# combinations = list(itertools.combinations(range(stockAmt), 2))
# stockListCombinations = list(itertools.combinations(stockList, 2))
# stockCombinations = stockUnzip(stock_dict, combinations)

# print(stockListCombinations)
# #To identify what is the corresponding percentages for the ideal portfolio.
# #for c in range(len(stockCombinations)):

# expReturnList = []
# for j in range (len(stockCombinations)):
#         expAppendList1 = expectedReturn(stockCombinations[j][0])
#         expAppendList2 = expectedReturn(stockCombinations[j][1])
#         expReturnList.append((expAppendList1, expAppendList2))
# print(expReturnList)

# stdReturnList = []
# for j in range (len(stockCombinations)):
#         stdAppendList1 = stdDev(stockCombinations[j][0])
#         stdAppendList2 = stdDev(stockCombinations[j][1])
#         stdReturnList.append((stdAppendList1, stdAppendList2))

# correlList = []
# for o in range(len(stockCombinations)):
#     correlation = correlCo([x for x in stockCombinations[o][0]['Daily Returns'].tolist() if str(x) != 'nan'], [x for x in stockCombinations[o][1]['Daily Returns'].tolist() if str(x) != 'nan'])
#     correlList.append(correlation)
# print(correlList)

# portfolioExpList = []
# for t in range(len(stockCombinations)):
#     portfolioExpAppend = portfolioExpReturn(stockCombinations[t][0], stockCombinations[t][1])
#     portfolioExpList.append(portfolioExpAppend)

# portfolioStdDevList = []
# for r in range(len(stockCombinations)):
#     portfolioStdDevAppend = portfolioStdDev(stockCombinations[r][0], stockCombinations[r][1], correlation)
#     portfolioStdDevList.append(portfolioStdDevAppend)
# portfolioStdDevList

# coorList = []
# perCoorList = []
# ratioList = []
# percentageList = []
# stockPercentileList = []

# #txt = "Best combination: "+ str(sharpeDict[maxRatio]) +  " " + str(percentageDict[sharpeDict[maxRatio]]) + "\n Worst combination is: "+ str(sharpeDict[minRatio]) +  " " + str(percentageDict[sharpeDict[minRatio]])
# for q in range(len(stockCombinations)):
#     coor = globalMin(stockCombinations[q][0], stockCombinations[q][1], plt, correlation, stockListCombinations, stockList, q)[0]
#     coor_dict = globalMin(stockCombinations[q][0], stockCombinations[q][1], plt, correlation, stockListCombinations, stockList, q)[1]
#     percentile_dict = globalMin(stockCombinations[q][0], stockCombinations[q][1], plt, correlation, stockListCombinations, stockList, q)[2]
#     coorList.append(coor)
#     perCoorList.append(coor_dict)
#     stockPercentileList.append(percentile_dict)
#     ratioList = sharpeRatio(coorList[q][0], coorList[q][1], ratioList)
#     percentageList.append(coorList[q][2])
# maxRatio = max(ratioList)
# minRatio = min(ratioList)
# sharpeDict = dict(zip(ratioList, stockListCombinations))
# percentageDict = dict(zip(stockListCombinations, percentageList))
# print("Best combination is: "+ str(sharpeDict[maxRatio]) +  " " + str(percentageDict[sharpeDict[maxRatio]]))
# print("Worst combination is: "+ str(sharpeDict[minRatio]) +  " " + str(percentageDict[sharpeDict[minRatio]]))

# #plt.grid()
# #plt.title('Minimum Variance Frontier ' + str(stockList))
# #plt.xlabel('Standard Deviation/Risk (%)', fontsize=11)
# #plt.ylabel('Daily Return (%)', fontsize=11)
# #plt.plot(portfolioStdDev(stockCombinations[q][0], stockCombinations[q][1], correlation), portfolioExpReturn(stockCombinations[q][0], stockCombinations[q][1]))
# solExp = globalMin(stockCombinations[q][0], stockCombinations[q][1], plt, correlation, stockListCombinations, stockList, q)[3]
# percentages = str(solExp[0]) + (stockListCombinations[q][0])  + " | "+ str(100 - solExp[0]) + str(stockListCombinations[q][1])
# #plt.plot(coor[0], coor[1], marker='o', label= 'Global Min | ' + percentages)
# #plt.rcParams.update({'font.': 11})
# #plt.annotate(percentages , (coor[0], coor[1]), fontsize=7.5)
# #plt.legend(bbox_to_anchor=(1.05, 1.0),loc='upper left')
# #plt.show()

# #____________________________________________________________________________________

# data = []

# for item in range(len(perCoorList)):
#     try:
#         bestCoor = perCoorList[item][percentageDict[sharpeDict[maxRatio]]]
#     except KeyError:
#         pass
# for h in range(len(stockPercentileList)):
#     try:
#         data = stockPercentileList[h][str(percentageDict[sharpeDict[maxRatio]])]
#     except KeyError:
#         pass
# print(str(percentageDict[sharpeDict[maxRatio]]))
# print(data)

# #________________________________________________________________________________________

# stk1 = str(data[0].values())[:-3:][14::]
# stk2 = str(data[1].values())[:-3:][14::]

# finalStockList = [stk1, stk2]

# dataList = []
# dataLog = getTicker(finalStockList, dataList)

# returnList = []
# returns = logReturnList(dataLog, returnList)[0]

# performanceList = []
# performance = newTSMAlgo(returnList, performanceList, returns, period=1, short=False)
# years = (performance[0].index.max() - performance[0].index.min()).days / 365
# perf_cum = [np.exp(performance[0].cumsum()), np.exp(performance[1].cumsum())]

# percent1 = (int(str(data[0])[1:3])/100)
# percent2 = (int(str(data[1])[1:3])/100)

# tot = (perf_cum[0][-1]*percent1 + perf_cum[1][-1]*percent2) - 1
# ann = (perf_cum[0][-1]*percent1 + perf_cum[1][-1]*percent2) ** (1 / years) - 1
# vol = bestCoor[0]
# #Continue to replicate the cell below.

# rfr = 0.7/365
# sharpe = (ann - rfr) / vol
# print(f"1-day TSM Strategy yields:" +
#       f"\n\t{tot*100:.2f}% total returns" + 
#       f"\n\t{ann*100:.2f}% annual returns" +
#       f"\n\t{sharpe:.2f} Sharpe Ratio")

# stock_ret1 = np.exp(returnList[0].cumsum())
# stock_ret2 = np.exp(returnList[1].cumsum())
# b_tot = (stock_ret1[-1]*percent1 + stock_ret2[-1]*percent2) - 1
# b_ann = (stock_ret1[-1]*percent1 + stock_ret2[-1]*percent2) ** (1 / years) - 1
# b_vol = bestCoor[0] * np.sqrt(252)
# b_sharpe = (b_ann - rfr) / b_vol
# print(f"Baseline Buy-and-Hold Strategy yields:" + 
#       f"\n\t{b_tot*100:.2f}% total returns" + 
#       f"\n\t{b_ann*100:.2f}% annual returns" +
#       f"\n\t{b_sharpe:.2f} Sharpe Ratio")

# periods = [3, 5, 15, 30, 90, 180, 365]
# fig = plt.figure(figsize=(21, 12))
# gs = fig.add_gridspec(6, 10)
# minPlot = fig.add_subplot(gs[:2, 6:])
# ax0 = fig.add_subplot(gs[:2, :6])
# ax1 = fig.add_subplot(gs[2:4, :5])
# ax2 = fig.add_subplot(gs[2:4, 5:])
# ax3 = fig.add_subplot(gs[4:,:5])
# ax4 = fig.add_subplot(gs[4:, 5:])


# for q in range(len(stockCombinations)):
#     graphData = globalMin(stockCombinations[q][0], stockCombinations[q][1], minPlot, correlation, stockListCombinations, stockList, q)[4]
#     minPlot.plot(graphData[1], graphData[2])
#     minPlot.plot(graphData[5], graphData[6], marker='o', label= 'Global Min | ' + str(graphData[4]))
#     minPlot.annotate(graphData[4] , graphData[3], fontsize=7.5)

# minPlot.grid()
# minPlot.set_title('Fig 1. Minimum Variance Frontier ' + graphData[0], fontsize=11)
# minPlot.set_ylabel('Daily Returns (%)')
# minPlot.set_xlabel('Standard Deviation/Risk (%)')

# ax0.plot((np.exp(returns.cumsum()) - 1)*100, label='B&H', linestyle='-')
# perf_dict = {'tot_ret': {'buy_and_hold': (np.exp((returns[0].sum()*percent1 + returns[1].sum()*percent2) - 1))}}
# perf_dict['ann_ret'] = {'buy_and_hold': b_ann}
# perf_dict['sharpe'] = {'buy_and_hold': b_sharpe}
# for p in periods:
#     log_perf = newTSMAlgo(returnList, performanceList, returns, period=p, short=False)[0]*percent1 + newTSMAlgo(returnList, performanceList, returns, period=p, short=False)[1]*percent2
#     perf = np.exp(log_perf.cumsum()) #Exponentialize the log returns.
#     perf_dict['tot_ret'][p] = (perf[-1] - 1) # Adding each period to the nested dictionary.
#     ann = (perf[-1] ** (1/years) - 1)
#     perf_dict['ann_ret'][p] = ann
#     vol = log_perf.std() * np.sqrt(252) #Volatility formula.
#     perf_dict['sharpe'][p] = (ann - rfr) / vol #Sharpe Ratio formula.
#     ax0.plot((perf - 1) * 100, label=f'{p}-Day Mean') #Plot first graph.
# ax0.set_ylabel('Returns (%)')
# ax0.set_xlabel('Date')
# ax0.set_title('Fig 2. Cumulative Returns | Best Pair: ' + str(percentageDict[sharpeDict[maxRatio]]))
# ax0.grid()
# ax0.legend()

# print(perf_dict)
# _ = [ax1.bar(i, v * 100) for i, v in enumerate(perf_dict['ann_ret'].values())]
# ax1.set_xticks([i for i, k in enumerate(perf_dict['ann_ret'])])
# ax1.set_xticklabels([f'{k}-Day Mean' 
#     if type(k) is int else 'B&H' for 
#     k in perf_dict['ann_ret'].keys()],
#     rotation=45)
# ax1.grid()
# ax1.set_ylabel('Returns (%)')
# ax1.set_xlabel('Strategy')
# ax1.set_title('Fig 3. Annual Returns | Best Pair: ' + str(percentageDict[sharpeDict[maxRatio]]))

# _ = [ax2.bar(i, v) for i, v in enumerate(perf_dict['sharpe'].values())]
# ax2.set_xticks([i for i, k in enumerate(perf_dict['sharpe'])])
# ax2.set_xticklabels([f'{k}-Day Mean' 
#     if type(k) is int else 'B&H' for 
#     k in perf_dict['sharpe'].keys()],
#     rotation=45)
# ax2.grid()
# ax2.set_ylabel('Sharpe Ratio')
# ax2.set_xlabel('Strategy')
# ax2.set_title('Fig 4. Sharpe Ratio | Best Pair: ' + str(percentageDict[sharpeDict[maxRatio]]))
# #plt.tight_layout()
# #plt.show()

# #------------------- FOR SMA STRATEGY --------------------------

# ann_retDict = perf_dict['ann_ret'].copy()
# sharpe_Dict = perf_dict['sharpe'].copy()

# del ann_retDict['buy_and_hold']
# del sharpe_Dict['buy_and_hold']

# #print(ann_retDict)
# #print(sharpe_Dict)

# combinedDict = {k: ann_retDict[k]*sharpe_Dict[k] for k in ann_retDict}
# maxValue = max(combinedDict, key=combinedDict.get)

# dataLog[0]['SMA90'] = SMA(dataLog[0].iloc[-500:], period=120)
# dataLog[0]['SMAMax'] = SMA(dataLog[0].iloc[-500:], period=maxValue)
# #dataLog[1]['SMA'] = SMA(dataLog[1])

# strat1 = SMAstrategy(dataLog[0])
# dataLog[0]['Buy'] = strat1[0]
# dataLog[0]['Sell'] = strat1[1]

# #ax2.figure(figsize=(16,8))
# ax3.set_title('Fig 5.')
# ax3.plot(dataLog[0].iloc[-500:]['Close'], label='Close Price')
# ax3.plot(dataLog[0].iloc[-500:]['SMA90'], label='SMA')
# ax3.plot(dataLog[0].iloc[-500:]['SMAMax'], label='SMA')
# ax3.scatter(dataLog[0].index, dataLog[0]['Buy'], color='green', label='Buy Signal', marker='^')
# ax3.scatter(dataLog[0].index, dataLog[0]['Sell'], color='red', label='Sell Signal', marker='v')
# ax3.set_xlabel('Date')
# ax3.set_ylabel('Close Price in USD')


# #plt.plot(dataLog[0]['SMA'])
# #plt.plot(dataLog[0]['Close'])
# coorText = []
# idx20 = np.argwhere(np.diff(np.sign(dataLog[0]['SMAMax'].values - dataLog[0]['SMA90'].values))).flatten()
# ax3.scatter(dataLog[0]['SMAMax'].index[idx20], dataLog[0]['SMA90'].values[idx20], color='red')
# intersectCoor = pd.DataFrame(dataLog[0]['SMA90'].values[idx20], dataLog[0]['SMAMax'].index[idx20], ['Coordinates'])
# print(intersectCoor.index)
# for l in range(len(intersectCoor)):
#     coorIndex = str(intersectCoor.index[l])[0:10]
#     coorY = str(dataLog[0]['SMA90'].values[idx20][l])[0:5]
#     coorText.append('('+coorIndex + ',' + coorY + ')')
#     ax3.annotate(coorText[l], (intersectCoor.index[l], dataLog[0]['SMA90'].values[idx20][l]), fontsize=9)

# #----------------

# combinedDict2 = {k: ann_retDict[k]*sharpe_Dict[k] for k in ann_retDict}
# maxValue2 = max(combinedDict2, key=combinedDict2.get)

# dataLog[1]['SMA90'] = SMA(dataLog[1].iloc[-500:], period=120)
# dataLog[1]['SMAMax'] = SMA(dataLog[1].iloc[-500:], period=maxValue2)

# strat2 = SMAstrategy(dataLog[1])
# dataLog[1]['Buy'] = strat2[0]
# dataLog[1]['Sell'] = strat2[1]

# ax4.set_title('Fig 6.')
# ax4.plot(dataLog[1].iloc[-500:]['Close'], label='Close Price')
# ax4.plot(dataLog[1].iloc[-500:]['SMA90'], label='SMA')
# ax4.plot(dataLog[1].iloc[-500:]['SMAMax'], label='SMA')
# ax4.scatter(dataLog[1].dropna(subset=['SMA90']).index, dataLog[1].dropna(subset=['SMA90'])['Buy'], color='green', label='Buy Signal', marker='^')
# ax4.scatter(dataLog[1].dropna(subset=['SMA90']).index, dataLog[1].dropna(subset=['SMA90'])['Sell'], color='red', label='Sell Signal', marker='v')
# ax4.set_xlabel('Date')
# ax4.set_label('Close Price in USD')


# #plt.plot(dataLog[0]['SMA'])
# #plt.plot(dataLog[0]['Close'])
# coorText = []
# idx20 = np.argwhere(np.diff(np.sign(dataLog[1]['SMAMax'].values - dataLog[1]['SMA90'].values))).flatten()
# ax4.scatter(dataLog[1]['SMAMax'].index[idx20], dataLog[1]['SMA90'].values[idx20], color='red')
# intersectCoor = pd.DataFrame(dataLog[1]['SMA90'].values[idx20], dataLog[1]['SMAMax'].index[idx20], ['Coordinates'])
# print(intersectCoor.index)
# for l in range(len(intersectCoor)):
#     coorIndex = str(intersectCoor.index[l])[0:10]
#     coorY = str(dataLog[1]['SMA90'].values[idx20][l])[0:5]
#     coorText.append('('+coorIndex + ',' + coorY + ')')
#     ax4.annotate(coorText[l], (intersectCoor.index[l], dataLog[1]['SMA90'].values[idx20][l]), fontsize=9)

# plt.tight_layout()
# plt.show()

# #--- FLASK WEBFRAME ------
# #app = Flask(__name__)

# #@app.route('/', methods = ['POST', 'GET'])
# #def feed():
#     #if request.method == 'POST':
#         #stockNo = request.form.get('amount')
#         #print(type(stockNo)) # for debugging
#         #return render_template("home.html", stockNo=stockNo)

# #if __name__ == '__main__':
#     # Threaded option to enable multiple instances for multiple user access support
#     #app.run(threaded=True, port=5000)