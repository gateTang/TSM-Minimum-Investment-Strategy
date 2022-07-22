# Description: A practice on building an efficeint frontier for two and three assets.
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize

from sympy.solvers import solve
import sympy
from sympy import Symbol

from dailyReturn import dailyReturn
from UI import stockAmt, portfolioSelect, stockUnzip
from expReturn import expectedReturn
from stats import stdDev, mean, standDev, correlCo
from portfolioStats import portfolioExpReturn, portfolioStdDev
def globalMin(stock1, stock2, minPlot, correlation, stockListCombinations, stockList, q):
    i = np.argmin(portfolioStdDev(stock1, stock2, correlation))
    x_min = portfolioStdDev(stock1, stock2, correlation)[i]
    y_min = portfolioExpReturn(stock1, stock2)[i]
    
    y = Symbol('y')
    pStd = sympy.sqrt(((y/100)**2)*((stdDev(stock1))**2) + (((y-100)/100)**2)*((stdDev(stock2))**2) + 2*(y/100)*((y-100)/100)*correlation*stdDev(stock1)*stdDev(stock2))
    solStd = solve(pStd - x_min, y)
    #print(solStd)
    
    x = Symbol('x')
    solExp = solve(stock1['Daily Returns'].mean()*(x/100) + stock2['Daily Returns'].mean()*((100-x)/100) - y_min, x)
    solExp[0] = round(solExp[0])
    #print(solExp[0])
    
    minPlot.grid()
    minPlot.title('Minimum Variance Frontier ' + str(stockList))
    minPlot.xlabel('Standard Deviation/Risk (%)', fontsize=11)
    minPlot.ylabel('Daily Return (%)', fontsize=11)
    minPlot.plot(portfolioStdDev(stock1, stock2, correlation), portfolioExpReturn(stock1, stock2))
    firstStock = {solExp[0]:stockListCombinations[q][0]}
    secondStock = {100 - solExp[0]:stockListCombinations[q][1]}
    percentages = str(solExp[0]) + (stockListCombinations[q][0])  + " | "+ str(100 - solExp[0]) + str(stockListCombinations[q][1])
    percentagesDict = {percentages:(firstStock, secondStock)}
    minPlot.plot(x_min, y_min, marker='o', label= 'Global Min | ' + percentages)
    
    minPlot.rcParams.update({'font.size': 11})
    minPlot.annotate(percentages , (x_min, y_min), fontsize=7.5)
    minPlot.legend(bbox_to_anchor=(1.05, 1.0),loc='upper left')
    return (x_min, y_min, percentages), {percentages: (x_min, y_min)}, percentagesDict, solExp



#Step 9 find the most optimized minimum variance with the highest Sharpe ratio.
def sharpeRatio(x_min, y_min, ratioList):
    ratio = y_min/x_min
    ratioList.append(ratio)
    return ratioList