import math
from stats import stdDev, mean, standDev, correlCo

def portfolioExpReturn (stock1, stock2):
    
    expReturnList = []

    stock1Mean = stock1['Daily Returns'].mean()
    stock2Mean = stock2['Daily Returns'].mean()
                                
    for x in range (0,105,5):
        weightedMean = stock1Mean*(x/100) + stock2Mean*((100-x)/100)
        expReturnList.append(weightedMean)
    return expReturnList

def portfolioStdDev(stock1, stock2, correlation):
    stdDevList = [] 
    for y in range (0,105,5):
        pStdDev = math.sqrt(((y/100)**2)*((stdDev(stock1))**2) + (((y-100)/100)**2)*((stdDev(stock2))**2) + 2*(y/100)*((y-100)/100)*correlation*stdDev(stock1)*stdDev(stock2))
        stdDevList.append(pStdDev)
    return stdDevList