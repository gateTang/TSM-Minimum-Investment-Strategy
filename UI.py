import yfinance as yf

from dailyReturn import dailyReturn

def stockAmt():
    while True:
        try:     
            stockAmt = int(input("Amount of stocks you are willing to invest in: " + ": "))
            break
        except ValueError:
            print("Invalid input. The input must be a integer between 1-9")
    return stockAmt

def portfolioSelect(portfolio, stockList):
    while True:
        try:
            stockChosen = input('Stock ' + str(len(portfolio)+1) + ': ')
            stockValid= yf.download(stockChosen, start='2022-01-01')
            stockValid = stockValid.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
            dailyReturn(stockValid)
            errorState = False
            portfolio.append(stockValid)
            stockList.append(stockChosen)
            #print(len(stockValid))
            counter = 0
            if len(stockValid) == 1:
                stockList.pop()
                raise ValueError
            break
        except ValueError:
            print("Not a stock. Choose another one")
            if counter == 0:
                portfolio.pop()
                counter = 1       
    return portfolio

def stockUnzip(dictionary, combinations):
    pairsList = []
    heading = []
    for f in range(len(combinations)):
        selection1 = dictionary[combinations[f][0]]
        selection2 = dictionary[combinations[f][1]]
        pairsList.append((selection1,selection2))
    return pairsList

def newPortfolioSelect(portfolio, stockList):
    newStockList = []
    for t in range(len(stockList)):
        stockChosen = stockList[t]
        stockValid= yf.download(stockChosen, start='2022-01-01')
        stockValid = stockValid.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
        dailyReturn(stockValid)
        errorState = False
        portfolio.append(stockValid)
        newStockList.append(stockChosen)
        #print(len(stockValid))
    return portfolio