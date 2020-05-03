import time

class Agent(object):

    def __init__(self, capital=1000 , shares = 0):
        self.capital = capital
        self.shares = shares
        self.valuation = capital
        
    def update(self, action, shares, sharePrice , transactionCost=0.1):
    
        if (action==0): #buy shares
            tradeStatus = self.buy(shares, sharePrice , transactionCost=0.1)
            self.valuation = self.capital+self.shares*sharePrice
            return tradeStatus 
            
        elif (action==1): #hold shares
            return True

        elif (action==2): #sell shares
            tradeStatus = self.sell(shares, sharePrice , transactionCost=0.1)
            self.valuation = self.capital+self.shares*sharePrice
            return tradeStatus
        else:
            raise Exception("Invalid Action")


    def buy(self, sharesBuy , sharePrice , transactionCost=0.1):
        totalsharePrice= sharesBuy*sharePrice*(transactionCost+1)
        if self.capital> totalsharePrice:
            self.capital-=totalsharePrice
            self.shares += sharesBuy
            return True
        else:
            return False
        
    def sell(self, sharesSell, sharePrice, transactionCost=0.1):
        if self.shares > sharesSell:
            self.capital+= sharesSell*sharePrice*(transactionCost+1)
            self.shares -= sharesSell
            return True
        else:
            return False
    
    def getCapital(self):
        return self.capital
        
    def getShares(self):
        return self.shares
        
    def getValuation(self):
        return self.valuation

  
if __name__ == "__main__":
    agent = Agent()
    while True:
        agent.update(0,2,30)
        agent.update(2,1,30)
        print(" ")
        print(agent.getCapital())
        print(agent.getShares())
        print(agent.getValuation())
        time.sleep(1)
        