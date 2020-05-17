import sys
import math
import numpy as np
import copy
import random
import gym
from gym import spaces

import DataReader

INITIAL_ACCOUNT_BALANCE = 10000

class StockTradingEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ):
        super(StockTradingEnvironment, self).__init__()
        MAX_ACCOUNT_BALANCE=10
        self.reward_range = (0, MAX_ACCOUNT_BALANCE) 
        
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box( low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        
        # self.action_space = spaces.Box( low=np.array([-1]), high=np.array([1]), dtype=np.float16)
        # Prices contains the OHCL values for the last five prices
        # self.observation_space = spaces.Box(
          # low=0, high=1, shape=(len(df.columns)+1,6), dtype=np.float16)
          
        self.maxBalance =10000
        self.maxSteps = 3e8
        self.numberOfInputDates =30
        self.technologyTickers = DataReader.getSavedData()
        self.ticker=None
        
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        self.rewardStep+=1
        done=False
        
        if self.current_step > (len(self.stockData.index) - self.numberOfInputDates-1):
            self.current_step = 0
            self.reset()
            self.rewardStep=0
            done = True

        profit = (self.net_worth - INITIAL_ACCOUNT_BALANCE)
        # if self.profit==0 or (self.profit<0 and profit>0) or (self.profit>0 and profit<0):
            # rewardStep = 0
        # delay_modifier = (self.rewardStep/ self.maxSteps)

        reward = profit*1e-5 #* delay_modifier
        self.profit = profit
        
        if done==False:
            done = self.net_worth <= 0
        obs = self._next_observation()
        return obs, reward, done, {}
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance =  self.maxBalance
        self.net_worth =  self.maxBalance
        self.max_net_worth =  self.maxBalance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        
        self.ticker=random.choice(['MSFT','AAPL','GOOGL','AMZN','FB'] )
        self.stockData,self.tickerMetrics =DataReader.dataReader(self.ticker)
        
        self.dfMax=self.stockData.max(axis = 0, skipna = True)
        dfMax=self.dfMax.copy()
        del dfMax['index']
        del dfMax['volume']
        self.maxSharePrice=max(dfMax)
        # Set the current step to a random point within the data frame
        self.current_step = np.random.randint(0, len(self.stockData.index) - self.numberOfInputDates-500)
        self.rewardStep = 0
        self.profit = 0
        return self._next_observation()

    def _next_observation(self):

        dfSliced=self.stockData[self.current_step: (self.current_step + self.numberOfInputDates)]
        filteredMetrics=self._getQuarterlyMetrics(dfSliced.index[-1])
        # print(dfSliced)
        # Get the data points for the last 5 days and scale to between 0-1
        frame = np.array([
        dfSliced['index'].to_numpy()/1000,
        dfSliced['open'].to_numpy()/100,
        dfSliced['high'].to_numpy()/100,
        dfSliced['low'].to_numpy()/100,
        dfSliced['close'].to_numpy()/100,
        dfSliced['adjClose'].to_numpy()/100,
        dfSliced['volume'].to_numpy(),
        dfSliced['change'].to_numpy()/100,
        dfSliced['changePercent'].to_numpy()/100,
        dfSliced['vwap'].to_numpy()/100,
        dfSliced['changeOverTime'].to_numpy(),
        ])
        frame=np.append(frame,filteredMetrics)
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [
        self.balance/ self.maxBalance,
        self.max_net_worth/ self.maxBalance,
        self.shares_held/ self.maxBalance,
        self.cost_basis/self.maxBalance,
        self.total_shares_sold/self.maxBalance,
        self.total_sales_value/(self.maxBalance*self.maxBalance),
        ])
        return (obs)
        
    def _getQuarterlyMetrics(self,startDate):
        endDate=startDate-np.timedelta64(4, 'M')
        filteredMetrics=self.tickerMetrics.loc[startDate:endDate]
        return filteredMetrics.iloc[0].to_numpy()
  
    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = np.random.uniform( self.stockData.iloc[self.current_step+self.numberOfInputDates,1], self.stockData.iloc[self.current_step+self.numberOfInputDates,4])   
        action_type = action[0]
        amount = action[1]
        if amount>1:
            amount=1
        if amount<0:
            amount=0
        if action_type <1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought+1.e-17)
            self.shares_held += shares_bought

        elif action_type >2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print(f'Step: {self.current_step}')
        print(f'Date: {self.stockData.iloc[self.current_step+6].name}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis}(Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth}(Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')

def main():
    dataReader=DataReader.dataReader()
    stockTradingEnvironment=StockTradingEnvironment(dataReader)
    state = stockTradingEnvironment.reset()

    while True:
        obs, rewards, done, info = stockTradingEnvironment.step([0,1])
        stockTradingEnvironment.render()
    
if __name__ == '__main__':
    main()