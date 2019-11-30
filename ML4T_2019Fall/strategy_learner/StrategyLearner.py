"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: shollister7 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903304661 (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util as ut  		   	  			  	 		  		  		    	 		 		   		 		  
import random
import QLearner as ql
import indicators as ind
import util as ut
import numpy as np 
import marketsimcode as ms  	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):
    LNG = 1
    HOLD = 0
    SHRT = -1 		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # constructor  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False, impact=0.0):  		   	  			  	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		   	  			  	 		  		  		    	 		 		   		 		  
        self.impact = impact
        self.q_l = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)		   	  			  	 		  		  		    	 		 		   		 		  

    def get_indicators(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1) ):
        df_p = ind.prepare_pricedf(symbol,sd, ed)
        ind.rolling_avg(df_p, symbol, df_p)
        ind.bollinger_bands(df_p, symbol, df_p)
        ind.momentum(df_p, symbol, df_p )
        ind.aroon(df_p, symbol, df_p)
        df_p.fillna(method='ffill', inplace=True)
        df_p.fillna(method='backfill', inplace=True)
        return df_p

    def discretize(self, indicators):
        indicators["price/sma"] = pd.cut(indicators['price/sma'], 10, labels=False)
        indicators["bb_num"] = pd.cut(indicators['bb_num'], 10, labels=False)
        indicators["momentum"] = pd.cut(indicators['momentum'], 10, labels=False)
        indicators["aroon_up"] = pd.cut(indicators['aroon_up'], 10, labels=False)
        indicators["aroon_down"] = pd.cut(indicators['aroon_down'], 10, labels=False)
        indicators = indicators.drop(['sma', 'upper_b', 'lower_b'], axis=1)
        indicators['state'] = indicators["price/sma"] + indicators["bb_num"] + indicators["momentum"]  + indicators['aroon_up']  + indicators['aroon_down']
        
        #return (bb * 1000) + (self.pp * 100) + (norm[date] * 10) + psma[date]
        return indicators

    # this method should create a QLearner, and train it for trading  		   	  			  	 		  		  		    	 		 		   		 		  
    def addEvidence(self, symbol = "JPM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        # add your code to do learning here  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        # example usage of the old backward compatible util function  		   	  			  	 		  		  		    	 		 		   		 		  
        syms=[symbol]  		   	  			  	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		   	  			  	 		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		   	  			  	 		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		   	  			  	 		  		  		    	 		 		   		 		  
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later  		   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(prices)  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        # example use with new colname  		   	  			  	 		  		  		    	 		 		   		 		  
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY  		   	  			  	 		  		  		    	 		 		   		 		  
        volume = volume_all[syms]  # only portfolio symbols  		   	  			  	 		  		  		    	 		 		   		 		  
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later  		   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(volume)

        df_p = self.get_indicators(symbol, sd, ed)
        feats = self.discretize(df_p)
        #print(df_p)
        pct_return = get_pctReturn(feats, symbol)
        init_state = feats.iloc[0]['state']
        self.q_l.querysetstate(int(float(init_state)))
        i = 0
        cum_returns = np.arange(500)
        for i in range(500):
            share_orders = []
            curr_shares = 0
            dates=[]
            count = []

            j = 0
            for index, row in feats.iterrows():
                #print(index)
                reward = curr_shares * pct_return.loc[index] * (1 - self.impact)
                action = self.q_l.query(int(float(feats.loc[index]['state'])), reward)
                if(action == 1):
                    if curr_shares == 0:
                        d = feats.index[j]
                        dates.append(d)
                        share_orders.append(1000)
                        curr_shares = curr_shares + 1000
                        count.append(curr_shares)
                    elif curr_shares == -1000:
                        d = feats.index[j]
                        dates.append(d)
                        #self.long_dates.append(d)
                        share_orders.append(2000)
                        curr_shares = curr_shares + 2000
                        count.append(curr_shares)
                elif (action == 2):
                    if curr_shares == 0:
                        d = feats.index[j]
                        dates.append(d)
                        share_orders.append(-1000)
                        curr_shares = curr_shares - 1000
                        count.append(curr_shares)
                    elif (curr_shares == 1000):
                        d = feats.index[j]
                        dates.append(d)
                        #self.long_dates.append(d)
                        share_orders.append(-2000)
                        curr_shares = curr_shares - 2000
                        count.append(curr_shares)
                j += 1


            if curr_shares != 0:
                share_orders.append(-curr_shares)
                dates.append(feats.index[len(feats.index)-2])
            
            buy_sell = []
            for order in share_orders:
                if order < 0:
                    buy_sell.append("SELL")
                elif order > 0:
                    buy_sell.append("BUY")
            abs_orders = [abs(x) for x in share_orders]
            symbols=[]
            for i in range(len(abs_orders)):
                symbols.append(symbol)
            

            df_trades = pd.DataFrame(data = symbols, index = dates, columns = ['Symbol'])
            df_trades["Order"] = buy_sell
            df_trades["Shares"] = abs_orders
            df_trades.index.name = "Date"
            
            #df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
            #df_trades.columns = ['Symbol', 'Order', 'Shares']

            strat_portvals = ms.compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
            cum_returns[i] = cumulative_return(strat_portvals)
            if (i > 20 and cum_returns[i] <= cum_returns[i - 5]):
                break

    def author(self):
        return 'shollister7'  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		   	  			  	 		  		  		    	 		 		   		 		  
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        # here we build a fake set of trades  		   	  			  	 		  		  		    	 		 		   		 		  
        # your code should return the same sort of data  		   	  			  	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		   	  			  	 		  		  		    	 		 		   		 		  
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		   	  			  	 		  		  		    	 		 		   		 		  
        trades = prices_all[[symbol,]]  # only portfolio symbols  		   	  			  	 		  		  		    	 		 		   		 		  
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[:,:] = 0 # set them all to nothing  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[0,:] = 1000 # add a BUY at the start  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[40,:] = -1000 # add a SELL  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[41,:] = 1000 # add a BUY  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[60,:] = -2000 # go short from long  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[61,:] = 2000 # go long from short  		   	  			  	 		  		  		    	 		 		   		 		  
        trades.values[-1,:] = -1000 #exit on the last day  		   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(type(trades)) # it better be a DataFrame!  		   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(trades)  		   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(prices_all)

        df_p = self.get_indicators(symbol, sd, ed)
        feats = self.discretize(df_p)
        init_state = feats.iloc[0]['state']
        self.q_l.querysetstate(int(float(init_state)))

        share_orders = []
        curr_shares = 0
        dates=[]
        count = []
        j = 0
        for index, row in feats.iterrows():
            action = self.q_l.querysetstate(int(float(feats.loc[index]['state'])))
            if(action == 1):
                if curr_shares == 0:
                    d = feats.index[j]
                    dates.append(d)
                    share_orders.append(1000)
                    curr_shares = curr_shares + 1000
                    count.append(curr_shares)
                elif curr_shares == -1000:
                    d = feats.index[j]
                    dates.append(d)
                    #self.long_dates.append(d)
                    share_orders.append(2000)
                    curr_shares = curr_shares + 2000
                    count.append(curr_shares)
            elif (action == 2):
                if curr_shares == 0:
                    d = feats.index[j]
                    dates.append(d)
                    share_orders.append(-1000)
                    curr_shares = curr_shares - 1000
                    count.append(curr_shares)
                elif curr_shares == 1000:
                    d = feats.index[j]
                    dates.append(d)
                    #self.long_dates.append(d)
                    share_orders.append(-2000)
                    curr_shares = curr_shares - 2000
                    count.append(curr_shares)
            j += 1
        
        if curr_shares != 0:
                share_orders.append(-curr_shares)
                dates.append(feats.index[len(feats.index)-2])
            
        buy_s = []
        for order in share_orders:
            if order < 0:
                buy_s.append("SELL")
            elif order > 0:
                buy_s.append("BUY")
        #abs_orders = [abs(x) for x in share_orders]
        symbols=[]
        for i in range(len(share_orders)):
            symbols.append(symbol)
        

        df_trades = pd.DataFrame(data = symbols, index = dates, columns = ['Symbol'])
        df_trades["Shares"] = share_orders
        df_trades.index.name = "Date"
        df_trades = df_trades.drop('Symbol', axis=1)
        trades = df_trades
        
        return trades  		   	  			  	 		  		  		    	 		 		   		 		  

def cumulative_return(portvals):
    cumulative_return, avg_daily, std_daily, sharpe = compute_port_stats(portvals)
    return cumulative_return

def get_pctReturn(feats,symbol):
    pct_return = feats[symbol]
    pct_return[1:] = (feats[symbol][1:] / feats[symbol][:-1].values) - 1
    return pct_return

def compute_port_stats(portvals):
        rfr = 0.0
        sr = 252.0
        portvals = portvals["portfolio_totals"]
        cumulative_return = (portvals[-1]/portvals[0]) - 1
        daily_ret = (portvals/portvals.shift(1)) - 1   
        
        avg_daily = daily_ret.mean()
        std_daily = daily_ret.std()
        diff = (daily_ret - rfr).mean()
        sharpe = np.sqrt(sr) * (diff / std_daily)
        return cumulative_return, avg_daily, std_daily, sharpe

if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")
    sd=dt.datetime(2009,1,1)
    ed=dt.datetime(2010,1,1)
    sl = StrategyLearner()
    sl.addEvidence(symbol="JPM")
    df_trades = sl.testPolicy(symbol="JPM")

    strat_portvals = ms.compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_port_stats(strat_portvals)
    strat_returns = strat_portvals["portfolio_totals"]

    print(f"Date Range: {sd} to {ed}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Manual Strat: {sharpe_ratio}")  		   	  			  	 		  		  		    	 		 		   		 		  
    #print(f"Sharpe Ratio of Benchmark : {sharpe_ratio_bench}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Manual Strat: {cum_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    #print(f"Cumulative Return of Benchmark : {cum_ret_bench}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Manual Strat: {std_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    #print(f"Standard Deviation of Benchmark : {std_daily_ret_bench}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Manual Strat: {avg_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    #print(f"Average Daily Return of Benchmark : {avg_daily_ret_bench}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value Manual Strat: {strat_returns[-1]}")
    #print(f"Final Portfolio Value Benchmark: {bench_returns[-1]}")  
    
    
    #print(df_trades)		  	 		  		  		    	 		 		   		 		  

#TODO: Spice Up code np.rolling() instead of np.iterrows for speed, stop Q learning when convergence reached etc
# experiment code (just keep it within Strat Learner)
# Spice up descritize (weight each indicator differently)