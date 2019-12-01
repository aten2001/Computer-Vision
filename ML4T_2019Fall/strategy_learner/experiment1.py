"""import pandas as pd
import util
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as msim 
import indicators as ind
import QLearner as ql
import matplotlib.pyplot as plt
import ManualStrategy as ms
import StrategyLearner as sl
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

symbol = "JPM"
ms = ms.ManualStrategy()
sd=dt.datetime(2008,1,1)
ed=dt.datetime(2009,12,31)

trades_df = ms.testPolicy()
#print(trades_df)
manual_portvals = msim.compute_portvals(trades_df, start_val = 100000, commission=9.95, impact=0.005)
cum_retM, avg_daily_retM, std_daily_retM, sharpe_ratioM = ms.compute_port_stats(manual_portvals)
man_returns = manual_portvals["portfolio_totals"]

sl = sl.StrategyLearner()
sl.addEvidence(symbol, sd, ed)
qtradesDf = sl.testPolicy(symbol, sd, ed)
#print(qtradesDf)
q_portvals = msim.compute_portvals(qtradesDf, start_val = 100000, commission=9.95, impact=0.005)
cum_retQ, avg_daily_retQ, std_daily_retQ, sharpe_ratioQ = ms.compute_port_stats(q_portvals)
qreturns = q_portvals["portfolio_totals"]
 


print(f"Date Range: {sd} to {ed}")  		   	  			  	 		  		  		    	 		 		   		 		  
print()  		   	  			  	 		  		  		    	 		 		   		 		  
print(f"Sharpe Ratio of Manual Strat: {sharpe_ratioM}")  		   	  			  	 		  		  		    	 		 		   		 		  
print(f"Sharpe Ratio of Q Learning Strat : {sharpe_ratioQ}")  		   	  			  	 		  		  		    	 		 		   		 		  
print()  		   	  			  	 		  		  		    	 		 		   		 		  
print(f"Cumulative Return of Manual Strat: {cum_retM}")  		   	  			  	 		  		  		    	 		 		   		 		  
print(f"Cumulative Return of Q Learning Strat : {cum_retQ}")  		   	  			  	 		  		  		    	 		 		   		 		  
print()  		   	  			  	 		  		  		    	 		 		   		 		  
print(f"Standard Deviation of Manual Strat: {std_daily_retM}")  		   	  			  	 		  		  		    	 		 		   		 		  
print(f"Standard Deviation of Q Learning Strat : {std_daily_retQ}")  		   	  			  	 		  		  		    	 		 		   		 		  
print()  		   	  			  	 		  		  		    	 		 		   		 		  
print(f"Average Daily Return of Manual Strat: {avg_daily_retM}")  		   	  			  	 		  		  		    	 		 		   		 		  
print(f"Average Daily Return of Q Learning Strat : {avg_daily_retQ}")  		   	  			  	 		  		  		    	 		 		   		 		  
print()  		   	  			  	 		  		  		    	 		 		   		 		  
print(f"Final Portfolio Value Manual Strat: {man_returns[-1]}")
print(f"Final Portfolio Value Q Learning Strat: {qreturns[-1]}")"""

import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd
import random

import QLearner as ql
import marketsimcode as ms
import ManualStrategy as mst
import indicators as ind
import util as ut
import StrategyLearner as sl
#from matplotlib import cm as cm
#from matplotlib import style
#import matplotlib.pyplot as plt

def plotStuff():
    slr = sl.StrategyLearner(verbose = False, impact=0.0)
    slr.addEvidence(symbol = "JPM")
    df_trades_sl = slr.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv = 100000)
    df_trades_sl['Symbol'] = 'JPM'
    df_trades_sl['Order'] = 'BUY'
    df_trades_sl.loc[df_trades_sl.Shares < 0, 'Order'] = 'SELL'
    df_trades_sl = df_trades_sl[df_trades_sl.Shares != 0]
    df_trades_sl = df_trades_sl[['Symbol', 'Order', 'Shares']]

    portvals_sl = ms.compute_portvals(df_trades_sl, start_val = 100000)
    #print(portvals_sl)
    man = mst.ManualStrategy()
    df_trades = man.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv = 100000)
    portvals = ms.compute_portvals(df_trades, start_val = 100000)

    syms = ['SPY']
    dates = pd.date_range(dt.datetime(2008,1,1), dt.datetime(2009,1,1))
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY_normalized = normalize_stocks(prices_SPY)
    prices_portval_normalized = normalize_stocks(portvals)
    prices_sl_normalized = normalize_stocks(portvals_sl)

    print(prices_sl_normalized)
    print(prices_portval_normalized)
    chart_df = pd.concat([prices_portval_normalized, prices_SPY_normalized, prices_sl_normalized], axis=1)
    chart_df.columns = ['Manual Strategy', 'Benchmark', 'Strategy Learner']
    chart_df.plot(grid=True, title='Comparing Manual strategy with Strategy Learner', use_index=True, color=['Black', 'Blue', 'Red'])
    plt.show()

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0, :]

def fill_missing_values(prices):
    """Fill missing values in data frame, in place."""
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

def author():
    return 'shollister7'

if __name__ == "__main__":
    plotStuff()


