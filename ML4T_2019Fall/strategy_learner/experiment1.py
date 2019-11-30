import pandas as pd
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
print(f"Final Portfolio Value Q Learning Strat: {qreturns[-1]}") 


