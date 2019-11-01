import pandas as pd
import util
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
from marketsimcode import *
import matplotlib.pyplot as plt

def look_ahead(df_prices, symbol="JPM"):
    share_orders = []
    dates = []
    count = []
    adj_closes = df_prices[symbol]
    curr_shares = 0
    #Max shares you can have is +- 1000
    #So, look ahead and if the next day closes higher, buy either 1000 or 2000 to get to + 1000
    # If next day closes lower, sell either 1000 or 2000
    for date in range(len(df_prices) - 1):
        today = adj_closes[date]
        tomorrow = adj_closes[date+1]
        if (tomorrow > today):
            if curr_shares == 0:
                dates.append(df_prices.index[date])
                share_orders.append(1000)
                curr_shares = curr_shares + 1000
                count.append(curr_shares)
            elif curr_shares == -1000:
                dates.append(df_prices.index[date])
                share_orders.append(2000)
                curr_shares = curr_shares + 2000
                count.append(curr_shares)
        elif (tomorrow < today):
            if curr_shares == 0:
                dates.append(df_prices.index[date])
                share_orders.append(-1000)
                curr_shares = curr_shares - 1000
                count.append(curr_shares)
            if curr_shares == 1000:
                dates.append(df_prices.index[date])
                share_orders.append(-2000)
                curr_shares = curr_shares - 2000
                count.append(curr_shares)
    
    if curr_shares != 0:
	    share_orders.append(-curr_shares)
	    dates.append(df_prices.index[len(df_prices.index)-2])
    
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
    return df_trades
    
def plot_ideal_trades(df_prices):
    trades_df = look_ahead(df_prices)
    portvals = compute_portvals(trades_df, start_val = 100000, commission=0, impact=0)
    portvals["Theoretical Returns"] = portvals["portfolio_totals"]
    portvals = portvals.drop("portfolio_totals", axis =1)

    bench_df = create_benchmark_tradesDF(df_prices)
    bench_portvals = compute_portvals(bench_df, start_val = 100000, commission=0, impact=0)
    bench_portvals["Benchmark Returns"] = bench_portvals["portfolio_totals"]

    total_df = pd.DataFrame(index=portvals.index)
    total_df["Theoretical Returns"] = portvals["Theoretical Returns"]
    total_df["Benchmark Returns"] = bench_portvals["Benchmark Returns"]
    total_df["Theoretical Returns"] = total_df["Theoretical Returns"] / total_df["Theoretical Returns"][0]
    total_df["Benchmark Returns"] = total_df["Benchmark Returns"] / total_df["Benchmark Returns"][0]

    curr_plt = plt.figure(0)
    plt.title("Returns of Theoretically Optimal Strategy vs Benchmark")
    plt.plot(total_df["Theoretical Returns"], label = "Theoretical Returns")
    plt.plot(total_df["Benchmark Returns"], label = "Benchmark Returns")
    plt.legend(loc="upper left")

    plt.show()

def create_benchmark_tradesDF(df_prices, symbol="JPM"):
    dates = []
    dates.append(df_prices.index[3])
    dates.append(df_prices.index[len(df_prices.index)-2])
    symbols = [symbol,symbol]
    df_trades = pd.DataFrame(data = symbols, index = dates, columns = ['Symbol'])
    df_trades["Order"] = ["BUY", "SELL"]
    df_trades["Shares"] = [1000, 1000]
    df_trades.index.name = "Date"
    return df_trades
    
def testPolicy(symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=10000):
    start = sd
    end = ed
    dates = pd.date_range(start,end)
    df_prices = util.get_data([symbol], dates, False)

    trades_df = look_ahead(df_prices)
    benchmark_df = create_benchmark_tradesDF(df_prices)


    #Plot Ideal Trades & Benchmark
    sv=100000
    portvals = compute_portvals(trades_df, start_val = 1000000, commission=0, impact=0)
    portvals = portvals["portfolio_totals"]	  			  	 		  		  		    	 		 		   		 		  
    #if isinstance(portvals, pd.DataFrame):  		   	  			  	 		  		  		    	 		 		   		 		  
    #    portvals = portvals[portvals.columns[0]]
    plot_ideal_trades(df_prices)

    
    #Print out Metrics
    start_date = start 		   	  			  	 		  		  		    	 		 		   		 		  
    end_date = end

    #JUST PLACEHOLDERS	   	  			  	 		  		  		    	 		 		   		 		  
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5] 			  	 		  		  		    	 		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")  
    
    return trades_df

testPolicy()