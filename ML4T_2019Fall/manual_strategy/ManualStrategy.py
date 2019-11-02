import pandas as pd
import util
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
from marketsimcode import *
from indicators import*
import matplotlib.pyplot as plt


def build_ind_df(sd, ed):
    df_p = prepare_pricedf(sd, ed)
    rolling_avg(df_p, "JPM", df_p)
    bollinger_bands(df_p, "JPM", df_p)
    momentum(df_p, "JPM", df_p )
    aroon(df_p, "JPM", df_p)
    return df_p

def check_long_conditions(date, sd, ed):
    df_ind = build_ind_df(sd, ed)
    sma = df_ind["price/sma"]
    bb_p = df_ind["bb_num"]
    momentum = df_ind["momentum"]
    aroon_up = df_ind["aroon_up"]

    if (sma[date] > 1.1 # 1.1
    and bb_p[date] > 1 #1 1
    and momentum[date] > 0.1 #0.1
    and aroon_up[date] > 80 #80 60
        ):
    #if (sma[date] > 1 and momentum[date] > 1):
        """results = []
        results.append(sma[date])
        results.append(momentum[date])
        results.append(aroon_up[date])
        print(results)"""
        return True
    else:
        return False

def check_short_conditions(date, sd, ed):
    df_ind = build_ind_df(sd, ed)
    sma = df_ind["price/sma"]
    bb_p = df_ind["bb_num"]
    momentum = df_ind["momentum"]
    aroon_down = df_ind["aroon_up"]

    if (sma[date] < 0.9 # 0.9 0.95-Dev values Man Strat
        and bb_p[date] < -0.08 # -0.08 -0.08
        and momentum[date] < -0.05 #-0.05 -0.00
        and aroon_down[date] > 70): #70 50
    #if (sma[date] < 1):
        return True
    else:
        return False    
        
def get_manualTrades_df(df_prices, sd, ed, symbol="JPM", ):
    """for date in range(len(df_prices)):
        if check_long_conditions(date):
            print(True)
        elif check_short_conditions(date):
            print(False)    """
    # Should enter long trade if:
    # 1. Price / sma > 1
    # 2. BB % > 1
    # 3. Momentum > 0
    # 4. Aroon Up Indicator is above 50
    curr_shares = 0
    share_orders = []
    dates = []
    count = []
    for date in range(len(df_prices)):
        if (check_long_conditions(date, sd, ed)):
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
        elif(check_short_conditions(date, sd, ed)):
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

    if len(df_trades.index) == 0:
        dates = []
        dates.append(df_prices.index[3])
        dates.append(df_prices.index[len(df_prices.index)-2])
        symbols = [symbol,symbol]
        df_trades = pd.DataFrame(data = symbols, index = dates, columns = ['Symbol'])
        df_trades["Order"] = ["SELL", "BUY"]
        df_trades["Shares"] = [1000, 1000]
        df_trades.index.name = "Date"
    
    return df_trades


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

def plot_man_trades(portvals, bench_portvals):
    portvals["Manual Strat Returns"] = portvals["portfolio_totals"]
    portvals = portvals.drop("portfolio_totals", axis =1)

    bench_portvals["Benchmark Returns"] = bench_portvals["portfolio_totals"]

    total_df = pd.DataFrame(index=portvals.index)
    total_df["Manual Strat Returns"] = portvals["Manual Strat Returns"]
    total_df["Benchmark Returns"] = bench_portvals["Benchmark Returns"]
    total_df["Manual Strat Returns"] = total_df["Manual Strat Returns"] / total_df["Manual Strat Returns"][0]
    total_df["Benchmark Returns"] = total_df["Benchmark Returns"] / total_df["Benchmark Returns"][0]

    curr_plt = plt.figure(0)
    plt.title("Returns of Manual Strategy vs Benchmark")
    plt.plot(total_df["Manual Strat Returns"], color="red", label = "Manual Strat Returns")
    plt.plot(total_df["Benchmark Returns"], color="green", label = "Benchmark Returns")
    plt.legend(loc="upper left")

    plt.show()

# in sample/development period is January 1, 2008 to December 31 2009.
# out of sample/testing period is January 1, 2010 to December 31 2011

def testPolicy(symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=10000):
    df_prices = prepare_pricedf(sd, ed)

    trades_df = get_manualTrades_df(df_prices, sd , ed)
    bench_df = create_benchmark_tradesDF(df_prices)
    bench_portvals = compute_portvals(bench_df, start_val = 100000, commission=9.95, impact=0.005)


    #Plot Ideal Trades & Benchmark
    sv=100000
    portvals = compute_portvals(trades_df, start_val = 100000, commission=9.95, impact=0.005)
    	  			  	 		  		  		    	 		 		   		 		  
    #if isinstance(portvals, pd.DataFrame):  		   	  			  	 		  		  		    	 		 		   		 		  
    #    portvals = portvals[portvals.columns[0]]
    plot_man_trades(portvals, bench_portvals)
    portvals = portvals["portfolio_totals"]

    
    #Print out Metrics
    start_date = sd 		   	  			  	 		  		  		    	 		 		   		 		  
    end_date = ed

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

if __name__ == "__main__":
    """sd = sd= dt.datetime(2010,1,1)
    ed = dt.datetime(2011,12,31)
    df_prices = prepare_pricedf(sd, ed)
    trades_df = get_manualTrades_df(df_prices)
    print(trades_df)"""
    testPolicy()

# TODO:
# Save all plots instead of plt.show
# green / red on Theoret optimal strat'
# change date for TheoretOptimal Strat
# Class for Man strat?
# def author() in all files!
# PRINT PORTFOLIO STATS FOR EACH RUN OF MAN and THEORET