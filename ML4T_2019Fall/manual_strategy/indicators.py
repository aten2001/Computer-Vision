import pandas as pd
import util
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import datetime as dt
matplotlib.use('Agg')

def author():
    return 'shollister7'

def rolling_avg(df_prices, ticker, results_df):
    adj_closes = df_prices[ticker]
    adj_closes = adj_closes/adj_closes[0]
    results_df["sma"] = adj_closes.rolling(10).mean()

def bollinger_bands(df_prices, ticker, results_df):
    adj_closes = df_prices[ticker]
    adj_closes = adj_closes/adj_closes[0]
    ma = adj_closes.rolling(10).mean()
    sd = adj_closes.rolling(10).std()
    higher_b = ma + (2* sd)
    lower_b = ma - (2 * sd)
    results_df["upper_b"] = higher_b
    results_df["lower_b"] = lower_b
    #bb_value???
def momentum(df_prices, ticker, results_df):
    adj_closes = df_prices[ticker]
    adj_closes = adj_closes/adj_closes[0]
    m = adj_closes.rolling(10).std()
    results_df["momentum"] = m

def main():
    start = dt.datetime(2010,1,1)
    end = dt.datetime(2011,12,31)
    time_period = pd.date_range(start, end)
    df_prices = util.get_data(["JPM"], time_period, addSPY=True)
    df_ind = pd.DataFrame(index=df_prices.index)
    
    rolling_avg(df_prices, "JPM", df_ind)
    bollinger_bands(df_prices, "JPM", df_ind)
    momentum(df_prices, "JPM", df_ind )


    #df_ind.plot()
    #plt.show()
    print(df_ind)


if __name__ == "__main__":
    main()
