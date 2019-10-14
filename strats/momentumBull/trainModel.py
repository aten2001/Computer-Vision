import numpy as np
import requests
import pandas as pd
import os
import bs4 as bs
import numpy as np
import pickle
from scipy.stats import linregress
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def get_sp500():
    url = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)  
    #with open("sp500tickers.pickle","wb") as f:
    #    pickle.dump(tickers,f)  
    return tickers

def compile_data():
    tickers = get_sp500()
    main_df = pd.DataFrame()
    for num, ticker in enumerate(tickers):
        if not os.path.exists("data/{}.csv".format(ticker)):
            continue
        df = pd.read_csv("data/{}.csv".format(ticker))
        df.set_index('Date', inplace=True)
        df.rename(columns = {'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
            
        if num % 10 == 0:
            print("{} csvs done".format(num))
        
    #print(main_df.head())
    main_df.to_csv('data/joined/sp500_joined_closes.csv')

def process_data_for_labels(ticker):
    days = 7
    df = pd.read_csv('data/joined/sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    for i in range(1, days + 1):
        df['{}_{}d'.format(ticker, i)] = ((df[ticker].shift(-i) - df[ticker]) / df[ticker])
    
    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    req = 0.02
    for col in cols:
        if col > 0.02:
            return 1
        if col < -0.035:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                                 df['{}_1d'.format(ticker)],
                                                 df['{}_2d'.format(ticker)],
                                                 df['{}_3d'.format(ticker)],
                                                 df['{}_4d'.format(ticker)],
                                                 df['{}_5d'.format(ticker)],
                                                 df['{}_6d'.format(ticker)],
                                                 df['{}_7d'.format(ticker)]))
   
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread: ', Counter(str_vals))
    
    df.fillna(0, inplace=True)
    df= df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0,inplace=True)
    
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    
    return X, y, df
    
def train_test(ticker):
    X, y, df = extract_featuresets(ticker)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    #print()
    #print()
    
    #with open("clf.pickle","wb") as f:
    #    pickle.dump(clf,f)
    return predictions[-1], confidence

def test_all():
    from statistics import mean

    tickers = get_sp500()

    accuracies = []
    for count,ticker in enumerate(tickers):
        if not os.path.exists("data/{}.csv".format(ticker)):
            continue

        if count%10==0:
            print(count)

        accuracy = train_test(ticker)
        accuracies.append(accuracy)
        print("{} accuracy: {}. Average accuracy:{}".format(ticker,accuracy,mean(accuracies)))
