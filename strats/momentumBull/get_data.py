import pandas_datareader as web
import bs4 as bs
import pickle
import requests
import datetime
import os
import time

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

def get_sp400():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text.strip()
        tickers.append(ticker)
        
    #with open("sp500tickers.pickle","wb") as f:
    #    pickle.dump(tickers,f)
        
    return tickers

def generate_csv(stocks):
    end = datetime.datetime.today()
    start = datetime.date(end.year-3,1,1)
    count = 0
    for stonk in stocks:
        if not os.path.exists('data/{}.csv'.format(stonk)):
            print("Getting {} data".format(stonk))
            try:
                df = web.DataReader(stonk, 'yahoo', start, end)
            except:
                print("Could not find data for {}".format(stonk)) 
                continue 
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            count = count + 1
            df.to_csv('data/{}.csv'.format(stonk))
        else:
            print('Already have {}'.format(stonk))
    return count

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    start_time = time.time()
    sp400tickers = get_sp400()
    sp500tickers = get_sp500()
    count = generate_csv(sp500tickers)
    print("Generated {} csv in {}".format(count, (time.time() - start_time)))