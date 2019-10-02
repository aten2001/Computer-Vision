import pandas_datareader as web
import bs4 as bs
import pickle
import requests
import datetime
import os

end = datetime.datetime.today()
start = datetime.date(end.year-15,1,1)
 
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
    start = datetime.date(end.year-15,1,1)
    for stock in stocks:
        if not os.path.exists('data/{}.csv'.format(stock)):
            df = web.DataReader(stock, 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('data/{}.csv'.format(stock))
        else:
            print('Already have {}'.format(stock))

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    sp400tickers = get_sp400()
    sp500Tickers = get_sp500()
    generate_csv(sp400tickers)