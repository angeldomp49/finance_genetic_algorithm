import pandas as pd
import yfinance as yf
from datetime import datetime

class StockInformation:
    def data_of_returns_of_tickers(self, all_data):
        
        stacked_data = pd.DataFrame()
        for data in all_data:
            stacked_data = pd.concat([stacked_data, data[["Close", "Volume", "Ticker"]]])
        
        data_of_returns = stacked_data.pivot_table(index='Date', columns='Ticker', values='Close')
        
        return data_of_returns.pct_change().dropna()
    
    def data_by_tickers(self, tickers, start_date, end_date):
        all_data = []
        
        for ticker in tickers:
            print("Downloading data for " + ticker)

            try:
                data = yf.download(ticker, start=start_date, end=end_date)

                if data.empty:
                    print("No data available for " + ticker)
                    return

                data['Ticker'] = ticker
                all_data.append(data)

            except Exception as e:
                print(e)
                
        return all_data
                
    def save_data(self, output_path, all_data):
        
        # datetime_suffix = datetime.today().strftime('%Y%m%d%H%M%S')
        
        for ticker in all_data:
            ticker.to_csv(f'{output_path}/{ticker["Ticker"][0]}.csv')
        
        