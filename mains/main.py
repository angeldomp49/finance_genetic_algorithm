from stock.StockInformation import StockInformation
from datetime import datetime

start_date = '2024-05-19'
end_date = '2025-05-19'

tickers = [
    'AAPL',
    'MSFT',
    'GOOGL',
    'AMZN',
    'TSLA',
    'NVDA'
]

output_path = "/Users/estefaniaramirez/Desktop/angel/development/practices/python/yahoo_finance/dist"

stock_information = StockInformation()

raw_data = stock_information.data_by_tickers(tickers, start_date, end_date)
stock_information.save_data(f'{output_path}/stock_data', raw_data)
daily_returns = stock_information.data_of_returns_of_tickers(raw_data)
datetime_suffix = datetime.today().strftime('%Y%m%d%H%M%S')
daily_returns.to_csv(f'{output_path}/generated/daily_returns_{datetime_suffix}.csv')
