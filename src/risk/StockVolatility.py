import numpy as np

class StockVolatility:
    
    def individual_stock_annual_volatility(self, daily_returns_dataframe):
        standard_deviation = daily_returns_dataframe.std()
        annualization_factor = np.sqrt(252)
        return annualization_factor * standard_deviation
    