import numpy as np

class PortfolioRisk:
    
    def calculate_portfolio_risk(self, weights, daily_returns_dataframe, risk_free_rate):
        
        expected_individual_daily_returns = daily_returns_dataframe.mean()
        annualization_factor = np.sqrt(252)
        covariance_matrix = daily_returns_dataframe.cov()

        portfolio_daily_return = np.dot(weights, expected_individual_daily_returns)
        portfolio_variance_daily = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_standard_deviation_daily = np.sqrt(portfolio_variance_daily)
        
        annual_portfolio_return = portfolio_daily_return * annualization_factor
        annual_portfolio_volatility = portfolio_standard_deviation_daily * np.sqrt(annualization_factor)
        
        if annual_portfolio_volatility == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (annual_portfolio_return - risk_free_rate) / annual_portfolio_volatility
            
        return annual_portfolio_return, annual_portfolio_volatility, sharpe_ratio

