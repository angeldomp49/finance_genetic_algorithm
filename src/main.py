import argparse
import os
import pandas as pd

from genetic_algorithm.GeneticAlgorithm import GeneticAlgorithm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_path', type=str, default='/opt/ml/processing/input/data/stock_daily_returns.csv')
    parser.add_argument('--output_data_path', type=str, default='/opt/ml/processing/output/results')

    parser.add_argument('--num_generations', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=50)
    parser.add_argument('--risk_free_rate', type=float, default=0.02) # Example: 2% annual risk-free rate
    parser.add_argument('--crossover_rate', type=float, default=0.8)
    parser.add_argument('--mutation_rate', type=float, default=0.1)
    parser.add_argument('--elitism_count', type=int, default=2) # Number of best individuals to carry over

    args = parser.parse_args()

    try:
        daily_returns_df = pd.read_csv(args.input_data_path, index_col='Date', parse_dates=True)
        print("Successfully loaded daily returns data from: " + args.input_data_path)
        print("Data shape: " + daily_returns_df.shape)
        print("Columns: "+ daily_returns_df.columns.tolist())
    except FileNotFoundError:
        print("Error: Input file not found at "+args.input_data_path)
        exit()


    genetic_algorithm = GeneticAlgorithm()
    best_portfolio_weights_series, final_return, final_volatility, final_sharpe = \
        genetic_algorithm.run_optimization(
            daily_returns_df,
            args.risk_free_rate,
            args.num_generations,
            args.population_size,
            args.crossover_rate,
            args.mutation_rate,
            args.elitism_count
        )

    # --- 3. Save Results ---
    os.makedirs(args.output_data_path, exist_ok=True) # Ensure output directory exists

    # Save best portfolio weights
    best_portfolio_weights_series.name = 'weight' # Name the series for better CSV column name
    best_portfolio_weights_series.to_csv(os.path.join(args.output_data_path, 'best_portfolio_weights.csv'), header=True)
    print("Best portfolio weights saved to: "+ os.path.join(args.output_data_path, 'best_portfolio_weights.csv'))

    # Save summary metrics
    summary_metrics = pd.DataFrame({
        'Metric': ['Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
        'Value': [final_return, final_volatility, final_sharpe]
    })
    summary_metrics.to_csv(os.path.join(args.output_data_path, 'portfolio_summary_metrics.csv'), index=False)
    print("Summary metrics saved to: "+ os.path.join(args.output_data_path, 'portfolio_summary_metrics.csv'))
