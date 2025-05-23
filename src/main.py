import argparse
import os
import random

import numpy as np
import pandas as pd

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

class GeneticAlgorithm:
    def initialize_population(self, population_size, assets_quantity):

        population = []

        for individual in range(population_size):
            weights = np.random.rand(assets_quantity)
            weights = self.normalize_weights(weights)
            population.append(weights)

        return population

    def fitness_function(self, weights, daily_returns_df, risk_free_rate):
        portfolioRisk = PortfolioRisk()
        _, _, sharpe_ratio = portfolioRisk.calculate_portfolio_risk(daily_returns_df, risk_free_rate)
        return sharpe_ratio

    def select_parents(self, population, fitness_scores, number_of_parents):
        zipped = zip(fitness_scores, population)
        sorted_zipped = sorted(zipped, key=lambda pair: pair[0], reverse=True)
        sorted_population = [x for _, x in sorted_zipped]

        return sorted_population[:number_of_parents]

    def crossover(self, parent1_weights, parent2_weights):
        number_of_assets = len(parent1_weights)
        crossover_point = np.random.randint(1, number_of_assets - 1)
        offspring1_weights = np.concatenate((parent1_weights[:crossover_point], parent2_weights[crossover_point:]))
        offspring2_weights = np.concatenate((parent1_weights[crossover_point:], parent2_weights[:crossover_point]))

        offspring1_weights = self.normalize_weights(offspring1_weights)
        offspring2_weights = self.normalize_weights(offspring2_weights)

        return offspring1_weights, offspring2_weights

    def mutate(self, weights, mutation_rate=0.05):

        mutation_strength = 0.1

        for i in range(len(weights)):
            if random.random() < mutation_rate:
                perturbation = (random.random() * 2 -1) * mutation_strength
                weights[i] += perturbation
                weights[i] = max(0, weights[i])

        return self.normalize_weights(weights)

    def run_optimization(self, daily_returns_dataframe, risk_free_rate, number_of_generations, population_size, crossover_rate, mutation_rate, elitism_count):

        number_of_assets = daily_returns_dataframe.shape[1]
        population = self.initialize_population(population_size, number_of_assets)

        best_portfolio = None
        best_sharpe = -np.inf
        portfolioRisk = PortfolioRisk()

        print("Starting GA for " + number_of_assets +" assets over " + number_of_generations + " generations...")

        for generation in range(number_of_generations):
            fitness_scores = []

            for individual_weight in population:
                sharpe = self.fitness_function(individual_weight, daily_returns_dataframe, risk_free_rate)
                fitness_scores.append(sharpe)

                current_best_idx = np.argmax(fitness_scores)
                current_best_sharpe = fitness_scores[current_best_idx]
                current_best_weights = population[current_best_idx]

                if current_best_sharpe > best_sharpe:
                    best_sharpe = current_best_sharpe
                    best_portfolio = current_best_weights.copy()

                print("Generation " + (generation+1) +"/" + number_of_generations + ": Best Sharpe Ratio = " + best_sharpe)

                new_population = []

                actual_elitism_count = min(elitism_count, population_size)
                elite_individuals = self.select_parents(population, fitness_scores, actual_elitism_count)
                new_population.extend(elite_individuals)

                num_offspring_needed = population_size - len(new_population)

                while len(new_population) < population_size:
                    parent1, parent2 = random.sample(population, 2)


                    if random.random() < crossover_rate:
                        offspring1, offspring2 = self.crossover(parent1, parent2)
                    else:
                        offspring1 = self.normalize_weights(parent1.copy())
                        offspring2 = self.normalize_weights(parent2.copy())


                    offspring1 = self.mutate(offspring1, mutation_rate)
                    offspring2 = self.mutate(offspring2, mutation_rate)

                    new_population.append(offspring1)
                    if len(new_population) < population_size:
                        new_population.append(offspring2)

                population = new_population

        final_annual_return, final_annual_volatility, final_sharpe_ratio = portfolioRisk.calculate_portfolio_risk(best_portfolio, daily_returns_dataframe, risk_free_rate)

        print("\n--- Optimization Complete ---")
        print("Best Portfolio Annual Return: "+final_annual_return)
        print("Best Portfolio Annual Volatility: "+final_annual_volatility)
        print("Best Portfolio Sharpe Ratio: "+final_sharpe_ratio)
        print("Best Portfolio Weights:")

        best_portfolio_series = pd.Series(best_portfolio, index=daily_returns_dataframe.columns)
        print(best_portfolio_series[best_portfolio_series > 0.001].sort_values(ascending=False)) # Only show non-negligible weights

        return best_portfolio_series, final_annual_return, final_annual_volatility, final_sharpe_ratio


    def normalize_weights(self, weights):
        total_sum = np.sum(weights)

        if total_sum == 0:
            return np.array([1.0 / len(weights)] * len(weights))

        return weights / total_sum

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
        # print("Data shape: " + daily_returns_df.shape)
        # print("Columns: "+ daily_returns_df.columns.tolist())
    except IOError:
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
