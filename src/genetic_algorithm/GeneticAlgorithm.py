import numpy as np
import random
import pandas as pd

from risk.PortfolioRisk import PortfolioRisk


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