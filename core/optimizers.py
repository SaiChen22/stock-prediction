"""
ML-based Parameter Optimizers

This module contains various optimization algorithms for finding optimal
hyperparameters for stock prediction models.
"""

from typing import Dict, Tuple, List, Optional, Callable
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import differential_evolution
import time
import warnings
warnings.filterwarnings('ignore')


class BaseOptimizer:
    """Base class for all parameter optimizers."""
    
    def __init__(self, stock_symbol: str, evaluation_func: Optional[Callable] = None):
        """Initialize optimizer.
        
        Args:
            stock_symbol: Stock symbol to optimize for
            evaluation_func: Function to evaluate parameter performance
        """
        self.stock_symbol = stock_symbol
        self.evaluation_func = evaluation_func or self._default_evaluation
    
    def optimize(self, **kwargs) -> Tuple[Dict, float]:
        """Run optimization and return best parameters and score.
        
        Returns:
            Tuple of (best_parameters, best_score)
        """
        raise NotImplementedError
    
    def _default_evaluation(self, k: int, t: float) -> float:
        """Default evaluation function - should be overridden."""
        # Simplified mock evaluation
        return np.random.uniform(0.45, 0.65)


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Optuna."""
    
    def __init__(self, stock_symbol: str, evaluation_func: Optional[Callable] = None):
        super().__init__(stock_symbol, evaluation_func)
    
    def objective(self, trial) -> float:
        """Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to maximize
        """
        k = trial.suggest_int('k', 1, 20)
        t = trial.suggest_float('t', 0.001, 0.3, log=True)
        
        score = self.evaluation_func(k, t)
        return score
    
    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None) -> Tuple[Dict, float]:
        """Run Bayesian optimization.
        
        Args:
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        print(f"🚀 Starting Bayesian optimization for {self.stock_symbol} (trials={n_trials})")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"✅ Best parameters: k={best_params['k']}, t={best_params['t']:.4f}")
        print(f"✅ Best score: {best_score:.4f}")
        
        return best_params, best_score


class GeneticOptimizer(BaseOptimizer):
    """Genetic algorithm optimization using differential evolution."""
    
    def __init__(self, stock_symbol: str, evaluation_func: Optional[Callable] = None):
        super().__init__(stock_symbol, evaluation_func)
    
    def objective_function(self, params: List[float]) -> float:
        """Objective function for genetic algorithm.
        
        Args:
            params: [k, t] parameter values
            
        Returns:
            Negative score (for minimization)
        """
        k, t = params
        k = int(max(1, min(20, k)))  # Constrain k
        t = max(0.001, min(0.3, t))  # Constrain t
        
        score = self.evaluation_func(k, t)
        return -score  # Negative because differential_evolution minimizes
    
    def optimize(self, maxiter: int = 30, popsize: int = 15) -> Tuple[Dict, float]:
        """Run genetic algorithm optimization.
        
        Args:
            maxiter: Maximum generations
            popsize: Population size
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        print(f"🧬 Starting genetic algorithm optimization for {self.stock_symbol} (generations={maxiter})")
        
        bounds = [(1, 20), (0.001, 0.3)]  # k and t bounds
        
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42
        )
        
        best_k = int(result.x[0])
        best_t = result.x[1]
        best_score = -result.fun
        
        print(f"✅ Best parameters: k={best_k}, t={best_t:.4f}")
        print(f"✅ Best score: {best_score:.4f}")
        
        return {'k': best_k, 't': best_t}, best_score


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimization."""
    
    def __init__(self, stock_symbol: str, evaluation_func: Optional[Callable] = None):
        super().__init__(stock_symbol, evaluation_func)
    
    def optimize(self, n_trials: int = 50) -> Tuple[Dict, float]:
        """Run random search optimization.
        
        Args:
            n_trials: Number of random trials
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        print(f"🎲 Starting random search optimization for {self.stock_symbol} (trials={n_trials})")
        
        best_score = 0
        best_params = {}
        
        for i in range(n_trials):
            # Random parameter sampling
            k = np.random.randint(1, 21)
            t = np.random.lognormal(np.log(0.01), 1.0)
            t = max(0.001, min(0.3, t))
            
            score = self.evaluation_func(k, t)
            
            if score > best_score:
                best_score = score
                best_params = {'k': k, 't': t}
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{n_trials}, Current best: {best_score:.4f}")
        
        print(f"✅ Best parameters: k={best_params['k']}, t={best_params['t']:.4f}")
        print(f"✅ Best score: {best_score:.4f}")
        
        return best_params, best_score


def compare_optimizers(stock_symbols: List[str], 
                      evaluation_func: Callable,
                      save_results: bool = True) -> pd.DataFrame:
    """Compare different optimization methods across multiple stocks.
    
    Args:
        stock_symbols: List of stock symbols to test
        evaluation_func: Function to evaluate parameter performance
        save_results: Whether to save results to CSV
        
    Returns:
        DataFrame with comparison results
    """
    optimizers = {
        'Bayesian': BayesianOptimizer,
        'Genetic': GeneticOptimizer,
        'Random': RandomSearchOptimizer
    }
    
    results = []
    
    for stock in stock_symbols:
        print(f"\n{'='*60}")
        print(f"🎯 Optimizing stock: {stock}")
        print(f"{'='*60}")
        
        stock_results = {'Stock': stock}
        
        for method_name, optimizer_class in optimizers.items():
            print(f"\n🔄 Running {method_name} optimization...")
            start_time = time.time()
            
            try:
                optimizer = optimizer_class(stock, evaluation_func)
                
                if method_name == 'Bayesian':
                    best_params, best_score = optimizer.optimize(n_trials=30)
                elif method_name == 'Genetic':
                    best_params, best_score = optimizer.optimize(maxiter=20)
                else:  # Random
                    best_params, best_score = optimizer.optimize(n_trials=30)
                
                elapsed_time = time.time() - start_time
                
                stock_results[f'{method_name}_Score'] = f"{best_score:.4f}"
                stock_results[f'{method_name}_K'] = best_params.get('k', 'N/A')
                stock_results[f'{method_name}_T'] = f"{best_params.get('t', 0):.4f}"
                stock_results[f'{method_name}_Time'] = f"{elapsed_time:.1f}s"
                
            except Exception as e:
                print(f"❌ {method_name} optimization failed: {e}")
                stock_results[f'{method_name}_Score'] = "ERROR"
        
        results.append(stock_results)
    
    df_results = pd.DataFrame(results)
    
    if save_results:
        df_results.to_csv('optimization_comparison.csv', index=False)
        print(f"\n💾 Results saved to: optimization_comparison.csv")
    
    return df_results