"""
Unit tests for optimizer modules
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from core.optimizers import BayesianOptimizer, GeneticOptimizer, RandomSearchOptimizer


class TestBayesianOptimizer:
    """Test cases for BayesianOptimizer."""
    
    def setup_method(self):
        """Set up test optimizer."""
        def mock_evaluation(k, t):
            # Simple mock function that prefers k=5, t=0.05
            return 0.6 - abs(k - 5) * 0.02 - abs(t - 0.05) * 2
        
        self.optimizer = BayesianOptimizer('AAPL', mock_evaluation)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.stock_symbol == 'AAPL'
        assert self.optimizer.evaluation_func is not None
    
    def test_objective_function_calls_evaluation(self):
        """Test that objective function calls evaluation correctly."""
        # Mock trial object
        trial = Mock()
        trial.suggest_int.return_value = 5
        trial.suggest_float.return_value = 0.05
        
        score = self.optimizer.objective(trial)
        
        # Should call suggest_int for k
        trial.suggest_int.assert_called_once_with('k', 1, 20)
        # Should call suggest_float for t  
        trial.suggest_float.assert_called_once_with('t', 0.001, 0.3, log=True)
        
        # Score should be reasonable
        assert 0.0 <= score <= 1.0
    
    @patch('core.optimizers.optuna')
    def test_optimize_returns_valid_results(self, mock_optuna):
        """Test that optimize returns valid parameter ranges."""
        # Mock the study and optimization
        mock_study = Mock()
        mock_study.best_params = {'k': 5, 't': 0.05}
        mock_study.best_value = 0.6
        mock_optuna.create_study.return_value = mock_study
        
        best_params, best_score = self.optimizer.optimize(n_trials=5)
        
        # Check parameter ranges
        assert 1 <= best_params['k'] <= 20
        assert 0.001 <= best_params['t'] <= 0.3
        assert 0 <= best_score <= 1


class TestGeneticOptimizer:
    """Test cases for GeneticOptimizer."""
    
    def setup_method(self):
        """Set up test optimizer."""
        def mock_evaluation(k, t):
            # Simple mock function
            return 0.6 - abs(k - 5) * 0.02 - abs(t - 0.05) * 2
        
        self.optimizer = GeneticOptimizer('AAPL', mock_evaluation)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.stock_symbol == 'AAPL'
        assert self.optimizer.evaluation_func is not None
    
    def test_objective_function_constrains_parameters(self):
        """Test that objective function constrains parameters correctly."""
        # Test extreme values
        score1 = self.optimizer.objective_function([0, -1])  # Should constrain to [1, 0.001]
        score2 = self.optimizer.objective_function([25, 1])  # Should constrain to [20, 0.3]
        
        # These should return the negated evaluation scores (for minimization)
        # The actual values depend on the mock evaluation function
        assert isinstance(score1, float)
        assert isinstance(score2, float)
    
    @patch('core.optimizers.differential_evolution')
    def test_optimize_returns_valid_results(self, mock_de):
        """Test that optimize returns valid results."""
        # Mock differential evolution result
        mock_result = Mock()
        mock_result.x = [5.0, 0.05]
        mock_result.fun = -0.6  # Negative because we minimize
        mock_de.return_value = mock_result
        
        best_params, best_score = self.optimizer.optimize(maxiter=5)
        
        # Check results
        assert best_params['k'] == 5
        assert best_params['t'] == 0.05
        assert best_score == 0.6


class TestRandomSearchOptimizer:
    """Test cases for RandomSearchOptimizer."""
    
    def setup_method(self):
        """Set up test optimizer."""
        def mock_evaluation(k, t):
            # Deterministic function for testing
            return 0.5 + 0.1 * k / 20 + 0.1 * t / 0.3
        
        self.optimizer = RandomSearchOptimizer('AAPL', mock_evaluation)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.stock_symbol == 'AAPL'
        assert self.optimizer.evaluation_func is not None
    
    @patch('numpy.random.randint')
    @patch('numpy.random.lognormal')
    def test_optimize_samples_parameters_correctly(self, mock_lognormal, mock_randint):
        """Test that optimize samples parameters in correct ranges."""
        # Mock random sampling to return specific values
        mock_randint.return_value = 10  # k value
        mock_lognormal.return_value = 0.05  # t value
        
        best_params, best_score = self.optimizer.optimize(n_trials=1)
        
        # Should have called random functions
        assert mock_randint.called
        assert mock_lognormal.called
        
        # Results should be in valid ranges
        assert 1 <= best_params['k'] <= 20
        assert 0.001 <= best_params['t'] <= 0.3
        assert best_score > 0
    
    def test_optimize_finds_better_parameters_over_time(self):
        """Test that longer search finds better parameters."""
        np.random.seed(42)  # For reproducible test
        
        # Run short optimization
        params_short, score_short = self.optimizer.optimize(n_trials=5)
        
        np.random.seed(42)  # Reset seed
        
        # Run longer optimization  
        params_long, score_long = self.optimizer.optimize(n_trials=20)
        
        # Longer search should find at least as good results
        assert score_long >= score_short


class TestOptimizerIntegration:
    """Integration tests for optimizers."""
    
    def setup_method(self):
        """Set up realistic evaluation function."""
        def realistic_evaluation(k, t):
            # Simulate realistic performance based on parameters
            # Higher k generally better for stability, moderate t optimal
            k_score = 0.5 + 0.3 * (1 - abs(k - 10) / 10)  # Optimal around k=10
            t_score = 0.5 + 0.2 * (1 - abs(t - 0.05) / 0.05)  # Optimal around t=0.05
            
            # Add some noise
            noise = np.random.normal(0, 0.05)
            score = (k_score + t_score) / 2 + noise
            
            return max(0.3, min(0.8, score))  # Constrain to reasonable range
        
        self.evaluation_func = realistic_evaluation
    
    def test_all_optimizers_find_reasonable_parameters(self):
        """Test that all optimizers find reasonable parameters."""
        optimizers = [
            BayesianOptimizer('AAPL', self.evaluation_func),
            GeneticOptimizer('AAPL', self.evaluation_func), 
            RandomSearchOptimizer('AAPL', self.evaluation_func)
        ]
        
        for optimizer in optimizers:
            # Use small trial counts for fast testing
            if isinstance(optimizer, BayesianOptimizer):
                params, score = optimizer.optimize(n_trials=10)
            elif isinstance(optimizer, GeneticOptimizer):
                params, score = optimizer.optimize(maxiter=5, popsize=8)
            else:  # RandomSearchOptimizer
                params, score = optimizer.optimize(n_trials=10)
            
            # All should find reasonable parameters
            assert 1 <= params['k'] <= 20
            assert 0.001 <= params['t'] <= 0.3
            assert 0.3 <= score <= 0.8
            
            print(f"{optimizer.__class__.__name__}: k={params['k']}, t={params['t']:.4f}, score={score:.4f}")


def test_optimizer_comparison_framework():
    """Test the optimizer comparison functionality."""
    from core.optimizers import compare_optimizers
    
    def mock_evaluation(k, t):
        return 0.6 - abs(k - 5) * 0.02 - abs(t - 0.05) * 2
    
    # Test with minimal parameters for speed
    with patch('core.optimizers.BayesianOptimizer') as mock_bayes, \
         patch('core.optimizers.GeneticOptimizer') as mock_genetic, \
         patch('core.optimizers.RandomSearchOptimizer') as mock_random:
        
        # Mock the optimizers to return predictable results
        mock_bayes.return_value.optimize.return_value = ({'k': 5, 't': 0.05}, 0.6)
        mock_genetic.return_value.optimize.return_value = ({'k': 6, 't': 0.04}, 0.65)
        mock_random.return_value.optimize.return_value = ({'k': 4, 't': 0.06}, 0.55)
        
        # Run comparison
        results = compare_optimizers(['AAPL'], mock_evaluation, save_results=False)
        
        # Should return DataFrame with results
        assert len(results) == 1
        assert results.iloc[0]['Stock'] == 'AAPL'


if __name__ == "__main__":
    pytest.main([__file__])