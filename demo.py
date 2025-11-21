#!/usr/bin/env python3
"""
Quick Demo of Advanced Stock Prediction System
Run this file to see a demonstration of the stock prediction capabilities.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Advanced Stock Prediction System - Quick Demo")
    print("="*55)
    
    try:
        # Import core functions
        from core import (
            extended_prediction_score, 
            BayesianOptimizer,
            compare_optimization_methods
        )
        
        print("✅ Successfully imported all modules!")
        
        # Demo 1: Single stock prediction
        print("\n📊 Demo 1: Single Stock Prediction")
        print("-" * 35)
        
        stock = 'AAPL'
        window_size = 5
        threshold = 0.01
        
        print(f"Testing {stock} with window_size={window_size}, threshold={threshold}")
        
        try:
            score = extended_prediction_score(stock, window_size, threshold)
            print(f"✅ Prediction Accuracy: {score:.2%}")
            
            if score > 0.60:
                print("🎉 Excellent performance!")
            elif score > 0.55:
                print("👍 Good performance!")
            else:
                print("📈 Room for improvement - try different parameters")
                
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            print("💡 Tip: Check your internet connection for stock data")
        
        # Demo 2: Parameter optimization
        print("\n🔧 Demo 2: Parameter Optimization")
        print("-" * 35)
        
        try:
            optimizer = BayesianOptimizer('NVDA')
            print("Running Bayesian optimization for NVDA (this may take a moment)...")
            
            best_params, best_score = optimizer.optimize(n_trials=10)
            print(f"✅ Best parameters found: k={best_params['k']}, t={best_params['t']:.4f}")
            print(f"✅ Best accuracy: {best_score:.2%}")
            
        except Exception as e:
            print(f"❌ Error in optimization: {e}")
        
        print("\n🎯 Demo completed!")
        print("📚 Next steps:")
        print("   • Explore examples/stock_prediction_demo.ipynb for detailed analysis")
        print("   • Run 'python -m core.ml_parameter_optimization' for full comparison")
        print("   • Check README.md for more usage examples")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Please install the package first:")
        print("   pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()