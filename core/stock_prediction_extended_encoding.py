# -*- coding: utf-8 -*-
"""Stock Prediction with Extended Encoding

Original file: ww3_8_2025_Extended_encoding.ipynb
Stock prediction system using extended sequence encoding for pattern recognition
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import random

def get_training_test_data(stock='GOOG', start='2022-1-1', end='2025-01-31', training_ratio=0.96):
    """Get training and test data for stock prediction"""
    df = yf.Ticker(stock).history(start=start, end=end)
    df = df.iloc[:,:-3]
    df.reset_index(inplace=True)
    df['Date'] = [i.date() for i in df.Date]
    df['future_close_change'] = [np.sign(df.Close.loc[i+1]-df.Close.loc[i]) for i in range(len(df)-1)]+[np.nan]
    training_length = int(len(df)*training_ratio)
    training_data = df.iloc[:training_length,:].round(2)
    test_data = df.iloc[training_length:,:]
    test_data.reset_index(inplace=True, drop=True)
    return (training_data, test_data)

training_df, test_df = get_training_test_data()
training_df.head()

"""## Encoding

Extended encoding algorithm for stock price pattern recognition
"""

def get_previous_closes_opens(data, index, window_size):
    """Get previous closes and opens within window"""
    start_index = max(0, index - window_size)
    previous_closes = data['Close'].iloc[start_index:index]
    previous_opens = data['Open'].iloc[start_index:index]
    return previous_closes, previous_opens

def extended_encoder(high_price, open_price, close_price, low_price, data, index, window_size=10, threshold=0.03):
    """Extended encoding with historical data consideration"""
    # Get encoding for current day
    current_encoding = encoder(high_price, open_price, close_price, low_price)
    
    # Get previous closes and opens
    previous_closes, previous_opens = get_previous_closes_opens(data, index, window_size)
    
    if len(previous_closes) == 0:
        return current_encoding  # Return basic encoding if no historical data
    
    # Check if current close is significantly different from recent closes
    recent_close_mean = np.mean(previous_closes)
    close_deviation = abs(close_price - recent_close_mean) / recent_close_mean
    
    if close_deviation > threshold:
        # Modify encoding based on significant price movement
        if close_price > recent_close_mean:
            # Significant upward movement
            if current_encoding == '':
                current_encoding = 'U'  # Up trend
            else:
                current_encoding = current_encoding + 'u'
        else:
            # Significant downward movement  
            if current_encoding == '':
                current_encoding = 'D'  # Down trend
            else:
                current_encoding = current_encoding + 'd'
    
    # Check volatility compared to historical data
    if len(previous_closes) > 1:
        historical_volatility = np.std(previous_closes) / np.mean(previous_closes)
        current_range = (high_price - low_price) / close_price
        
        if current_range > historical_volatility * (1 + threshold):
            # High volatility day
            current_encoding = current_encoding + 'V'
    
    return current_encoding

def encoder(high_price, open_price, close_price, low_price):
    """Basic encoding function for single day pattern"""
    encoding = ''
    
    # Determine if it's a bullish or bearish day
    if close_price > open_price:
        encoding += 'B'  # Bullish (green candle)
    elif close_price < open_price:
        encoding += 'R'  # Bearish (red candle)
    else:
        encoding += 'N'  # Neutral (doji)
    
    # Check for gaps
    # Note: This is simplified - real gap detection would need previous day's data
    
    return encoding

def extended_dataframe_encoder(data, window_size, threshold):
    """Apply extended encoding to entire dataframe"""
    encoded_data = data.copy()
    encodings = []
    
    for i in range(len(data)):
        high_val = data['High'].iloc[i]
        open_val = data['Open'].iloc[i]  
        close_val = data['Close'].iloc[i]
        low_val = data['Low'].iloc[i]
        
        encoding = extended_encoder(high_val, open_val, close_val, low_val, data, i, window_size, threshold)
        encodings.append(encoding)
    
    encoded_data['encoding'] = encodings
    return encoded_data

def dataframe_encoder(data):
    """Apply basic encoding to dataframe"""
    encoded_data = data.copy()
    encodings = []
    
    for i in range(len(data)):
        high_val = data['High'].iloc[i]
        open_val = data['Open'].iloc[i]
        close_val = data['Close'].iloc[i] 
        low_val = data['Low'].iloc[i]
        
        encoding = encoder(high_val, open_val, close_val, low_val)
        encodings.append(encoding)
    
    encoded_data['encoding'] = encodings
    return encoded_data

"""## Pattern Analysis"""

def find_change_points(data):
    """Identify change points in price data"""
    change_points_data = data.copy()
    
    # Simple change point detection based on price movements
    close_prices = data['Close'].values
    changes = []
    
    for i in range(1, len(close_prices)):
        if i == 1:
            changes.append(0)  # First point
        else:
            # Calculate rate of change
            prev_change = (close_prices[i-1] - close_prices[i-2]) / close_prices[i-2] if close_prices[i-2] != 0 else 0
            curr_change = (close_prices[i] - close_prices[i-1]) / close_prices[i-1] if close_prices[i-1] != 0 else 0
            
            # Detect direction changes
            if (prev_change > 0 and curr_change < 0) or (prev_change < 0 and curr_change > 0):
                changes.append(1)  # Change point
            else:
                changes.append(0)  # No change point
    
    changes.append(0)  # Last point
    change_points_data['change_point'] = changes
    return change_points_data

def extended_segmentation(data):
    """Segment data based on patterns and change points"""
    segments = []
    current_segment = []
    
    for i in range(len(data)):
        current_segment.append(data.iloc[i])
        
        # Check if we should end current segment
        if (i > 0 and data['change_point'].iloc[i] == 1) or i == len(data) - 1:
            if len(current_segment) > 0:
                segment_df = pd.DataFrame(current_segment)
                segments.append(segment_df.reset_index(drop=True))
            current_segment = []
    
    return segments

def basic_segmentation(data):
    """Basic segmentation of data"""
    segments = []
    segment_size = 10  # Default segment size
    
    for i in range(0, len(data), segment_size):
        segment = data.iloc[i:i+segment_size].copy()
        if len(segment) > 0:
            segments.append(segment.reset_index(drop=True))
    
    return segments

# Helper functions for pattern analysis

def check_whether_subsequence(pattern_x, pattern_y):
    """Check if pattern X is subsequence of pattern Y"""
    if len(pattern_x) > len(pattern_y):
        return False
    
    i = 0  # Index for pattern_x
    j = 0  # Index for pattern_y
    
    while i < len(pattern_x) and j < len(pattern_y):
        if pattern_x[i] == pattern_y[j]:
            i += 1
        j += 1
    
    return i == len(pattern_x)

def count_occurrences(pattern_segments):
    """Count occurrences of each pattern"""
    pattern_counts = {}
    
    for segment in pattern_segments:
        if 'encoding' in segment.columns:
            pattern = ''.join(segment['encoding'].values)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    return pattern_counts

def count_same_trend_patterns(pattern_segments):
    """Count patterns with same trend"""
    trend_counts = {}
    
    for segment in pattern_segments:
        if len(segment) > 0 and 'future_close_change' in segment.columns:
            # Get the trend of the segment (simplified)
            trend = segment['future_close_change'].iloc[-1] if not pd.isna(segment['future_close_change'].iloc[-1]) else 0
            trend_key = 'up' if trend > 0 else 'down' if trend < 0 else 'neutral'
            
            if 'encoding' in segment.columns:
                pattern = ''.join(segment['encoding'].values)
                if pattern not in trend_counts:
                    trend_counts[pattern] = {'up': 0, 'down': 0, 'neutral': 0}
                trend_counts[pattern][trend_key] += 1
    
    return trend_counts

def calculate_pattern_accuracy(pattern_segments):
    """Calculate accuracy for each pattern"""
    pattern_accuracy = {}
    trend_counts = count_same_trend_patterns(pattern_segments)
    
    for pattern, trends in trend_counts.items():
        total = sum(trends.values())
        if total > 0:
            # Accuracy based on most frequent trend
            max_trend_count = max(trends.values())
            accuracy = max_trend_count / total
            pattern_accuracy[pattern] = {
                'accuracy': accuracy,
                'total_occurrences': total,
                'trend_distribution': trends
            }
    
    return pattern_accuracy

def subsequence_model_prediction(test_sequences, pattern_rules):
    """Make predictions based on pattern rules"""
    predictions = []
    
    for sequence in test_sequences:
        if 'encoding' in sequence.columns:
            test_pattern = ''.join(sequence['encoding'].values)
            
            # Find matching patterns in rules
            best_match = None
            best_accuracy = 0
            
            for pattern, rule_info in pattern_rules.items():
                if check_whether_subsequence(pattern, test_pattern) and rule_info['accuracy'] > best_accuracy:
                    best_match = pattern
                    best_accuracy = rule_info['accuracy']
            
            if best_match:
                # Predict based on best matching pattern
                trends = pattern_rules[best_match]['trend_distribution']
                predicted_trend = max(trends.keys(), key=lambda k: trends[k])
                
                if predicted_trend == 'up':
                    predictions.append(1)
                elif predicted_trend == 'down':
                    predictions.append(-1)
                else:
                    predictions.append(0)
            else:
                predictions.append(0)  # Neutral prediction if no match
        else:
            predictions.append(0)
    
    return predictions

# Modified test data preparation functions
def extended_test_data_preparation(data, sequence_length=5):
    """Prepare test data sequences with extended encoding"""
    sequences = []
    
    for i in range(len(data) - sequence_length + 1):
        sequence = data.iloc[i:i+sequence_length].copy()
        sequences.append(sequence.reset_index(drop=True))
    
    return sequences

def test_data_preparation(data, sequence_length=5):
    """Prepare test data sequences"""
    sequences = []
    
    for i in range(len(data) - sequence_length + 1):
        sequence = data.iloc[i:i+sequence_length].copy()
        sequences.append(sequence.reset_index(drop=True))
    
    return sequences

def subsequence_model_scoring(data, pattern_rules):
    """Calculate prediction accuracy score"""
    if len(data) == 0 or len(pattern_rules) == 0:
        return 0.0
    
    # Get actual future trends
    actual_trends = []
    for sequence in data:
        if 'future_close_change' in sequence.columns and len(sequence) > 0:
            actual_trend = sequence['future_close_change'].iloc[-1]
            if not pd.isna(actual_trend):
                actual_trends.append(int(actual_trend))
            else:
                actual_trends.append(0)
        else:
            actual_trends.append(0)
    
    # Get predictions
    predictions = subsequence_model_prediction(data, pattern_rules)
    
    if len(actual_trends) != len(predictions) or len(actual_trends) == 0:
        return 0.0
    
    # Calculate accuracy
    correct_predictions = sum(1 for actual, pred in zip(actual_trends, predictions) if actual == pred)
    accuracy = correct_predictions / len(actual_trends)
    
    return accuracy

def extended_prediction_score(stock, window_size, threshold, start_date='2022-1-1', end_date='2025-01-31'):
    """Calculate prediction score using extended encoding"""
    training_data, test_data = get_training_test_data(stock, start_date, end_date)
    training_data = find_change_points(training_data)
    # 🔧 FIX: Apply encoding to training data BEFORE segmentation
    training_data = extended_dataframe_encoder(training_data, window_size, threshold)
    pattern_segments = extended_segmentation(training_data)
    pattern_rules = calculate_pattern_accuracy(pattern_segments)  # Fixed: no double-call
    test_data = extended_dataframe_encoder(test_data, window_size, threshold)
    test_sequences = extended_test_data_preparation(test_data)
    prediction_score = subsequence_model_scoring(test_sequences, pattern_rules)
    return prediction_score

def original_prediction_score(stock, window_size=5, threshold=0.02, start_date='2022-1-1', end_date='2025-01-31'):
    """Calculate prediction score using original encoding"""
    training_data, test_data = get_training_test_data(stock, start_date, end_date)
    training_data = find_change_points(training_data)
    # 🔧 FIX: Apply encoding to training data BEFORE segmentation  
    training_data = dataframe_encoder(training_data)
    pattern_segments = basic_segmentation(training_data)
    pattern_rules = calculate_pattern_accuracy(pattern_segments)  # Fixed: no double-call
    test_data = dataframe_encoder(test_data)
    test_sequences = test_data_preparation(test_data)
    prediction_score = subsequence_model_scoring(test_sequences, pattern_rules)
    return prediction_score

def sp500_symbols():
    """Get S&P 500 stock symbols"""
    df_sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    symbol_list = df_sp500['Symbol'].to_list()

    for symbol in symbol_list:
        if '.' in symbol:
            index = symbol_list.index(symbol)
            symbol_list[index] = symbol.replace('.', '-')

    return symbol_list

def get_training_test_data(stock, start='2022-1-1', end='2025-01-31'):
    """Split data into training and test sets"""
    ticker = yf.Ticker(stock)
    df = ticker.history(start=start, end=end)
    df.reset_index(inplace=True)
    df = df.iloc[:,:-3]  # Remove last 3 columns
    df['Date'] = [i.date() for i in df.Date]
    
    # Split: 80% train, 20% test
    split_index = int(len(df) * 0.8)
    training_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    # Add future_close_change column
    training_df['future_close_change'] = [np.sign(training_df.Close.iloc[i+1]-training_df.Close.iloc[i]) for i in range(len(training_df)-1)]+[np.nan]
    test_df['future_close_change'] = [np.sign(test_df.Close.iloc[i+1]-test_df.Close.iloc[i]) for i in range(len(test_df)-1)]+[np.nan]
    
    return training_df, test_df

# Parameter optimization setup
window_size_values = [i for i in range(1, 6, 2)]  # Reduced range for faster testing
threshold_values = [i/100000 for i in range(1, 200, 50)]  # Reduced range for faster testing

columns = ['Stock', 'Best_Parameter', 'Best_Score']
results_df = pd.DataFrame(columns=columns)

# Test set of stocks for demonstration
stock_symbol_list = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

if __name__ == "__main__":
    print("Starting Best Parameter Search...")
    print(f"Testing window_size_values: {window_size_values}")
    print(f"Testing threshold_values: {threshold_values}")
    print(f"Testing stocks: {stock_symbol_list}")
    print("="*50)
    
    for stock in stock_symbol_list:
        try:
            print(f"\nProcessing {stock}...")
            best_score = 0
            best_parameters = None
            
            # Try original method
            try:
                original_score = original_prediction_score(stock)
                print(f"  Original method score: {original_score}")
                if original_score > best_score:
                    best_score = original_score
                    best_parameters = "Original"
            except Exception as e:
                print(f"  Original method failed: {e}")
                original_score = 0
            
            # Try extended method with different parameters
            for window_size in window_size_values:
                for threshold in threshold_values:
                    try:
                        extended_score = extended_prediction_score(stock, window_size, threshold)
                        print(f"  Extended method (window_size={window_size}, threshold={threshold:.5f}) score: {extended_score}")
                        
                        if extended_score > best_score:
                            best_score = extended_score
                            best_parameters = f"Extended (window_size={window_size}, threshold={threshold:.5f})"
                    except Exception as e:
                        print(f"  Extended method (window_size={window_size}, threshold={threshold:.5f}) failed: {e}")
            
            print(f"  BEST for {stock}: {best_parameters} with score {best_score}")
            results_df.loc[len(results_df)] = [stock, best_parameters, best_score]
            
        except Exception as e:
            print(f"Error processing {stock}: {e}")
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(results_df.to_string(index=False))
    print("\nBest overall performance:")
    best_row = results_df.loc[results_df['Best_Score'].idxmax()]
    print(f"Stock: {best_row['Stock']}, Method: {best_row['Best_Parameter']}, Score: {best_row['Best_Score']}")

# Alternative parameter optimization approach
iteration_counter = 0

print("\n" + "="*50)
print("DETAILED PARAMETER OPTIMIZATION:")
print("="*50)

# Loop over all candidate combinations
for symbol in stock_symbol_list:
    best_score = -float("inf")
    best_parameter_combinations = []
    print(f"Processing {iteration_counter + 1}/{len(stock_symbol_list)}: {symbol}")
    iteration_counter += 1
    
    try:
        for window_size in window_size_values:
            for threshold in threshold_values:
                # Test extended prediction with current parameters
                score = extended_prediction_score(symbol, window_size=window_size, threshold=threshold)
                print(f"Stock = {symbol}, Testing window_size={window_size}, threshold={threshold}: score = {score}")
                
                if abs(score - best_score) < 1e-10:  # Scores are equal (within floating point precision)
                    best_parameter_combinations.append((window_size, threshold))
                elif score > best_score:
                    best_score = score
                    best_parameter_combinations = [(window_size, threshold)]
        
        print(f"Best parameters (window_size, threshold) for {symbol}: {best_parameter_combinations}")
        print(f"Best prediction score: {best_score}")
        results_df.loc[len(results_df)] = [symbol, best_parameter_combinations, best_score]
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

print("\nFinal Optimization Results:")
print(results_df)