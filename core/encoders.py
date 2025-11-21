"""
Stock Price Pattern Encoders

This module contains various encoding methods to transform raw OHLCV data
into symbolic sequences for pattern recognition.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


class BaseEncoder:
    """Base class for all encoders."""
    
    def __init__(self):
        pass
    
    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode OHLCV data into symbolic sequences.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added 'code' column containing symbols
        """
        raise NotImplementedError


class SimpleEncoder(BaseEncoder):
    """Simple price direction encoder."""
    
    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode based on simple price direction patterns.
        
        Args:
            data: DataFrame with Open, High, Low, Close columns
            
        Returns:
            DataFrame with 'code' column added
        """
        data_copy = data.copy()
        codes = []
        
        for i in range(len(data_copy)):
            if i == 0:
                codes.append('eM')  # Default for first row
            else:
                hp = data_copy['High'].iloc[i]
                op = data_copy['Open'].iloc[i] 
                cp = data_copy['Close'].iloc[i]
                lp = data_copy['Low'].iloc[i]
                
                # Simple encoding logic
                if hp > op and cp > lp:
                    codes.append('aM')  # ascending medium
                elif hp < op and cp < lp:
                    codes.append('eM')  # descending medium  
                elif hp > op:
                    codes.append('aS')  # ascending small
                else:
                    codes.append('eS')  # descending small
        
        data_copy['code'] = codes
        return data_copy


class ExtendedEncoder(BaseEncoder):
    """Extended encoder with advanced pattern recognition."""
    
    def __init__(self, k: int = 10, t: float = 0.03):
        """Initialize extended encoder.
        
        Args:
            k: Look-back window for pattern analysis
            t: Threshold for significant moves
        """
        self.k = k
        self.t = t
    
    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode using extended algorithm with volatility awareness.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with 'code' column added
        """
        data_copy = data.copy()
        codes = []
        
        for i in range(len(data_copy)):
            if i == 0:
                codes.append('eM')
            else:
                hp = data_copy['High'].iloc[i]
                op = data_copy['Open'].iloc[i]
                cp = data_copy['Close'].iloc[i] 
                lp = data_copy['Low'].iloc[i]
                
                # Extended encoding with volatility consideration
                code = self._extended_encode_single(hp, op, cp, lp, data_copy, i)
                codes.append(code)
        
        data_copy['code'] = codes
        return data_copy
    
    def _extended_encode_single(self, hp: float, op: float, cp: float, lp: float, 
                               data: pd.DataFrame, i: int) -> str:
        """Extended encoding logic for a single data point."""
        
        # Get historical context
        lookback_start = max(0, i - self.k)
        recent_data = data.iloc[lookback_start:i]
        
        if len(recent_data) < 2:
            return 'eM'
        
        # Calculate volatility-adjusted features
        recent_volatility = recent_data['Close'].pct_change().std()
        price_change = (cp - op) / op if op > 0 else 0
        
        # Extended pattern recognition
        if abs(price_change) > self.t:
            if price_change > 0:
                return 'aL' if recent_volatility > 0.02 else 'aM'
            else:
                return 'eL' if recent_volatility > 0.02 else 'eM'
        else:
            return 'eS' if price_change < 0 else 'aS'


def create_test_sequences(data: pd.DataFrame, N: int = 5) -> List[Tuple[str, float]]:
    """Create test sequences from encoded data.
    
    Args:
        data: DataFrame with 'code' and 'fcc' columns
        N: Sequence length (number of codes, not characters)
        
    Returns:
        List of (sequence, target) tuples
    """
    test_data = []
    data_reset = data.reset_index(drop=True)
    
    for i in range(len(data_reset) - N):
        # Build input sequence - join N consecutive codes
        sequence = ''.join(data_reset['code'].iloc[i:i+N].astype(str))
        # Get target (future close change after the sequence)
        target_idx = i + N
        if target_idx < len(data_reset):
            target = data_reset['fcc'].iloc[target_idx]
        else:
            continue  # Skip this sequence if no target available
        
        if not pd.isna(target):
            test_data.append((sequence, float(target)))
    
    return test_data