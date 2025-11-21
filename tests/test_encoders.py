"""
Unit tests for encoder modules
"""

import pytest
import pandas as pd
import numpy as np
from core.encoders import SimpleEncoder, ExtendedEncoder, create_test_sequences


class TestSimpleEncoder:
    """Test cases for SimpleEncoder."""
    
    def setup_method(self):
        """Set up test data."""
        self.encoder = SimpleEncoder()
        self.test_data = pd.DataFrame({
            'Open': [100, 101, 99, 102, 98],
            'High': [105, 103, 101, 105, 102], 
            'Low': [98, 99, 97, 101, 96],
            'Close': [104, 100, 102, 103, 99]
        })
    
    def test_encode_returns_dataframe_with_code_column(self):
        """Test that encode returns DataFrame with code column."""
        result = self.encoder.encode(self.test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'code' in result.columns
        assert len(result) == len(self.test_data)
    
    def test_first_row_gets_default_code(self):
        """Test that first row gets default 'eM' code."""
        result = self.encoder.encode(self.test_data)
        
        assert result['code'].iloc[0] == 'eM'
    
    def test_encoding_logic(self):
        """Test specific encoding logic cases."""
        # Create test case where hp > op and cp > lp -> should be 'aM'
        test_case = pd.DataFrame({
            'Open': [100, 100],
            'High': [105, 110],  # hp > op
            'Low': [95, 95],
            'Close': [102, 98]   # cp > lp for first case
        })
        
        result = self.encoder.encode(test_case)
        assert result['code'].iloc[1] == 'aM'


class TestExtendedEncoder:
    """Test cases for ExtendedEncoder."""
    
    def setup_method(self):
        """Set up test data."""
        self.encoder = ExtendedEncoder(k=3, t=0.02)
        
        # Create more realistic test data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Open': [100 + i + np.random.normal(0, 1) for i in range(10)],
            'High': [105 + i + np.random.normal(0, 1) for i in range(10)],
            'Low': [95 + i + np.random.normal(0, 1) for i in range(10)],
            'Close': [102 + i + np.random.normal(0, 1) for i in range(10)],
            'Volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(10)]
        })
    
    def test_encode_returns_dataframe_with_code_column(self):
        """Test that encode returns DataFrame with code column."""
        result = self.encoder.encode(self.test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'code' in result.columns
        assert len(result) == len(self.test_data)
    
    def test_parameters_affect_encoding(self):
        """Test that different parameters produce different results."""
        encoder1 = ExtendedEncoder(k=2, t=0.01)
        encoder2 = ExtendedEncoder(k=5, t=0.05)
        
        result1 = encoder1.encode(self.test_data)
        result2 = encoder2.encode(self.test_data)
        
        # Results should potentially be different
        # (This test might occasionally fail due to randomness, but should generally pass)
        codes_different = not result1['code'].equals(result2['code'])
        assert codes_different or len(self.test_data) <= 3  # Allow for small datasets


class TestCreateTestSequences:
    """Test cases for create_test_sequences function."""
    
    def setup_method(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'code': ['aM', 'eM', 'aS', 'eS', 'aM', 'eM'],
            'fcc': [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        })
    
    def test_creates_correct_number_of_sequences(self):
        """Test that correct number of sequences are created."""
        sequences = create_test_sequences(self.test_data, N=3)
        
        # With 6 rows and N=3, we should get 6-3=3 sequences
        assert len(sequences) == 3
    
    def test_sequence_structure(self):
        """Test that sequences have correct structure."""
        sequences = create_test_sequences(self.test_data, N=3)
        
        for seq, target in sequences:
            assert isinstance(seq, str)
            assert len(seq) == 6  # N=3 codes, each 2 characters = 6 total characters
            assert isinstance(target, float)
    
    def test_sequence_content(self):
        """Test that sequence content is correct."""
        sequences = create_test_sequences(self.test_data, N=3)
        
        # First sequence should be first 3 codes: 'aMeMaS'
        first_seq, first_target = sequences[0]
        assert first_seq == 'aMeMaS'
        assert first_target == -1.0  # fcc at position 3
    
    def test_handles_nan_targets(self):
        """Test that NaN targets are filtered out."""
        data_with_nan = self.test_data.copy()
        data_with_nan.loc[3, 'fcc'] = np.nan
        
        sequences = create_test_sequences(data_with_nan, N=3)
        
        # Should exclude sequences with NaN targets
        for seq, target in sequences:
            assert not pd.isna(target)


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    base_price = 100
    
    data = []
    for i, date in enumerate(dates):
        price_change = np.random.normal(0, 2)
        open_price = base_price + price_change
        high_price = open_price + abs(np.random.normal(0, 1))
        low_price = open_price - abs(np.random.normal(0, 1))
        close_price = open_price + np.random.normal(0, 1)
        volume = np.random.randint(900000, 1100000)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high_price, 
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
        
        base_price = close_price
    
    return pd.DataFrame(data)


def test_integration_simple_encoder(sample_stock_data):
    """Integration test with realistic stock data."""
    encoder = SimpleEncoder()
    result = encoder.encode(sample_stock_data)
    
    # Should have all original columns plus 'code'
    expected_columns = list(sample_stock_data.columns) + ['code']
    assert list(result.columns) == expected_columns
    
    # All codes should be valid
    valid_codes = {'aM', 'eM', 'aS', 'eS'}
    assert set(result['code']) <= valid_codes
    
    # Should not have any NaN codes
    assert not result['code'].isna().any()


def test_integration_extended_encoder(sample_stock_data):
    """Integration test for ExtendedEncoder with realistic data."""
    encoder = ExtendedEncoder(k=5, t=0.02)
    result = encoder.encode(sample_stock_data)
    
    # Basic structure tests
    assert 'code' in result.columns
    assert len(result) == len(sample_stock_data)
    assert not result['code'].isna().any()


if __name__ == "__main__":
    pytest.main([__file__])