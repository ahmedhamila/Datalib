import pytest
import pandas as pd
import numpy as np
from datalib_ha.data_manipulation import DataManipulation
import os
import tempfile

class TestDataManipulation:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 75000, 90000, 100000],
            'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney']
        })
    
    def test_load_csv(self, sample_dataframe):
        """Test CSV loading functionality."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            sample_dataframe.to_csv(temp_file.name, index=False)
        
        # Load the CSV
        loaded_df = DataManipulation.load_csv(temp_file.name)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        # Assert loaded data matches original
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    def test_save_csv(self, sample_dataframe):
        """Test CSV saving functionality."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            # Save DataFrame to CSV
            DataManipulation.save_csv(sample_dataframe, temp_file.name)
            
            # Reload and verify
            loaded_df = pd.read_csv(temp_file.name)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        # Assert loaded data matches original
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    def test_filter_data(self, sample_dataframe):
        """Test data filtering functionality."""
        # Filter by age condition
        filtered_df = DataManipulation.filter_data(
            sample_dataframe, 
            {'age': lambda x: x > 30}
        )
        
        # Expected result
        expected_df = sample_dataframe[sample_dataframe['age'] > 30]
        
        pd.testing.assert_frame_equal(filtered_df, expected_df)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create DataFrame with missing values
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [np.nan, 2, 3, 4]
        })
        
        # Test drop method
        dropped_df = DataManipulation.handle_missing_values(df, method='drop')
        assert len(dropped_df) == 1
        
        # Test fill method
        filled_df = DataManipulation.handle_missing_values(df, method='fill', fill_value=0)
        assert filled_df.isna().sum().sum() == 0
    
    def test_normalize_data(self, sample_dataframe):
        """Test data normalization."""
        normalized_df = DataManipulation.normalize_data(
            sample_dataframe, 
            columns=['age', 'income']
        )
        
        # Check normalization ranges
        for column in ['age', 'income']:
            assert normalized_df[column].min() >= 0
            assert normalized_df[column].max() <= 1