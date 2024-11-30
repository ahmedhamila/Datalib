import pytest
import numpy as np
import pandas as pd
from datalib_ha.statistics import StatisticalAnalysis

class TestStatisticalAnalysis:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [1, 2, 3, 4, 5]
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for correlation testing."""
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [1, 3, 5, 7, 9]
        })
    
    def test_descriptive_stats(self, sample_data):
        """Test descriptive statistics calculation."""
        stats = StatisticalAnalysis.descriptive_stats(sample_data)
        
        # Verify key statistical properties
        assert stats['mean'] == 3
        assert stats['median'] == 3
        assert stats['std_dev'] == pytest.approx(1.5811, 0.001)
        assert stats['min'] == 1
        assert stats['max'] == 5
    
    def test_correlation(self, sample_dataframe):
        """Test correlation matrix calculation."""
        corr_matrix = StatisticalAnalysis.correlation(sample_dataframe)
        
        # Check correlation matrix properties
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        
        # Check perfect correlation between B and C
        assert corr_matrix.loc['B', 'C'] == pytest.approx(1.0)
    
    def test_t_test(self):
        """Test independent t-test."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [2, 4, 6, 8, 10]
        
        t_test_result = StatisticalAnalysis.t_test(group1, group2)
        
        # Verify t-test results
        assert 't_statistic' in t_test_result
        assert 'p_value' in t_test_result
        assert 'significant' in t_test_result
    
    def test_chi_square_test(self):
        """Test chi-square goodness of fit test."""
        observed = np.array([10, 20, 30])
        
        chi2_result = StatisticalAnalysis.chi_square_test(observed)
        
        # Verify chi-square test results
        assert 'chi2_statistic' in chi2_result
        assert 'p_value' in chi2_result
        assert 'significant' in chi2_result