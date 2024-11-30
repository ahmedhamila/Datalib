import pandas as pd
import numpy as np
from scipy import stats
from typing import Union, List

class StatisticalAnalysis:
    """
    A class providing statistical analysis methods for DataLib.
    
    Offers methods for calculating basic and advanced statistical measures,
    including descriptive statistics and hypothesis testing.
    """
    
    @staticmethod
    def descriptive_stats(data: Union[pd.Series, List[float], np.ndarray]) -> dict:
        """
        Calculate comprehensive descriptive statistics for a dataset.
        
        Args:
            data (Union[pd.Series, List[float], np.ndarray]): Input data.
        
        Returns:
            dict: Dictionary containing descriptive statistics.
        """
        # Convert input to numpy array for consistent processing
        arr_data = np.array(data)
        
        return {
            'mean': np.mean(arr_data),
            'median': np.median(arr_data),
            'mode': stats.mode(arr_data)[0][0],
            'std_dev': np.std(arr_data),
            'variance': np.var(arr_data),
            'min': np.min(arr_data),
            'max': np.max(arr_data),
            'range': np.ptp(arr_data),
            'skewness': stats.skew(arr_data),
            'kurtosis': stats.kurtosis(arr_data)
        }
    
    @staticmethod
    def correlation(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix between numeric columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            method (str, optional): Correlation method. 
                Defaults to 'pearson'. Other options: 'spearman', 'kendall'.
        
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        numeric_columns = df.select_dtypes(include=[np.number])
        return numeric_columns.corr(method=method)
    
    @staticmethod
    def t_test(group1: Union[pd.Series, List[float]], 
               group2: Union[pd.Series, List[float]], 
               equal_var: bool = True) -> dict:
        """
        Perform independent t-test between two groups.
        
        Args:
            group1 (Union[pd.Series, List[float]]): First group of data.
            group2 (Union[pd.Series, List[float]]): Second group of data.
            equal_var (bool, optional): Assume equal variances. Defaults to True.
        
        Returns:
            dict: T-test results including t-statistic and p-value.
        """
        t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        return {
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def chi_square_test(observed: np.ndarray) -> dict:
        """
        Perform chi-square goodness of fit test.
        
        Args:
            observed (np.ndarray): Observed frequencies.
        
        Returns:
            dict: Chi-square test results.
        """
        expected = np.ones_like(observed) * np.mean(observed)
        chi2_statistic, p_value = stats.chisquare(observed, f_exp=expected)
        
        return {
            'chi2_statistic': chi2_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }