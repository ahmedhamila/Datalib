import pandas as pd
import numpy as np
from typing import Union, List, Optional

class DataManipulation:
    """
    A class for handling data manipulation tasks in DataLib.
    
    This class provides methods for loading, processing, and transforming data,
    with a focus on CSV files and general data cleaning operations.
    """
    
    @staticmethod
    def load_csv(filepath: str, 
                 delimiter: str = ',', 
                 encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.
        
        Args:
            filepath (str): Path to the CSV file to be loaded.
            delimiter (str, optional): Delimiter used in the CSV file. Defaults to ','.
            encoding (str, optional): File encoding. Defaults to 'utf-8'.
        
        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        
        Raises:
            FileNotFoundError: If the specified file cannot be found.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        try:
            return pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {filepath}")
    
    @staticmethod
    def save_csv(dataframe: pd.DataFrame, 
                 filepath: str, 
                 delimiter: str = ',', 
                 encoding: str = 'utf-8') -> None:
        """
        Save a pandas DataFrame to a CSV file.
        
        Args:
            dataframe (pd.DataFrame): DataFrame to be saved.
            filepath (str): Destination path for the CSV file.
            delimiter (str, optional): Delimiter to use. Defaults to ','.
            encoding (str, optional): File encoding. Defaults to 'utf-8'.
        """
        dataframe.to_csv(filepath, sep=delimiter, encoding=encoding, index=False)
    
    @staticmethod
    def filter_data(dataframe: pd.DataFrame, 
                    conditions: Optional[dict] = None) -> pd.DataFrame:
        """
        Filter DataFrame based on specified conditions.
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame to filter.
            conditions (dict, optional): Dictionary of column:value filtering conditions.
        
        Returns:
            pd.DataFrame: Filtered DataFrame.
        
        Example:
            filter_data(df, {'age': lambda x: x > 25, 'city': 'Paris'})
        """
        if conditions is None:
            return dataframe
        
        filtered_df = dataframe.copy()
        for column, condition in conditions.items():
            if callable(condition):
                filtered_df = filtered_df[filtered_df[column].apply(condition)]
            else:
                filtered_df = filtered_df[filtered_df[column] == condition]
        
        return filtered_df
    
    @staticmethod
    def handle_missing_values(dataframe: pd.DataFrame, 
                               method: str = 'drop', 
                               fill_value: Optional[Union[int, float, str]] = None) -> pd.DataFrame:
        """
        Handle missing values in a DataFrame.
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame.
            method (str, optional): Method to handle missing values. 
                Defaults to 'drop'. Other options: 'fill'.
            fill_value (optional): Value to use for filling missing data.
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        if method == 'drop':
            return dataframe.dropna()
        elif method == 'fill':
            return dataframe.fillna(fill_value)
        else:
            raise ValueError("Method must be 'drop' or 'fill'")
    
    @staticmethod
    def normalize_data(dataframe: pd.DataFrame, 
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize numerical columns using min-max scaling.
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame.
            columns (list, optional): Columns to normalize. If None, normalizes all numeric columns.
        
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        normalized_df = dataframe.copy()
        
        if columns is None:
            columns = normalized_df.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            min_val = normalized_df[column].min()
            max_val = normalized_df[column].max()
            
            normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
        
        return normalized_df