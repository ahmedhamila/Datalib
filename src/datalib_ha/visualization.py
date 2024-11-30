import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union

class DataVisualization:
    """
    A class for creating data visualizations in DataLib.
    
    Provides methods for generating various types of plots and charts
    to help users understand and explore their data.
    """
    
    @staticmethod
    def bar_plot(df: pd.DataFrame, 
                 x_column: str, 
                 y_column: str, 
                 title: Optional[str] = None,
                 output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a bar plot from DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            x_column (str): Column to use for x-axis.
            y_column (str): Column to use for y-axis.
            title (str, optional): Plot title.
            output_path (str, optional): File path to save the plot.
        
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(df[x_column], df[y_column])
        plt.title(title or f'{y_column} by {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=45)
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def histogram(data: Union[pd.Series, List[float]], 
                  bins: int = 10, 
                  title: Optional[str] = None,
                  output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a histogram of the data.
        
        Args:
            data (Union[pd.Series, List[float]]): Input data.
            bins (int, optional): Number of histogram bins. Defaults to 10.
            title (str, optional): Plot title.
            output_path (str, optional): File path to save the plot.
        
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins)
        plt.title(title or 'Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def scatter_plot(df: pd.DataFrame, 
                     x_column: str, 
                     y_column: str, 
                     hue: Optional[str] = None,
                     title: Optional[str] = None,
                     output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a scatter plot from DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            x_column (str): Column to use for x-axis.
            y_column (str): Column to use for y-axis.
            hue (str, optional): Column to use for color differentiation.
            title (str, optional): Plot title.
            output_path (str, optional): File path to save the plot.
        
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        plt.figure(figsize=(10, 6))
        
        if hue:
            for category in df[hue].unique():
                subset = df[df[hue] == category]
                plt.scatter(subset[x_column], subset[y_column], label=category)
            plt.legend()
        else:
            plt.scatter(df[x_column], df[y_column])
        
        plt.title(title or f'{y_column} vs {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def correlation_heatmap(correlation_matrix: pd.DataFrame, 
                             title: Optional[str] = None,
                             output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a correlation heatmap from a correlation matrix.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix.
            title (str, optional): Plot title.
            output_path (str, optional): File path to save the plot.
        
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(title or 'Correlation Heatmap')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        
        return plt.gcf()