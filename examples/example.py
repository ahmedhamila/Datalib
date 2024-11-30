"""
DataLib Comprehensive Usage Example

This script demonstrates the key functionalities of the DataLib library
across data manipulation, statistical analysis, visualization, 
and advanced analysis modules.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datalib_ha.data_manipulation import DataManipulation
from datalib_ha.statistics import StatisticalAnalysis
from datalib_ha.visualization import DataVisualization
from datalib_ha.advanced_analysis import AdvancedAnalysis

def main():
    # 1. Data Manipulation
    print("1. Data Manipulation Demonstration")
    
    # Create sample dataset
    data = {
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'income': [30000, 45000, 50000, 60000, 70000, 80000, 85000, 90000],
        'education_years': [12, 14, 16, 16, 18, 18, 20, 20],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Berlin', 'Toronto', 'Madrid']
    }
    df = pd.DataFrame(data)
    
    # Filter data
    filtered_df = DataManipulation.filter_data(df, {'age': lambda x: x > 35})
    print("Filtered Data (Age > 35):")
    print(filtered_df)
    
    # Normalize data
    normalized_df = DataManipulation.normalize_data(df, ['age', 'income'])
    print("\nNormalized Data:")
    print(normalized_df)
    
    # 2. Statistical Analysis
    print("\n2. Statistical Analysis Demonstration")
    
    # Descriptive statistics
    age_stats = StatisticalAnalysis.descriptive_stats(df['age'])
    print("Age Statistics:")
    print(age_stats)
    
    # Correlation analysis
    correlation_matrix = StatisticalAnalysis.correlation(df[['age', 'income', 'education_years']])
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # 3. Data Visualization
    print("\n3. Data Visualization Demonstration")
    
    # Bar plot
    plt1 = DataVisualization.bar_plot(df, 'city', 'income', 'Income by City')
    
    # Scatter plot
    plt2 = DataVisualization.scatter_plot(df, 'age', 'income', title='Income vs Age')
    
    # Correlation heatmap
    plt3 = DataVisualization.correlation_heatmap(correlation_matrix, 'Correlation Heatmap')
    
    # 4. Advanced Analysis
    print("\n4. Advanced Analysis Demonstration")
    
    # Linear Regression
    X = df[['age', 'education_years']]
    y = df['income']
    
    regression_results = AdvancedAnalysis.linear_regression(X, y)
    print("Linear Regression Results:")
    print(f"R-squared: {regression_results['r_squared']}")
    print("Coefficients:", regression_results['coefficients'])
    
    # K-means Clustering
    clustering_results = AdvancedAnalysis.kmeans_clustering(X, n_clusters=2)
    print("\nK-means Clustering Labels:")
    print(clustering_results['cluster_labels'])
    
    # PCA
    pca_results = AdvancedAnalysis.principal_component_analysis(X)
    print("\nPCA Explained Variance Ratio:")
    print(pca_results['explained_variance_ratio'])
    
    # Show plots (comment out if running in script mode)
    plt.show()

if __name__ == "__main__":
    main()