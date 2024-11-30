# DataLib: Python Data Analysis Library

## Overview

DataLib is a comprehensive Python library designed to simplify data manipulation, statistical analysis, visualization, and machine learning tasks. It provides an intuitive and powerful set of tools for data scientists, researchers, and analysts.

## Features

### Data Manipulation
- CSV file loading and saving
- Data filtering
- Missing value handling
- Data normalization

### Statistical Analysis
- Descriptive statistics
- Correlation analysis
- T-tests
- Chi-square tests

### Data Visualization
- Bar plots
- Histograms
- Scatter plots
- Correlation heatmaps

### Advanced Analysis
- Linear and Polynomial Regression
- Classification Algorithms (KNN, Decision Trees)
- Clustering (K-means)
- Dimensionality Reduction (PCA)

## Installation

```bash
pip install datalib
```

## Quick Examples

### Data Manipulation
```python
from datalib.data_manipulation import DataManipulation

# Load CSV
df = DataManipulation.load_csv('data.csv')

# Filter data
filtered_df = DataManipulation.filter_data(df, {'age': lambda x: x > 25})
```

### Statistical Analysis
```python
from datalib.statistics import StatisticalAnalysis

# Calculate descriptive stats
stats = StatisticalAnalysis.descriptive_stats(df['column'])

# Correlation matrix
corr_matrix = StatisticalAnalysis.correlation(df)
```

### Visualization
```python
from datalib.visualization import DataVisualization

# Create bar plot
DataVisualization.bar_plot(df, 'category', 'value')

# Scatter plot
DataVisualization.scatter_plot(df, 'x_column', 'y_column')
```

## Contributing

Contributions are welcome! Please check our GitHub repository for guidelines.

## License

This project is licensed under the MIT License.