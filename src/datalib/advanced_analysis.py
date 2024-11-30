import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Tuple, Optional

class AdvancedAnalysis:
    """
    A class providing advanced data analysis methods for DataLib.
    
    Includes regression, classification, clustering, and dimensionality 
    reduction techniques.
    """
    
    @staticmethod
    def linear_regression(X: pd.DataFrame, 
                           y: pd.Series, 
                           test_size: float = 0.2) -> dict:
        """
        Perform linear regression analysis.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
        
        Returns:
            dict: Regression analysis results including model, coefficients, and performance metrics.
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'model': model,
            'coefficients': dict(zip(X.columns, model.coef_)),
            'intercept': model.intercept_,
            'mean_squared_error': mse,
            'r_squared': model.score(X_test, y_test)
        }
    
    @staticmethod
    def polynomial_regression(X: pd.DataFrame, 
                               y: pd.Series, 
                               degree: int = 2, 
                               test_size: float = 0.2) -> dict:
        """
        Perform polynomial regression analysis.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            degree (int, optional): Polynomial degree. Defaults to 2.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
        
        Returns:
            dict: Polynomial regression analysis results.
        """
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=42)
        
        # Fit polynomial regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'model': model,
            'mean_squared_error': mse,
            'r_squared': model.score(X_test, y_test)
        }
    
    @staticmethod
    def knn_classification(X: pd.DataFrame, 
                            y: pd.Series, 
                            n_neighbors: int = 5, 
                            test_size: float = 0.2) -> dict:
        """
        Perform K-Nearest Neighbors classification.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            n_neighbors (int, optional): Number of neighbors. Defaults to 5.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
        
        Returns:
            dict: KNN classification results.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Fit KNN classifier
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    @staticmethod
    def decision_tree_classification(X: pd.DataFrame, 
                                     y: pd.Series, 
                                     max_depth: Optional[int] = None, 
                                     test_size: float = 0.2) -> dict:
        """
        Perform Decision Tree classification.
        
        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target variable.
            max_depth (int, optional): Maximum tree depth. Defaults to None.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
        
        Returns:
            dict: Decision Tree classification results.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Fit Decision Tree classifier
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'feature_importance': dict(zip(X.columns, model.feature_importances_)),
            'predictions': y_pred
        }
    
    @staticmethod
    def kmeans_clustering(X: pd.DataFrame, 
                           n_clusters: int = 3, 
                           random_state: int = 42) -> dict:
        """
        Perform K-means clustering.
        
        Args:
            X (pd.DataFrame): Input features.
            n_clusters (int, optional): Number of clusters. Defaults to 3.
            random_state (int, optional): Random seed for reproducibility.
        
        Returns:
            dict: K-means clustering results.
        """
        # Fit K-means model
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        model.fit(X)
        
        # Cluster labels and centroids
        labels = model.labels_
        centroids = model.cluster_centers_
        
        return {
            'model': model,
            'cluster_labels': labels,
            'centroids': centroids
        }
    
    @staticmethod
    def principal_component_analysis(X: pd.DataFrame, 
                                     n_components: Optional[int] = None) -> dict:
        """
        Perform Principal Component Analysis (PCA).
        
        Args:
            X (pd.DataFrame): Input features.
            n_components (int, optional): Number of components to keep. 
                Defaults to None (min of features or samples).
        
        Returns:
            dict: PCA analysis results.
        """
        # Fit PCA model
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)
        
        return {
            'transformed_data': X_transformed,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_
        }