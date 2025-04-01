import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_features(df, target_col=None, categorical_cols=None, numerical_cols=None):
    """
    Prepare features for machine learning
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str, optional): Target column name
        categorical_cols (list, optional): List of categorical column names
        numerical_cols (list, optional): List of numerical column names
        
    Returns:
        tuple: X (features), y (target if provided), preprocessor (column transformer)
    """
    # If columns not specified, infer them
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
            
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Prepare X and y
    if target_col:
        X = df[numerical_cols + categorical_cols]
        y = df[target_col]
        return X, y, preprocessor
    else:
        X = df[numerical_cols + categorical_cols]
        return X, None, preprocessor

def perform_clustering(X, preprocessor, n_clusters=None, method='kmeans', plot_results=True, save_path=None):
    """
    Perform clustering on the data
    
    Args:
        X (pd.DataFrame): Features
        preprocessor (ColumnTransformer): Feature preprocessor
        n_clusters (int, optional): Number of clusters for KMeans
        method (str): Clustering method ('kmeans' or 'dbscan')
        plot_results (bool): Whether to plot results
        save_path (str, optional): Path to save the plot
        
    Returns:
        tuple: Cluster labels, model
    """
    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)
    
    # For KMeans, find optimal number of clusters if not provided
    if method == 'kmeans':
        if n_clusters is None:
            # Elbow method to find optimal number of clusters
            inertia = []
            silhouette_scores = []
            k_range = range(2, 11)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_processed)
                inertia.append(kmeans.inertia_)
                
                # Calculate silhouette score
                labels = kmeans.labels_
                silhouette_scores.append(silhouette_score(X_processed, labels))
            
            # Plot elbow curve
            if plot_results:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                ax1.set_xlabel('Number of clusters')
                ax1.set_ylabel('Inertia', color='tab:blue')
                ax1.plot(k_range, inertia, 'o-', color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                
                ax2 = ax1.twinx()
                ax2.set_ylabel('Silhouette Score', color='tab:red')
                ax2.plot(k_range, silhouette_scores, 'o-', color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                
                plt.title('Elbow Method and Silhouette Score for Optimal k')
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path)
                    plt.close()
                else:
                    plt.show()
            
            # Choose optimal k (could be more sophisticated)
            n_clusters = k_range[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters: {n_clusters}")
        
        # Fit KMeans with optimal/provided number of clusters
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_processed)
        
    elif method == 'dbscan':
        # DBSCAN parameters
        eps = 0.5  # Distance threshold
        min_samples = 5  # Minimum number of samples in a neighborhood
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_processed)
        
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    return labels, model

def train_classification_model(X, y, preprocessor, model_type='rf', test_size=0.2, random_state=42):
    """
    Train a classification model
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        preprocessor (ColumnTransformer): Feature preprocessor
        model_type (str): Model type ('rf' for Random Forest, 'lr' for Logistic Regression)
        test_size (float): Test set size
        random_state (int): Random state
        
    Returns:
        tuple: Trained model, test predictions, test metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create pipeline
    if model_type == 'rf':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=random_state))
        ])
        
        # Parameter grid for grid search
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30]
        }
        
    elif model_type == 'lr':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
        ])
        
        # Parameter grid for grid search
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs']
        }
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    return best_model, y_pred, metrics

def train_regression_model(X, y, preprocessor, model_type='rf', test_size=0.2, random_state=42):
    """
    Train a regression model
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        preprocessor (ColumnTransformer): Feature preprocessor
        model_type (str): Model type ('rf' for Random Forest, 'lr' for Linear Regression)
        test_size (float): Test set size
        random_state (int): Random state
        
    Returns:
        tuple: Trained model, test predictions, test metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create pipeline
    if model_type == 'rf':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=random_state))
        ])
        
        # Parameter grid for grid search
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 10, 20, 30]
        }
        
    elif model_type == 'lr':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # No hyperparameters to tune for simple linear regression
        param_grid = {}
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Grid search if there are parameters to tune
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    return best_model, y_pred, metrics

def plot_feature_importance(model, X, preprocessor, top_n=10, save_path=None):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features
        preprocessor (ColumnTransformer): Feature preprocessor
        top_n (int): Number of top features to show
        save_path (str, optional): Path to save the plot
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model[-1], 'feature_importances_'):
        print("Model doesn't support feature importance visualization")
        return
    
    # Get feature names
    feature_names = []
    
    # Get numerical feature names
    num_transformer = preprocessor.named_transformers_['num']
    if hasattr(preprocessor, 'transformers_'):
        num_cols = preprocessor.transformers_[0][2]
        feature_names.extend(num_cols)
    
    # Get one-hot encoded feature names
    cat_transformer = preprocessor.named_transformers_['cat']
    if hasattr(cat_transformer, 'get_feature_names_out'):
        cat_cols = preprocessor.transformers_[1][2]
        cat_features = cat_transformer.get_feature_names_out(cat_cols)
        feature_names.extend(cat_features)
    
    # Get feature importances
    importances = model[-1].feature_importances_
    
    # Create DataFrame for plotting
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
