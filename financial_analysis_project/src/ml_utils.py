import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_pca(X, n_components=None, variance_threshold=0.95):
    """
    Perform PCA dimensionality reduction
    
    Args:
        X: Feature matrix
        n_components: Number of components to keep, if None, use variance_threshold
        variance_threshold: Minimum cumulative explained variance
        
    Returns:
        X_pca: Transformed data
        pca: Fitted PCA object
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # If n_components is not specified, use variance threshold
    if n_components is None:
        # First fit PCA with all components
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Find number of components that explain variance_threshold of variance
        explained_variance_ratio_cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(explained_variance_ratio_cumsum >= variance_threshold) + 1
        
        print(f"Selected {n_components} components explaining {variance_threshold*100:.1f}% of variance")
    
    # Fit PCA with selected number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Total explained variance: {sum(explained_variance):.4f}")
    
    return X_pca, pca

def plot_pca_components(pca, feature_names=None, output_path=None):
    """
    Plot PCA component loadings
    
    Args:
        pca: Fitted PCA object
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    # Get the number of components and features
    n_components = pca.n_components_
    n_features = pca.components_.shape[1]
    
    # Set feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Create a figure with a subplot for each component
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))
    
    # If there's only one component, axes is not an array
    if n_components == 1:
        axes = [axes]
    
    # Plot each component
    for i, (ax, component) in enumerate(zip(axes, pca.components_)):
        indices = np.argsort(np.abs(component))[::-1]
        ax.barh(range(n_features), component[indices])
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([feature_names[j] for j in indices])
        ax.set_xlabel('Component loading')
        ax.set_title(f'PCA Component {i+1} (Explained variance: {pca.explained_variance_ratio_[i]:.4f})')
        ax.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_pca_variance(pca, output_path=None):
    """
    Plot explained variance by PCA components
    
    Args:
        pca: Fitted PCA object
        output_path: Path to save the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot explained variance
    explained_variance = pca.explained_variance_ratio_
    components = range(1, len(explained_variance) + 1)
    
    ax1.bar(components, explained_variance)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Explained Variance by Component')
    ax1.grid(True)
    
    # Plot cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    
    ax2.plot(components, cumulative_variance, marker='o')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.grid(True)
    
    # Add horizontal lines at 0.8, 0.9, 0.95
    for threshold in [0.8, 0.9, 0.95]:
        ax2.axhline(y=threshold, linestyle='--', color='r', alpha=0.5)
        # Add text annotation
        component_idx = np.argmax(cumulative_variance >= threshold)
        ax2.text(component_idx + 1.5, threshold, f'{threshold:.0%} at {component_idx + 1} components', 
                 verticalalignment='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_pca_scatter(X_pca, labels=None, output_path=None):
    """
    Plot scatter plot of first two PCA components
    
    Args:
        X_pca: PCA-transformed data
        labels: Labels for coloring points (optional)
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Color by labels
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            plt.scatter(
                X_pca[labels == label, 0],
                X_pca[labels == label, 1],
                c=[colors[i]],
                label=f'Cluster {label}',
                alpha=0.7
            )
        plt.legend()
    else:
        # No labels, single color
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: First Two Principal Components')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def perform_kmeans_clustering(X, max_clusters=10, random_state=42):
    """
    Perform K-means clustering with elbow method to find optimal number of clusters
    
    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to try
        random_state: Random seed for reproducibility
        
    Returns:
        kmeans: Fitted KMeans object with optimal number of clusters
        labels: Cluster labels for each data point
        inertias: List of inertias for each number of clusters
    """
    # Calculate inertia for different numbers of clusters
    inertias = []
    models = []
    
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        models.append(kmeans)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('K-means Elbow Method')
    plt.grid(True)
    plt.show()
    
    # Ask for optimal number of clusters
    optimal_k = int(input("Enter the optimal number of clusters based on the elbow curve: "))
    
    # Return the model with optimal number of clusters
    optimal_model = models[optimal_k - 1]
    labels = optimal_model.labels_
    
    print(f"K-means clustering completed with {optimal_k} clusters")
    
    return optimal_model, labels, inertias

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create train-test split with proper stratification
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Train and test splits
    """
    from sklearn.model_selection import train_test_split
    
    # Check if classification or regression task
    if len(np.unique(y)) < 10:  # Assume classification if fewer than 10 unique values
        # Use stratified split for classification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        # Regular split for regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def perform_cross_validation(model, X, y, cv=5, scoring=None):
    """
    Perform cross-validation and return scores
    
    Args:
        model: The model/estimator
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        scoring: Scoring metric(s) to evaluate
        
    Returns:
        cv_results: Dictionary with cross-validation results
    """
    from sklearn.model_selection import cross_validate
    
    # Default scoring metrics based on problem type
    if scoring is None:
        if len(np.unique(y)) < 10:  # Classification
            scoring = ['accuracy', 'f1_weighted', 'roc_auc']
        else:  # Regression
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    # Print results
    print("Cross-validation results:")
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        print(f"{metric}: {np.mean(test_scores):.4f} (±{np.std(test_scores):.4f})")
    
    return cv_results

def plot_roc_curve(y_true, y_score, output_path=None):
    """
    Plot ROC curve for binary classification
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates of the positive class)
        output_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_score, output_path=None):
    """
    Plot precision-recall curve for binary classification
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates of the positive class)
        output_path: Path to save the plot
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Calculate precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return avg_precision

def plot_feature_distributions(df, features, hue=None, output_path=None):
    """
    Plot distributions of selected features
    
    Args:
        df: DataFrame containing the data
        features: List of features to plot
        hue: Column to use for color grouping (e.g., 'Cluster')
        output_path: Path to save the plot
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    plt.figure(figsize=(14, 4 * n_rows))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        if hue is not None and hue in df.columns:
            sns.histplot(data=df, x=feature, hue=hue, kde=True, alpha=0.6)
        else:
            sns.histplot(data=df, x=feature, kde=True)
            
        plt.title(f'Distribution of {feature}')
        plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_cluster_profiles(df, cluster_col, features, output_path=None):
    """
    Plot radar charts showing cluster profiles across features
    
    Args:
        df: DataFrame containing the data
        cluster_col: Column name containing cluster labels
        features: List of features to include in the profile
        output_path: Path to save the plot
    """
    from matplotlib.path import Path as MplPath
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D
    
    # Calculate mean values for each cluster and feature
    cluster_profiles = df.groupby(cluster_col)[features].mean()
    
    # Normalize the data for radar chart
    min_max_scaler = lambda x: (x - x.min()) / (x.max() - x.min())
    cluster_profiles_norm = cluster_profiles.apply(min_max_scaler)
    
    # Number of variables
    N = len(features)
    
    # Create angles for radar chart
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add feature labels
    plt.xticks(angles[:-1], features, size=12)
    
    # Draw cluster profiles
    for cluster in cluster_profiles_norm.index:
        values = cluster_profiles_norm.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    plt.title('Cluster Profiles Across Features', size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return cluster_profiles

def detect_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Detect outliers in specified columns
    
    Args:
        df: DataFrame containing the data
        columns: List of columns to check for outliers
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)
        
    Returns:
        outliers_df: DataFrame containing outlier information
    """
    outliers_info = []
    
    for col in columns:
        if method == 'iqr':
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(df)) * 100
            
            outliers_info.append({
                'Column': col,
                'Method': 'IQR',
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound,
                'Outlier_Count': outlier_count,
                'Outlier_Percent': outlier_percent
            })
            
        elif method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = df[z_scores > threshold]
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(df)) * 100
            
            outliers_info.append({
                'Column': col,
                'Method': 'Z-Score',
                'Threshold': threshold,
                'Outlier_Count': outlier_count,
                'Outlier_Percent': outlier_percent
            })
    
    outliers_df = pd.DataFrame(outliers_info)
    return outliers_df

def plot_decision_boundaries(X, y, model, feature_names=None, output_path=None):
    """
    Plot decision boundaries for a classifier (works best with 2D data)
    
    Args:
        X: Feature matrix (should be 2D for visualization)
        y: Target labels
        model: Trained classifier model
        feature_names: Names of the two features
        output_path: Path to save the plot
    """
    # Only use first two dimensions if X has more than 2 features
    if X.shape[1] > 2:
        print("Warning: Only using first two features for visualization")
        X_plot = X[:, :2]
    else:
        X_plot = X
    
    # Set feature names
    if feature_names is None or len(feature_names) < 2:
        feature_names = [f'Feature {i+1}' for i in range(2)]
    
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot data points
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, edgecolors='k', 
                         cmap=plt.cm.coolwarm, alpha=0.8)
    plt.legend(*scatter.legend_elements(), title="Classes")
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Boundaries')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def perform_grid_search(model, param_grid, X, y, cv=5, scoring=None):
    """
    Perform grid search for hyperparameter tuning
    
    Args:
        model: The model/estimator
        param_grid: Dictionary with parameters names as keys and lists of parameter values
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        scoring: Scoring metric to evaluate
        
    Returns:
        best_model: Model with best parameters
        cv_results: DataFrame with cross-validation results
    """
    from sklearn.model_selection import GridSearchCV
    
    # Set default scoring based on problem type
    if scoring is None:
        if len(np.unique(y)) < 10:  # Classification
            scoring = 'f1_weighted'
        else:  # Regression
            scoring = 'r2'
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best {scoring} score: {grid_search.best_score_:.4f}")
    
    # Create DataFrame with results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    return grid_search.best_estimator_, cv_results

def plot_grid_search_results(cv_results, param_name, output_path=None):
    """
    Plot grid search results for a specific parameter
    
    Args:
        cv_results: DataFrame with cross-validation results
        param_name: Parameter name to plot
        output_path: Path to save the plot
    """
    # Extract parameter values and scores
    param_values = cv_results[f'param_{param_name}'].astype(str)
    mean_scores = cv_results['mean_test_score']
    std_scores = cv_results['std_test_score']
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(param_values, mean_scores, yerr=std_scores, marker='o', linestyle='-')
    plt.xlabel(param_name)
    plt.ylabel('Mean CV Score')
    plt.title(f'Grid Search Results for {param_name}')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def save_model(model, model_path, metadata=None):
    """
    Save a trained model and metadata
    
    Args:
        model: Trained model to save
        model_path: Path to save the model
        metadata: Dictionary with additional metadata
    """
    import joblib
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create model info dictionary
    model_info = {
        'model': model,
        'metadata': metadata or {}
    }
    
    # Add timestamp to metadata
    from datetime import datetime
    model_info['metadata']['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save model
    joblib.dump(model_info, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """
    Load a saved model and metadata
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        model: Loaded model
        metadata: Model metadata
    """
    import joblib
    
    # Load model info
    model_info = joblib.load(model_path)
    
    # Extract model and metadata
    model = model_info['model']
    metadata = model_info['metadata']
    
    print(f"Model loaded from {model_path}")
    print(f"Model metadata: {metadata}")
    
    return model, metadata

def create_feature_pipeline(numeric_features, categorical_features, ordinal_features=None, ordinal_categories=None):
    """
    Create a preprocessing pipeline for mixed feature types
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        ordinal_features: List of ordinal feature names
        ordinal_categories: List of lists containing categories in order
        
    Returns:
        preprocessor: ColumnTransformer for preprocessing
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    
    # Numeric features pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Initialize transformers list
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
    
    # Add ordinal features if provided
    if ordinal_features and ordinal_categories:
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=ordinal_categories))
        ])
        transformers.append(('ord', ordinal_transformer, ordinal_features))
    
    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor

def plot_feature_distributions_by_target(df, features, target, output_path=None):
    """
    Plot feature distributions grouped by target variable
    
    Args:
        df: DataFrame containing the data
        features: List of features to plot
        target: Target variable name
        output_path: Path to save the plot
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    plt.figure(figsize=(14, 4 * n_rows))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        if df[feature].dtype.kind in 'ifc':  # Numeric features
            sns.kdeplot(data=df, x=feature, hue=target, common_norm=False)
        else:  # Categorical features
            sns.countplot(data=df, x=feature, hue=target)
            plt.xticks(rotation=45)
            
        plt.title(f'Distribution of {feature} by {target}')
        plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_cluster_scatter_matrix(df, features, cluster_col, output_path=None):
    """
    Plot scatter matrix of features colored by cluster
    
    Args:
        df: DataFrame containing the data
        features: List of features to include
        cluster_col: Column name containing cluster labels
        output_path: Path to save the plot
    """
    # Limit to a reasonable number of features
    if len(features) > 5:
        print(f"Warning: Limiting to first 5 features out of {len(features)}")
        features = features[:5]
    
    # Create scatter matrix
    plt.figure(figsize=(12, 12))
    scatter_matrix = pd.plotting.scatter_matrix(
        df[features], 
        c=df[cluster_col], 
        figsize=(12, 12), 
        marker='o',
        hist_kwds={'bins': 20},
        s=30, 
        alpha=0.8
    )
    
    # Add title to each subplot
    for ax in scatter_matrix.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    
    plt.suptitle('Scatter Matrix by Cluster', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_silhouette_analysis(X, cluster_labels, output_path=None):
    """
    Plot silhouette analysis for clustering evaluation
    
    Args:
        X: Feature matrix
        cluster_labels: Cluster labels
        output_path: Path to save the plot
        
    Returns:
        silhouette_avg: Average silhouette score
    """
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    # Get number of clusters
    n_clusters = len(np.unique(cluster_labels))
    
    # Create silhouette plot
    plt.figure(figsize=(12, 8))
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate silhouette scores for samples belonging to cluster i
        ith_cluster_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_values.sort()
        
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with cluster numbers
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
    
    plt.title(f"Silhouette Analysis (Average Score: {silhouette_avg:.3f})")
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster Label")
    
    # The vertical line for average silhouette score
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    plt.yticks([])  # Clear y-axis labels
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return silhouette_avg

def generate_model_report(model, X_train, X_test, y_train, y_test, is_classification=True):
    """
    Generate a comprehensive model performance report
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        is_classification: Whether this is a classification task
        
    Returns:
        report: Dictionary containing performance metrics
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Initialize report dictionary
    report = {
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'features': X_train.shape[1]
    }
    
    if is_classification:
        # Classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        
        # Training metrics
        report['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        report['train_precision'] = precision_score(y_train, y_train_pred, average='weighted')
        report['train_recall'] = recall_score(y_train, y_train_pred, average='weighted')
        report['train_f1'] = f1_score(y_train, y_train_pred, average='weighted')
        
        # Test metrics
        report['test_accuracy'] = accuracy_score(y_test, y_test_pred)
        report['test_precision'] = precision_score(y_test, y_test_pred, average='weighted')
        report['test_recall'] = recall_score(y_test, y_test_pred, average='weighted')
        report['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
        
        # Detailed classification report
        report['classification_report'] = classification_report(y_test, y_test_pred)
        
        # Print summary
        print("Classification Model Performance:")
        print(f"Training Accuracy: {report['train_accuracy']:.4f}")
        print(f"Test Accuracy: {report['test_accuracy']:.4f}")
        print(f"Test F1 Score: {report['test_f1']:.4f}")
        print("\nClassification Report:")
        print(report['classification_report'])
        
    else:
        # Regression metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Training metrics
        report['train_mse'] = mean_squared_error(y_train, y_train_pred)
        report['train_rmse'] = np.sqrt(report['train_mse'])
        report['train_mae'] = mean_absolute_error(y_train, y_train_pred)
        report['train_r2'] = r2_score(y_train, y_train_pred)
        
        # Test metrics
        report['test_mse'] = mean_squared_error(y_test, y_test_pred)
        report['test_rmse'] = np.sqrt(report['test_mse'])
        report['test_mae'] = mean_absolute_error(y_test, y_test_pred)
        report['test_r2'] = r2_score(y_test, y_test_pred)
        
        # Print summary
        print("Regression Model Performance:")
        print(f"Training R²: {report['train_r2']:.4f}")
        print(f"Test R²: {report['test_r2']:.4f}")
        print(f"Test RMSE: {report['test_rmse']:.4f}")
        print(f"Test MAE: {report['test_mae']:.4f}")
    
    return report

def plot_confusion_matrix_with_metrics(y_true, y_pred, class_names=None, output_path=None):
    """
    Plot confusion matrix with precision, recall, and F1-score metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Set class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix')
    
    # Plot metrics
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }, index=class_names)
    
    sns.heatmap(metrics_df, annot=True, cmap='Greens', ax=ax2)
    ax2.set_title('Classification Metrics by Class')
    ax2.set_ylabel('Class')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return metrics_df

def plot_feature_importance_comparison(models, model_names, feature_names, output_path=None):
    """
    Compare feature importances across multiple models
    
    Args:
        models: List of trained models with feature_importances_ attribute
        model_names: List of model names
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    # Extract feature importances from each model
    all_importances = []
    
    for i, model in enumerate(models):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            print(f"Warning: Model {model_names[i]} doesn't have feature importances")
            continue
            
        # Create DataFrame for this model
        model_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Model': model_names[i]
        })
        
        all_importances.append(model_importances)
    
    # Combine all importances
    if all_importances:
        importances_df = pd.concat(all_importances, ignore_index=True)
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', hue='Model', data=importances_df)
        plt.title('Feature Importance Comparison Across Models')
        plt.grid(axis='x')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        plt.show()
        
        return importances_df
    else:
        print("No feature importances available for the provided models")
        return None

def plot_model_comparison(models_performance, metric_name, higher_is_better=True, output_path=None):
    """
    Plot performance comparison across multiple models
    
    Args:
        models_performance: Dictionary with model names as keys and performance metrics as values
        metric_name: Name of the metric to compare
        higher_is_better: Whether higher metric values are better
        output_path: Path to save the plot
    """
    # Create DataFrame from performance dictionary
    performance_df = pd.DataFrame({
        'Model': list(models_performance.keys()),
        metric_name: [perf[metric_name] for perf in models_performance.values()]
    })
    
    # Sort by performance
    performance_df = performance_df.sort_values(
        by=metric_name, 
        ascending=not higher_is_better
    )
    
    # Plot performance comparison
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x='Model', y=metric_name, data=performance_df)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.01,
            f'{bar.get_height():.4f}',
            ha='center'
        )
    
    plt.title(f'Model Comparison by {metric_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return performance_df

def plot_learning_curves_comparison(estimators, estimator_names, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), 
                                   scoring='r2', output_path=None):
    """
    Plot learning curves for multiple models for comparison
    
    Args:
        estimators: List of models/estimators
        estimator_names: List of model names
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        train_sizes: Array of training set sizes to evaluate
        scoring: Scoring metric
        output_path: Path to save the plot
    """
    from sklearn.model_selection import learning_curve
    
    plt.figure(figsize=(12, 8))
    
    for i, (estimator, name) in enumerate(zip(estimators, estimator_names)):
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=-1)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes_abs, train_mean, 'o-', color=f'C{i}', label=f'{name} (train)')
        plt.plot(train_sizes_abs, test_mean, 's--', color=f'C{i}', label=f'{name} (test)')
        
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                         alpha=0.1, color=f'C{i}')
        plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, 
                         alpha=0.1, color=f'C{i}')
    
    plt.title('Learning Curves Comparison')
    plt.xlabel('Training Examples')
    plt.ylabel(f'{scoring} Score')
    plt.legend(loc='best')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_validation_curve(estimator, X, y, param_name, param_range, cv=5, scoring='r2', output_path=None):
    """
    Plot validation curve for a model parameter
    
    Args:
        estimator: The model/estimator
        X: Feature matrix
        y: Target vector
        param_name: Parameter name to vary
        param_range: Range of parameter values to try
        cv: Number of cross-validation folds
        scoring: Scoring metric
        output_path: Path to save the plot
    """
    from sklearn.model_selection import validation_curve
    
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Validation Curve for {param_name}")
    plt.xlabel(param_name)
    plt.ylabel(f"{scoring} Score")
    plt.grid(True)
    
    plt.plot(param_range, train_mean, 'o-', color="r", label="Training score")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.plot(param_range, test_mean, 's--', color="g", label="Cross-validation score")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    plt.legend(loc="best")
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    # Find best parameter value
    best_idx = np.argmax(test_mean)
    best_param = param_range[best_idx]
    best_score = test_mean[best_idx]
    
    print(f"Best {param_name}: {best_param} with score: {best_score:.4f}")
    
    return best_param, best_score

def create_ensemble_model(base_models, meta_model, X_train, y_train, X_test=None, y_test=None):
    """
    Create and evaluate a stacking ensemble model
    
    Args:
        base_models: List of (name, model) tuples for base models
        meta_model: Meta-learner model
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        y_test: Test target (optional)
        
    Returns:
        ensemble: Trained ensemble model
    """
    from sklearn.ensemble import StackingClassifier, StackingRegressor
    
    # Determine if classification or regression
    if len(np.unique(y_train)) < 10:  # Assume classification if fewer than 10 unique values
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
    else:  # Regression
        ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
    
    # Train the ensemble
    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate if test data is provided
    if X_test is not None and y_test is not None:
        y_pred = ensemble.predict(X_test)
        
        if len(np.unique(y_train)) < 10:  # Classification
            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(f"Ensemble Test Accuracy: {accuracy:.4f}")
            print(f"Ensemble Test F1 Score: {f1:.4f}")
        else:  # Regression
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"Ensemble Test R²: {r2:.4f}")
            print(f"Ensemble Test RMSE: {rmse:.4f}")
    
    return ensemble

def plot_permutation_importance(model, X, y, feature_names=None, n_repeats=10, random_state=42, output_path=None):
    """
    Plot permutation feature importance
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        n_repeats: Number of times to permute a feature
        random_state: Random seed
        output_path: Path to save the plot
        
    Returns:
        result: Permutation importance results
    """
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    
    # Set feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Create DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Permutation Importance')
    plt.grid(axis='x')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return result, importance_df

def plot_partial_dependence(model, X, features, feature_names=None, output_path=None):
    """
    Plot partial dependence plots for specified features
    
    Args:
        model: Trained model
        X: Feature matrix
        features: List of feature indices or names to plot
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    from sklearn.inspection import plot_partial_dependence
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate partial dependence plots
    display = plot_partial_dependence(
        model, X, features, 
        feature_names=feature_names,
        ax=ax
    )
    
    # Set title
    fig.suptitle('Partial Dependence Plots', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_calibration_curve(y_true, y_prob, n_bins=10, output_path=None):
    """
    Plot calibration curve for binary classification
    
    Args:
        y_true: True binary labels
        y_prob: Probability estimates of the positive class
        n_bins: Number of bins for calibration curve
        output_path: Path to save the plot
    """
    from sklearn.calibration import calibration_curve
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_cumulative_gain_curve(y_true, y_prob, output_path=None):
    """
    Plot cumulative gain curve for binary classification
    
    Args:
        y_true: True binary labels
        y_prob: Probability estimates of the positive class
        output_path: Path to save the plot
    """
    # Sort by probability
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate cumulative gain
    total_positive = np.sum(y_true)
    cum_gain = np.cumsum(y_true_sorted) / total_positive
    
    # Create percentiles
    percentiles = np.arange(1, len(y_true) + 1) / len(y_true)
    
    # Plot cumulative gain curve
    plt.figure(figsize=(10, 6))
    plt.plot(percentiles, cum_gain, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
    
    plt.xlabel('Percentage of sample')
    plt.ylabel('Percentage of positive outcomes')
    plt.title('Cumulative Gain Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_lift_curve(y_true, y_prob, n_bins=10, output_path=None):
    """
    Plot lift curve for binary classification
    
    Args:
        y_true: True binary labels
        y_prob: Probability estimates of the positive class
        n_bins: Number of bins for lift curve
        output_path: Path to save the plot
    """
    # Sort by probability
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate lift
    baseline = np.mean(y_true)
    bin_size = len(y_true) // n_bins
    lift_values = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
        bin_positive_rate = np.mean(y_true_sorted[start:end])
        lift_values.append(bin_positive_rate / baseline)
    
    # Plot lift curve
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_bins + 1), lift_values)
    plt.axhline(y=1, color='r', linestyle='--', label='Baseline')
    
    plt.xlabel('Decile')
    plt.ylabel('Lift')
    plt.title('Lift Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return lift_values

def plot_feature_clustering(X, feature_names=None, method='correlation', output_path=None):
    """
    Plot feature clustering based on correlation or mutual information
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        method: Clustering method ('correlation' or 'mutual_info')
        output_path: Path to save the plot
    """
    # Set feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    if method == 'correlation':
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Convert to distance matrix
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, dendrogram
        
        linkage_matrix = linkage(distance_matrix, method='average')
        
        # Plot dendrogram
        plt.figure(figsize=(14, 10))
        dendrogram(
            linkage_matrix,
            labels=feature_names,
            orientation='right',
            leaf_font_size=12
        )
        plt.title('Feature Clustering based on Correlation')
        plt.xlabel('Distance (1 - |Correlation|)')
        
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        # Create synthetic target for mutual information calculation
        y_synth = np.mean(X, axis=1)
        
        # Calculate mutual information
        mi_values = mutual_info_regression(X, y_synth)
        
        # Create DataFrame for plotting
        mi_df = pd.DataFrame({
            'Feature': feature_names,
            'Mutual Information': mi_values
        }).sort_values('Mutual Information', ascending=False)
        
        # Plot mutual information
        plt.figure(figsize=(12, 8))
        plt.barh(mi_df['Feature'], mi_df['Mutual Information'])
        plt.xlabel('Mutual Information')
        plt.title('Feature Mutual Information')
        plt.grid(axis='x')
    
    else:
        raise ValueError("Method must be 'correlation' or 'mutual_info'")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_tsne_visualization(X, labels=None, perplexity=30, n_iter=1000, output_path=None):
    """
    Plot t-SNE visualization of high-dimensional data
    
    Args:
        X: Feature matrix
        labels: Labels for coloring points (optional)
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        output_path: Path to save the plot
    """
    from sklearn.manifold import TSNE
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Plot t-SNE results
    plt.figure(figsize=(12, 10))
    
    if labels is not None:
        # Color by labels
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            plt.scatter(
                X_tsne[labels == label, 0],
                X_tsne[labels == label, 1],
                c=[colors[i]],
                label=f'Cluster {label}',
                alpha=0.7
            )
        plt.legend()
    else:
        # No labels, single color
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
    
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return X_tsne

def plot_umap_visualization(X, labels=None, n_neighbors=15, min_dist=0.1, output_path=None):
    """
    Plot UMAP visualization of high-dimensional data
    
    Args:
        X: Feature matrix
        labels: Labels for coloring points (optional)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        output_path: Path to save the plot
    """
    try:
        import umap
        
        # Apply UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        X_umap = reducer.fit_transform(X)
        
        # Plot UMAP results
        plt.figure(figsize=(12, 10))
        
        if labels is not None:
            # Color by labels
            unique_labels = np.unique(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                plt.scatter(
                    X_umap[labels == label, 0],
                    X_umap[labels == label, 1],
                    c=[colors[i]],
                    label=f'Cluster {label}',
                    alpha=0.7
                )
            plt.legend()
        else:
            # No labels, single color
            plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.7)
        
        plt.title('UMAP Visualization')
        plt.xlabel('UMAP feature 1')
        plt.ylabel('UMAP feature 2')
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path)
        plt.show()
        
        return X_umap
        
    except ImportError:
        print("UMAP is not installed. Please install it with 'pip install umap-learn'")
        return None

def plot_elbow_method(X, max_clusters=10, random_state=42, output_path=None):
    """
    Plot the elbow method to determine the optimal number of clusters for KMeans.
    
    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to try
        random_state: Random seed for reproducibility
        output_path: Path to save the plot (optional)
    
    Returns:
        optimal_k: Optimal number of clusters (user input).
    """
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)
    
    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path)
    
    plt.show()
    
    # Ask the user to input the optimal number of clusters based on the elbow curve
    optimal_k = 2
    return optimal_k


def plot_clusters(X, labels, output_path=None):
    """
    Plot clusters in 2D space.
    Args:
        X: Feature matrix (must be 2D for visualization)
        labels: Cluster labels
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f'Cluster {label}', alpha=0.7)
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Cluster Visualization')
    plt.grid(True)
    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_feature_importance(model, feature_names, output_path):
    """
    Plots feature importance from a trained Random Forest model and returns a DataFrame.
    
    Args:
        model (RandomForestClassifier): Trained Random Forest model.
        feature_names (list): List of feature names.
        output_path (str): Path to save the plot.
    
    Returns:
        pd.DataFrame: DataFrame containing feature names and their importances.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort features by importance

    # Create a DataFrame for the top features
    feature_importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    return feature_importance_df  # Return the DataFrame

def evaluate_classification_model(y_true, y_pred, output_path):
    """
    Evaluates a classification model and plots the confusion matrix.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        output_path (str): Path to save the plot.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def evaluate_regression_model(y_true, y_pred, output_path):
    """
    Evaluates a regression model and saves the results.
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        output_path (str): Path to save the evaluation metrics.
    
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Save metrics to the specified output path
    with open(output_path, 'w') as f:
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
    
    # Return metrics as a dictionary
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R² Score': r2
    }

def plot_residuals(y_true, y_pred, output_path):
    """
    Plots the residuals (errors) of a regression model.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# Add a main section to demonstrate usage if the script is run directly
if __name__ == "__main__":
    print("Machine Learning Utility Functions for Financial Analysis")
    print("Import this module to use the functions in your analysis.")
