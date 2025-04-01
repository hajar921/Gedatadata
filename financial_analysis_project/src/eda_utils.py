"""
Utility functions for exploratory data analysis (EDA).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union

def generate_summary_statistics(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Generate summary statistics for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to analyze
        
    Returns:
        DataFrame with summary statistics
    """
    summary = df[columns].describe()
    
    # Add additional statistics
    for col in columns:
        summary.loc['skew', col] = df[col].skew()
        summary.loc['kurtosis', col] = df[col].kurtosis()
        summary.loc['median', col] = df[col].median()
        summary.loc['iqr', col] = df[col].quantile(0.75) - df[col].quantile(0.25)
        summary.loc['range', col] = df[col].max() - df[col].min()
    
    return summary

def plot_distribution(df: pd.DataFrame, column: str, bins: int = 30, 
                     figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of a column with histogram, KDE, and boxplot.
    
    Args:
        df: Input DataFrame
        column: Column name to plot
        bins: Number of bins for histogram
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Create subplot with 1 row and 2 columns
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True, bins=bins)
    plt.title(f'Distribution of {column}')
    plt.axvline(df[column].mean(), color='red', linestyle='--', label=f'Mean: {df[column].mean():.2f}')
    plt.axvline(df[column].median(), color='green', linestyle='--', label=f'Median: {df[column].median():.2f}')
    plt.legend()
    
    # Add boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                           figsize: Tuple[int, int] = (12, 10), 
                           save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Plot correlation matrix for numeric columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to include (optional, defaults to all numeric columns)
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Correlation matrix DataFrame
    """
    # Select columns for correlation
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return corr_matrix

def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str, 
                figsize: Tuple[int, int] = (12, 6), 
                save_path: Optional[str] = None) -> None:
    """
    Create a boxplot to compare a numeric variable across categories.
    
    Args:
        df: Input DataFrame
        x_col: Column name for categories (x-axis)
        y_col: Column name for numeric values (y-axis)
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title(f'{y_col} by {x_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_time_series(df: pd.DataFrame, date_col: str, value_cols: List[str],
                    freq: str = 'M', figsize: Tuple[int, int] = (14, 10),
                    save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Plot time series data for specified columns.
    
    Args:
        df: Input DataFrame
        date_col: Column name containing dates
        value_cols: List of column names to plot
        freq: Frequency for resampling ('D' for daily, 'M' for monthly, etc.)
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Resampled DataFrame with time series data
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Set date as index
    df_ts = df.set_index(date_col)
    
    # Resample data
    df_resampled = df_ts[value_cols].resample(freq).mean()
    
    # Plot time series
    plt.figure(figsize=figsize)
    
    for i, col in enumerate(value_cols):
        plt.subplot(len(value_cols), 1, i+1)
        plt.plot(df_resampled.index, df_resampled[col])
        plt.title(f'{col} Over Time')
        plt.ylabel(col)
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return df_resampled

def identify_outliers(df: pd.DataFrame, column: str, method: str = 'iqr',
                     threshold: float = 1.5) -> Tuple[pd.DataFrame, float, float]:
    """
    Identify outliers in a column using IQR or Z-score method.
    
    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for Z-score)
        
    Returns:
        Tuple of (outliers DataFrame, lower bound, upper bound)
    """
    if method.lower() == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    elif method.lower() == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers_idx = np.where(z_scores > threshold)[0]
        outliers = df.iloc[outliers_idx]
        
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    return outliers, lower_bound, upper_bound

def plot_categorical_distribution(df: pd.DataFrame, column: str, 
                                 figsize: Tuple[int, int] = (12, 6),
                                 top_n: Optional[int] = None,
                                 save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of a categorical column.
    
    Args:
        df: Input DataFrame
        column: Categorical column name to plot
        figsize: Figure size as (width, height)
        top_n: Number of top categories to display (optional)
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    
    # Get value counts
    value_counts = df[column].value_counts()
    
    # Limit to top N if specified
    if top_n is not None and top_n < len(value_counts):
        value_counts = value_counts.head(top_n)
        title_suffix = f" (Top {top_n})"
    else:
        title_suffix = ""
    
    # Create bar plot
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f'Distribution of {column}{title_suffix}')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_grouped_metrics(df: pd.DataFrame, group_col: str, metric_cols: List[str],
                        agg_func: str = 'mean', top_n: Optional[int] = None,
                        figsize: Tuple[int, int] = (14, 10),
                        save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Plot metrics grouped by a categorical column.
    
    Args:
        df: Input DataFrame
        group_col: Column name to group by
        metric_cols: List of metric column names to aggregate
        agg_func: Aggregation function ('mean', 'sum', 'median', etc.)
        top_n: Number of top groups to display (optional)
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Grouped DataFrame with aggregated metrics
    """
    # Group data
    grouped_data = df.groupby(group_col)[metric_cols].agg(agg_func).reset_index()
    
    # Sort by the first metric column
    grouped_data = grouped_data.sort_values(by=metric_cols[0], ascending=False)
    
    # Limit to top N if specified
    if top_n is not None and top_n < len(grouped_data):
        grouped_data = grouped_data.head(top_n)
        title_suffix = f" (Top {top_n})"
    else:
        title_suffix = ""
    
    # Plot metrics
    plt.figure(figsize=figsize)
    
    for i, col in enumerate(metric_cols):
        plt.subplot(len(metric_cols), 1, i+1)
        sns.barplot(x=group_col, y=col, data=grouped_data)
        plt.title(f'{agg_func.capitalize()} {col} by {group_col}{title_suffix}')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return grouped_data.set_index(group_col)

