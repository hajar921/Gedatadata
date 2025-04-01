import pandas as pd
import numpy as np
import re
from datetime import datetime

def load_raw_data(file_path):
    """
    Load raw financial data from CSV file
    
    Args:
        file_path (str): Path to the raw CSV file
        
    Returns:
        pd.DataFrame: Raw dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_currency(x):
    """
    Clean currency values by removing $, commas, and whitespace
    
    Args:
        x: Value to clean
        
    Returns:
        float: Cleaned numeric value
    """
    if isinstance(x, str):
        # Remove $, commas, and whitespace
        return float(re.sub(r'[^\d.]', '', x))
    return x

def standardize_date(x):
    """
    Standardize date format to YYYY-MM-DD
    
    Args:
        x: Date value to standardize
        
    Returns:
        str: Standardized date string
    """
    if pd.isna(x) or x == '-' or x == 'None':
        return np.nan
        
    if isinstance(x, str):
        # Try different date formats
        date_formats = [
            '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', 
            '%d-%m-%Y', '%B %d, %Y', '%d %B %Y'
        ]
        
        # Remove any extra whitespace
        x = x.strip()
        
        for fmt in date_formats:
            try:
                return datetime.strptime(x, fmt).strftime('%Y-%m-%d')
            except:
                continue
                
        # If it's just a month name
        months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        x_lower = x.lower()
        for month, num in months.items():
            if month in x_lower:
                # Extract year if present
                year_match = re.search(r'\d{4}', x)
                year = year_match.group(0) if year_match else '2023'  # Default to current year if not found
                return f"{year}-{num}-01"  # Default to first day of month
                
    return x

def normalize_categorical(df, column):
    """
    Normalize inconsistent categorical labels
    
    Args:
        df (pd.DataFrame): DataFrame containing the column
        column (str): Column name to normalize
        
    Returns:
        pd.Series: Normalized column
    """
    if column not in df.columns:
        return df[column]
        
    # Create a mapping dictionary for common inconsistencies
    # This should be expanded based on the actual data
    mappings = {
        'segment': {
            'midmarket': 'Mid Market',
            'mid market': 'Mid Market',
            'mid-market': 'Mid Market',
            'enterprise': 'Enterprise',
            'small business': 'Small Business',
            'smallbusiness': 'Small Business',
            'small-business': 'Small Business'
        },
        'country': {
            'usa': 'United States',
            'us': 'United States',
            'united states': 'United States',
            'uk': 'United Kingdom',
            'united kingdom': 'United Kingdom',
            'can': 'Canada',
            'aus': 'Australia'
        },
        'product': {
            'prod a': 'Product A',
            'product-a': 'Product A',
            'prod b': 'Product B',
            'product-b': 'Product B'
        }
    }
    
    col_lower = column.lower()
    if col_lower in mappings:
        # Apply mapping to normalize values
        return df[column].apply(lambda x: mappings[col_lower].get(str(x).lower(), x) if pd.notna(x) else x)
    
    return df[column]

def clean_financial_data(df):
    """
    Clean the financial dataset
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Replace placeholder values with NaN
    cleaned_df.replace(['-', 'None', 'NaN', 'nan', 'NULL'], np.nan, inplace=True)
    
    # Clean currency columns (assuming Revenue, Cost, Profit are currency)
    currency_columns = ['Revenue', 'Cost', 'Profit']
    for col in currency_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(clean_currency)
    
    # Standardize date columns (assuming there's a Date column)
    date_columns = [col for col in cleaned_df.columns if 'date' in col.lower()]
    for col in date_columns:
        cleaned_df[col] = cleaned_df[col].apply(standardize_date)
    
    # Normalize categorical columns
    categorical_columns = ['Segment', 'Country', 'Product']
    for col in categorical_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = normalize_categorical(cleaned_df, col)
    
    # Handle missing values
    # For numeric columns, impute with median
    numeric_columns = cleaned_df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        if cleaned_df[col].isna().any():
            median_value = cleaned_df[col].median()
            cleaned_df[col].fillna(median_value, inplace=True)
    
    # For categorical columns, impute with mode
    cat_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        if cleaned_df[col].isna().any():
            mode_value = cleaned_df[col].mode()[0]
            cleaned_df[col].fillna(mode_value, inplace=True)
    
    return cleaned_df

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        output_path (str): Path to save the cleaned CSV
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
