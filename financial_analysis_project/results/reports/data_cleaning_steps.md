## Data Cleaning Steps
1. **Handling Missing Values**
    Identified missing values in Sales, COGS, and Profit columns.
    Replaced missing values in numeric columns with median values.
    Replaced missing values in categorical columns with mode values.
    Note: In the financial cleaning process, rows with NaN values in Sales, COGS, or Profit were dropped to ensure data integrity during initial cleaning.

2. **Standardizing Formats**
    Column Names : Stripped leading/trailing spaces from column names and replaced spaces within column names with underscores for consistency.
    Removed currency symbols ($), commas, and extra spaces from monetary values in the Sales, COGS, and Profit columns.
    Converted all monetary values to numeric format, coercing invalid entries to NaN.
    Standardized date formats to YYYY-MM-DD.
    Example of column renaming:
        If a column was named "Sales Revenue", it was renamed to Sales_Revenue.

3. **Correcting Inconsistencies**
    Standardized segment names (e.g., "Midmarket" vs "Mid Market").
    Corrected country name inconsistencies.
    Standardized product names.
    Ensured no spaces or special characters remain in column names after renaming.
4. **Data Validation**
    Verified that Sales - COGS = Profit. Any discrepancies were flagged for further review.
    Checked for and removed duplicate records.
    Validated that all categorical variables have consistent values across the dataset.
    Data Types After Cleaning:
        The Sales, COGS, and Profit columns were successfully converted to numeric types, ensuring accurate calculations.
    Cleaned Financial Columns (First 5 Rows):
    Displayed the cleaned financial columns to confirm successful transformation.

5. Feature Engineering
    Created ROA (Return on Assets) metric.
    Created Profit_Margin feature using the formula (Profit / Sales) * 100.
    Derived time-based features from date columns, such as year, month, and quarter.
    Dataset Shape After Cleaning:
        The final dataset shape is [rows, columns], with all financial columns free of non-numeric characters and missing values handled appropriately.

    Columns:
        A list of all columns in the cleaned dataset was generated for reference.

    Full Data Types:
        All column data types were verified and printed to ensure consistency.

## Summary of Key Steps:
    Loaded the cleaned data from ../data/processed/cleaned_data.csv.
    Stripped spaces from column names and replaced spaces with underscores where necessary.
    Cleaned financial columns (Sales, COGS, Profit) by removing $, commas, and spaces,  then converting them to numeric format.
    Dropped rows with NaN values in financial columns to maintain data quality.
    Verified data types and displayed basic information about the cleaned dataset.
    This ensures the dataset is ready for further analysis or modeling.