# Data Cleaning Steps

## 1. Handling Missing Values
- Identified missing values in Revenue, Cost, and Profit columns
- Replaced missing values in numeric columns with median values
- Replaced missing values in categorical columns with mode values

## 2. Standardizing Formats
- Removed currency symbols ($) and commas from monetary values
- Converted all monetary values to numeric format
- Standardized date formats to YYYY-MM-DD

## 3. Correcting Inconsistencies
- Standardized segment names (e.g., "Midmarket" vs "Mid Market")
- Corrected country name inconsistencies
- Standardized product names

## 4. Data Validation
- Verified that Revenue - Cost = Profit
- Checked for and removed duplicate records
- Validated that all categorical variables have consistent values

## 5. Feature Engineering
- Created ROA (Return on Assets) metric
- Created Profit_Margin feature
- Derived time-based features from date columns