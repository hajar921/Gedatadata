# Financial Analysis Project

## Overview

This project analyzes financial data to identify patterns, customer segments, and predictive factors for profitability. It implements a comprehensive pipeline from data cleaning through exploratory analysis to machine learning models for clustering, classification, and regression.

## Repository Structure

```
financial_analysis_project/
├── data/
│   ├── raw/              # Original data files
│   └── processed/        # Cleaned and transformed data
├── notebooks/
│   ├── 1_Data_Cleaning.ipynb   # Data preparation and cleaning
│   ├── 2_EDA.ipynb             # Exploratory data analysis
│   ├── 3_Clustering.ipynb      # Customer segmentation
│   ├── 4_Classification.ipynb  # Predictive classification
│   └── 5_Regression.ipynb      # Profit prediction
├── results/
│   ├── plots/            # Visualizations
│   │   ├── eda/          # Exploratory analysis plots
│   │   ├── clustering/   # Cluster analysis visualizations
│   │   ├── classification/ # Model performance charts
│   │   └── regression/   # Regression analysis plots
│   ├── models/           # Saved model files
│   └── reports/          # Generated reports and findings
│       ├── final_report.md           # Comprehensive project findings
│       ├── data_cleaning_steps.md    # Detailed data preparation process
│       └── model_limitations.md      # Analysis of model constraints
├── src/                  # Reusable code modules
│   ├── data_loader.py    # Functions for loading and cleaning data
│   ├── eda_utils.py      # Utilities for exploratory analysis
│   ├── ml_utils.py       # Machine learning utility functions
│   └── ml_models.py      # Model implementation functions
└── README.md
```

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/hajar921/Gedatadata.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Execution Order

Run the notebooks in the following order:
1. `1_Data_Cleaning.ipynb` - Prepares and cleans the raw financial data
2. `2_EDA.ipynb` - Performs exploratory analysis to identify patterns and relationships
3. `3_Clustering.ipynb` - Segments customers using unsupervised learning
4. `4_Classification.ipynb` - Builds classification models to predict categorical outcomes
5. `5_Regression.ipynb` - Develops regression models to predict profit and other metrics

## Key Documentation

### [Final Report](results/reports/final_report.md)
Comprehensive summary of all analyses, findings, and recommendations from the project. Includes executive summary, methodology overview, key insights, and strategic recommendations.

### [Data Cleaning Steps](results/reports/data_cleaning_steps.md)
Detailed documentation of the data preparation process, including handling of missing values, standardization of formats, correction of inconsistencies, and feature engineering.

### [Model Limitations](results/reports/model_limitations.md)
Analysis of constraints and limitations in the modeling approaches used, including assumptions of clustering algorithms, classification challenges, regression limitations, and general data constraints.

## Key Features

- **Data Cleaning Pipeline**: Handles missing values, standardizes formats, and corrects inconsistencies
- **Comprehensive EDA**: Analyzes distributions, correlations, and segment/region trends
- **Advanced Visualizations**: Includes distribution plots, correlation matrices, and cluster profiles
- **Machine Learning Models**: Implements clustering, classification, and regression with proper validation
- **Modular Code Structure**: Organizes reusable functions in the `src` directory for maintainability

## Machine Learning Capabilities

- **Clustering**: K-means and DBSCAN for customer segmentation
- **Classification**: Random Forest and Logistic Regression with hyperparameter tuning
- **Regression**: Linear and Random Forest regression with feature importance analysis
- **Model Evaluation**: Includes cross-validation, ROC curves, and feature importance visualization

## Dependencies

- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualization
- scikit-learn: Machine learning algorithms
- jupyter: Interactive notebook environment


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial data analysis best practices
- Scikit-learn documentation and examples
- Data visualization techniques from the Python community
