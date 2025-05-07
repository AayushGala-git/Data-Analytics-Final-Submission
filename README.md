# Advanced Count Regression Modeling Project

## Overview

This project implements and compares various count regression models for predicting article publication counts among biochemists, including:
- Poisson Regression
- Negative Binomial Regression
- Zero-Inflated Poisson (ZIP) Models
- Count Regression Trees (CORE Trees)

The analysis includes model comparison, cross-validation, variable importance assessment, sensitivity analysis, tree visualization, and residual diagnostics.

## Project Structure

- **Data Files:**
  - `biochemists_data.csv`: Main dataset with article counts and predictors
  - `ireland_data.csv`: Auxiliary dataset (not used in the main analysis)

- **Core Scripts:**
  - `load_biochemists.py`: Script to load data from R packages and save to CSV
  - `biochemists_analysis.py`: Initial exploratory data analysis
  - `model_evaluation_visualization.py`: Model comparison, validation, and diagnostics (Tasks 7-9)
  - `advanced_model_analysis.py`: Variable importance, sensitivity analysis, and reporting (Tasks 10-12)

- **Output Directories:**
  - `model_evaluation_results_20250507_183854/`: Results from model evaluation
  - `advanced_model_analysis_20250507_184607/`: Results from advanced analysis

- **Result Files:**
  - `count_modeling_results_20250507_182656.txt`: Text output from initial modeling
  - Various PNG files for plots and visualizations
  - CSV files with metrics and tables

## How to Reproduce the Analysis

### Prerequisites

Required Python packages:
```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
networkx
rpy2 (for loading data from R)
```

Install these packages using pip:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn networkx rpy2
```

### Step 1: Data Loading and Exploration

```bash
# Load data from R packages and save to CSV
python load_biochemists.py

# Run exploratory data analysis
python biochemists_analysis.py
```

This generates the initial visualization plots (plot1_*.png to plot8_*.png).

### Step 2: Model Comparison and Validation (Tasks 7-9)

```bash
# Run model comparison, validation and diagnostics
python model_evaluation_visualization.py
```

This performs:
- 10-fold cross-validation of all four models
- Calculation of performance metrics (MAE, MSE, RMSE, AIC, BIC)
- Tree structure visualization and interpretation
- Residual diagnostics

Results are saved to a timestamped directory `model_evaluation_results_*/`.

### Step 3: Advanced Analysis and Reporting (Tasks 10-12)

```bash
# Run variable importance, sensitivity analysis, and reporting
python advanced_model_analysis.py
```

This performs:
- Variable importance assessment across all models
- Partial dependence plots for top predictors
- Sensitivity analysis with different tree parameters
- Split stability assessment via bootstrapping
- Creation of final report and consolidated results

Results are saved to a timestamped directory `advanced_model_analysis_*/`.

## Key Results

Based on the analysis:

1. **Model Performance:**
   - Best model by MAE: CORE Tree (MAE = 1.317)
   - Best model by RMSE: Poisson (RMSE = 1.855)
   - Best model by AIC/BIC: Negative Binomial

2. **Top Predictors:**
   - Number of mentors
   - PhD quality
   - Marital status

3. **Key Findings:**
   - The data shows evidence of over-dispersion (Negative Binomial has better AIC/BIC)
   - CORE trees provide comparable accuracy with better interpretability
   - Number of mentors consistently emerged as the strongest predictor
   - Gender, marital status, and having young children all influence publication counts

## Files to Review

For a comprehensive overview of the results:

1. `advanced_model_analysis_*/final_report.md`: Complete summary report
2. `advanced_model_analysis_*/key_figures/`: Directory with the most important visualizations
3. `model_evaluation_results_*/analysis_summary.txt`: Summary of model evaluation

## Recommendations

1. **For prediction accuracy:** Use the Poisson model
2. **For interpretability:** Use the CORE tree model
3. **For policy implications:** Focus on mentorship programs
4. **For future research:** Consider additional variables related to research funding and institutional support

## Author

Aayush Gala
May 2025

## References

- Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data (Vol. 53). Cambridge University Press.
- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC Press.
- Long, J. S. (1997). Regression models for categorical and limited dependent variables. Sage Publications.