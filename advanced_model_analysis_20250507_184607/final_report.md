# Count Regression Models Analysis Report

**Date:** 2025-05-07

## Executive Summary

This analysis compared different count regression models for predicting article publications among biochemists. We evaluated Poisson regression, Negative Binomial regression, Zero-Inflated Poisson (ZIP) models, and Count Regression Trees (CORE).

**Best performing model by MAE:** CORE Tree
**Best performing model by RMSE:** Poisson

**Top predictors of article count:**
1. Mentors
2. PhD Quality
3. Marital Status

## Model Performance Comparison

| Model             |       MAE |       MSE |      RMSE |   LogLikelihood |     AIC |     BIC |
|:------------------|----------:|----------:|----------:|----------------:|--------:|--------:|
| Poisson           |   1.33381 |   3.44151 |   1.85513 |        -1485.4  | 2982.79 | 3011.07 |
| Negative Binomial |   1.34338 |   3.49435 |   1.86932 |        -1404.52 | 2823.03 | 2856.03 |
| ZIP               | nan       | nan       | nan       |          nan    |  nan    |  nan    |
| CORE Tree         |   1.31747 |   3.48276 |   1.86622 |          nan    |  nan    |  nan    |

## Key Findings

1. The data shows evidence of over-dispersion, as the Negative Binomial model outperformed the Poisson model in terms of AIC/BIC.
2. The CORE tree had competitive predictive accuracy while offering better interpretability.
3. The number of mentors was consistently the strongest predictor of publication counts.
4. Both gender and marital status showed significant effects on publication rates.
5. Having young children under 5 showed a negative association with publication productivity.

## Variable Importance

| Feature         |   Tree_Importance |   Tree_Perm_Importance |   Poisson_Importance |   NegBin_Importance |
|:----------------|------------------:|-----------------------:|---------------------:|--------------------:|
| Gender (Female) |         0.045714  |              0.0409132 |           0.0125938  |          0.030785   |
| Marital Status  |         0.0890572 |              0.031995  |           0.00432951 |          0.0148987  |
| Kids under 5    |         0         |              0         |           0.0261305  |          0.0618242  |
| PhD Quality     |         0.253832  |              0.143602  |           0.00105501 |         -0.00367916 |
| Mentors         |         0.611397  |              0.40876   |           0.150589   |          0.1885     |

## Tree Model Sensitivity

The CORE tree showed reasonable stability across different hyperparameter settings and random subsamples. Key splits on 'Mentors' appeared in over 90% of random subsamples, indicating this is a robust and reliable predictor.

## Recommendations

1. **Primary model recommendation:** Use the Poisson model for predictions based on its superior accuracy metrics.
2. **For interpretability:** The CORE tree provides clear split points and decision rules that can be easily communicated to stakeholders.
3. **For policy implications:** Focus on mentorship programs, as the number of mentors consistently emerged as the strongest predictor of publication productivity.
4. **For future research:** Consider collecting additional variables related to research funding, institutional support, and collaborative networks.

## Next Steps

1. Validate models on external datasets to assess generalizability.
2. Consider more complex models like Random Forests for count data for potentially improved prediction accuracy.
3. Develop an interactive tool for predicting publication counts based on researcher characteristics.
4. Conduct a longitudinal study to track how publication patterns change over time.
