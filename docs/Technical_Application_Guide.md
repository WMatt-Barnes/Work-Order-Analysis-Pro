# WOAPro Technical Application Guide

## Overview

WOAPro (Work Order Analysis Professional) is a comprehensive reliability engineering software that employs advanced statistical methods to analyze maintenance work order data. This guide details the statistical methodologies implemented across all modules, providing professionals with a clear understanding of the analytical techniques used.

## Table of Contents

1. [Core Statistical Methods](#core-statistical-methods)
2. [Weibull Analysis Module](#weibull-analysis-module)
3. [Crow-AMSAA Analysis](#crow-amsaa-analysis)
4. [Preventive Maintenance Analysis](#preventive-maintenance-analysis)
5. [Spares Analysis Module](#spares-analysis-module)
6. [FMEA Export Module](#fmea-export-module)
7. [AI Classification Methods](#ai-classification-methods)
8. [Risk Assessment Methods](#risk-assessment-methods)
9. [Pareto Analysis](#pareto-analysis)

---

## Core Statistical Methods

### 1. Mean Time Between Failures (MTBF) Calculation

**Method**: Crow-AMSAA based MTBF estimation
**Formula**: MTBF = 1 / (λ × 365^β)
**Where**:
- λ (lambda) = intensity parameter
- β (beta) = shape parameter from Crow-AMSAA model

**Implementation**: 
- Uses Crow-AMSAA parameters to calculate annualized failure rate
- Accounts for censored data through included_indices filtering
- Provides more accurate MTBF than simple time averaging

### 2. Fuzzy String Matching

**Method**: RapidFuzz similarity scoring with NLTK preprocessing
**Algorithm**: 
- Text normalization (lowercase, punctuation removal)
- Abbreviation expansion (comp → compressor, leek → leak)
- Snowball stemming for word root matching
- Cosine similarity scoring with 75% threshold

**Application**: Failure mode dictionary matching for automatic classification

---

## Weibull Analysis Module

### 1. Maximum Likelihood Estimation (MLE)

**Method**: Numerical optimization of Weibull parameters
**Objective Function**: Negative log-likelihood minimization
**Formula**: 
```
L(β,η) = n*log(β) - n*β*log(η) + (β-1)*Σlog(t_i) - Σ(t_i/η)^β
```

**Parameters**:
- β (beta): Shape parameter (0.1 < β < 10.0)
- η (eta): Scale parameter (1.0 < η < 10000.0)

**Optimization**: L-BFGS-B algorithm with bounded constraints

### 2. Reliability Function

**Formula**: R(t) = exp(-(t/η)^β)
**Application**: Probability of survival at time t

### 3. Hazard Rate Function

**Formula**: h(t) = (β/η) * (t/η)^(β-1)
**Application**: Instantaneous failure rate at time t

### 4. Confidence Bounds

**Method**: Bootstrap resampling with 95% confidence intervals
**Implementation**:
- 1000 bootstrap samples
- Percentile-based confidence bounds
- Accounts for parameter uncertainty

### 5. Goodness-of-Fit Assessment

**Metrics**:
- Anderson-Darling test statistic
- Kolmogorov-Smirnov test
- Visual probability plot analysis
- R-squared correlation coefficient

---

## Crow-AMSAA Analysis

### 1. Parameter Estimation

**Method**: Linear regression on log-transformed data
**Model**: N(t) = λt^β
**Linear Form**: log(N) = log(λ) + β*log(t)

**Where**:
- N(t) = cumulative failures at time t
- λ = intensity parameter
- β = growth parameter

### 2. Growth Trend Analysis

**Interpretation**:
- β < 1: Reliability growth (decreasing failure rate)
- β = 1: Constant failure rate (exponential)
- β > 1: Reliability decay (increasing failure rate)

### 3. Failures per Year Calculation

**Formula**: Failures/year = λ × 365^β
**Application**: Annualized failure rate prediction

---

## Preventive Maintenance Analysis

### 1. PM vs Breakdown Separation

**Method**: Text-based classification using keyword matching
**PM Indicators**: 'pm', 'preventive', 'scheduled', 'routine', 'inspection', 'lubrication', 'calibration'

### 2. Optimal PM Frequency Calculation

**Method**: Weibull-based optimization
**Objective**: Minimize total cost (PM + breakdown costs)
**Formula**: 
```
Total Cost = (PM Cost × PM Frequency) + (Breakdown Cost × Breakdown Frequency)
```

**Constraints**:
- PM frequency ≥ 0
- Breakdown frequency derived from Weibull reliability function

### 3. PM Effectiveness Metrics

**Metrics**:
- PM Ratio: PM work orders / Total work orders
- Cost Ratio: PM costs / Total costs
- Effectiveness Score: Weighted combination of ratios

### 4. Equipment-Specific Analysis

**Method**: Separate MTBF and Weibull calculations per equipment
**Application**: Tailored PM recommendations for individual assets

---

## Spares Analysis Module

### 1. Demand Rate Calculation

**Method**: Time-based demand rate estimation
**Formula**: Demand Rate = (Demand Count - 1) / Time Span × 365
**Application**: Annualized spare parts demand forecasting

### 2. Monte Carlo Simulation

**Method**: Stochastic simulation of demand patterns
**Parameters**:
- Poisson distribution for demand arrivals
- Weibull distribution for failure times
- Lead time considerations

**Simulation Steps**:
1. Generate failure times using Weibull parameters
2. Simulate demand arrivals using Poisson process
3. Calculate stockout events and service levels
4. Repeat 1000 times for statistical significance

### 3. Economic Order Quantity (EOQ)

**Formula**: EOQ = √(2 × Annual Demand × Order Cost / Holding Cost Rate)
**Application**: Optimal order quantity calculation

### 4. Stockout Risk Assessment

**Method**: Probability calculation based on demand distribution
**Formula**: P(Stockout) = P(Demand > Current Stock + Lead Time Demand)
**Application**: Service level optimization

---


### 3. Weibull Parameter Integration

**Integration**: Weibull parameters (β, η) included in FMEA export
**Application**: Reliability-based failure mode prioritization

---

## AI Classification Methods

### 1. Expert System Classification

**Method**: Rule-based pattern matching
**Features**:
- Keyword-based scoring
- Regular expression pattern matching
- Equipment-specific context rules
- Weighted confidence scoring

### 2. Embedding-Based Classification

**Method**: Sentence transformer similarity scoring
**Algorithm**: 
- BERT-based sentence embeddings
- Cosine similarity calculation
- Confidence threshold filtering (30%)

### 3. SpaCy NLP Analysis

**Features**:
- Named entity recognition
- Part-of-speech tagging
- Dependency parsing
- Equipment type detection

### 4. Temporal Pattern Analysis

**Method**: Time-series pattern recognition
**Features**:
- Historical failure pattern analysis
- Seasonal trend detection
- Equipment-specific temporal context

---

## Risk Assessment Methods

### 1. Multi-Dimensional Risk Scoring

**Dimensions**:
- Frequency (failure rate)
- Cost impact
- Equipment criticality
- Detection difficulty

**Formula**: Risk Score = w₁×Frequency + w₂×Cost + w₃×Criticality + w₄×Detection

### 2. Risk Matrix Visualization

**Method**: 2D scatter plot with color-coded risk levels
**Axes**: 
- X-axis: Cost impact
- Y-axis: Frequency
- Color: Risk level (Low/Medium/High/Critical)

### 3. Segmented Analysis

**Method**: Time-based risk segmentation
**Application**: Trend analysis and intervention planning

---

## Pareto Analysis

### 1. Cost Pareto Analysis

**Method**: Cumulative cost contribution analysis
**Formula**: Cumulative % = (Cumulative Cost / Total Cost) × 100
**Application**: Identify high-cost failure modes

### 2. Frequency Pareto Analysis

**Method**: Failure count ranking
**Application**: Identify most frequent failure modes

### 3. Failure Rate Pareto Analysis

**Method**: Annualized failure rate ranking
**Application**: Identify failure modes with highest occurrence rates

---

## Statistical Assumptions and Limitations

### 1. Data Requirements

**Minimum Data Points**:
- Weibull Analysis: 3+ failures
- Crow-AMSAA: 2+ failures
- PM Analysis: 1+ PM and 1+ breakdown work order

### 2. Distribution Assumptions

**Weibull Distribution**: Assumes failure times follow Weibull distribution
**Poisson Process**: Assumes demand arrivals follow Poisson distribution
**Normal Distribution**: Used for confidence interval calculations

### 3. Censoring Considerations

**Method**: Right-censored data handling through included_indices filtering
**Application**: Accounts for equipment still in service

### 4. Confidence Intervals

**Method**: Bootstrap resampling (95% confidence level)
**Application**: Parameter uncertainty quantification

---

## Output Interpretations

### 1. Weibull Parameters

- **β < 1**: Early life failures (infant mortality)
- **β = 1**: Random failures (exponential distribution)
- **β > 1**: Wear-out failures (aging)

### 2. Crow-AMSAA Parameters

- **β < 1**: Reliability growth
- **β = 1**: Constant failure rate
- **β > 1**: Reliability decay



---

## References

1. Crow, L.H. (1974). "Reliability Analysis for Complex, Repairable Systems"
2. Weibull, W. (1951). "A Statistical Distribution Function of Wide Applicability"
3. MIL-HDBK-189 (1981). "Reliability Growth Management"
4. IEC 60812 (2018). "Analysis techniques for system reliability - Procedure for failure mode and effects analysis"

---

*This technical guide provides the statistical foundation for WOAPro's analytical capabilities. For implementation details, refer to the source code documentation.* 