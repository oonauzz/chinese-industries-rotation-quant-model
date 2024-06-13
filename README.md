# Quantitative Industry Rotation Model README

This Python script implements a quantitative industry rotation model for financial analysis. The model is designed to analyze industry-level data and provide insights for portfolio management and trading strategies. Below is an overview of the functionalities and usage instructions for the script:

## Prerequisites

- Python 3.x
- Required libraries: `scipy`, `numpy`, `pandas`, `matplotlib`, `statsmodels`

## How to Use

1. Make sure Python and the required libraries are installed.
2. Run the script in a Python environment.

### Portfolio Performance Evaluation

The script includes functions to evaluate portfolio performance, such as:

- Computing cumulative returns
- Calculating maximum drawdown
- Estimating annualized volatility
- Determining annual return
- Calculating the Sharpe ratio
- Computing the Calmar ratio
- Determining the win ratio
- Estimating the high watermark (HWM)

### Factor Analysis

The script provides functions for factor analysis, including:

- Computing IC (Information Coefficient) and Rank IC
- Grouping factors and calculating returns for each group

### Data Preprocessing

The script preprocesses financial data, including:

- Reading financial data from Excel files
- Resampling daily data to monthly data
- Computing various types of returns (intraday, overnight, all day)

### Visualization

The script offers visualizations for better understanding of the data, such as:

- Plotting cumulative returns for different industry groups
- Visualizing portfolio performance over time

## Usage Examples

The script includes usage examples for each function, demonstrating how to utilize them with sample industry data.
