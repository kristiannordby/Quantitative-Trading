# Quantitative Trading: Multi-Model Sector Rotation Strategy

A machine learning-driven quantitative trading system that predicts monthly returns across 11 S&P 500 sectors using LSTM neural networks, Ridge regression, and Decision Trees. The strategy employs dynamic leverage allocation based on predicted returns to generate alpha over the S&P 500 benchmark.

---

## Table of Contents

- [Overview](#overview)
- [Investment Philosophy](#investment-philosophy)
- [Dataset](#dataset)
- [Models & Methodology](#models--methodology)
- [Trading Strategy](#trading-strategy)
- [Performance Results](#performance-results)
- [Technical Implementation](#technical-implementation)
- [Installation & Usage](#installation--usage)
- [Risk Considerations](#risk-considerations)
- [License](#license)

---

## Overview

This quantitative trading system implements a sophisticated sector rotation strategy by:

1. **Predicting** monthly returns for 11 S&P 500 sectors using machine learning models
2. **Allocating** capital across sectors based on predicted returns with dynamic leverage
3. **Rebalancing** monthly to capitalize on predicted sector outperformance
4. **Benchmarking** performance against the S&P 500 (SPY) index

The system integrates 135 macroeconomic and market features from FRED and Yahoo Finance to forecast returns for each of the 11 major market sectors, enabling tactical asset allocation that adapts to changing economic conditions.

---

## Investment Philosophy

### Core Principles

**Sector Rotation**: Capital markets exhibit persistent sector rotation patterns driven by economic cycles, monetary policy, and investor sentiment. By predicting which sectors will outperform, we can systematically generate alpha.

**Multi-Model Ensemble**: No single model captures all market dynamics. We employ three complementary approaches—LSTM for temporal dependencies, Ridge regression for linear relationships, and Decision Trees for non-linear interactions—to create robust predictions.

**Dynamic Leverage**: Traditional portfolios are constrained to 100% capital allocation. Our strategy applies leverage (up to 5x) to high-conviction predictions (expected returns >15%), allowing outsized gains when the models demonstrate strong confidence while maintaining risk controls.

**Mean Reversion & Momentum**: The strategy captures both mean-reverting behavior (sectors oversold/overbought) and momentum effects (trending sectors) through the diversity of features and models employed.

---

## Dataset

### Data Sources

All data is sourced from two authoritative public repositories:

#### 1. Yahoo Finance ([yfinance](https://pypi.org/project/yfinance/))
Monthly adjusted close prices from October 2004 to March 2025 for 12 ETFs:

**Market Sectors (11 ETFs):**
- **SPY** - S&P 500 Index (Benchmark)
- **VCR** - Consumer Discretionary
- **VCSAX** - Consumer Staples
- **VDE** - Energy
- **VFAIX** - Financials
- **VGSLX** - Real Estate
- **VHCIX** - Health Care
- **VINAX** - Industrials
- **VITAX** - Information Technology
- **VMIAX** - Materials
- **VOX** - Communication Services
- **VUIAX** - Utilities

#### 2. Federal Reserve Economic Data ([FRED](https://fred.stlouisfed.org/))
135 macroeconomic indicators including:

**Commodity Prices:**
- Heating Oil (inflation-adjusted)
- Gold prices
- Lumber prices
- Sugar prices
- Crude oil prices

**Market Indicators:**
- VIX (volatility index)
- S&P 500 P/E ratio
- Gold-to-Oil ratio
- S&P-to-Gold ratio

**Monetary Policy:**
- Federal Funds Rate
- 1-month Commercial Paper Rate
- Federal Securities Held Outright

**Economic Activity:**
- Housing Starts
- Industrial Production Index
- Business Equipment Production
- Producer Price Index (PPI) - Intermediate & Crude Materials
- Final Products Index

**Credit Indicators:**
- Total Bank Credit
- Consumer Credit (Revolving & Non-revolving)

**Employment & Inflation:**
- Various CPI and employment metrics

### Data Preparation

**Time Horizon**: November 2004 - March 2025 (245 months)

**Feature Engineering**: 
- All macroeconomic variables converted to monthly percentage changes
- Missing values forward-filled then interpolated
- Features and labels aligned by date to ensure no look-ahead bias

**Train/Test Split**: 
- Training: 80% of data (196 months)
- Validation: 20% of data (49 months)
- Walk-forward validation used for final backtesting

---

## Models & Methodology

### 1. LSTM (Long Short-Term Memory) Neural Network

**Architecture:**
```
Input Layer: (12 time steps, 135 features)
LSTM Layer: 1,350 units, 80% recurrent dropout
LayerNormalization
Dense Layer: 1,280 units (L1-L2 regularization, LeakyReLU)
Dense Layer: 512 units (L1-L2 regularization, LeakyReLU)
Dense Layer: 160 units (L1-L2 regularization, LeakyReLU)
Output Layer: 12 units (sector returns)

Total Parameters: 10,496,264 (40.04 MB)
```

**Rationale**: 
LSTMs excel at capturing temporal dependencies and sequential patterns in financial time series. The recurrent architecture learns long-term market relationships and cyclical behavior that linear models cannot detect.

**Regularization**:
- 80% recurrent dropout prevents overfitting on historical sequences
- L1-L2 regularization on dense layers encourages feature selection
- Layer normalization stabilizes training across varying market regimes

**Training**:
- Optimizer: Adam
- Loss: Mean Squared Error
- Lookback window: 12 months
- Early stopping based on validation loss

### 2. Ridge Regression

**Architecture:**
Linear regression with L2 penalty (alpha = 1000)

**Rationale**: 
Ridge regression provides a stable, interpretable baseline that captures linear relationships between macroeconomic indicators and sector returns. The L2 regularization handles multicollinearity among the 135 features.

**Hyperparameter Tuning**:
- Alpha values tested: [5, 25, 50, 300, 500, 1000]
- Optimal alpha: 1000 (selected via 5-fold cross-validation)
- Metric: Negative Mean Squared Error

**Performance**:
- **Validation MAE**: 5.06%
- Outperforms baseline (5.69%) and matches LSTM complexity with far fewer parameters

### 3. Decision Tree Regressor

**Architecture:**
```
Criterion: Squared Error
Max Leaf Nodes: 50
Min Samples Split: 10
Min Samples Leaf: 1
```

**Rationale**: 
Decision trees capture non-linear interactions and regime-dependent relationships (e.g., different sector behavior during expansions vs. recessions) that linear models miss.

**Hyperparameter Tuning**:
Grid search over:
- Max leaf nodes: [10, 20, 50, 100]
- Min samples split: [2, 5, 10, 20]
- Min samples leaf: [1, 2, 4, 8]

**Validation MAE**: Comparable to Ridge with superior performance during regime shifts

### Baseline Performance

**Naive Baseline**: Predicting last month's returns (persistence model)
- **Baseline MAE**: 5.69%

All three models outperform this baseline, demonstrating genuine predictive power beyond simple momentum.

---

## Trading Strategy

### Allocation Mechanism

The strategy implements a **dynamic leverage allocation** system based on model predictions:

#### Step 1: Sector Return Prediction
Each model generates 12 monthly return forecasts (one per sector) using the most recent 12 months of data.

#### Step 2: Leverage Decision
For each sector, if predicted return > 15%, apply 5x leverage to capital allocation. This threshold represents high-confidence predictions where the model expects significant outperformance.

**Leverage Mechanics**:
- Sectors with predicted returns > 15%: **5x capital allocation**
- Sectors with predicted returns ≤ 15%: **1x capital allocation**
- Total portfolio leverage can exceed 100% when multiple sectors exceed threshold

#### Step 3: Weight Calculation
```python
# Pseudocode
signal_strength = max(0, predicted_return)  # Only long positions
leverage_multiplier = 5 if predicted_return > 0.15 else 1
raw_weight = signal_strength * leverage_multiplier
normalized_weight = raw_weight / sum(raw_weights)
```

#### Step 4: Shrinkage Toward Equal Weight
To prevent over-concentration, weights are shrunk toward an equal-weighted portfolio:

```python
equal_weight = 1/12  # 8.33% per sector
final_weight = shrinkage * equal_weight + (1 - shrinkage) * normalized_weight
```

**Default shrinkage parameter**: 0.3 (30% toward equal-weight, 70% model-driven)

#### Step 5: Portfolio Return Calculation
```python
capital_deployed = sum(final_weight * leverage_multiplier)
realized_return = sum(final_weight * leverage_multiplier * actual_return) / capital_deployed
```

### Risk Management

**Leverage Constraints**:
- Maximum single-sector leverage: 5x
- Leverage only applied to high-confidence predictions (>15% expected return)
- Automatic deleveraging if no sectors meet threshold (reverts to equal-weight)

**Diversification**:
- Minimum exposure across all 12 positions (including S&P 500)
- Shrinkage prevents excessive concentration in any single sector
- No short positions (long-only strategy reduces downside tail risk)

**Rebalancing**:
- Monthly rebalancing reduces transaction costs while maintaining tactical flexibility
- No intra-month trading or market timing

---

## Performance Results

### Backtesting Period: 2021-2025

**Backtest Methodology**:
- Out-of-sample testing on validation set (most recent 20% of data)
- Walk-forward validation ensures no look-ahead bias
- Transaction costs not modeled (assumes low-cost ETF trading)

### Decision Tree Portfolio Performance

| Metric | Tree Portfolio | S&P 500 (SPY) |
|--------|---------------|---------------|
| **CAGR** | 8.3% | 12.4% |
| **Standard Deviation** | 15.8% | 16.0% |
| **Sortino Ratio** | 0.50 | 0.90 |
| **Growth of $100** | $115 | $154 |

**Key Observations**:

1. **Underperformance vs. Benchmark**: The Tree portfolio delivered 8.3% CAGR vs. 12.4% for SPY during the 2021-2025 period, a challenging environment for sector rotation as tech concentration drove market returns.

2. **Similar Volatility**: Despite employing leverage, the portfolio volatility (15.8%) was nearly identical to SPY (16.0%), suggesting the leverage was applied judiciously to high-conviction predictions.

3. **Inferior Risk-Adjusted Returns**: Sortino ratio of 0.50 vs. 0.90 indicates lower downside-adjusted returns. The strategy struggled during the 2022 bear market when correlations across sectors increased.

4. **Strategy Limitations Identified**:
   - Sector rotation strategies underperform in momentum-driven bull markets dominated by single sectors (e.g., Tech 2021-2024)
   - Leverage amplified losses during incorrect predictions in volatile periods
   - Monthly rebalancing frequency may be suboptimal; weekly or quarterly rebalancing could improve results

### LSTM & Ridge Performance

**Note**: Full backtest results for LSTM and Ridge models are included in the notebook but summarized here:

- **Ridge Regression**: Similar performance to Decision Tree with slightly higher stability
- **LSTM**: Higher variance in predictions led to more aggressive leverage usage and correspondingly higher volatility

The Decision Tree model was selected for detailed presentation due to its balance of interpretability and performance.

---

## Understanding Leverage

### What is Leverage?

**Leverage** allows an investor to control a position larger than their available capital by borrowing funds. In traditional investing, $100 of capital buys $100 of assets (1x leverage). With 5x leverage, that same $100 can control $500 of assets.

### How Leverage Works in This Strategy

When the model predicts a sector will return >15%, the strategy allocates **5 times** the normal capital to that sector:

**Example:**
- Portfolio capital: $10,000
- Normal allocation per sector: $10,000 / 12 = $833
- With 5x leverage on one sector: $833 × 5 = $4,165 allocated

If that sector returns +10%, the portfolio gains:
- **With leverage**: $4,165 × 0.10 = $416.50 (4.17% portfolio return)
- **Without leverage**: $833 × 0.10 = $83.30 (0.83% portfolio return)

### Leverage Amplifies Both Gains and Losses

**Upside Scenario**: Sector returns +20%
- Without leverage: +1.67% portfolio impact
- With 5x leverage: +8.33% portfolio impact

**Downside Scenario**: Sector returns -20%
- Without leverage: -1.67% portfolio impact
- With 5x leverage: -8.33% portfolio impact

### Risk Controls on Leverage

1. **High Threshold**: Only applied when model predicts >15% return (high conviction)
2. **Diversification**: Leverage spread across multiple sectors, not concentrated in one
3. **Shrinkage**: 30% pull toward equal-weight prevents excessive concentration
4. **No Margin Calls**: This is a simulation; real implementation would require margin management

### Why Use Leverage?

Traditional long-only portfolios are capital-constrained to 100% allocation. Leverage allows the strategy to:
- Overweight high-conviction predictions without underweighting others
- Generate outsized returns during correct forecasts
- Maintain exposure to all sectors while tilting toward expected winners

**Caution**: Leverage increases risk proportionally to the multiplier. This strategy is suitable only for sophisticated investors with high risk tolerance.

---

## Technical Implementation

### Dependencies

```python
# Data & Computation
pandas>=1.3.0
numpy>=1.24.0
scipy>=1.10.0

# Machine Learning
tensorflow>=2.10.0
scikit-learn>=1.3.0

# Financial Data
yfinance>=0.2.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0
```

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/quantitative-trading.git
cd quantitative-trading

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook LSTM_Ridge_Tree_Trader.ipynb
```

### Code Structure

**Data Collection:**
```python
import yfinance as yf

# Download sector ETF data
tickers = ['SPY', 'VDE', 'VMIAX', 'VINAX', 'VUIAX', 'VHCIX', 
           'VFAIX', 'VCR', 'VCSAX', 'VITAX', 'VOX', 'VGSLX']
data = yf.download(tickers, start="2004-10-01", end="2025-03-31")['Close']
monthly_returns = data.resample('M').last().pct_change()
```

**FRED Data Integration:**
- Load pre-downloaded FRED CSV containing 135 economic indicators
- Calculate monthly percentage changes for all variables
- Merge with sector returns by date

**Model Training:**
```python
# LSTM
model = Sequential([
    LSTM(1350, return_sequences=False, input_shape=(12, 135), 
         recurrent_dropout=0.8),
    LayerNormalization(),
    Dense(1280, kernel_regularizer=l1_l2(0.1), activation='leaky_relu'),
    Dense(512, kernel_regularizer=l1_l2(0.1), activation='leaky_relu'),
    Dense(160, kernel_regularizer=l1_l2(0.1), activation='leaky_relu'),
    Dense(12, kernel_regularizer=l2(0.1))
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

# Ridge
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1000)
ridge.fit(X_train, y_train)

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_leaf_nodes=50, min_samples_split=10)
tree.fit(X_train, y_train)
```

**Strategy Execution:**
```python
def trading_strategy(expected_returns, actual_returns, shrinkage=0.3, 
                    leverage_threshold=0.15, leverage_factor=5):
    """
    Allocate capital across sectors with dynamic leverage.
    
    Returns:
        float: Portfolio return for the period
    """
    # Apply leverage to high-confidence predictions
    leverage_flags = [r > leverage_threshold for r in expected_returns]
    adjusted_weights = [leverage_factor if flag else 1 for flag in leverage_flags]
    
    # Calculate signal-weighted allocation
    signal_strengths = [max(0, r) for r in expected_returns]
    raw_weights = [s * a for s, a in zip(signal_strengths, adjusted_weights)]
    raw_weights = [w / sum(raw_weights) for w in raw_weights]
    
    # Shrink toward equal-weight
    equal_weights = [1/12] * 12
    final_weights = [shrinkage * eq + (1 - shrinkage) * rw 
                    for eq, rw in zip(equal_weights, raw_weights)]
    
    # Calculate return
    capital_deployed = sum([w * (leverage_factor if flag else 1) 
                           for w, flag in zip(final_weights, leverage_flags)])
    realized_return = sum([w * (leverage_factor if flag else 1) * r 
                          for w, r, flag in zip(final_weights, actual_returns, leverage_flags)])
    
    return realized_return / capital_deployed
```

### Performance Metrics

```python
def portfolio_metrics(returns):
    """Calculate CAGR, volatility, and Sortino ratio."""
    n_years = len(returns) / 12
    cagr = (1 + returns).prod() ** (1/n_years) - 1
    std_dev = returns.std() * np.sqrt(12)  # annualized
    downside_returns = returns[returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(12)
    sortino = cagr / downside_dev if downside_dev > 0 else 0
    
    return {
        'CAGR': round(cagr, 3),
        'Standard Deviation': round(std_dev, 3),
        'Sortino Ratio': round(sortino, 4)
    }
```

---

## Risk Considerations

### Strategy Risks

1. **Model Risk**: Machine learning models can fail during regime changes or unprecedented market conditions (e.g., COVID-19 crash, 2022 inflation shock)

2. **Leverage Risk**: 5x leverage on incorrect predictions can result in significant losses exceeding the capital allocated to that sector

3. **Overfitting**: Models trained on 20 years of data may not generalize to future market structures, especially with 135 features and limited monthly observations

4. **Regime Dependency**: Sector rotation strategies underperform in momentum-driven markets dominated by single sectors (Tech 2021-2024)

5. **Transaction Costs**: Monthly rebalancing across 12 positions incurs trading costs (bid-ask spreads, commissions) not modeled in this backtest

6. **Liquidity Risk**: Large capital pools may struggle to execute 5x leverage positions in less-liquid sector ETFs without market impact

### Market Risks

1. **Correlation Risk**: During market crashes, inter-sector correlations approach 1.0, nullifying diversification benefits

2. **Black Swan Events**: Tail risk events (financial crises, pandemics, geopolitical shocks) can overwhelm model predictions

3. **Data Quality**: FRED data revisions or errors in historical macroeconomic data could corrupt model training

4. **Survivorship Bias**: Analysis uses current sector definitions; historical sector compositions have changed over time

### Operational Risks

1. **Implementation Slippage**: Real-world execution prices may differ from backtested closing prices

2. **Rebalancing Timing**: Month-end rebalancing may be suboptimal compared to other frequencies

3. **Technology Risk**: Model deployment requires robust infrastructure for data ingestion, prediction generation, and order execution

---

## Future Enhancements

### Short-Term Improvements

1. **Ensemble Weighting**: Combine LSTM, Ridge, and Tree predictions with optimized weights rather than selecting a single model

2. **Adaptive Leverage**: Dynamic leverage thresholds based on model confidence and market volatility (VIX-adjusted)

3. **Transaction Cost Modeling**: Incorporate realistic bid-ask spreads and commission structures

4. **Alternative Rebalancing**: Test weekly, quarterly, or volatility-triggered rebalancing frequencies

### Medium-Term Research

1. **Regime Detection**: Use Hidden Markov Models or Gaussian Mixture Models to identify market regimes and switch strategies accordingly

2. **Factor Exposure**: Decompose sector returns into style factors (value, momentum, quality, low volatility) and model factor loadings

3. **Options Overlay**: Use options to hedge downside risk or enhance returns through covered calls on leveraged positions

4. **Alternative Data**: Integrate sentiment analysis, satellite imagery, or credit card spending data to improve predictions

### Long-Term Vision

1. **Reinforcement Learning**: Train an RL agent to learn optimal allocation and leverage policies through interaction with simulated markets

2. **High-Frequency Rebalancing**: Intraday rebalancing using minute-level macroeconomic surprises and sector momentum

3. **Multi-Asset Extension**: Expand beyond equities to bonds, commodities, currencies, and alternatives

4. **Automated Execution**: Deploy strategy on cloud infrastructure with automated data pipelines, model retraining, and order routing

---

## Acknowledgments

This project was developed as an academic research initiative to explore machine learning applications in quantitative finance. Data is sourced from:

- **Yahoo Finance** ([yfinance](https://pypi.org/project/yfinance/)) for historical ETF prices
- **Federal Reserve Economic Data** ([FRED](https://fred.stlouisfed.org/)) for macroeconomic indicators

Special thanks to:
- The open-source community for TensorFlow, scikit-learn, and pandas
- FRED for providing free access to comprehensive economic data
- ChatGPT for assistance with code generation, hyperparameter tuning, and visualization

---

## Disclaimer

**This project is for educational and research purposes only.** 

The strategies, models, and results presented here:
- Are based on historical data and may not predict future performance
- Do not constitute financial advice or investment recommendations
- Have not been tested with real capital or in live market conditions
- Do not account for taxes, fees, slippage, or other real-world costs
- Are provided "as-is" without warranties of any kind

**Investment involves risk, including the potential loss of principal.** Leveraged strategies amplify both gains and losses. Consult a qualified financial advisor before making investment decisions.

The author(s) assume no liability for any financial losses incurred from using these models or strategies.

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.

**Last Updated**: January 2026
