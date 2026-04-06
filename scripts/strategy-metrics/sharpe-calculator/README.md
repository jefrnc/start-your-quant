> 🇪🇸 [Leer en Español](README.es.md) | 🇺🇸 **English**

# Sharpe Ratio Calculator

A complete script to calculate the Sharpe ratio and related metrics for quantitative trading strategies.

## Features

- **Multiple input formats**: Direct returns or trades with P&L
- **Comprehensive analysis**: Sharpe, Sortino, Calmar, Information Ratio
- **Benchmark comparison**: SPY, QQQ, sector ETFs
- **Rolling Sharpe ratio**: For temporal analysis
- **Robust validations**: Missing data and outlier handling

## Quick Usage

```bash
# Basic analysis from returns
python calculate_sharpe.py --returns daily_returns.csv

# Analysis from trades
python calculate_sharpe.py --trades my_trades.csv --capital 10000

# Comparison with benchmark
python calculate_sharpe.py --returns strategy.csv --benchmark spy.csv
```

## Data Format

### Returns CSV
```csv
date,returns
2024-01-01,0.025
2024-01-02,-0.012
2024-01-03,0.018
```

### Trades CSV
```csv
date,pnl
2024-01-01,150.50
2024-01-02,-75.25
2024-01-03,200.00
```

## Calculated Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 = Good, > 2.0 = Excellent |
| **Sortino Ratio** | Only considers downside risk | Better for asymmetric strategies |
| **Calmar Ratio** | Annual return / Max Drawdown | Efficiency vs maximum loss |
| **Information Ratio** | Excess return vs benchmark | Trader skill vs market |

## Sharpe Ratio Interpretation

```
> 2.0   : Excellent
1.0-2.0 : Very good
0.5-1.0 : Good
0.0-0.5 : Poor
< 0.0   : Destroys value
```

## Practical Examples

### Gap & Go Strategy Analysis
```bash
python calculate_sharpe.py \
  --trades gap_go_trades.csv \
  --benchmark spy_returns.csv \
  --capital 10000 \
  --rf-rate 0.045
```

### Rolling Sharpe for Monitoring
```bash
python calculate_sharpe.py \
  --returns daily_returns.csv \
  --rolling-window 30 \
  --period daily
```

## Important Considerations

**Sharpe Ratio Limitations**:
- Assumes normal distribution of returns
- Sensitive to extreme outliers
- Does not capture tail risk
- Measurement period affects the result

**Best Practices**:
- Use multiple metrics (Sharpe + Sortino + Calmar)
- Always compare with relevant benchmarks
- Analyze rolling Sharpe to detect degradation
- Consider market regime (bull vs bear)

## Advanced Configuration

```python
from calculate_sharpe import SharpeCalculator

# Customize risk-free rate
calc = SharpeCalculator(risk_free_rate=0.045)

# Detailed analysis
analysis = calc.detailed_analysis(returns, period='daily')
print(f"Sharpe: {analysis['sharpe_ratio']:.4f}")
print(f"Max DD: {analysis['max_drawdown']:.2%}")
```

## Integration with Other Scripts

This script is part of the **Quant Playbook** and integrates with:

- `../max-drawdown/`: Detailed drawdown analysis
- `../profit-factor/`: Profit factor calculation
- `../../backtesting/`: Strategy validation
- `../../data-collection/`: Data pipelines

## Troubleshooting

### Error: "Insufficient data"
- Minimum 2 valid observations
- Verify date format
- Remove rows with NaN

### Sharpe ratio = infinity
- Volatility = 0 (all returns equal)
- Use a longer period
- Verify data quality

### Benchmark comparison fails
- Align dates between strategy and benchmark
- Use the same time period
- Verify column format
