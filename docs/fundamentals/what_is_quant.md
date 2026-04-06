> 🇪🇸 [Leer en Español](what_is_quant.es.md) | 🇺🇸 **English**

# What is Quantitative Trading?

## Definition

Quantitative trading is a systematic approach to operating in financial markets that relies on mathematical models, statistical analysis, and computational algorithms to identify trading opportunities and execute trades.

> **Focus of this guide**: We specialize in **small caps** (market cap < $2B), which offer greater volatility and opportunities than large caps, but require specific risk management techniques.

## Key Characteristics

### 1. **Data-Driven**
- Decisions grounded in historical and real-time analysis
- Elimination of emotional biases
- Rigorous backtesting before implementing strategies

### 2. **Systematic and Reproducible**
- Clear, defined rules
- Consistent and predictable results
- Ability to scale operations

### 3. **Automatable**
- Automatic order execution
- 24/7 market monitoring
- Immediate response to signals

## Advantages of Quantitative Trading

1. **Objectivity**: Decisions are based on data, not emotions
2. **Speed**: Ability to analyze thousands of assets simultaneously
3. **Discipline**: The system follows rules without exception
4. **Scalability**: A strategy can be applied across multiple markets
5. **Continuous Improvement**: Results are measurable and optimizable

## Essential Components

### 1. **Data**
```python
# Example: Get historical data
import yfinance as yf
data = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
```

### 2. **Strategy**
```python
# Example: Simple moving average crossover strategy
def moving_average_strategy(data, short_window=20, long_window=50):
    data['SMA20'] = data['Close'].rolling(window=short_window).mean()
    data['SMA50'] = data['Close'].rolling(window=long_window).mean()
    data['Signal'] = 0
    data.loc[data['SMA20'] > data['SMA50'], 'Signal'] = 1
    data.loc[data['SMA20'] < data['SMA50'], 'Signal'] = -1
    return data
```

### 3. **Risk Management**
```python
# Example: Position size calculation
def calculate_position_size(capital, risk_per_trade, stop_loss_pct):
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    return position_size
```

### 4. **Execution**
```python
# Example: Basic execution framework
def execute_trade(signal, ticker, quantity):
    if signal == 1:
        # Buy
        order = broker.buy(ticker, quantity)
    elif signal == -1:
        # Sell
        order = broker.sell(ticker, quantity)
    return order
```

## Common Myths

### "It's only for mathematicians"
**Reality**: With today's tools, any trader can get started with basic concepts

### "It guarantees profits"
**Reality**: Like any form of trading, it carries risks and requires proper management

### "It completely replaces the trader"
**Reality**: The trader designs, supervises, and improves the systems

## Getting Started

1. **Learn basic Python**: It's the most widely used language in quant trading
2. **Understand basic statistics**: Mean, standard deviation, correlation
3. **Practice with paper trading**: Test strategies without real risk
4. **Start simple**: A basic strategy well executed is better than a complex one poorly implemented

## Complete Example: My First Quant Strategy

```python
import pandas as pd
import numpy as np
import yfinance as yf

# 1. Get data
ticker = 'SPY'
data = yf.download(ticker, start='2023-01-01', end='2023-12-31')

# 2. Calculate indicators
data['Returns'] = data['Close'].pct_change()
data['SMA20'] = data['Close'].rolling(window=20).mean()
data['Upper_Band'] = data['SMA20'] + (data['Close'].rolling(window=20).std() * 2)
data['Lower_Band'] = data['SMA20'] - (data['Close'].rolling(window=20).std() * 2)

# 3. Generate signals
data['Signal'] = 0
data.loc[data['Close'] < data['Lower_Band'], 'Signal'] = 1  # Buy
data.loc[data['Close'] > data['Upper_Band'], 'Signal'] = -1  # Sell

# 4. Calculate strategy returns
data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']

# 5. Performance metrics
total_return = (1 + data['Strategy_Returns']).cumprod()[-1] - 1
sharpe_ratio = data['Strategy_Returns'].mean() / data['Strategy_Returns'].std() * np.sqrt(252)

print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
```

## Recommended Resources

- **Books**: "Algorithmic Trading" by Ernest Chan
- **Courses**: QuantConnect Learning, Coursera Financial Engineering
- **Practice**: Kaggle trading competitions
- **Communities**: r/algotrading, QuantConnect forums

## Next Step

Continue with [Differences Between Discretionary and Quant](discretionary_vs_quant.md) to better understand the advantages of the quantitative approach.
