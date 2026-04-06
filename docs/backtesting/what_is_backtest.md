> 🇪🇸 [Leer en Español](what_is_backtest.es.md) | 🇺🇸 **English**

# What Is a Backtest?

## The Trading Time Machine

A backtest is basically a time machine. You take your strategy and run it on historical data to see how it would have performed. It's the difference between gambling and investing.

## Why It Matters

### The Harsh Reality
- 90% of traders lose money
- Most trade without prior testing
- "It looks good on the chart" ≠ Profitable strategy
- A backtest saves you from losing real money

### What You're Really Measuring
```python
# It's not just: "Did I make money?"
# It's: "Can I repeat this consistently?"

backtest_questions = {
    'profitability': 'Is it profitable?',
    'consistency': 'Does it work across different periods?',
    'risk': 'How much can I lose?',
    'frequency': 'How many opportunities are there?',
    'drawdown': 'Can I psychologically handle the losses?',
    'market_conditions': 'Does it work in bull and bear markets?'
}
```

## Anatomy of a Backtest

### 1. Historical Data
```python
# Data quality = Results quality
data_requirements = {
    'timeframe': '2+ years minimum',
    'resolution': 'Match your strategy (1min for day trading)',
    'quality': 'Adjusted for splits, dividends',
    'survivorship_bias': 'Include delisted stocks',
    'universe': 'Representative of where you will trade'
}
```

### 2. Trading Rules
```python
def example_strategy_rules():
    """Example of clear, testable rules"""
    entry_rules = {
        'signal': 'close > vwap AND rvol > 2',
        'timing': 'Market hours only',
        'size': '1% risk per trade',
        'max_positions': '3 concurrent'
    }
    
    exit_rules = {
        'stop_loss': '2% below entry',
        'take_profit': '4% above entry (2:1 R/R)',
        'time_stop': 'End of day',
        'market_stop': 'VIX > 30'
    }
    
    return {'entry': entry_rules, 'exit': exit_rules}
```

### 3. Costs and Frictions
```python
def realistic_costs():
    """Include all real costs"""
    return {
        'commission': 0.005,  # $5 per 1000 shares
        'spread': 0.0002,     # 2 basis points
        'slippage': 0.0001,   # 1 basis point average
        'borrowing_costs': 0.0003,  # For shorts
        'platform_fees': 50,  # Monthly
        'data_fees': 100      # Monthly
    }
```

## Example: My First Backtest

```python
import pandas as pd
import numpy as np
import yfinance as yf

def simple_vwap_backtest(ticker, start_date, end_date):
    """Simple VWAP strategy backtest"""
    
    # 1. Get data
    data = yf.download(ticker, start=start_date, end=end_date, interval='5m')
    
    # 2. Calculate indicators
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['above_vwap'] = data['Close'] > data['vwap']
    
    # 3. Generate signals
    data['signal'] = data['above_vwap'] & ~data['above_vwap'].shift(1)  # Cross above
    
    # 4. Simulate trades
    initial_capital = 10000
    position = 0
    cash = initial_capital
    trades = []
    
    for i in range(len(data)):
        if data['signal'].iloc[i] and position == 0:
            # Entry
            shares = int(cash * 0.95 / data['Close'].iloc[i])
            position = shares
            cash -= shares * data['Close'].iloc[i]
            entry_price = data['Close'].iloc[i]
            
        elif position > 0:
            # Check exits
            current_price = data['Close'].iloc[i]
            
            # Stop loss: 2%
            if current_price < entry_price * 0.98:
                cash += position * current_price
                trades.append(current_price - entry_price)
                position = 0
                
            # Take profit: 4%
            elif current_price > entry_price * 1.04:
                cash += position * current_price
                trades.append(current_price - entry_price)
                position = 0
    
    # 5. Calculate metrics
    if trades:
        win_rate = len([t for t in trades if t > 0]) / len(trades)
        avg_win = np.mean([t for t in trades if t > 0])
        avg_loss = np.mean([t for t in trades if t < 0])
        profit_factor = abs(sum([t for t in trades if t > 0]) / sum([t for t in trades if t < 0]))
        
        final_value = cash + (position * data['Close'].iloc[-1] if position > 0 else 0)
        total_return = (final_value - initial_capital) / initial_capital
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'final_value': final_value
        }
    else:
        return {'error': 'No trades generated'}

# Run
results = simple_vwap_backtest('AAPL', '2023-01-01', '2023-12-31')
print(results)
```

## Types of Backtesting

### 1. Vectorized Backtesting
```python
# Fast but less realistic
def vectorized_backtest(data, signals):
    """Entire time series at once"""
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = signals.shift(1) * data['returns']
    
    cumulative_returns = (1 + data['strategy_returns']).cumprod()
    return cumulative_returns
```

### 2. Event-Driven Backtesting
```python
# Slower but more realistic
class EventDrivenBacktest:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        
    def process_bar(self, bar):
        """Process each bar individually"""
        # Check signals
        # Manage positions
        # Execute trades
        pass
```

### 3. Monte Carlo Simulation
```python
def monte_carlo_backtest(strategy, num_simulations=1000):
    """Multiple simulations with altered data"""
    results = []
    
    for i in range(num_simulations):
        # Shuffle or resample data
        shuffled_data = shuffle_returns(original_data)
        result = run_backtest(strategy, shuffled_data)
        results.append(result)
    
    return analyze_distribution(results)
```

## Common Backtesting Mistakes

### 1. Look-Ahead Bias
```python
# ❌ BAD: Using future information
data['signal'] = data['close'] > data['close'].shift(-1)  # Peek into future

# ✅ GOOD: Only information available at the time
data['signal'] = data['close'] > data['close'].shift(1)
```

### 2. Survivorship Bias
```python
# ❌ BAD: Only stocks that survived
universe = ['AAPL', 'MSFT', 'GOOGL']  # Only winners

# ✅ GOOD: Include delisted stocks
universe = get_historical_universe('Russell3000', start_date)
```

### 3. Data Mining Bias
```python
# ❌ BAD: Optimize until it works
for sma in range(5, 100):
    for rsi_threshold in range(20, 80):
        if backtest_return > 0.3:  # Cherry picking
            print(f"Found winning combo: SMA={sma}, RSI={rsi_threshold}")
```

### 4. Overfitting
```python
# ❌ BAD: Too many parameters
def overfitted_strategy(data, p1, p2, p3, p4, p5, p6, p7, p8):
    # 8 parameters = too specific to historical data
    pass

# ✅ GOOD: Keep it simple
def simple_strategy(data, short_ma=9, long_ma=20):
    # 2 parameters = more generalizable
    pass
```

## In-Sample vs Out-of-Sample

```python
def proper_backtesting_workflow(data):
    """Correct workflow to avoid overfitting"""
    
    # Split data
    total_length = len(data)
    in_sample_end = int(total_length * 0.7)  # 70% for development
    
    in_sample = data.iloc[:in_sample_end]
    out_sample = data.iloc[in_sample_end:]
    
    # 1. Develop strategy on in-sample
    strategy = develop_strategy(in_sample)
    
    # 2. One time only: test on out-of-sample
    out_sample_results = test_strategy(strategy, out_sample)
    
    # 3. If it fails out-of-sample, go back to step 1
    if out_sample_results['sharpe'] < 1.0:
        return "Strategy needs work"
    else:
        return "Strategy ready for paper trading"
```

## Walk-Forward Analysis

```python
def walk_forward_backtest(data, window_size=252, rebalance_freq=21):
    """Backtest with periodic re-optimization"""
    results = []
    
    for start in range(0, len(data) - window_size, rebalance_freq):
        # Training window
        train_end = start + window_size
        train_data = data.iloc[start:train_end]
        
        # Test period
        test_start = train_end
        test_end = min(test_start + rebalance_freq, len(data))
        test_data = data.iloc[test_start:test_end]
        
        # Optimize strategy on training data
        best_params = optimize_strategy(train_data)
        
        # Test on out-of-sample period
        period_result = test_strategy(best_params, test_data)
        results.append(period_result)
    
    return combine_results(results)
```

## Red Flags in Results

```python
def validate_backtest_results(results):
    """Identify suspicious results"""
    red_flags = []
    
    # Too good to be true
    if results['annual_return'] > 0.5:  # +50% annual
        red_flags.append("Returns too high - likely overfitted")
    
    # Unrealistic win rate
    if results['win_rate'] > 0.8:  # 80%+ win rate
        red_flags.append("Win rate too high - check for look-ahead bias")
    
    # Drawdown too low
    if results['max_drawdown'] < 0.05:  # Less than 5%
        red_flags.append("Drawdown too low - not realistic")
    
    # Too few trades
    if results['total_trades'] < 100:
        red_flags.append("Not enough trades for statistical significance")
    
    # Unrealistic profit factor
    if results['profit_factor'] > 3:
        red_flags.append("Profit factor too high - likely curve-fitted")
    
    return red_flags
```

## Paper Trading: The Next Step

```python
def transition_to_paper_trading(backtest_results):
    """How to go from backtest to paper trading"""
    
    if backtest_results['sharpe_ratio'] > 1.5:
        return {
            'recommendation': 'Start paper trading',
            'position_size': 'Use 1/4 of planned size initially',
            'duration': 'Paper trade for 2-3 months minimum',
            'success_criteria': {
                'correlation_with_backtest': '>0.7',
                'sharpe_ratio': '>1.0',
                'max_drawdown': '<15%'
            }
        }
    else:
        return {
            'recommendation': 'Improve strategy first',
            'issues_to_address': analyze_weaknesses(backtest_results)
        }
```

## Backtesting Tools

```python
# Popular frameworks
backtesting_tools = {
    'basic': 'pandas + numpy (custom)',
    'intermediate': 'backtrader, zipline',
    'advanced': 'vectorbt, quantconnect',
    'professional': 'QuantLib, custom C++'
}
```

## Next Step

Now that you understand what a backtest is, let's move on to [Simple Backtest Engine](simple_engine.md) to build one from scratch.
