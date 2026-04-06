> 🇪🇸 [Leer en Español](strategy_types.es.md) | 🇺🇸 **English**

# Types of Quantitative Strategies

## The 4 Main Categories

### 1. Momentum / Trend Following
**Philosophy**: "The trend is your friend"

```python
# Simple example: 20-day Breakout
def momentum_strategy(data):
    data['20_day_high'] = data['High'].rolling(20).max()
    data['signal'] = data['Close'] > data['20_day_high'].shift(1)
    return data
```

**Characteristics**:
- Low win rate (35-45%) but high reward/risk
- Work best in trending markets
- Easy to code and backtest
- Suffer in sideways markets

**Common variants**:
- Range breakouts
- Moving average crossovers  
- Relative strength (RS) vs index
- Gap and Go (for small caps)

### 2. Mean Reversion
**Philosophy**: "What goes up, comes down"

```python
# Example: Bollinger Bands reversal
def mean_reversion_strategy(data):
    data['SMA'] = data['Close'].rolling(20).mean()
    data['STD'] = data['Close'].rolling(20).std()
    data['Lower_Band'] = data['SMA'] - (2 * data['STD'])
    data['Upper_Band'] = data['SMA'] + (2 * data['STD'])
    
    # Buy on oversold
    data['signal'] = data['Close'] < data['Lower_Band']
    return data
```

**Characteristics**:
- High win rate (60-70%) but lower reward
- Better in sideways markets
- Require good stops
- Dangerous in strong trends

**Common variants**:
- RSI oversold/overbought
- Bollinger Bands squeeze
- Pairs trading
- VWAP reversion

### 3. Statistical Arbitrage
**Philosophy**: "Exploit temporary inefficiencies"

```python
# Example: Pairs trading
def pairs_trading(stock_a, stock_b, window=20):
    # Calculate spread
    ratio = stock_a / stock_b
    mean_ratio = ratio.rolling(window).mean()
    std_ratio = ratio.rolling(window).std()
    
    # Z-score of the spread
    z_score = (ratio - mean_ratio) / std_ratio
    
    # Signals
    signals = pd.DataFrame()
    signals['long_a_short_b'] = z_score < -2
    signals['short_a_long_b'] = z_score > 2
    return signals
```

**Characteristics**:
- Market neutral (direction-independent)
- Requires significant capital
- Competition with HFT
- Small margins, high volume

**Common variants**:
- ETF arbitrage
- Index arbitrage
- Cross-exchange crypto arb
- Options arbitrage

### 4. Market Making / Liquidity
**Philosophy**: "Provide liquidity and capture the spread"

```python
# Conceptual example (simplified)
def simple_market_making(ticker, spread_target=0.02):
    mid_price = (bid + ask) / 2
    
    # Place orders on both sides
    place_buy_order(price=mid_price - spread_target/2)
    place_sell_order(price=mid_price + spread_target/2)
    
    # Adjust inventory if imbalanced
    if inventory > max_inventory:
        adjust_prices_to_reduce_inventory()
```

**Characteristics**:
- Thousands of small gains
- Requires low latency
- Critical inventory management
- Risk in directional moves

## Small Cap-Specific Strategies

### Gap and Go
```python
def gap_and_go_scanner(universe):
    candidates = []
    
    for ticker in universe:
        gap_pct = (ticker.open - ticker.prev_close) / ticker.prev_close
        
        if (gap_pct > 0.10 and  # Gap >10%
            ticker.float < 50_000_000 and  # Low float
            ticker.premarket_vol > 1_000_000):  # Volume
            
            candidates.append({
                'ticker': ticker.symbol,
                'gap': gap_pct,
                'rvol': ticker.volume / ticker.avg_volume
            })
    
    return sorted(candidates, key=lambda x: x['rvol'], reverse=True)
```

### VWAP Reclaim
```python
def vwap_reclaim_setup(data):
    # Calculate VWAP
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # Detect reclaim
    data['below_vwap'] = data['Low'] < data['VWAP']
    data['reclaim'] = (data['below_vwap'].shift(1) & 
                      (data['Close'] > data['VWAP']))
    
    # Confirm with volume
    data['volume_spike'] = data['Volume'] > data['Volume'].rolling(20).mean() * 2
    data['signal'] = data['reclaim'] & data['volume_spike']
    
    return data
```

### Parabolic Short
```python
def parabolic_exhaustion(data, lookback=10):
    # Detect parabolic move
    returns = data['Close'].pct_change()
    cumulative = (1 + returns).cumprod()
    
    # Exhaustion conditions
    data['parabolic'] = (
        (cumulative > cumulative.shift(lookback) * 1.5) &  # +50% in 10 days
        (data['RSI'] > 80) &  # Extreme RSI
        (data['Volume'] > data['Volume'].rolling(20).mean() * 3)  # Climax volume
    )
    
    return data['parabolic']
```

## How to Choose Your Strategy

### Based on your personality:
- **Patient + Analytical** -> Mean Reversion
- **Aggressive + Fast** -> Momentum
- **Mathematical + Detail-oriented** -> Arbitrage
- **Multitasking + Tech** -> Market Making

### Based on your capital:
- **< $25k**: Momentum small caps (PDT rule)
- **$25k - $100k**: Mix momentum + mean reversion
- **$100k - $500k**: Add pairs trading
- **> $500k**: All strategies

### Based on your time:
- **Part-time**: Swing momentum
- **Full-time**: Day trading + scalping
- **Automated**: Market making + arbitrage

## Framework for Evaluating Strategies

```python
def evaluate_strategy(backtest_results):
    metrics = {
        'win_rate': backtest_results['wins'] / backtest_results['total_trades'],
        'avg_win': backtest_results['gross_profits'] / backtest_results['wins'],
        'avg_loss': backtest_results['gross_losses'] / backtest_results['losses'],
        'profit_factor': backtest_results['gross_profits'] / abs(backtest_results['gross_losses']),
        'sharpe_ratio': calculate_sharpe(backtest_results['returns']),
        'max_drawdown': calculate_max_drawdown(backtest_results['equity_curve']),
        'expectancy': calculate_expectancy(backtest_results)
    }
    
    # Final score
    if (metrics['profit_factor'] > 1.5 and 
        metrics['sharpe_ratio'] > 1.0 and 
        metrics['max_drawdown'] < 0.20):
        return "VIABLE"
    else:
        return "NEEDS WORK"
```

## My Personal Mix

```python
# Strategy portfolio
STRATEGIES = {
    'morning_momentum': {
        'allocation': 0.40,
        'timeframe': '9:30-10:30',
        'instruments': ['small_cap_gappers']
    },
    'vwap_reversion': {
        'allocation': 0.30,
        'timeframe': '10:30-15:30',
        'instruments': ['liquid_stocks']
    },
    'pairs_trading': {
        'allocation': 0.20,
        'timeframe': 'all_day',
        'instruments': ['sector_etfs']
    },
    'overnight_swing': {
        'allocation': 0.10,
        'timeframe': 'multi_day',
        'instruments': ['large_caps']
    }
}
```

## Common Mistakes

1. **Over-optimizing a single strategy** instead of diversifying
2. **Ignoring transaction costs** in backtests
3. **Not considering liquidity** in small caps
4. **Assuming perfect fills** in high-frequency strategies
5. **Not having circuit breakers** for adverse conditions

## Resources for Further Learning

- **Momentum**: "Following the Trend" - Andreas Clenow
- **Mean Reversion**: "Mean Reversion Trading" - Ernest Chan  
- **Arbitrage**: "Statistical Arbitrage" - Andrew Pole
- **Market Making**: Avellaneda & Stoikov papers

## Next Step

Now you know the types of strategies. Let's move on to [Data Sources](../data/data_sources.md) to see where to get the raw material for your backtests.
