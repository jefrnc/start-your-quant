> 🇪🇸 [Leer en Español](Trading-Systems-Anatomy.es.md) | 🇺🇸 **English**

# Anatomy of a Trading System

A trading system is a set of objective, ordered, and programmable rules that generate entry and exit signals. If you can't write it in code, it's not a system -- it's an opinion.

## The 4 Stages of a Trader

Nearly every trader goes through these evolutionary phases -- a concept widely discussed in systematic trading literature. Knowing which stage you're in saves you months of frustration.

### Stage 1: Discretionary Trader

Trades by intuition, Twitter tips, and "gut feeling." Depends on external opinions. Searches for the magic signal that always works.

**Key symptom**: can't explain their strategy in 3 clear rules.

Most traders lose money at this stage and never move past it.

### Stage 2: Technical Trader

Starts using indicators (RSI, MACD, moving averages). Searches for the perfect combination of indicators -- the "holy grail."

**Key symptom**: switches indicators every week looking for the one that "always works."

Real progress comes when they accept that no indicator is magic and start thinking in terms of **fixed rules**.

### Stage 3: Systems Trader

Trades with objective rules. Backtests. Measures. Has numerical expectations for their system (win rate, profit factor, expected drawdown).

**Key symptom**: can watch 10 losing trades in a row and not change the system, because they know it's statistically expected.

```python
# This is the Stage 3 mindset:
# "My system has a 40% win rate and a 2.5:1 ratio.
#  A streak of 10 losses has a 0.6% probability.
#  It's rare but possible. I change nothing."

def expected_value(win_rate, avg_win, avg_loss):
    """Expectancy per trade -- if positive, the system is viable."""
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

# 40% win rate, average win $250, average loss $100
ev = expected_value(0.40, 250, 100)
print(f"Expectancy per trade: ${ev:.2f}")  # $40.00
```

### Stage 4: Portfolio Manager

Stops thinking about "the system" and starts thinking about **the portfolio of systems**. Multiple strategies, multiple markets, multiple timeframes. Liquidity management and correlation between systems.

**Key symptom**: cares more about the correlation between their systems than the Sharpe of any individual one.

**The key transition**: moving from optimizing ONE system to optimizing the COMBINATION of systems. A portfolio of 5 mediocre but uncorrelated systems outperforms a single "perfect" system.

## System Classification by Strategy

### Trend Following (Momentum/Trend Following)

Buy high to sell higher. Follow the trend until it exhausts itself.

| Characteristic | Typical Value |
|---|---|
| Win rate | 30-45% |
| Win/loss ratio | 2:1 to 5:1 |
| Best market | Strong trending |
| Worst market | Sideways/choppy |
| Psychology | Difficult -- many small losses |

```python
def trend_following_signal(prices, fast=20, slow=50):
    """
    Classic trend signal: moving average crossover.
    Few signals, many false ones in sideways markets,
    but captures the big moves.
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()

    signal = 0
    if fast_ma.iloc[-1] > slow_ma.iloc[-1] and fast_ma.iloc[-2] <= slow_ma.iloc[-2]:
        signal = 1   # buy
    elif fast_ma.iloc[-1] < slow_ma.iloc[-1] and fast_ma.iloc[-2] >= slow_ma.iloc[-2]:
        signal = -1  # sell
    return signal
```

### Counter-Trend (Mean Reversion)

Buy cheap near support, sell high near resistance. Assume the price reverts to its mean.

| Characteristic | Typical Value |
|---|---|
| Win rate | 55-70% |
| Win/loss ratio | 0.5:1 to 1.5:1 |
| Best market | Sideways/range-bound |
| Worst market | Strong trending |
| Psychology | More manageable -- many winners |

```python
def mean_reversion_signal(prices, lookback=20, z_threshold=2.0):
    """
    Mean reversion signal using z-score.
    Buys when price is 2 standard deviations below the mean.
    """
    mean = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    z_score = (prices.iloc[-1] - mean.iloc[-1]) / std.iloc[-1]

    if z_score < -z_threshold:
        return 1   # oversold -> buy
    elif z_score > z_threshold:
        return -1  # overbought -> sell
    return 0
```

### Volatility Breakout / ORB

Enter on the breakout of a range (opening, previous range, etc.) and close quickly. Hybrids between trend following and mean reversion.

```python
def orb_signal(open_price, high_first_15min, low_first_15min, current_price):
    """
    Opening Range Breakout: enters if price breaks
    the first 15 minutes' range.
    """
    range_size = high_first_15min - low_first_15min

    if current_price > high_first_15min:
        return 1, high_first_15min - range_size  # long, stop below range
    elif current_price < low_first_15min:
        return -1, low_first_15min + range_size   # short, stop above range
    return 0, None
```

### Other Types Worth Knowing

| Type | Core Idea | Complexity |
|---|---|---|
| **Rotational** | Rotates capital between assets based on relative strength | Medium |
| **Market Making** | Provides liquidity by buying the bid / selling the ask | High |
| **Pairs Trading** | Long one asset + short another correlated one | Medium-High |
| **Statistical Arbitrage** | Exploits temporary price discrepancies | High |
| **Seasonal** | Patterns that repeat on specific dates | Low |

## There Is No Holy Grail (Individually)

A system with 40% accuracy can be more profitable than one with 70%. What matters is the **expectancy** (expected value per trade) and how it behaves in combination with other systems.

### The real holy grail: system diversification

```python
import numpy as np

def portfolio_sharpe(returns_matrix, weights):
    """
    The Sharpe of a portfolio of uncorrelated systems
    is greater than that of any individual system.
    """
    portfolio_return = np.dot(weights, returns_matrix.mean(axis=0)) * 252
    portfolio_vol = np.sqrt(
        np.dot(weights, np.dot(returns_matrix.cov() * 252, weights))
    )
    return portfolio_return / portfolio_vol

# Example: 3 systems with individual Sharpe of ~1.0
# but low correlation between them -> portfolio Sharpe > 1.5
```

The key is not finding the perfect system. It's building a portfolio where:
- Systems are **individually profitable** (positive expectancy)
- They have **low correlation with each other** (they don't all lose at the same time)
- They operate in **different markets or timeframes** (stocks + futures, intraday + swing)

## Algo Trading: Myths vs Reality

### Myth: "Algo trading = high frequency"

**Reality**: an algorithm can trade on monthly charts. What makes it algorithmic is that the rules are coded, not the speed of execution.

### Myth: "Markets are random, you can't win systematically"

The efficient market hypothesis (popularized by Malkiel in *A Random Walk Down Wall Street*, 1973) holds that prices reflect all available information and that you cannot beat the market systematically and consistently after costs. But:

- The market does not have a perfectly normal distribution -- there are fat tails, volatility clusters, and asymmetries (drops are fast with high volatility, rallies are gradual)
- Information is neither perfect nor instantaneous for all participants
- Exploitable inefficiencies exist, especially in smaller-cap instruments

You don't need to predict the future. You need to find patterns with a **statistical edge** and manage risk so that edge materializes over thousands of trades.

### Myth: "You need Wall Street infrastructure"

**Reality**: the technology gap between institutional firms and individual traders has never been smaller. With Python, a broker with an API, and affordable market data, you can build and run profitable systems. It's estimated that in the US, over 60% of volume is algorithmic (including market makers and HFT), but a large portion of that volume comes from relatively simple, well-executed systems, not nanosecond infrastructure.

## Algorithmic vs Discretionary Trading

| Dimension | Algorithmic | Discretionary |
|---|---|---|
| **Emotions** | Minimized -- the code executes | Present in every trade |
| **Expectations** | Estimable via backtest (with limitations) | Estimated, subjective |
| **Discipline** | Inherent to the code | Requires willpower |
| **Adapting to changes** | Requires re-development | Immediate (for expert traders) |
| **Diversification** | Easy -- run N systems in parallel | Difficult -- one brain, one market |
| **Drawdown** | Expected and quantified | Surprising and emotionally tough |
| **Scalability** | High | Limited by the trader |

Algorithmic trading is not "better" -- it's **different**. An expert discretionary trader can outperform many algorithms. But the learning curve for discretionary trading is longer, the emotional toll is greater, and consistency is harder to maintain.

The real advantage of algorithmic trading: you can know in advance, with historical data, how bad things can get. Knowing that your system can have 12 consecutive losses and that this is normal is infinitely more manageable than discovering it live with no context.
