> 🇪🇸 [Leer en Español](Filters-Asset-Selection-Examples.es.md) | 🇺🇸 **English**

# Filters, Asset Selection, and Practical Examples

## Filters: Power and Danger

A filter is a rule that eliminates trades. The goal is to remove losers without touching winners. Sounds perfect -- and that's why it's dangerous.

### Why Filters Overfit

Every filter you add reduces the sample. With fewer trades, your statistical evaluation loses significance. And the worst part: it's extremely easy to improve a backtest with filters.

```
No filter:  100 trades, 45% win rate, PF 1.3
+ Remove Mondays: 82 trades, 48% win rate, PF 1.5
+ Remove October: 74 trades, 51% win rate, PF 1.7
+ If yesterday's high < day before's high: 58 trades, 55% win rate, PF 2.1
```

Each filter "improves" the system in the backtest. But what you're doing is fitting the system to historical data. In live trading, those patterns probably won't repeat.

### Rules for Filtering Without Destroying

1. **One filter per system, maximum.** If you need two filters for it to work, the entry signal probably has no edge
2. **It must have market logic.** "I don't trade on Mondays" needs a reason (e.g., lower liquidity at weekly open), not just a better backtest
3. **Evaluate across the entire history.** The filter must act on enough cases over time, not just in one period
4. **The system must work reasonably WITHOUT the filter.** If without the filter the system is a disaster, the filter is masking a broken system

## Every Asset Has a Personality

A system that works on the Nasdaq doesn't necessarily work on soybeans, and one that works on daily charts may fail on 5-minute charts.

### General Rule by Timeframe

| Timeframe | Noise | Extrapolability | Trades | Best for |
|---|---|---|---|---|
| 1-5 min | Very high | Low -- asset-specific | Many | A single asset, well calibrated |
| 15-60 min | High | Moderate | Moderate | Group of similar assets |
| Daily | Moderate | High | Few-moderate | Multiple assets, universal ideas |
| Weekly/Monthly | Low | Very high | Very few | Baskets of 50-100 assets for statistical significance |

**The lower the timeframe, the more noise, the harder to extrapolate to other assets, and the faster the edge degrades.** Ideas that work on daily and weekly charts tend to be more universal and robust.

### How to Measure Trendiness and Volatility

Before applying a system to an asset, measure whether that asset has the characteristics your system needs.

**ADX (Average Directional Index)**: measures whether there is a trend, regardless of direction. Above ~20 is considered trending. It doesn't tell you if it's bullish or bearish -- for that, there are the DI+ (buying pressure) and DI- (selling pressure) lines that accompany it.

**Normalized ATR (ATR%)**: ATR divided by price, expressed as a percentage. Allows comparing volatility across assets with very different prices. See [KISS: Design Principles](./KISS-Design-Principles.md) for the full implementation.

Indicative values comparing assets on long-term daily charts:

| Asset | Average ADX | Average ATR% | Profile |
|---|---|---|---|
| Nasdaq 100 | ~23 | ~1.6% | Volatile, moderate trend |
| S&P 500 | ~25 | ~1.2% | Less volatile, more trending |
| Gold (GLD) | ~23 | ~1.0% | Low volatility, moderate trend |
| Soybeans | ~24 | ~2.2% | High volatility, good trend |
| Crude Oil | ~26 | ~3.0% | Very volatile, very trending |

*These values are indicative and vary depending on the period analyzed. Always verify with your own data.*

### Patterns by Asset Type

**Major stock indices** (S&P 500, Nasdaq): on intraday they are highly mean-reverting -- it's hard to capture trends. On daily and weekly they do show clear trends. Volatility is asymmetric: it increases in declines, decreases in rallies.

**Less traded commodities** (soybeans, natural gas, coffee): tend to be more trending. Less algorithmic arbitrage, more directional moves.

**Fixed income** (bonds, TLT): trendiness similar to equities but lower volatility. Trend-following systems can work well.

**General rule**: the more massive and liquid the asset, the less intraday trend and the more mean reversion. The less traded, the more trend.

## Four Examples That Teach

These are not systems to trade -- they're for understanding entry/exit structure and how different rules produce different risk profiles.

### 1. Buy and Hold: The Benchmark

The simplest possible setup. Buy and hold. No exit, no stop, nothing.

```python
# Entry setup: buy on the first bar
# Exit setup: none (until the end of the backtest)
# Result on SPY (~25 years): ~7.5% annual (without dividends)
# Drawdown: severe (50%+ in 2008)
# Time invested: 100%
```

**What it's for**: it's your benchmark. Any system that doesn't beat buy & hold on a risk-adjusted basis has no reason to exist.

### 2. 200-Day Moving Average: The Classic Trend Follower

Buy when the close is above the 200-day moving average. Sell when it falls below.

```python
def sma200_signal(close, sma200):
    """The most basic trend follower there is."""
    if close > sma200:
        return 'BUY'
    elif close < sma200:
        return 'SELL'
    return 'HOLD'

# Result on SPY:
# Win rate: ~30% -- most signals are false
# Profit factor: high -- the few winners are huge
# Trades: ~105 across the entire history
# Time invested: ~71%
# Drawdown: lower than buy & hold
```

**Lesson**: a system can be right only 30% of the time and still be profitable. What matters is the size of winners vs losers. This system fails constantly in sideways markets but captures the big trends.

### 3. Golden Cross: Less Noise, Fewer Trades

50-day moving average crosses above the 200-day -> buy. Crosses below -> sell.

```python
def golden_cross_signal(sma50, sma50_prev, sma200, sma200_prev):
    """Slower than the 200 MA alone, but much cleaner."""
    if sma50 > sma200 and sma50_prev <= sma200_prev:
        return 'BUY'   # golden cross
    elif sma50 < sma200 and sma50_prev >= sma200_prev:
        return 'SELL'  # death cross
    return 'HOLD'

# Result on SPY:
# Win rate: ~84% -- very few failures
# Trades: ~13 across the entire history
# Return: ~4.5% annual
# Problem: so few trades that it lacks statistical significance
```

**Lesson**: an 84% win rate is impressive, but with only 13 trades you can't conclude anything statistically. A system needs hundreds of trades to be evaluable. This example shows why long-term systems need to be tested across multiple assets.

### 4. Connors TPS: Averaging Down with Method

TPS stands for **Time Price Scale** -- a system published by Larry Connors that scales positions by buying more as price corrects within an uptrend. It combines a filter (200 MA), entry signal (2-period RSI), and position scaling.

**Long side:**
- **Filter**: price above 200 MA (bull market)
- **Level 1**: RSI(2) < 25 for 2 consecutive days -> buy 10%
- **Level 2**: if price drops from previous entry -> add 20%
- **Level 3**: if it drops more -> add 30%
- **Level 4**: if it drops more -> add 40% (total: 100%)
- **Exit**: RSI(2) closes above 70

**Short side**: symmetric (below 200 MA, RSI(2) > 75)

```python
def tps_connors_signal(close, sma200, rsi2, rsi2_prev, position_level):
    """
    Connors TPS logic (simplified).
    Buys corrections within an uptrend.
    Scales position as the correction deepens.
    """
    levels = {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.40}

    # Filter: only long if above 200
    if close <= sma200:
        return None, 0

    # Entry: RSI(2) below 25, two consecutive days
    if position_level == 0:
        if rsi2 < 25 and rsi2_prev < 25:
            return 'BUY', levels[0]  # 10%
        return None, 0

    # Scaling: if price dropped since last entry
    if position_level < 4:
        # (simplified -- in practice, compare against the close of the last entry)
        return 'ADD', levels.get(position_level, 0)

    return None, 0

def tps_exit(rsi2):
    """Exit when RSI(2) exceeds 70."""
    return rsi2 > 70
```

**Typical results on SPY:**
- Long win rate: ~71%
- Time invested: ~18% (long only)
- Profit factor: high
- No explicit stop loss

**Lessons:**

1. **Averaging down can work in diversified ETFs within an uptrend** -- it breaks the classic rule of "cut losses quickly," but the logic holds: a broad ETF like SPY historically recovers corrections within an uptrend confirmed by the 200 MA. Outside an uptrend (price below the 200 MA), this logic does NOT apply

2. **Low exposure is an advantage**. Only 18% time invested means 82% of capital is free for other systems. This enables leverage or a portfolio of strategies

3. **Risk exists even without a stop.** Without a stop, a black swan can generate a drawdown larger than anything in the historical data. Connors acknowledges this and recommends using TPS only on diversified ETFs (like SPY), never on individual stocks, because a broad ETF has a much lower probability of collapse than a single stock

4. **Short side: more profitable per trade but riskier.** Higher profit factor but drawdown ~2x vs the long side. Volatility is asymmetric -- declines are fast and violent

## Pragmatism Over Purism

The scientific method says to evaluate entries and exits separately. And that's ideal. But if you're starting out and found a system in a book that you want to test as a whole -- test it as a whole. Don't let perfectionism paralyze you.

What matters is:

- **Understand what each part does.** Even if you don't evaluate them separately, know what your entry is, what your exit is, and why
- **Be critical.** If a book says "RSI is for overbought/oversold," try using it as a trend filter. If they say "never average down," test with data. Question everything
- **Be pragmatic.** If lunar cycles work for you and you can demonstrate it statistically, go for it. What matters is a demonstrable edge, not theoretical elegance

With more experience, you'll be less dogmatic. "It all depends" is the most honest phrase in algorithmic trading.
