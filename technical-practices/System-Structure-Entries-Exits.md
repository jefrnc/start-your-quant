> 🇪🇸 [Leer en Español](System-Structure-Entries-Exits.es.md) | 🇺🇸 **English**

# System Structure: Entries, Exits, and Stops

A trading system breaks down into three parts: **how you enter**, **how you exit with a loss**, and **how you exit with a profit**. Separating them isn't just a formality -- it's what lets you evaluate them with the [scientific method](./Scientific-Method-System-Development.md), isolate what works and what doesn't, and optimize without contaminating results.

## Fundamental Principle: Simplicity

If you can't explain in two minutes to anyone why your system buys and why it sells, it's probably too complex. You don't need machine learning or thousands of lines of code. Simple things work in algorithmic markets -- and they're more robust.

One entry, one filter at most. For exits, you can (and should) use multiple combined methods.

## Order Types for Entries

### Stop (for Trend Entries)

Triggers when price exceeds a level upward (buy) or falls below (sell). The natural entry for breakout and trend-following systems.

```python
def entry_stop_order(current_price, stop_level, side='long'):
    """
    Stop entry: buy if it exceeds a level, sell if it breaks below.
    Always stop market -- we want to guarantee execution.
    """
    if side == 'long' and current_price >= stop_level:
        return 'BUY_MARKET'  # executes at best available price
    elif side == 'short' and current_price <= stop_level:
        return 'SELL_MARKET'
    return None

# Advantage: guarantees the entry, the filter is implicit
# Disadvantage: slippage because you enter in the direction of the move
# and many orders may be triggering at the same level
```

**Always stop market, never stop limit for entries.** If you're using a stop it's because you want guaranteed execution. A stop limit may not fill if price jumps quickly.

### Limit (for Counter-Trend Entries)

You buy when price falls to your price. You sell when it rises to your price. The natural entry for support/resistance and mean-reversion systems.

```python
def entry_limit_order(current_price, limit_level, side='long'):
    """
    Limit entry: buy if price falls to my level or better.
    No negative slippage, but may not fill.
    """
    if side == 'long' and current_price <= limit_level:
        return 'BUY_LIMIT'
    elif side == 'short' and current_price >= limit_level:
        return 'SELL_LIMIT'
    return None

# Advantage: no negative slippage -- you buy at your price or better
# Disadvantage: may not fill if price doesn't reach the exact level
```

**Backtesting problem**: the fact that price touched your level in historical data doesn't guarantee you would have been filled. There may have been thousands of orders ahead in the queue. And the trades that "don't fill" tend to be winners, which overestimates your backtest.

### Market Order (Next Bar Open)

Executes at the best available price immediately. Used when the signal doesn't have an implicit price (e.g., an oscillator crosses a threshold).

```python
# Typical pattern: signal at bar close -> entry at next bar open
if signal_at_close:
    order = 'BUY_MARKET_NEXT_BAR'
    # The system marks the theoretical price as the next bar's open
    # In live trading, you'll execute slightly above or below (bidirectional slippage)
```

### When to Use Each Type

| Entry model | Recommended order |
|---|---|
| High/low breakout | Stop |
| Support/resistance, pivots | Limit |
| Moving average crossover, channel breakout | Stop |
| Oscillator (RSI, stochastic) without implicit price | Market (next bar) |
| Mean reversion | Limit |

## Classic Entry Setups

### Donchian Channels (N-Bar Breakout)

Buy when price exceeds the high of N bars. Sell when it breaks the low. Created in the 1960s, the basis of Richard Dennis's Turtle system. Still works on trending assets.

```python
def donchian_entry(highs, lows, close, period=20):
    upper = highs.rolling(period).max()
    lower = lows.rolling(period).min()

    if close.iloc[-1] > upper.iloc[-2]:  # exceeds yesterday's high
        return 'LONG'
    elif close.iloc[-1] < lower.iloc[-2]:
        return 'SHORT'
    return None
```

### Moving Averages

The most widely used indicator. Three entry variants:
- **Single MA crossover**: price crosses the MA -> signal
- **Dual MA crossover**: fast MA crosses the slow MA
- **MA slope**: direction change -> signal

There's no clear evidence that one type of moving average (simple, exponential, weighted) is consistently better than another. Test on your specific system.

### Bollinger Bands

20-period moving average +/- 2 standard deviations. Two opposite uses:

- **Trend**: buy when it breaks the upper band (volatility expansion). Enters late, with wide stops, but captures big moves
- **Reversion**: buy when it recovers the lower band after losing it. More precise entry, tighter stop

### Momentum Indicators

**RSI** (Wilder): measures the magnitude of gains vs losses in successive closes, normalized from 0 to 100. Above 50 = dominant buying pressure. Also used as an overbought/oversold oscillator.

**MACD** (Gerald Appel): the difference between two EMAs. An acceleration indicator -- measures whether the averages are diverging or converging.

**Stochastic** (George Lane): normalizes price relative to the period's range, from 0 to 100. Smoother than RSI because it includes smoothing. Typically for overbought/oversold, but explore its trend use with high period values.

**ATR** (Wilder): not a signal indicator but a volatility one. Measures the average range including gaps. Essential for sizing stops and profits.

> Don't stick to the typical use of indicators. RSI isn't just for overbought/oversold -- it can work as a trend filter. A Donchian channel isn't just for breakouts -- it can be used for reversion by buying at the lower band. Question everything and test.

## Exit Setups

### The Entry-Exit Asymmetry

For entries: a simple setup, not many rules. For exits: multiple combined methods. You can (and should) exit by stop, by take profit, by opposing signal AND by time, all in the same system.

### Take Profit Exit

Three ways to calculate it:

```python
def take_profit(entry_price, method='volatility', **kwargs):
    """Calculate take profit level."""
    if method == 'fixed':
        return entry_price + kwargs['amount']

    elif method == 'percentage':
        return entry_price * (1 + kwargs['pct'])

    elif method == 'volatility':
        # ATR adjusts the TP to current market volatility
        return entry_price + kwargs['atr'] * kwargs['multiplier']
```

**Recommendation**: adjust by volatility. A fixed $1,000 TP doesn't mean the same thing when the asset moves 3% per day versus when it moves 0.5%.

### Stop Loss Exit

The stop loss **is not for making money -- it's for protecting it**. If a stop loss increases your system's profit, that's a bad sign: there's probably overfitting.

**The stop loss is not free.** Almost always, a system earns less with a stop than without one. What the stop gives you is protection against aberrant events.

```python
def stop_loss(entry_price, method='volatility', **kwargs):
    if method == 'fixed':
        return entry_price - kwargs['amount']
    elif method == 'percentage':
        return entry_price * (1 - kwargs['pct'])
    elif method == 'volatility':
        return entry_price - kwargs['atr'] * kwargs['multiplier']
```

### Opposing Signal Exit

In almost all systems, the signal opposite to the entry takes you out of the market. If you bought when the MA crossed up and now it crosses down, you close.

It can be the same indicator as the entry or a different one, although adding different indicators increases complexity and overfitting risk.

### Time-Based Exit

Sounds strange but it works: exit after N bars if the trade hasn't moved enough. The logic is simple -- being invested is a risk. If you achieve the same profit while spending less time in the market, your risk/return ratio improves.

```python
def temporal_exit(bars_in_trade, max_bars, current_pnl=None):
    """
    Time-based exit: if after N bars the trade hasn't gone anywhere,
    exit. Reduce risk without losing profit.
    """
    if bars_in_trade >= max_bars:
        return True
    # Variant: if at a loss after N bars, exit earlier
    if current_pnl is not None and current_pnl < 0 and bars_in_trade >= max_bars // 2:
        return True
    return False
```

You can also use seasonal cycles: don't trade on Mondays, close before macro news, avoid certain months.

### Trailing Stop

Follows price in favor of your position. Sounds ideal in theory -- you protect accumulated profit. In practice, **they are hard to calibrate correctly** and tend to overfit to backtest data.

The main problem in trend-following systems: the trailing can take you out of the trades with the biggest moves. Price pulls back a little (normal in any trend), the trailing triggers, and price keeps going up without you. In markets with gentle pullbacks, they can work better.

If you use them, adjust by volatility or percentage -- never by absolute value, because a fixed $500 trailing doesn't have the same meaning at high prices versus low prices.

### Catastrophic Stop

A stop that ideally never triggers. It covers black swans -- events that aren't in your historical data.

Not everything is in the backtest. A 9% gap in the DAX from Brexit was three times larger than any previous gap in the data. No backtest would have captured it.

**Considerations**:
- Circuit breakers (7%, 14%, 20% in US) exist but don't guarantee execution -- in a panic there may be no counterparty
- A catastrophic stop doesn't improve the system's profit -- it's there so you survive the event your model didn't foresee
- In a portfolio of 15+ systems, some may not have an explicit stop if they exit by opposing signal quickly. With 1-2 systems, the stop is essential

## Basic Money Management from the Start

Before evaluating anything, incorporate basic money management to equalize results over time.

```python
def equalized_position_size(account_value, price, atr, risk_per_trade_pct=0.01):
    """
    Volatility-adjusted position sizing so that results
    are comparable across the entire history.

    Without this, 2024 trades (Nasdaq at 18,000) dominate the backtest
    vs 2009 trades (Nasdaq at 1,500). That biases the optimization.
    """
    risk_dollars = account_value * risk_per_trade_pct
    shares = int(risk_dollars / atr) if atr > 0 else 0
    return shares
```

**$1,000 is not the same with the Nasdaq at 5,000 versus 15,000.** If you don't equalize, recent trades (with higher prices) dominate the analysis and bias the optimization. This money management is only for equalizing data -- the final money management algorithm is chosen at the end of the process.

## Isolated Evaluation: Entries and Exits Separately

The scientific method requires isolating variables. But you can't evaluate an entry without an exit (you need complete trades to measure).

**To evaluate entries**: use standardized exits (fixed stop and TP adjusted by ATR) and don't touch them. Compare different entries with the same exits.

**To evaluate exits**: use already-validated entries, or better yet, random entries. If your exit doesn't beat random entries, it has no edge.

```python
import random

def random_entry(data, probability=0.05):
    """
    Random entry: on each bar, 5% probability of entering.
    Any entry method should beat this.
    If it doesn't beat it, your entry has no edge.
    """
    return random.random() < probability
```

## The Complete Process

1. Define your profile -> what type of system you're looking for
2. Research ideas -> books, platforms, your own experience
3. Design the entry -> a simple setup, one primary indicator
4. Evaluate the entry with standardized exits
5. Design the exits -> multiple: TP, stop, opposing signal, time-based
6. Evaluate each exit in isolation
7. Combine everything -> entry + exits + basic money management
8. Optimize with protocol (in-sample, validation, out-of-sample)
9. Evaluate stop loss last -> not first, so you don't bias the evaluation of the pure edge
