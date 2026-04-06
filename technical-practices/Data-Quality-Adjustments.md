> 🇪🇸 [Leer en Español](Data-Quality-Adjustments.es.md) | 🇺🇸 **English**

# Data Quality and Adjustments for Backtesting

Data is your raw material. If your database doesn't faithfully represent what happened in the market, your backtest is worthless -- no matter how sophisticated your algorithm. This is probably the most underestimated topic in algorithmic trading.

## The Roll Problem in Futures

Futures expire. The Mini S&P 500 expires quarterly (third Friday of March, June, September, December). To build a historical chart longer than one quarter, you need to **splice expirations** into what's called a continuous chart.

The problem: between expirations there's almost always a **price gap**. This gap isn't the product of real supply and demand -- it's an artifact of linking two different contracts. If you don't handle it correctly, that gap contaminates your indicators and distorts your backtest.

### Where the Gap Comes From

When a new futures contract starts trading (e.g., September), it trades **above** the expiring contract (e.g., June). The difference is explained by:

- **Interest rates**: the time value of money. Money today is worth more than money in 3 months
- **Dividends**: in index futures, the dividends that stocks will pay between now and expiration
- **Storage costs**: in commodities (oil, grains), the cost of storing the physical asset

In US indices with high interest rates, this difference can be 40-45 points. In bonds (e.g., Bund), the gaps are enormous.

### When to Roll

The optimal time to switch contracts in your continuous chart is **when the new expiration has more volume than the old one**:

```python
def detect_roll_date(front_volume, next_volume):
    """
    Detects the day when the next expiration surpasses the current one in volume.
    That is the optimal day for the splice in the continuous chart.
    """
    for date in front_volume.index:
        if date in next_volume.index:
            if next_volume[date] > front_volume[date]:
                return date
    return None

# IMPORTANT: timing varies by market
# - US Indices (S&P, Nasdaq, Dow): 5-7 days before expiration
# - DAX and European futures: varies, can be a few days before or even close to expiration
# - Bonds: varies by market
# Knowing the dates for each future you trade is your responsibility
```

**Key note**: the operational roll timing (closing a position and opening another in the new contract) does NOT have to coincide exactly with the splice point in the continuous chart. They are two different things.

## Three Adjustment Methods

### 1. Unadjusted

Simply splice the contracts as they trade. The gap stays.

**Advantage**: historical prices are real -- what traded, traded.

**Problem**: the artificial gap contaminates any indicator that uses more than one day of data. And it's not just the roll day -- a 50-period EMA can be affected for weeks because its recursive formula propagates the error.

```python
# Indicators especially sensitive to unadjusted gaps:
# - EMA (exponential moving average): the recursive formula propagates the error
# - ATR: uses true range which includes the gap between closes
# - ADX: derived from ATR
# - RSI: uses price changes that include the false gap
# - Stochastic: compares current price to range, altered by the gap
#
# They can be distorted for MORE bars than their calculation period
# due to recursive formulas that use prior data.
```

### 2. Absolute Value Adjustment (Points)

Subtract (or add) the gap difference from the entire prior history.

```python
def adjust_absolute(data, roll_gaps):
    """
    Backward adjustment by absolute value.
    Keeps the current contract at real price and modifies the past.
    """
    adjusted = data.copy()
    cumulative_adjustment = 0

    # Process from most recent to oldest
    for roll_date, gap in sorted(roll_gaps.items(), reverse=True):
        cumulative_adjustment += gap
        mask = adjusted.index < roll_date
        for col in ['open', 'high', 'low', 'close']:
            adjusted.loc[mask, col] -= cumulative_adjustment

    return adjusted
```

**Advantage**: preserves the minimum tick of the asset. If the Mini S&P moves in 0.25 increments, the adjusted prices do too.

**Disadvantage**: over long histories, the proportional price relationship gets distorted. A 40-point adjustment doesn't mean the same thing when the S&P was at 2,000 versus when it's at 5,000.

### 3. Ratio Adjustment (Percentage) -- Recommended

Adjust proportionally, preserving the percentage relationship of prices.

```python
def adjust_ratio(data, roll_dates_and_prices):
    """
    Backward adjustment by ratio.
    Preserves percentage relationships -- the most correct method
    for long histories.
    """
    adjusted = data.copy()
    cumulative_ratio = 1.0

    for roll_date, old_close, new_close in sorted(
        roll_dates_and_prices, reverse=True
    ):
        ratio = new_close / old_close
        cumulative_ratio *= ratio
        mask = adjusted.index < roll_date
        for col in ['open', 'high', 'low', 'close']:
            adjusted.loc[mask, col] /= cumulative_ratio

    return adjusted
```

**Advantage**: a 1% move in 2003 is represented the same as a 1% move in 2024. This is correct because markets move in percentages, not absolute points.

**Disadvantage**: prices lose the minimum tick (decimals appear that don't exist in the real market). Solution: round orders to the real tick before sending them.

### Which One to Use

| Situation | Recommendation |
|---|---|
| Backtest with percentage indicators (ATR%, ROC) | Ratio |
| Backtest with absolute point indicators | Absolute value works |
| Short history (< 2 years) | Either one, the difference is minimal |
| Long history (> 5 years) | Ratio, without question |
| I want to see real historical prices | Unadjusted (only for visualization, not for backtesting) |

**The ideal combination**: adjust by ratio and work with percentage-based indicators.

```python
# INSTEAD OF:
atr = calculate_atr(data, 14)  # ATR in points -- inconsistent over time

# USE:
atr_pct = calculate_atr(data, 14) / data['close'] * 100  # ATR in % -- consistent
```

## Dividends in Stocks: The Same Problem

When a stock pays a dividend, the price is automatically discounted by the dividend amount. Your net worth position doesn't change (you have the stock worth less + the dividend cash), but your chart shows a drop that wasn't the product of supply and demand.

```
Example: stock trades at $100, pays $2 dividend
- Before: 1 share x $100 = $100
- After: 1 share x $98 + $2 cash = $100
- The chart shows a 2% drop that is NOT a real loss
```

For backtesting, it's better to **adjust for dividends** (backward, by ratio) so that your indicators and signals aren't contaminated by artificial drops.

### Total Return Indices

"Total Return" indices already incorporate this adjustment. They reinvest dividends into the index, showing the real return of an investor who reinvests.

The difference can be enormous: the IBEX 35 standard index may appear far from its nominal all-time highs, while the IBEX with dividends may have surpassed them by a wide margin -- the gap accumulates year after year. In European stocks that pay high dividends, the effect is very pronounced. In the Nasdaq, where dividends are smaller, the effect exists but is less dramatic.

**For backtesting strategies on indices**: use the Total Return or the dividend-adjusted ETF (such as backward-adjusted SPY). They are the most faithful representation of real returns.

## Common Data Errors

### 1. Assuming All Data Is Equal

Different providers can have different data for the same asset. Compare databases before blindly trusting one.

Even reference providers have problems with very old data. If your provider has gold data since 1975, verify the quality of those early years -- data capture technology was rudimentary.

### 2. Not Verifying Delay in Real-Time Data

Different real-time data sources may not be synchronized. If your signal uses data from one feed and your order goes to another, a delay of seconds can matter.

### 3. Using Backtest and Real-Time Data of Different Quality

If your backtest uses clean, high-quality data but your live execution uses an inferior feed, results will diverge. The properties of both datasets must be analogous.

### 4. Ignoring Survivorship Bias in Stocks

If you backtest a rotational strategy on the S&P 500 from 2010, you need the index composition **at each historical moment**, including companies that were removed. Those that survived to today have an inherent positive bias.

## Pre-Backtesting Checklist

- [ ] Is my futures data adjusted? By ratio or by absolute value?
- [ ] Do I know the expiration dates and roll pattern for each future?
- [ ] Is my stock data adjusted for dividends and splits?
- [ ] Is my historical data quality comparable to my real-time data?
- [ ] Have I verified the provider's quality, especially for old data?
- [ ] Are my indicators using percentages instead of absolute values when possible?
- [ ] Am I aware of survivorship bias if using stock universes?
