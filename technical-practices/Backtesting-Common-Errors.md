> 🇪🇸 [Leer en Español](Backtesting-Common-Errors.es.md) | 🇺🇸 **English**

# Common Backtesting Errors

Your backtest is only useful if it's reproducible in live trading. For the three levels of validation (classic, forward, walk-forward), see [Backtesting: From Classic to Walk-Forward](./Backtesting-Three-Levels.md). For data-specific issues, see [Data Quality and Adjustments](./Data-Quality-Adjustments.md). A system that looks spectacular on historical data but has a configuration error, a non-tradable asset, or a look-ahead bias is worthless. These are the most common errors and how to detect them.

## 1. Trading a Non-Tradable Asset

Indices (VIX, S&P 500, IBEX 35, Nasdaq) are synthetic assets -- they cannot be bought or sold directly. To invest, you need a derivative: futures, ETFs, options, or CFDs.

**The VIX case**: a mean-reversion system on the VIX index can look spectacular (profit factor 5+, near-perfect equity curve). But the VIX index is not tradable. When you apply the same system to VIX futures, it may stop working or require significant adjustments.

**Why**: VIX futures are typically in strong **contango** -- each successive expiration trades at a premium to the current one (typically 2% to 8% between months, depending on market conditions). This creates a persistent artificial decline in the continuous chart that destroys the clean cyclicality you see in the index.

**Rule**: before backtesting, verify that the asset is tradable. If it's an index, use the corresponding futures or ETF.

### Contango and Backwardation

- **Contango** (the norm): future expirations are more expensive than spot. Reflects the cost of time: interest rates, storage (in commodities), uncertainty
- **Backwardation**: future expirations are cheaper than spot. In commodities, it indicates current scarcity or strong immediate demand. In the VIX, a shift to backwardation signals market panic (high demand for immediate protection)

Contango affects all futures, but in most cases it's small (~1% between quarters for equity indices). In the VIX it's extreme and destroys strategies that work on the pure index.

## 2. Look-Ahead Bias

Using information that would not have been available at the time of the decision.

**Common examples:**

```python
# WRONG: buying at today's open based on today's close
# By the close you can no longer buy at the open -- it already happened
if close_today > sma200_today:
    buy_at(open_today)  # IMPOSSIBLE -- open_today already passed when you have close_today

# RIGHT: signal at today's close -> buy at tomorrow's open
if close_today > sma200_today:
    buy_at(open_tomorrow)  # CORRECT -- decision today, execution tomorrow
```

**External data**: the COT (Commitment of Traders) report is published on Fridays but is dated the previous Tuesday. If your system uses the data date (Tuesday) instead of the publication date (Friday), you have look-ahead bias.

The same applies to earnings reports, macro data (GDP, employment), or any data where the reference date differs from the availability date.

**Rule**: all information you use must be 100% available at the time it's evaluated. When in doubt, use the publication date, not the data date.

## 3. Look Inside Bar (Execution Order Within the Bar)

From a historical bar you only know 4 values: open, high, low, close. You don't know the order in which price moved internally.

**The problem**: if you have a stop at 100 and a take profit at 105, and the bar has low=99 and high=106, which triggered first? If it went down first -> stop. If it went up first -> TP. The outcome is opposite depending on the order.

```
Scenario A: drops -> rises    ->  stop loss triggers first  ->  loss
Scenario B: rises -> drops    ->  take profit triggers first ->  profit
```

The backtest engine **infers** the order, but it can be wrong.

**Solution**: enable Look Inside Bar (TradeStation) or Bar Magnifier (MultiCharts), which loads a lower timeframe (1 minute) to simulate how each bar was formed.

**Trap**: even with 1-minute data, if your stops are very tight, they can trigger within a single 1-minute bar and the problem repeats. Verify that no orders trigger within the same bar on the lower timeframe.

**Real case**: a system with a very tight trailing stop showed a spectacular equity curve. When Look Inside Bar was enabled, it turned into losses. The trailing stop was triggering within the bar before price reached the TP.

## 4. Market Rules You Don't Know

Each market has specific rules that can make your system inoperable.

| Rule | Example |
|---|---|
| **Restricted order types** | VIX does not accept market orders or stops in extended hours -- limit only |
| **Trading hours** | Globex does not accept stops in premarket |
| **Circuit breakers** | US stocks halt on declines of 7%, 14%, 20% |
| **Position limits** | Each futures contract has a maximum number of contracts allowed |
| **Liquidity by time of day** | An asset may have 1-tick spreads at 10 AM and 20-tick spreads at 3 AM |
| **Short selling** | Some stocks are Hard-To-Borrow (HTB) -- you can't short them or it's expensive |

**Where to find this info**: exchange websites (CME, CBOE, NYSE). Search for "contract specifications" for futures. The CME has excellent free courses.

**Rule**: before trading any new asset, read the contract specifications. It's a one-time effort -- after that, you know.

## 5. Limit Orders in Backtests

Limit orders have a problem that stops and market orders don't: **they may not fill even though price touched your level**.

In live trading, when you place a limit buy at 100.00, there's an order queue. If there are 500 orders ahead and only 400 fill, you get left out. But the backtest marks the trade as executed because price touched 100.00.

**The bias**: trades that "don't fill" would almost always have been winners (price touched your level and bounced). This inflates the backtest.

**Conservative solution**: configure the backtest to fill limits only when price **exceeds** your level (one tick beyond), not when it touches it. It's more pessimistic but more realistic.

## 6. Slippage and Commissions

Slippage is the difference between the system's theoretical price and the actual execution price. Commissions are what the broker charges.

**General rule**: 1-2 ticks of slippage per trade under normal conditions. More in:
- High-volatility moments
- Illiquid assets
- Trading around news or breakouts (many stops triggering at the same time)

If your system has an average profit of 3 ticks per trade and slippage + commissions total 2 ticks, you're left with 1 tick -- any variation puts you in the red. Viable systems need comfortable margin above transaction costs.

## Preliminary Evaluation Checklist

Before moving a system to formal optimization, verify:

- [ ] The asset is tradable (not a pure index)
- [ ] I know the contract specifications (hours, allowed orders, expirations, limits)
- [ ] There is no look-ahead bias in the code (all info available at decision time)
- [ ] External data uses publication date, not data date
- [ ] Look Inside Bar enabled if using tight stops/TPs
- [ ] Limits configured with "exceeds" fill logic, not "touches"
- [ ] Slippage and commissions included (even if estimated)
- [ ] I've reviewed the chart at different times: high/low volatility, trending, sideways, crash
- [ ] Signals are visually consistent with what the system should be doing
- [ ] Everything is 100% reproducible in live trading
