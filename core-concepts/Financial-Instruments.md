> 🇪🇸 [Leer en Español](Financial-Instruments.es.md) | 🇺🇸 **English**

# Financial Instruments for Quant Traders

Before building an algorithmic system, you need to understand **what you're trading**. Each instrument has different rules that directly affect how you design, backtest, and execute your strategy.

## Stocks (Equities)

A stock represents partial ownership of a company. When you buy a stock, you're a partner -- with rights to dividends and voting at shareholder meetings.

They're called "equities" (or "variable income" in some traditions) because the total return (price + dividends) is uncertain -- unlike a bond where the coupon is fixed in advance.

### What matters for your system

- **Chart representation**: plotted by the *last* (the most recent price at which a trade was executed)
- **Hours**: regular market 9:30-16:00 ET, premarket from 4:00 AM, afterhours until 20:00
- **Liquidity**: varies enormously -- a large cap like AAPL vs a $2 small cap are completely different worlds
- **Short selling**: requires locating borrowed shares (locate), not always available in small caps
- **No expiration**: you can hold a position indefinitely

### Why the US market

The US market has the highest volume, the largest number of listed instruments, and the most mature APIs for algorithmic trading. If your account is small, commissions from brokers like IBKR ($0.005/share) are manageable. Starting with European stocks limits your opportunities and your tools.

```python
import yfinance as yf

# Stock data -- it's this simple to access
data = yf.download('AAPL', period='1mo', interval='1d')
print(f"Last price: ${data['Close'].iloc[-1]:.2f}")
print(f"Average volume: {data['Volume'].mean():,.0f} shares/day")
```

## Bonds (Fixed Income)

Debt instruments: you lend money to a company or government and they return the principal plus a coupon (fixed interest). They're called "fixed income" because the coupon is predetermined.

### What matters for your system

- **The coupon doesn't change, but the bond price does**: there's a secondary market where bonds trade at a discount or premium
- **Inverse relationship with interest rates**: when rates go up, bond prices go down -- this is algorithmically tradeable
- **Higher coupon = higher risk**: an Argentine bond paying 15% is riskier than a US Treasury at 4%
- **Fixed maturity**: if you wait until maturity, you collect exactly what was agreed (barring default)

### Quant application

Bonds are fundamental for **yield curve** strategies, **spread trading** (e.g., long corporate bonds / short treasuries), and as a macro indicator for equity systems. A momentum system in stocks that ignores the bond market is operating with incomplete information.

## Futures

Standardized contracts to buy/sell an asset at an agreed price on a future date. They trade on regulated markets (CME, EUREX, etc.).

### What matters for your system

- **They have expiration**: if you don't close before expiry, you may receive physical delivery (commodities) or cash settlement (financials)
- **Rollover**: to maintain continuous exposure, you need to "roll" from the nearest contract to the next -- your backtest MUST account for this
- **Chart representation**: plotted by the *last*, same as stocks
- **Margin**: you don't pay 100% of the value -- you trade on margin, which amplifies gains and losses
- **Native short position**: you can sell futures without locate restrictions

### Common backtesting pitfalls

For a complete treatment of data adjustment in futures, see [Data Quality and Adjustments](../technical-practices/Data-Quality-Adjustments.md).

```python
# WRONG: using a continuous chart without adjusting for rollover
# Gaps between contracts generate false signals

# RIGHT: backward-adjust by difference at rollover time
# Simplified concept -- in practice, specialized tools
# or trading platform functions are used.
#
# The idea: on the roll date, calculate the price difference
# between the new and old contract, and subtract that difference
# from the entire prior history to eliminate the artificial gap.
```

### Specifications you must know before trading

Each future has unique specifications. Before including any future in your system, verify:

| Specification | Why it matters |
|---|---|
| Contract size | Defines how much capital you actually need |
| Minimum tick and value | Affects your minimum stop loss in dollars |
| Last trading day | If your system doesn't close before this, the broker will (with a penalty) |
| Deliverable vs. cash settlement | You never want 1,000 barrels of oil delivered to you |
| Trading hours | Some futures trade nearly 24h, others don't |

## Options

They give the buyer a **right** (not obligation) to buy or sell an asset at a specific price (strike). The seller has the **obligation** if the buyer exercises.

- **Call**: right to buy
- **Put**: right to sell
- The buyer pays a **premium** for that right (like insurance)

### The 4 basic positions

| Position | Outlook | Maximum risk | Maximum gain |
|---|---|---|---|
| Buy Call | Bullish | Premium paid | Unlimited |
| Sell Call | Bearish/Neutral | Unlimited | Premium collected |
| Buy Put | Bearish | Premium paid | Strike - Premium |
| Sell Put | Bullish/Neutral | Strike - Premium | Premium collected |

### Why options are hard to algorithmize

Options depend on multiple variables simultaneously (the "Greeks"):

- **Delta**: sensitivity to the underlying's price
- **Theta**: time decay (the option loses value every day)
- **Vega**: sensitivity to implied volatility
- **Gamma**: acceleration of delta

Additionally, each underlying has dozens of strikes and active expirations simultaneously. This multiplies the complexity of data, backtesting, and execution. If you're starting in algorithmic trading, options are **not** the best entry point.

### American vs European options

- **American**: can be exercised at any time -- more flexibility but more variables to model
- **European**: only at expiration -- simpler to model algorithmically

## CFDs (Contracts for Difference)

A purely speculative derivative product. You buy/sell the price difference without owning the asset. **They don't trade on regulated markets** -- they're OTC products negotiated directly with your broker.

### What matters for your system

- **Your broker is your counterparty**: they "create" the market, which generates a structural conflict of interest
- **There's no single price**: each broker can have different spreads and prices for the same CFD
- **Overnight swaps**: holding positions open from one day to the next has a cost -- this destroys slow swing trading strategies
- **No expiration**: unlike futures, they don't expire
- **Variable spread**: during high volatility moments, the broker can widen the spread significantly or even close positions

### Chart representation

CFDs (and forex) are typically negotiated and charted based on **bid or ask** (or a calculated mid-price), not the last as in centralized exchanges. This means:

```
# A CFD backtest that uses "close" data without distinguishing bid/ask
# is ignoring the broker's real spread.
#
# If your strategy has an average profit of 5 pips and the spread
# is 2 pips, the spread eats 40% of your profit.
# You don't see it in backtest. You do in live trading.
```

### When CFDs make sense

With very small accounts (< $5,000) where you can't access futures due to margin, CFDs provide access to markets with small position sizes. But if you can trade futures, prefer futures: they're regulated, have a clearinghouse, and the price is transparent.

## Forex (Foreign Exchange Market)

A decentralized market where currency pairs are traded. It's the largest market in the world (~USD 6-7 trillion daily volume) and operates 24 hours on business days.

### What matters for your system

- **Pairs**: you always trade one currency against another (EUR/USD, GBP/JPY). If EUR/USD goes up, the euro strengthens vs. the dollar
- **OTC like CFDs**: there's no central exchange, each liquidity provider can have a different price
- **Sessions**: it moves by time zones -- Tokyo -> London -> New York. Liquidity and volatility change depending on the session
- **Bid/ask representation**: like CFDs, your backtest must account for the real spread
- **No expiration**: similar to CFDs, positions are held indefinitely (with swaps)

### Advantage for algorithmic systems

Forex is one of the most algorithm-friendly markets due to its massive liquidity, continuous operation, and predictable volatility by session. Many quant firms trade forex as their first market.

## Comparison Table for Decision Making

| Criterion | Stocks | Futures | Options | CFDs | Forex |
|---|---|---|---|---|---|
| **Practical minimum capital** | $500+ | $5,000+ | $2,000+ | $200+ | $200+ |
| **Regulation** | High | High | High | Low | Low |
| **Algorithmic complexity** | Medium | Medium | High | Low | Medium |
| **Backtest data** | Easy | Medium | Difficult | Difficult | Medium |
| **Data by** | Last | Last | Last | Bid/Ask | Bid/Ask |
| **Expiration** | No | Yes | Yes | No | No |
| **Short selling** | Limited | Native | Via puts | Native | Native |
| **Best for starting algo** | Yes | With capital | No | With small account | Yes |

## Implications for Your System

### If you're just starting

Start with **US stocks** or **forex majors**. Abundant data, accessible brokers, and the complexity is manageable. Build your first functional system before adding more complex instruments.

### If you already have a profitable system

Diversifying by instrument is as important as diversifying by strategy. A portfolio with systems in stocks + futures + forex has lower correlations than one solely in stocks. Decorrelation between instruments is the true "holy grail" of portfolio management.

### Common mistakes

1. **Backtesting CFDs with exchange data**: the prices aren't the same -- your broker has its own feed
2. **Ignoring rollover costs in futures**: a system that rolls 12 times a year has 12 extra slippage events
3. **Assuming you can short any stock**: in small caps, locates are expensive or nonexistent
4. **Using the same backtesting framework for options as for stocks**: options need to model time decay, implied volatility, and multiple simultaneous strikes
