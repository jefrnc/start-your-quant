> 🇪🇸 [Leer en Español](Compound-Growth-and-Risk.es.md) | 🇺🇸 **English**

# Compound Growth and the 10 Real Risks of Trading

## The Growth Engine: Compound Interest in Trading

The most powerful concept in finance isn't any indicator -- it's compound interest. In trading, it's equivalent to **reinvesting your gains into your system** so that your position sizes grow with your account.

### Simple vs Compound: The Real Difference

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_growth(initial_capital, monthly_return, months):
    """Compare simple vs compound growth."""
    simple = np.zeros(months + 1)
    compound = np.zeros(months + 1)
    simple[0] = compound[0] = initial_capital

    monthly_gain = initial_capital * monthly_return

    for m in range(1, months + 1):
        simple[m] = simple[m-1] + monthly_gain              # always gains the same
        compound[m] = compound[m-1] * (1 + monthly_return)   # gains on the accumulated

    return simple, compound

capital = 10_000
rate = 0.03  # 3% monthly
months = 60  # 5 years

simple, compound = simulate_growth(capital, rate, months)

print(f"Initial capital: ${capital:,.0f}")
print(f"After {months} months at {rate*100}% monthly:")
print(f"  Simple:   ${simple[-1]:,.0f} (+{(simple[-1]/capital - 1)*100:.0f}%)")
print(f"  Compound: ${compound[-1]:,.0f} (+{(compound[-1]/capital - 1)*100:.0f}%)")
# Simple:    $28,000 (+180%)
# Compound:  $58,916 (+489%)
```

The difference amplifies over time. Over 5 years at 3% monthly, compounding generates **almost 3 times more** than simple growth. This is why the first months "don't feel" like progress, but after a year the curves separate dramatically.

### Practical Application: Position Sizing that Grows with Your Account

```python
def dynamic_position_size(account_balance, risk_per_trade_pct, entry_price, stop_price):
    """
    Position sizing is recalculated with each trade based
    on the current balance -- not the initial balance.
    This IS compound interest applied to trading.
    """
    risk_amount = account_balance * risk_per_trade_pct
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share == 0:
        return 0

    shares = int(risk_amount / risk_per_share)
    return shares

# Month 1: $10,000 account
shares_m1 = dynamic_position_size(10_000, 0.01, 5.00, 4.80)
# Month 12: account grew to $14,000
shares_m12 = dynamic_position_size(14_000, 0.01, 5.00, 4.80)

print(f"Month 1:  {shares_m1} shares (risk ${10_000 * 0.01:.0f})")
print(f"Month 12: {shares_m12} shares (risk ${14_000 * 0.01:.0f})")
# Same percentage risk, more capital at work.
```

### The Trap: Compounding Also Works Against You

Compound interest amplifies losses just like gains. If you lose 20% of your account, you need to gain 25% to get back to the starting point. If you lose 50%, you need to gain 100%.

```python
def recovery_needed(drawdown_pct):
    """How much you need to gain to recover from a drawdown."""
    return (1 / (1 - drawdown_pct) - 1) * 100

for dd in [0.10, 0.20, 0.30, 0.50, 0.70]:
    recovery = recovery_needed(dd)
    print(f"Loss {dd*100:.0f}% -> need to gain {recovery:.0f}% to recover")

# Loss 10% -> need to gain 11%
# Loss 20% -> need to gain 25%
# Loss 30% -> need to gain 43%
# Loss 50% -> need to gain 100%
# Loss 70% -> need to gain 233%
```

**Conclusion**: protecting capital isn't conservatism -- it's mathematics. A 50% drawdown puts you in a position where you need to double your account just to get back to zero.

## The 10 Real Risks of Algorithmic Trading

Most traders only think about market risk. But there are at least 10 types of risk, and the less obvious ones do the most damage.

### 1. Market Risk

The most evident: price moves against you. Overnight gaps, flash crashes, macro events.

**Mitigation**: stop losses, position sizing, temporal diversification (not all capital at the same time).

### 2. Design Risk

Your algorithm has a bug or flawed logic. It backtests well but for the wrong reasons (lookahead bias, overfitting, survivorship bias).

**Mitigation**: walk-forward analysis, out-of-sample testing, peer code review, paper trading before real capital.

### 3. Liquidity Risk

There isn't enough volume to enter or exit at the price you want. Slippage eats your edge. Especially relevant in small caps.

**Mitigation**: filter by minimum volume, limit position size as a percentage of daily volume, use limit orders instead of market orders.

```python
def max_position_by_liquidity(avg_daily_volume, max_pct_of_volume=0.01):
    """
    Never trade more than 1% of average daily volume.
    In small caps, even 1% can move the price.
    """
    return int(avg_daily_volume * max_pct_of_volume)

# Stock with 500k daily volume -> maximum 5,000 shares
max_shares = max_position_by_liquidity(500_000)
print(f"Maximum position by liquidity: {max_shares:,} shares")
```

### 4. Breakdown Risk

Your system stops working. The market changed regimes and the edge it was exploiting no longer exists. Every system has a lifespan.

**Mitigation**: monitor rolling metrics (30/60/90-day Sharpe), define deactivation conditions before launching the system, don't depend on a single system.

### 5. Operational Risk

Technical failures: internet goes down in the middle of an open position, the server crashes, the broker API stops responding, a deploy goes wrong.

**Mitigation**: UPS for power, backup connection, monitoring alerts, ability to close positions from your phone, stops on the broker's server (not just local ones).

### 6. Credit Risk

Being unable to meet financial obligations to the broker -- for example, a margin call you can't cover because losses exceeded your available capital. Different from market risk (price moving against you): credit risk is not having the funds to respond.

**Mitigation**: maintain ample margin, avoid overnight positions in instruments with high gap risk, and size positions so that the worst reasonable scenario doesn't compromise the account.

### 7. Counterparty Risk

Your broker being unable to respond. Broker bankruptcy, frozen funds, inability to execute orders during a crash.

**Mitigation**: use regulated brokers (SIPC in the US), don't keep all capital at a single broker, verify the broker's solvency and regulation.

### 8. Regulatory Risk

Changes in regulation that affect your operations. New margin rules, taxes, short selling restrictions (like the temporary bans in 2008 and 2020), changes to the PDT rule.

**Mitigation**: stay informed, design systems that don't depend on a single regulatory mechanism.

### 9. Legal Risk

Lawsuits, intellectual property issues if you use third-party code, unintentional regulatory violations.

**Mitigation**: understand the rules of your jurisdiction, have appropriate licenses if managing third-party capital.

### 10. Reputational Risk

Relevant if you manage external capital or publish results. A public drawdown can destroy your ability to raise future capital.

**Mitigation**: be transparent about risks, don't promise returns, document your track record including the bad periods.

### Risk by Trading Style

Not all risks weigh equally depending on how you trade:

| Risk | Intraday | Swing/Daily |
|---|---|---|
| Market | Lower (no overnight) | Higher (gaps) |
| Liquidity | Higher (need to enter/exit fast) | Lower |
| Breakdown | Higher (more sensitive to noise) | Lower (clearer signals) |
| Operational | Higher (depend on constant uptime) | Lower |
| Counterparty | Higher (more interaction with broker) | Lower |
| Credit | Lower (short-duration positions) | Higher (overnight margin calls) |

## The Complete Framework: Sustainable Growth

Putting it all together -- compound growth is your engine, but risks are the brakes. A profitable system that ignores risks eventually blows up. An ultra-conservative system that ignores compounding never grows.

The balance:

To implement this as part of your operations, see [Trading Plan](./Trading-Plan-Framework.md).

1. **Positive expectancy**: your system must win more than it loses on average
2. **Dynamic position sizing**: that grows with your account (compounding)
3. **Capital protection**: defined maximum drawdown where size is reduced or trading stops
4. **Diversification**: multiple systems, multiple instruments, multiple timeframes
5. **Continuous monitoring**: rolling metrics that detect degradation before it becomes catastrophic

```python
def should_reduce_risk(rolling_sharpe_30d, rolling_sharpe_90d, max_drawdown_current):
    """
    Simple control framework: if metrics degrade,
    reduce exposure before the drawdown becomes unrecoverable.
    """
    if max_drawdown_current > 0.15:  # drawdown > 15%
        return "STOP -- pause system, review"
    if rolling_sharpe_30d < 0 and rolling_sharpe_90d > 0:
        return "REDUCE -- lower position size to 50%"
    if rolling_sharpe_30d < 0 and rolling_sharpe_90d < 0:
        return "STOP -- the edge may have disappeared"
    return "NORMAL -- trade at full size"
```

Controlling risk isn't optional -- it's what keeps you in the game long enough for compounding to work its magic.
