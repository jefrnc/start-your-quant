> 🇪🇸 [Leer en Español](Expected-Growth-Metrics-Hierarchy.es.md) | 🇺🇸 **English**

# Expected Growth: The Metric Almost Nobody Uses

Most traders evaluate their systems with Sharpe ratio, profit factor, or win rate. These metrics are useful but incomplete -- none of them captures what really matters: **how much your account grows trade by trade when you reinvest profits**.

Expected Growth (EG) does capture this. And it reveals a counterintuitive truth: two systems with the same arithmetic expectancy can produce radically different results when compounded.

## The Metrics Hierarchy

```
Expected Growth (EG)  >  Expectancy (Edge)  >  Win Rate  >  Profit Ratio
       ↑                      ↑                    ↑              ↑
   Real growth           Average gain         Frequency     Relative size
   with compounding      per trade            of winners    of gains
```

Each metric on the right feeds into the next one on the left, but **does not determine it**. You can have an excellent profit ratio and win rate, good expectancy, and still grow very little geometrically. The reason is that compounding is not linear.

## The Metrics Step by Step

Before getting to EG, you need to understand the three metrics that feed into it. Each one measures something different, and each one alone is insufficient.

### Win Rate

**What it measures**: the percentage of trades that end in profit.

**How it's calculated**:

```python
win_rate = winning_trades / total_trades
# 75 winners out of 100 trades -> win_rate = 0.75 (75%)
```

**How to interpret it**: it indicates how frequently the system is right. A 60% win rate means that out of every 10 trades, approximately 6 are winners and 4 are losers.

**Typical values by system type**:
- Trend following systems: 30-45%
- Mean reversion systems: 55-70%
- Breakout / volatility systems: 40-55%

**Why it's not enough on its own**: a system with 90% accuracy can lose money if the 10% of losses are catastrophic (e.g., wins $1 nine times, loses $20 once -> loses $11 net). A system with 30% can be highly profitable if the winners are huge. Win rate says nothing about the **size** of wins and losses.

### Profit Ratio (Win/Loss Ratio)

**What it measures**: the relationship between what you gain when you're right and what you lose when you're wrong. Also known as **reward-to-risk ratio** or **payoff ratio**.

**How it's calculated**:

```python
profit_ratio = average_win / average_loss
# If on average you win $200 and lose $100 -> profit_ratio = 2.0 (2:1)
```

**How to interpret it**: a 2:1 ratio means each winning trade recovers what two losing trades took away. A 0.5:1 ratio means you need to win twice as often as you lose just to break even.

**Typical values**:
- Trend following systems: 2:1 to 5:1 (few winners but large)
- Mean reversion systems: 0.5:1 to 1.5:1 (many winners but small)
- Breakout: 1:1 to 3:1

**Why it's not enough on its own**: a 10:1 ratio sounds spectacular, but if your win rate is 5%, you lose 95 out of every 100 trades. 95 x $100 lost = $9,500 in losses, 5 x $1,000 won = $5,000. Incredible ratio, disastrous result.

### Expectancy / Edge

**What it measures**: the average expected gain for every dollar you risk. It's the first metric that combines win rate and profit ratio into a single number. Also known as **mathematical expectation**, **edge**, or simply **expectancy**.

**How it's calculated**:

```python
def expectancy(win_rate, profit_ratio):
    """
    Expected gain per dollar risked.
    Positive = edge. Negative = the market eats you alive.
    Zero = neutral game (like a casino with no house edge).
    """
    return win_rate * profit_ratio - (1 - win_rate)

# Examples:
expectancy(0.50, 2.0)  # = 0.50 -> you gain $0.50 per $1 risked
expectancy(0.75, 1.0)  # = 0.50 -> you gain $0.50 per $1 risked
expectancy(0.40, 1.0)  # = -0.20 -> you LOSE $0.20 per $1 (don't trade)
expectancy(0.30, 5.0)  # = 0.80 -> you gain $0.80 per $1 risked
```

**How to interpret it**:
- **Positive**: the system has a statistical edge. In the long run, you make money
- **Zero**: neutral game. Commissions will make you lose
- **Negative**: the system loses money systematically. No position sizing or risk management can save it

**Practical threshold**: an expectancy of at least 0.10-0.20 (10-20 cents per dollar risked) is needed to cover transaction costs and slippage. Below that, real-world costs eat up the edge.

**Why it's not enough on its own**: expectancy is an arithmetic mean. It assumes you always bet the same fixed amount. But if you reinvest profits (compounding), your position sizes grow with your account. And when you compound, the frequency of wins (win rate) matters in a way that arithmetic expectancy doesn't capture. That's exactly what Expected Growth measures.

### Expected Growth (EG)

**What it measures**: the expected geometric growth rate per trade when position size is adjusted to available capital. It's the metric that captures the **real growth of your account with compounding**.

**Why it's different from expectancy**: expectancy tells you "on average you gain X per trade." EG tells you "your account grows X% per trade when you reinvest." The difference is enormous because compounding is not linear -- a 50% loss requires a 100% gain to recover.

**How it's calculated**: it uses the Kelly fraction (the optimal proportion of capital to risk per trade) and applies it to the geometric growth formula.

```python
def expected_growth(win_rate, profit_ratio):
    """
    Expected geometric growth per trade with Kelly sizing.
    
    Formula: EG = (1 + f*R)^p * (1 - f)^(1-p) - 1
    where f = Kelly fraction = expectancy / profit_ratio
    """
    p = win_rate
    R = profit_ratio
    edge = p * R - (1 - p)
    
    if edge <= 0:
        return 0  # no edge, no growth
    
    f = edge / R  # Kelly fraction
    eg = (1 + f * R) ** p * (1 - f) ** (1 - p) - 1
    return eg
```

## The Example That Changes Everything

Two systems with **exactly the same arithmetic expectancy** (0.50 per dollar risked):

```python
# System 1: few winners but large
eg1 = expected_growth(win_rate=0.50, profit_ratio=2.0)

# System 2: many winners but small  
eg2 = expected_growth(win_rate=0.75, profit_ratio=1.0)
```

| Metric | System 1 | System 2 |
|---|---|---|
| Win Rate | 50% | 75% |
| Profit Ratio | 2:1 | 1:1 |
| Expectancy | 0.50 | 0.50 |
| Kelly fraction | 25% | 50% |
| **Expected Growth** | **6.1%** | **14.0%** |

Same expectancy, but System 2 grows **2.3 times faster**.

### Why This Happens

The win rate enters the EG formula as an **exponent**, not as a multiplier:

```
EG = (1 + f*R)^p * (1 - f)^(1-p) - 1
                 ↑              ↑
              exponent        exponent
```

When the win rate is high (75%), the first term `(1 + f*R)^p` dominates -- you win frequently and each gain compounds on top of the previous one. The second term `(1 - f)^(1-p)` has little impact because losses are infrequent.

When the win rate is low (50%), even though individual gains are larger (2:1 ratio), the intervening losses slow down compounding. Each loss reduces the base on which the next gain is calculated.

**In compounding, the frequency of gains matters more than their size.**

### Winning Streaks as the Engine of Compounding

The win rate as an exponent has a direct consequence: it determines the probability of consecutive winning streaks. And streaks are the engine of geometric growth.

| Win Rate | P(10 consecutive wins) | Effect on compounding |
|---|---|---|
| 50% | 0.1% -- almost never happens | Flat, grows slowly |
| 75% | 5.6% -- happens regularly | Strong, exponential curve |
| 83% | ~15% -- happens often | Explosive |

With a 75% WR, a streak of 10 consecutive winners occurs in 1 out of every ~18 sequences of 10 trades. Each of those streaks is a compounding "boost" where capital grows without interruption from losses. With a 50% WR, those streaks practically don't exist.

### The Trap: EG per Trade vs EG per Day

A system with 14% EG per trade that trades 0.5 times per day produces less real growth than one with 6% EG that trades 3 times per day:

```python
# Real daily growth = (1 + EG_per_trade) ^ trades_per_day - 1
daily_growth_A = (1 + 0.14) ** 0.5 - 1   # ≈ 6.8% daily
daily_growth_B = (1 + 0.06) ** 3 - 1     # ≈ 19.1% daily
# System B grows ~3x faster despite having lower EG per trade
```

**EG per trade x trade frequency = real portfolio growth.** When comparing systems, don't just look at EG -- multiply it by the trading frequency.

## Simulation: 1000 Trades

```python
import numpy as np

def simulate_system(win_rate, profit_ratio, trades=1000, simulations=10000):
    """
    Simulates account growth with Kelly sizing.
    Shows the real distribution, not just the average.
    """
    edge = win_rate * profit_ratio - (1 - win_rate)
    kelly_fraction = edge / profit_ratio
    
    # Use half-Kelly (more conservative, standard in practice)
    f = kelly_fraction * 0.5
    
    final_values = []
    for _ in range(simulations):
        capital = 1.0
        outcomes = np.random.random(trades) < win_rate
        for win in outcomes:
            if win:
                capital *= (1 + f * profit_ratio)
            else:
                capital *= (1 - f)
        final_values.append(capital)
    
    return {
        'median_growth': np.median(final_values),
        'mean_growth': np.mean(final_values),
        'pct_profitable': np.mean(np.array(final_values) > 1.0) * 100,
        'worst_5pct': np.percentile(final_values, 5),
    }

# With half-Kelly (more conservative than full Kelly):
# System 1 (50% WR, 2:1): moderate median growth
# System 2 (75% WR, 1:1): significantly higher median growth
```

The simulation confirms what the formula predicts: System 2 not only grows faster on average -- it has lower variance and a higher probability of being profitable in any window of N trades.

## Practical Implications

### 1. Don't Dismiss High Win Rate Systems with Low Ratios

Conventional wisdom says "aim for a 2:1 ratio or better." But a system with 70% accuracy and a 1:1 ratio can be superior to one with 40% accuracy and a 3:1 ratio, even if the arithmetic expectancy is similar. EG reveals this.

### 2. Watch Out for "Spectacular" Low Win Rate Systems

A trend following system with 30% accuracy and a 5:1 ratio has good expectancy (0.80). But its EG can be modest because losing streaks slow down compounding. You need to survive 7-10 consecutive losses before the big winner arrives -- and each loss reduces your capital base.

```python
# Aggressive trend following: 30% WR, 5:1 ratio
eg_trend = expected_growth(0.30, 5.0)  # Edge=0.80, EG≈7.8%

# Conservative mean reversion: 65% WR, 1.2:1 ratio
eg_mr = expected_growth(0.65, 1.2)     # Edge=0.43, EG≈6.6%

# The trend system has NEARLY DOUBLE the expectancy (0.80 vs 0.43)
# but only 18% more EG (7.8% vs 6.6%)
# Compounding "penalizes" the low win rate
```

### 3. EG as a Portfolio Selection Criterion

When you need to choose between systems for your portfolio, EG is a better criterion than Sharpe ratio or profit factor:

- **Sharpe ratio**: penalizes upside volatility (a huge gain lowers the Sharpe, which is absurd)
- **Profit factor**: doesn't distinguish between frequency and size of trades
- **EG**: captures exactly what you want to maximize -- the geometric growth of your account

## Kelly Criterion: How Much to Risk per Trade

EG depends on how much you risk per trade. Risking too little wastes the edge. Risking too much destroys it. The **Kelly Criterion** (John Kelly, 1956, Bell Labs) gives you the optimal fraction of capital to risk in order to maximize long-term geometric growth.

### The Basic Formula

```python
def kelly_fraction(win_rate, profit_ratio):
    """
    Optimal fraction of capital to risk per trade.
    Maximizes long-term geometric growth.
    """
    edge = win_rate * profit_ratio - (1 - win_rate)
    if edge <= 0:
        return 0  # no edge, don't risk anything
    return edge / profit_ratio

# System with 60% WR and 1.5:1 ratio
f = kelly_fraction(0.60, 1.5)
# Edge = 0.60*1.5 - 0.40 = 0.50
# Kelly = 0.50 / 1.5 = 0.333 -> risk 33% of capital per trade
```

**What it means**: if your system has 60% accuracy with a 1.5:1 ratio, Kelly tells you to risk 33% of your capital on each trade to grow as fast as possible.

33% sounds enormous. And it is. That's exactly the trap of full Kelly.

### Why Full Kelly Is Dangerous

The relationship between position size and volatility **is not linear -- it's exponential**. Doubling the size doesn't double the volatility; it quadruples it or more.

```python
def eg_at_fraction(win_rate, profit_ratio, fraction):
    """EG for any fraction of capital (not just Kelly optimal)."""
    p = win_rate
    R = profit_ratio
    f = fraction
    if f <= 0 or f >= 1:
        return 0
    return (1 + f * R) ** p * (1 - f) ** (1 - p) - 1

# System: 60% WR, 1.5:1 ratio, Kelly optimal = 33%
wr, ratio = 0.60, 1.5
kelly = kelly_fraction(wr, ratio)  # 0.333

fractions = [0.05, 0.10, 0.167, 0.25, 0.333, 0.50, 0.667]
for f in fractions:
    eg = eg_at_fraction(wr, ratio, f)
    label = ""
    if abs(f - kelly) < 0.01: label = " <- KELLY OPTIMAL"
    if abs(f - kelly*0.5) < 0.01: label = " <- HALF KELLY"
    if abs(f - kelly*1.5) < 0.02: label = " <- 1.5x KELLY"
    print(f"  f={f:.1%}: EG={eg*100:.2f}%{label}")

# Result:
#   f=5.0%:  EG=2.31%
#   f=10.0%: EG=4.26%
#   f=16.7%: EG=6.29%  <- HALF KELLY
#   f=25.0%: EG=7.90%
#   f=33.3%: EG=8.45%  <- KELLY OPTIMAL
#   f=50.0%: EG=6.03%  <- 1.5x KELLY (SIMILAR EG to Half Kelly!)
#   f=66.7%: EG=-2.33% <- 2x KELLY (LOSS!)
```

Look at what happens:

| Fraction | Relation to Kelly | EG | Observation |
|---|---|---|---|
| 5% | 0.15x Kelly | 2.3% | Very conservative, grows slowly |
| 16.7% | **Half Kelly** | 6.3% | **~74% of optimal EG, manageable volatility** |
| 33.3% | **Full Kelly** | 8.5% | Theoretical maximum, extreme volatility |
| 50% | 1.5x Kelly | 6.0% | **Similar EG to Half Kelly, but with massive volatility** |
| 66.7% | 2x Kelly | -2.3% | **You lose money.** Oversizing destroys the edge |

### The Three Key Insights

**1. At 1.5x Kelly you get the same return as at 0.5x Kelly, but with brutal volatility.**

This is the deadly asymmetry of sizing. Going above Kelly is far worse than staying below it. If you err on the low side, you grow slower. If you err on the high side, you can blow up the account.

**2. Volatility scales exponentially with size.**

It's not that doubling the position doubles the risk. It quadruples it. That's why a small error in estimating your parameters (win rate, ratio) can be catastrophic with full Kelly -- if your actual win rate is 55% instead of 60%, you've gone from optimal Kelly to being oversized.

**3. The expected max drawdown is approximately equal to the Kelly percentage.**

If you use half Kelly (16.7% of capital per trade), expect drawdowns of up to ~16-17%. If you use full Kelly (33%), expect drawdowns of ~33%. This is a rule of thumb, not exact, but useful for calibrating expectations.

### Half Kelly: The Industry Standard

Most practitioners use **half Kelly** (half the optimal fraction). The math justifies why:

- You get **~74% of the growth** of optimal Kelly (varies by system, but consistently between 70-80%)
- With **significantly lower volatility**
- Expected drawdown is cut in half
- You have a margin of error: if your estimates of win rate or ratio are off, you're still on the safe side of the curve

```python
def practical_kelly(win_rate, profit_ratio, fraction_of_kelly=0.5):
    """
    Kelly adjusted for real-world use.
    fraction_of_kelly=0.50 -> half Kelly (standard)
    fraction_of_kelly=0.25 -> quarter Kelly (for limited data)
    """
    edge = win_rate * profit_ratio - (1 - win_rate)
    if edge <= 0:
        return 0
    full_kelly = edge / profit_ratio
    return full_kelly * fraction_of_kelly

# Half Kelly for a 60% WR, 1.5:1 system
f = practical_kelly(0.60, 1.5, fraction_of_kelly=0.50)
# = 0.333 * 0.5 = 0.167 -> risk 16.7% per trade
```

### Quarter Kelly: For When You're Not Sure

If you have limited data (few trades in the backtest), parameters estimated with uncertainty, or a new system you haven't yet validated live, **quarter Kelly** (25% of optimal) is more prudent:

- You grow slower (~56% of optimal EG)
- But you survive much larger estimation errors
- Ideal for the first 6-12 months of a new system in production

### Kelly with Stop Loss: Adjusting for Real Risk

Basic Kelly assumes you lose 100% of what you risked on each losing trade. But if you use a stop loss, your actual loss is smaller. That allows for larger positions:

```python
def kelly_stop_adjusted(win_rate, profit_ratio, stop_loss_pct):
    """
    Kelly adjusted for stop loss.
    If your stop is 2% of price, you can have larger positions
    than if you were risking 100%.
    
    Parameters:
    - stop_loss_pct: max loss per trade as a fraction (0.02 = 2%)
    """
    edge = win_rate * profit_ratio - (1 - win_rate)
    if edge <= 0:
        return 0
    
    kelly_base = edge / profit_ratio
    kelly_adjusted = kelly_base / stop_loss_pct
    return kelly_adjusted

# System 60% WR, 1.5:1 ratio, 2% stop
position = kelly_stop_adjusted(0.60, 1.5, stop_loss_pct=0.02)
# Kelly base = 33.3%
# Adjusted = 33.3% / 2% = 16.67x capital
# That is: with 2% stops, you can lever up to ~16x

# In practice, with half Kelly adjusted:
position_half = position * 0.5  # ~8x
```

**The logic**: a tighter stop limits the loss per trade, allowing larger positions for the same dollar amount of risk. This keeps you more frequently exposed when the trade moves in your favor.

**Caution**: this assumes the stop always executes at the exact price. In reality, there's slippage, overnight gaps, and markets that skip right past your stop. Never size assuming perfect stop execution.

### Scaling by Account Size

Theoretical Kelly doesn't consider market constraints. In practice, account size limits how much Kelly you can use:

| Account Size | Practical Kelly | Why |
|---|---|---|
| < $25K | Up to 50% of Kelly | Little diversification, each trade weighs heavily |
| $25K - $100K | 33-50% | You start to have room to diversify |
| $100K - $200K | 25-33% | Slippage starts to matter in small caps |
| $200K - $500K | 12.5-25% | Fill probability drops, you move price on entry |
| $500K+ | Fixed stake or < 12.5% | In small caps, your order IS the market |

The reason: slippage, fill probability, and liquidity constraints scale with position size. A $500K position in a $2 small cap will move the price significantly on entry and exit. Theoretical Kelly doesn't know this.

### Common Mistake: Adjusting Kelly by "Setup Quality"

"I use 50% Kelly on A+ setups and 25% on B setups." This is incorrect.

Kelly **already incorporates setup quality** through the win rate and profit ratio. An A+ setup naturally has a better WR and/or better ratio, which produces a higher Kelly fraction. A B setup has worse metrics, which produces a lower Kelly.

If you manually adjust on top of that, you're overriding the math with your opinion. The only valid reason to reduce Kelly is uncertainty in the parameters (limited data, new system) -- and that's what half Kelly and quarter Kelly are for.

### Alternative: Ralph Vince's Optimal-f

Kelly assumes a binary distribution (you win R or lose 1). **Optimal-f** by Ralph Vince uses the complete distribution of historical returns to find the optimal fraction. It's conceptually superior because it doesn't simplify the distribution, but it's computationally more expensive and requires enough historical trades for the empirical distribution to be representative.

In practice, Kelly with half/quarter adjustment is sufficient for most systems. Optimal-f is relevant if you trade with highly asymmetric distributions (shorting small caps, for example).

## How to Incorporate EG and Kelly into Your Process

1. **Calculate EG for all your systems** and compare against their arithmetic expectancy. You'll find surprises -- systems that looked equivalent by expectancy are not equivalent by EG

2. **Use EG as the objective function in optimization** instead of net profit or Sharpe. The optimizer will seek parameters that maximize real geometric growth

3. **Compare systems by EG before building a portfolio**. A portfolio of systems with individually high EG, and low correlation between them, is the most powerful combination

4. **Remember that EG assumes Kelly sizing**. If you use fixed position sizing (always the same amount), arithmetic expectancy is sufficient. EG matters when you compound -- and if you're not compounding, you're leaving growth on the table

## Anti-Scalping: The Math of Not Cutting Winners

Most traders (and many algos) cut profits too early. Instinct says "lock in the gain." The math says the opposite.

### Halfway Probability: Conditional Probabilities in Action

If your system has an 80% win rate and a trade is already at +5%, what's the probability it reaches +10%?

Intuition says "I've already made 5%, better close it." But the math says the opposite. Every tick in your favor is **Bayesian evidence** that the trade thesis is correct. The conditional probability (given that you're already in profit) of reaching the target **increases** as the trade progresses:

Example with empirical data from a system with ~80% WR in small caps:

| Your Current Gain | P(doubling to the next level) | Implication |
|---|---|---|
| +5% | ~94% of reaching +10% | Covering is throwing money away |
| +7.5% | ~80% of reaching +15% | Still extremely likely |
| +10% | ~70% of reaching +20% | Hold |
| +15% | Probability stabilizes | Home runs -- HOLD |

*These values are specific to a particular system. Calculate your own using the conditional simulation function below.*

This is **Bayesian updating** applied to trading: each move in your favor updates your estimate of the probability of success upward.

These values come from simulations with real data from systems with ~80% WR. There's no simple closed-form formula that reproduces them -- they depend on the specific return distribution of the system. The way to calculate them for your system is with **conditional simulation**:

```python
import numpy as np

def estimate_conditional_probability(trade_returns, current_pct, target_pct, n_sims=50000):
    """
    Estimates the probability of reaching target_pct given that
    you're already at current_pct, using the real trade distribution.
    
    Simulates trajectories starting at current_pct and counts
    how many reach target_pct before returning to 0%.
    """
    reached_target = 0
    for _ in range(n_sims):
        pnl = current_pct
        for _ in range(50):  # max 50 trades to get there
            trade = np.random.choice(trade_returns) * 100  # to percentage
            pnl += trade
            if pnl >= target_pct:
                reached_target += 1
                break
            if pnl <= 0:
                break
    return reached_target / n_sims

# Usage: estimate_conditional_probability(my_trades, 5.0, 10.0)
# With your real data, you'll get the specific probabilities
# for your system -- not a generic approximation.
```

### The Bias of Cutting Winners

Think about 100 trades from your system. Some will be big losers (max loss). Others will be big winners (home runs). The system's natural distribution produces both.

If you absorb the full max losses (because the stop executes and you can't avoid them) but cut the home runs prematurely (because you "lock in profits"), you're doing something very specific: **skewing the distribution of outcomes against yourself**.

```
Natural distribution of the system:
[big loss] [small loss] [small gain] [big gain]
    <- you absorb these fully ->    <- you cut these short ->

Result: your real system has worse metrics than the backtest
because you eliminated the positive tails but kept the negative ones.
```

### Cumulative Probability

The probability of not having a single home run in N trades is `(1 - P_homerun)^N`. This drops exponentially:

```python
def prob_at_least_one_homerun(p_homerun_per_trade, n_trades):
    """P(at least 1 home run in N trades)"""
    return 1 - (1 - p_homerun_per_trade) ** n_trades

# If each trade has a 20% probability of being a home run:
for n in [5, 10, 20, 50]:
    p = prob_at_least_one_homerun(0.20, n)
    print(f"  {n} trades: {p:.0%} probability of at least 1 home run")

# 5 trades: 67%
# 10 trades: 89%
# 20 trades: 99%
# 50 trades: 99.99%
```

In 10 trades, you have ~90% probability of at least one home run. But if you cut every trade at +5% instead of letting them run to +10%, that home run never materializes in your account.

### Implications for Trailing Stops and Targets

1. **Very tight trailing stops kill home runs.** If your trailing protects the +3% but the system regularly produces +15% trades, the trailing takes you out before the positive tail materializes

2. **Fixed targets limit upside.** A TP at 2:1 when the system naturally produces 5:1 trades is giving away the difference

3. **The solution is not "don't use stops/targets"** -- it's calibrating them with the real distribution of your system. If the backtest shows that 15% of your trades produce gains > 3R, your trailing or target shouldn't cut at 2R

4. **Evaluate the opportunity cost**: how much EG are you sacrificing for the "peace of mind" of locking in gains early? Calculate it using the EG formula with the real profit ratio vs the truncated profit ratio

### Scalping Produces Negative Expectancy

The hardest data point against scalping: in simulations with real data, systematically covering at +5% produces **negative Sim-EG**. You literally lose money in the long run by scalping a system that's profitable if you let it run.

| Exit Strategy | Result |
|---|---|
| Cover everything at +5% | Sim-EG **negative** -- you lose money |
| Cover everything at +10% | ~2% profit, Sim-EG ~0.4% -- marginal |
| Set and forget (hold to close) | Best possible result |

The empirical conclusion is consistent: **no partial covering strategy was found that beats the full-day hold**. The cost of locking in gains early exceeds the benefit of avoiding pullbacks.

This doesn't mean you should never close a trade before the target. It means that if your system has edge, the default decision should be to hold, and the burden of proof is on demonstrating that closing early improves Sim-EG -- not the other way around.

## Sim-EG: Monte Carlo as a Metric

The closed-form EG formula assumes a binary distribution (you win R or lose 1). Your real system has a continuous distribution of outcomes -- trades that win a little, trades that win a lot, trades that lose different amounts each time. To capture this, we use **Monte Carlo simulation not as validation, but as the metric itself**.

### The Process

```python
import numpy as np

def sim_eg(trade_returns, n_simulated_trades=10000, n_runs=3):
    """
    Simulated Expected Growth: estimates the real EG of the system
    using the empirical trade distribution (not the theoretical one).
    
    More robust than the closed-form formula when:
    - The distribution is not binary (most cases)
    - There is skew in the returns
    - There are fat tails
    
    Parameters:
    - trade_returns: array of per-trade returns from the backtest
      (e.g., [0.02, -0.01, 0.05, -0.008, ...])
    - n_simulated_trades: trades to simulate per run (10K is standard)
    - n_runs: runs to average (3 is sufficient with 10K trades)
    """
    eg_estimates = []

    for _ in range(n_runs):
        # Resample with replacement from the real distribution
        sampled = np.random.choice(trade_returns, size=n_simulated_trades, replace=True)

        # Calculate geometric growth
        # Each trade multiplies capital by (1 + return)
        growth_factors = 1 + sampled
        final_value = np.prod(growth_factors)

        # EG = growth per trade = Nth root of final value - 1
        eg = final_value ** (1 / n_simulated_trades) - 1
        eg_estimates.append(eg)

    return {
        'sim_eg': np.mean(eg_estimates),
        'eg_std': np.std(eg_estimates),
        'eg_runs': eg_estimates,
    }

# Example usage:
# trades = np.array([results from your backtest])
# result = sim_eg(trades)
# print(f"Sim-EG: {result['sim_eg']*100:.2f}% per trade")
```

### Why 10,000 Trades

With 1,000-2,000 simulated trades, results vary considerably between runs (e.g., 3.8%, 5.1%, 4.2%). With 10,000, they converge (e.g., 5.0%, 5.2%, 5.07%). Three runs of 10K give 30K effective trades, enough for the estimator to be stable.

### Why Sim-EG > Closed-Form Formula

The closed-form EG formula assumes each trade wins exactly R or loses exactly 1. In reality:

- A trade can win 0.5R, 1R, 2R, or 5R
- A trade can lose 0.3R, 0.7R, or 1R (if it has a stop)
- The distribution can have positive skew (longer right tail)
- There can be fat tails that the binary formula doesn't capture

Sim-EG uses the **real empirical distribution** of your trades. It's essentially **bootstrap resampling** applied to compound growth. It makes no assumptions about the shape of the distribution -- it directly uses what your system produced.

### Sim-EG as a Quality Gate

Use Sim-EG as a minimum quality filter: **if Sim-EG < 2% over 10K trades, the edge is too fragile to trade.** Real-world costs (slippage, commissions, execution errors) will consume such a thin edge.

### Worst-Case Simulations

The real power of Sim-EG is exploring the extremes. With thousands of bootstrap runs, you can find the worst possible scenario with your trade distribution:

```python
def sim_eg_worst_case(trade_returns, n_trades=100, n_simulations=10000):
    """
    What is the worst plausible scenario for N trades?
    Searches through thousands of simulations for the worst trajectory.
    """
    final_values = []
    for _ in range(n_simulations):
        sampled = np.random.choice(trade_returns, size=n_trades, replace=True)
        capital = np.prod(1 + sampled)
        final_values.append(capital)

    return {
        'median': np.median(final_values),
        'worst_5pct': np.percentile(final_values, 5),
        'worst_found': min(final_values),
        'best_found': max(final_values),
        'pct_profitable': np.mean(np.array(final_values) > 1.0) * 100,
    }

# If in 10,000 simulations of 100 trades, the WORST case
# still doubles capital, you have real conviction.
# If the worst case loses money, the edge is fragile.
```

This provides a level of confidence no other metric offers: "even in the worst bootstrap scenario, do I survive?"

### Block Bootstrap: Preserving Streaks

Standard bootstrap (i.i.d.) assumes each trade is independent of the previous one. But in reality there can be **temporal autocorrelation** -- winning or losing streaks that depend on market regime.

**Block bootstrap** solves this: instead of resampling individual trades, it resamples blocks of consecutive trades (e.g., blocks of 5-10 trades). This preserves the temporal structure.

```python
def sim_eg_block_bootstrap(trade_returns, block_size=5, n_trades=10000, n_runs=3):
    """
    Block bootstrap: resamples consecutive blocks of trades
    to preserve temporal autocorrelation.
    
    If there is regime dependency (the market alternates good/bad phases),
    i.i.d. bootstrap hides it. Block bootstrap preserves it.
    """
    n = len(trade_returns)
    eg_estimates = []

    for _ in range(n_runs):
        sampled = []
        while len(sampled) < n_trades:
            start = np.random.randint(0, n - block_size)
            sampled.extend(trade_returns[start:start + block_size])
        sampled = np.array(sampled[:n_trades])

        final_value = np.prod(1 + sampled)
        eg = final_value ** (1 / n_trades) - 1
        eg_estimates.append(eg)

    return np.mean(eg_estimates)

# If Sim-EG i.i.d. ≈ Sim-EG block -> no significant autocorrelation
# If Sim-EG block << Sim-EG i.i.d. -> regime dependency was inflating the result
```

If the difference between standard Sim-EG and block bootstrap is large, your system likely depends on a specific market regime and will suffer when the regime changes.

### When to Use Each One

| Situation | Use |
|---|---|
| Compare ideas quickly | Closed-form EG formula |
| Evaluate a system with a complete backtest | Sim-EG |
| Few trades (< 100) | Closed-form (Sim-EG doesn't have enough data to resample) |
| Distribution with fat tails or skew | Sim-EG (captures the real shape) |
| Optimization (thousands of evaluations) | Closed-form (faster) |

### Practical Integration

Add Sim-EG as another column in your system evaluation, alongside Sharpe, profit factor, and max drawdown:

```python
def full_system_evaluation(trade_returns):
    """Complete system evaluation."""
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    wr = len(wins) / len(trade_returns)
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    profit_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    edge = wr * profit_ratio - (1 - wr)
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('inf')

    # Sim-EG captures what the other metrics cannot
    seg = sim_eg(trade_returns)

    return {
        'trades': len(trade_returns),
        'win_rate': f"{wr:.1%}",
        'profit_ratio': f"{profit_ratio:.2f}",
        'expectancy': f"{edge:.3f}",
        'profit_factor': f"{pf:.2f}",
        'sim_eg': f"{seg['sim_eg']*100:.2f}% per trade",
    }
```

## Limitations of EG and Sim-EG

- **Closed-form EG**: assumes binary distribution, perfect knowledge of parameters, independence between trades. Useful for quick comparisons, not for final decisions
- **Sim-EG**: more robust but needs enough historical trades (minimum ~200 for the resampling to be representative). Doesn't capture market regime changes
- **Both**: don't capture temporal correlation between trades (streaks), nor the impact of variable costs, nor events that aren't in the historical data (black swans)

Even with these limitations, EG and Sim-EG are more informative metrics than arithmetic expectancy, Sharpe ratio, or profit factor for any system that operates with position sizing proportional to capital. The reason is simple: they are the only ones that measure what you actually want to maximize -- the geometric growth of your account.
