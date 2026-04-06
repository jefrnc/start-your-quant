> 🇪🇸 [Leer en Español](Cognitive-Biases-Algo-Trading.es.md) | 🇺🇸 **English**

# Cognitive Biases in Algorithmic Trading

"If I automate, emotions won't affect me." Wrong. Your bot executes, but you decide when to activate it, when to stop it, what data to use for designing it, and how to react when it loses 12 times in a row. Psychology remains the most fragile component of algorithmic trading.

## Alexander Elder's Three M's

Alexander Elder, in *Trading for a Living* (1993), broke trading down into three pillars:

- **Mind**: the trader's psychology
- **Money**: position management and capital protection
- **Method**: the rules and strategies

In algorithmic trading, we tend to think that only Method (the code) and Money (position sizing) matter. But Mind remains the most important pillar because:

- You decide whether to stop a system in drawdown or let it run
- You choose what data to use and how to validate
- You interpret backtest results
- You can fall into the temptation of manually intervening when you "know" what the market is going to do

Without a solid Mind, the best Method sabotages itself.

## Your Risk Profile: The Test that Reveals Your Bias

Before designing any system, you need to understand how your mind processes gains and losses. These two scenarios, based on Kahneman and Tversky's prospect theory, reveal it.

### Scenario 1: Losses

Choose ONE option:
- **A**: Certain loss of $9,000
- **B**: 95% probability of losing $10,000 + 5% probability of losing nothing

### Scenario 2: Gains

Choose ONE option:
- **A**: Certain gain of $9,000
- **B**: 95% probability of gaining $10,000 + 5% probability of gaining nothing

### The Mathematically Correct Answers

```python
def expected_value(outcomes):
    """Expected value: sum of (probability x value) for each outcome."""
    return sum(prob * value for prob, value in outcomes)

# Scenario 1
ev_loss_sure = expected_value([(1.0, -9000)])           # = -9,000
ev_loss_game = expected_value([(0.95, -10000), (0.05, 0)])  # = -9,500
print(f"Scenario 1: Certain loss EV={ev_loss_sure}, Gamble EV={ev_loss_game}")
# Better option: certain loss (-9,000 > -9,500)

# Scenario 2
ev_gain_sure = expected_value([(1.0, 9000)])             # = 9,000
ev_gain_game = expected_value([(0.95, 10000), (0.05, 0)])  # = 9,500
print(f"Scenario 2: Certain gain EV={ev_gain_sure}, Gamble EV={ev_gain_game}")
# Better option: gamble (9,500 > 9,000)
```

**What most people choose**: in Scenario 1, gamble (incorrect). In Scenario 2, the certain gain (incorrect).

**What this means for your trading**:
- **Scenario 1**: we struggle to accept certain losses -- in the market, this translates to not cutting losses, moving the stop, or "waiting for it to come back"
- **Scenario 2**: we struggle to let winners run -- we close winning positions prematurely out of fear of losing what we've gained

If you chose correctly on both, you have a real psychological edge. If not, it's not a problem -- it's the normal reaction of 70-80% of people. But now you know, and you can design your systems with rules that protect you from yourself.

## Biases by Development Phase

Biases don't all appear at once. Different phases of your work as a quant are vulnerable to different biases.

### Phase 1: Idea Generation

| Bias | What it is | How it affects you in algo trading |
|---|---|---|
| **Optimism/Pessimism** | Predisposition to see everything as positive or negative, without data basis | You discard viable ideas out of pessimism or fall in love with bad ideas out of optimism |
| **Overconfidence** | Overestimating your predictions and abilities | You take more risk than necessary, underestimate possible drawdowns |
| **Loss aversion** | Giving more weight to losses than to equivalent gains | You don't cut losses, you close gains prematurely |

### Phase 2: Research and Analysis

| Bias | What it is | How it affects you in algo trading |
|---|---|---|
| **Illusion of control** | Believing your decisions influence outcomes more than they actually do | You over-optimize believing you can "control" the market |
| **Confirmation** | Seeking only information that confirms what you already believe | You only test favorable conditions, ignoring contradictory evidence |
| **Guru effect** (Pygmalion) | Giving excessive authority to a person or source | You copy systems from an "expert" without validating them yourself |
| **Availability** | Giving more importance to what you easily remember | You only trade well-known assets (AAPL, TSLA) without evaluating whether they're the best |
| **Anchoring** | Fixating on a specific number or idea and deciding around it | You believe a round number ($100, $50) has special significance without evidence |
| **Herding** | Following the majority opinion | You choose "trendy" strategies instead of those supported by data |

**Confirmation bias is the most dangerous in this phase.** It's extremely easy to backtest only the conditions that favor your hypothesis and ignore those that contradict it. Protocol, protocol, protocol.

### Phase 3: Biases Specific to Algorithmic Trading

These biases are inherent to working with data and backtesting. They don't exist (or are less relevant) in discretionary trading. For the protocol that prevents these biases, see [Scientific Method in System Development](../technical-practices/Scientific-Method-System-Development.md).

**Selection Bias**

Choosing data subsets arbitrarily. "I'll only test from 2020 to 2023 because the market was bullish then." Training, validation, and out-of-sample data must be selected with protocol, not convenience.

**Look-Ahead Bias**

Using information that wouldn't be available in real time. It's the most technical and the most treacherous.

```python
# INCORRECT: look-ahead bias
# The day's RSI is calculated with the day's close,
# but your buy signal is generated DURING the day
data['rsi'] = calculate_rsi(data['close'], 14)
data['signal'] = data['rsi'] < 30  # you use today's RSI to buy that same day

# CORRECT: use data available at the time of the decision
data['rsi'] = calculate_rsi(data['close'], 14)
data['signal'] = data['rsi'].shift(1) < 30  # signal based on the previous day's RSI
```

Some backtesting languages prevent this by design. Python/pandas does not -- you're responsible for avoiding it yourself.

**Data Snooping (Torturing the Data)**

As Ronald Coase said: "If you torture the data long enough, it will eventually confess to whatever you want." Also known as **p-hacking** or **data mining bias** -- exhaustively searching for patterns until you find something that fits, without real statistical significance.

If you test 1,000 parameter combinations, by pure chance ~50 will appear profitable. That's not a system -- it's noise. The solution: rigorous out-of-sample validation and walk-forward analysis.

**Survivorship Bias**

Backtesting with stocks that exist TODAY, ignoring those that went bankrupt or were delisted. Today's S&P 500 is not the same as 10 years ago -- the companies that went bankrupt were replaced. If your historical backtest only uses survivors, the results are inflated.

```python
# If you backtest a rotational strategy on S&P 500 stocks
# from 2010, you need the index composition AT EACH POINT IN TIME,
# including those that were removed (Lehman, Enron, etc.)
#
# Sources with survivorship bias-free data:
# - Sharadar (Nasdaq Data Link)
# - CRSP
# - Norgate Data
```

### Phase 4: Evaluation and Optimization

| Bias | What it is | How it affects you |
|---|---|---|
| **Insufficient validation** | Samples too small to be significant | 30 trades prove nothing. You need hundreds for valid conclusions |
| **Normality bias** | Assuming returns follow a normal distribution | You underestimate black swans. Real tails are much fatter |

The normality bias deserves special attention. If you design your risk management assuming a normal distribution, you're underestimating the frequency of extreme events by a factor of 10x or more.

### Phase 5: Live Trading

| Bias | What it is | How it affects you |
|---|---|---|
| **Gambler's fallacy** | Believing independent events are correlated | "It's had 8 losses in a row, the next one HAS to be a winner" -- no, it doesn't |
| **Status quo** | Resistance to change | Not updating systems that no longer work because "they've always done this" |
| **Sunk cost** | Keeping something only because you've already invested a lot in it | Continuing to run a broken system because it took you 6 months to develop |
| **Endowment** | Overvaluing what you already have | Believing your systems are better than new alternatives simply because they're yours |

```python
def is_gambler_fallacy(consecutive_losses, expected_loss_streaks):
    """
    After N consecutive losses, the probability of the
    next trade remains the same. Trades are
    (generally) independent events.
    """
    # A system with 40% win rate can have streaks of:
    # 5 losses: ~7.8% probability in any sequence of 5
    # 10 losses: ~0.6% - rare but expected in 1000+ trades
    # 15 losses: ~0.05% - very rare but possible

    from math import pow
    loss_rate = 0.60  # 40% win rate = 60% loss rate
    prob_streak = pow(loss_rate, consecutive_losses)

    return {
        'prob_this_streak': f"{prob_streak*100:.2f}%",
        'message': "The probability of the next trade does NOT change because of previous ones."
    }
```

## Anti-Bias Protocol

You can't eliminate biases -- they're part of how the human brain works. But you can build protocols that neutralize them:

1. **Write your rules BEFORE seeing the results.** Define what you're going to test, with what data, and what success criteria you'll use. If you define it after seeing the backtest, you're biased.

2. **Always use out-of-sample.** Reserve a data period that you NEVER use for optimization. It's your reality check.

3. **Record your decisions.** An algorithmic trading journal doesn't record trades (the system log does that). It records decisions: "Today I paused system X because..." Re-read those entries a month later -- you'll be surprised at how much your perspective changed.

4. **Define the clip point before launching.** How much drawdown or how many consecutive losses will make you stop the system? Define it when you're NOT in drawdown. Write it down. Don't change it during live trading.

5. **Ask for a second opinion on your data**, not on your market opinion. Show your backtest methodology to someone and ask: "Do you see any bias in how I'm testing this?"

6. **Accept that every system has a lifespan.** There's no such thing as an eternal system. Design from the start a monitoring protocol with rolling metrics that tell you when the edge is degrading, before it becomes obvious.
