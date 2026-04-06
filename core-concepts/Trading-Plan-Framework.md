> 🇪🇸 [Leer en Español](Trading-Plan-Framework.es.md) | 🇺🇸 **English**

# The Trading Plan: A Professional Framework

If you don't have a written trading plan, you don't have a business -- you have an expensive hobby. The trading plan is the document that defines who you are as a trader, how you operate, and what you do when things go wrong. You write it when you're calm, and you follow it when you're not.

## Why a Written Plan

When you're not trading, you think clearly. You define rational rules, evaluate risk objectively, and make data-driven decisions.

When you're in the middle of a 12% drawdown, your mind changes. Rules that seemed obvious now seem questionable. The stop loss you defined feels too tight. The system you validated with 3 years of data "maybe doesn't work for this market."

**The written plan is your anchor.** It's not modified on the fly. It's reviewed periodically, with a cool head, using data -- never during a losing streak.

Real fact: traders who keep a decision journal discover, when re-reading it weeks later, that their perspective changed significantly without them being aware of it. Paper doesn't lie.

## The 6 Sections of the Plan

### 1. Trading Philosophy

This is your mission and vision as a trader. It's not philosophical in the abstract sense -- it's concrete:

**What to define:**
- **Dedication**: Full-time or part-time? If part-time, how many realistic weekly hours?
- **Available capital**: How much capital do you have to trade? This conditions everything -- instruments, strategies, expectations
- **Source of capital**: Your own or third-party? Trading with other people's money completely changes the psychological pressure
- **Required vs desired financial goals**: "I need $2,000/month to live" is very different from "I want to earn $10,000/month"
- **Markets**: US stocks? Futures? Forex? The choice depends on your capital, schedule, and profile

**The necessity trap**: if you depend financially on your trading from day 1, the emotional pressure compromises your decisions. The recommendation, whenever possible, is to start as a supplementary activity until your track record and capital justify the transition.

```yaml
# Example of documented philosophy
philosophy:
  dedication: part_time  # 2h/day, mornings before work
  initial_capital: 15000
  source: own
  required_goal: 0  # I don't depend on this for a living
  desired_goal: 500_monthly  # first year
  markets: [us_smallcap_stocks, index_futures]
  horizon: long_term  # minimum 3 years to evaluate
  preferred_timeframes: [daily, weekly]
  # If my dedication is part-time, I do NOT trade aggressive intraday
  # because I can't monitor constantly
```

### 2. Psychology

The mind is the most fragile component of trading, even algorithmic trading. Your bot executes, but you decide when to activate it, when to stop it, and how to react when things don't go as the backtest promised.

**What to include:**
- **Self-assessment**: your risk profile, your known biases (see [Cognitive Biases](./Cognitive-Biases-Algo-Trading.md))
- **Emotional protocol**: what do you do when you're at maximum drawdown -- do you review the data or start impulsively changing parameters?
- **Maintenance**: exercise, rest, disconnection. 20 years of daily trading requires sustained care of body and mind
- **Support network**: do you have someone to talk to objectively about trading? A mentor, a group, a professional?

**What doesn't seem important until it is**: a 10% drawdown in a backtest is a number. A 10% drawdown in your real account, when you need that money, is a completely different experience. The psychological plan is written for the second case, not the first.

### 3. Rules and Systems

The operational heart of the plan. This is where your trading systems, their specifications, and the portfolio live.

**For each system, document:**

```yaml
system:
  name: "Gap_SmallCap_Long_v2"
  market: us_stocks
  universe: smallcap_0.50_10.99
  timeframe: premarket_5min
  type: momentum  # trend-following

  entry_rules:
    - gap_up > 10%
    - premarket_volume > 500000
    - price > vwap
  exit_rules:
    - stop_loss: -3%
    - take_profit: +8%
    - max_time: 2h

  backtest_metrics:
    period: "2021-01-01 to 2024-12-31"
    trades: 847
    win_rate: 0.42
    profit_factor: 1.85
    max_drawdown: -12.3%
    sharpe: 1.45
    max_consecutive_losses: 11

  risk_management:
    risk_per_trade: 1%  # of capital
    max_simultaneous_positions: 3
    max_exposure: 30%  # of total capital
```

**The portfolio matters more than any individual system.** Document how your systems combine:
- What correlation do they have with each other?
- Do they operate in the same markets/hours?
- How is capital allocated between them?

This is the most dynamic section of the plan -- systems change, get added, and get retired.

### 4. Launch (Infrastructure)

Algorithmic trading requires infrastructure. You don't need a datacenter, but you do need a reliable setup.

**What to define:**

| Component | Key decisions |
|---|---|
| **Hardware** | Local or cloud? For heavy optimization, temporary cloud servers are more efficient than buying hardware |
| **Software** | Backtesting platform, language (Python, etc.), broker API |
| **Data** | Provider, quality, cost. Do you need tick data or are 1-minute candles enough? |
| **Connectivity** | Primary internet + backup. An outage during an open position is a real risk |
| **Recurring costs** | Data, hosting, broker, tools. Calculate the monthly break-even |

```yaml
# Example cost analysis
monthly_costs:
  polygon_data: 29     # USD
  cloud_vps: 40        # for 24/7 execution
  broker_data: 0       # included with IBKR
  tools: 0             # Python + open source
  total: 69
  # Break-even: I need to generate > $69/month just to cover costs
  # With $15,000 capital that's 0.46% monthly
```

### 5. Monitoring and Recycling

Every system has a lifespan. The market changes regimes, volatility evolves, new participants alter the microstructure. Your job doesn't end when you launch a system -- it begins.

**Clip Point (Kill Switch)**

Define, BEFORE launching, under what conditions you pause or retire a system. In the industry this is known as the "clip point" or "kill switch" -- the threshold where you decide the system has stopped working:

```python
def evaluate_system_health(rolling_metrics):
    """
    Monitoring protocol: evaluate monthly.
    Define BEFORE launching, not during a drawdown.
    """
    checks = {
        'drawdown_breach': rolling_metrics['current_dd'] > rolling_metrics['max_expected_dd'] * 1.5,
        'sharpe_degraded': rolling_metrics['sharpe_90d'] < 0.3,
        'win_rate_collapsed': rolling_metrics['win_rate_60d'] < rolling_metrics['expected_wr'] * 0.6,
        'consecutive_losses': rolling_metrics['current_streak'] > rolling_metrics['max_expected_streak'],
    }

    triggered = [k for k, v in checks.items() if v]

    if len(triggered) >= 2:
        return "PAUSE -- multiple degradation signals"
    elif len(triggered) == 1:
        return f"MONITOR -- alert signal: {triggered[0]}"
    return "NORMAL"
```

**Continuous education**: the market evolves and so must you. New techniques, new data, new regulations. Dedicate regular time to learning, not just trading.

**System recycling**: when a system is retired, document why. That information is valuable for designing future systems.

### 6. Crisis Plan

What do you do when things go seriously wrong?

**Scenarios to plan for:**

| Scenario | Protocol |
|---|---|
| **Internet outage** | Do you have stops on the broker's server (not just local)? Can you close from your phone? |
| **Server crash** | Is there redundancy? How long can you be without execution? |
| **Flash crash** | Are your stops on the market or simulated? Simulated ones don't execute if your software goes down |
| **Broker unresponsive** | Do you have an account at a second broker? Can you hedge the position? |
| **Maximum drawdown** | At what percentage do you stop everything? Who decides -- you or the code? |
| **Power outage** | A 30-minute UPS costs little and saves a lot |
| **Code error** | How do you detect a bug in production? Are there automatic alerts for anomalous trades? |

## Business Mindset

Algorithmic trading is a business with low barriers to entry -- which is good for access but means a lot of competition. Having a good system (product) is a necessary but not sufficient condition.

What distinguishes a professional trader from one who's just playing:

- **Written protocols** for each phase (design, testing, launch, monitoring, retirement)
- **Realistic cost analysis** (not just commissions -- also data, hosting, time)
- **Business metrics**, not just trading metrics: how much time do you invest per dollar earned? Is it scalable?
- **Contingency plan** for adverse scenarios
- **Periodic review** of the complete plan, not just the systems

Whether you dedicate 2 hours per week or 12 hours per day, treat your trading with the seriousness of a business. The markets don't distinguish between professionals and amateurs -- but long-term results do.

## When to Review the Plan

| Frequency | What to review |
|---|---|
| **Weekly** | Active system metrics, monitoring protocol alerts |
| **Monthly** | Complete portfolio performance, correlations between systems |
| **Quarterly** | Philosophy, costs, are the plan's assumptions still valid? |
| **Life changes** | New job, capital change, family changes -> review sections 1 and 2 |
| **Market crisis** | Don't change the plan during the crisis. Review it AFTER, with a cool head, using data |
