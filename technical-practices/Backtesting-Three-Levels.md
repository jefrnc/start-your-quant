> 🇪🇸 [Leer en Español](Backtesting-Three-Levels.es.md) | 🇺🇸 **English**

# Backtesting: From Classic to Walk-Forward

There are three ways to simulate a system on historical data. Each has its time and purpose. They are not mutually exclusive -- they are progressive.

## Level 1: Classic Backtest

Evaluate the strategy on all available historical data, without separating datasets.

```
[========== All historical data ==========]
         Optimize and evaluate here
```

**What it does**: shows how the strategy would have performed in the past.

**What it's for**: preliminary evaluation. Discard ideas that don't show a minimum of viability before investing time in deeper analysis.

**Limitation**: it's extremely easy to overfit. If you optimize on the entire history and pick the best parameters, you're almost certainly fitting the system to past data that won't repeat.

**When to use it**: research and preliminary evaluation phase. If it doesn't work here, there's no point continuing.

## Level 2: Forward Testing (Out-of-Sample)

Optimize on one period and evaluate on another that the optimizer never saw.

```
[==== In-Sample (optimization) ====][==== Out-of-Sample (test) ====]
         2000 - 2015                      2015 - 2024
```

**What it does**: simulates what would have happened if you had developed the system in 2015 and let it run untouched.

**What it's for**: more realistic validation. If the out-of-sample results are consistent with the in-sample (not identical -- consistent), the signal is probably real.

**Limitation**: a single forward test guarantees nothing. And it's easy to cheat without clear protocols (e.g., adjusting the cutoff point between IS and OOS until it works).

**When to use it**: after the preliminary evaluation shows potential.

## Level 3: Walk-Forward Testing

Multiple cycles of optimization + testing, advancing through the history.

```
[IS-1][OOS-1]
      [IS-2][OOS-2]
            [IS-3][OOS-3]
                  [IS-4][OOS-4]
                        [IS-5][OOS-5]
```

Each IS (In-Sample) block is optimized. The winning parameters are tested on the following OOS (Out-of-Sample) block. Then the window advances and it repeats.

**What it does**: generates a complete curve of out-of-sample results across nearly all the historical data. It's the closest test to simulating what would have actually happened if you had been re-optimizing periodically.

**What it's for**: it's the test that best allows you to **objectively measure** robustness. If the system passes a well-executed walk-forward, the probability that it works in live trading is significantly higher.

**Advantages**:
- Produces many out-of-sample data points, not just one
- Objectively measures robustness
- Allows you to choose the parameters for live trading (from the last cycle)
- Dramatically reduces the risk of over-optimization

**Limitations**:
- Long, slow, and computationally intensive process
- Few systems pass it -- it's a very tough test
- Even so, it's not infallible. With bad practice, you can still cheat

**When to use it**: when the system has already passed preliminary evaluation and basic forward testing.

## Summary: When to Use Each One

| Phase | Method | Objective |
|---|---|---|
| Research / initial idea | Classic backtest | Does it show a minimum of viability? |
| Initial validation | Forward testing | Does it work on unseen data? |
| Serious validation | Walk-forward | Is it truly robust? |
| Live trading | Paper trading -> reduced live | Does it execute as the backtest predicted? |

There's no point running a walk-forward on a system that can't even pass a classic backtest. Use each level as a progressive filter.

## The Golden Rule

A backtest tells you how it would have gone in the past. **It doesn't tell you how it will go in the future.** It's a necessary condition for trading a system, but never sufficient on its own. Walk-forward is the closest thing to a guarantee we can get -- and even so, it's not a guarantee.
