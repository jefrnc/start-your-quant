> 🇪🇸 [Leer en Español](Scientific-Method-System-Development.es.md) | 🇺🇸 **English**

# The Scientific Method Applied to System Development

Developing a trading system is not "trying things until something works." It's a scientific process with protocol, variable isolation, and statistical evaluation. If you don't follow a structured method, you're doing data snooping disguised as research.

## The Three Pillars of the Scientific Method in Trading

### 1. Object of Study

Your trading system -- or one of its parts. Each component must be objectively observable, without relying on your subjective judgment.

This means that everything you evaluate must be expressible in numbers: win rate, profit factor, drawdown, Sharpe ratio. "It seems to work well" is not an evaluation -- it's an opinion.

### 2. Standardized Procedure

An orderly method for evaluating the system's behavior. The procedure must be **reproducible**: if you do it or someone else does it with the same data and the same protocol, the results should be compatible.

If your evaluation process gives different results depending on who runs it, there's subjectivity infiltrating some phase.

### 3. Statistical Evaluation

Results are evaluated with statistical tools, not intuition. A streak of 15 winning trades doesn't prove a system works, and a streak of 10 losses doesn't prove it's broken. What matters is statistical significance over a sufficient sample.

## The Protocol: The Key Word

The scientific method applied to trading boils down to one word: **protocol**. All the processes you follow when developing a system must be protocolized. You don't change the rules midway, you don't make exceptions "because this time is different," you don't adjust the process after seeing the results.

```python
class SystemDevelopmentProtocol:
    """
    Framework for system development following the scientific method.
    Each phase has defined inputs, standardized process, and measurable outputs.
    """

    def __init__(self):
        self.phases = [
            "1_hypothesis",      # idea with market logic
            "2_data_preparation", # clean data, defined period
            "3_in_sample_test",   # backtest on training data
            "4_parameter_selection",  # optimization with protocol
            "5_out_of_sample",    # validation on UNSEEN data
            "6_walk_forward",     # progressive temporal validation
            "7_robustness_tests", # Monte Carlo, sensitivity
            "8_paper_trading",    # execution without real capital
            "9_live_deployment",  # real capital, reduced size
        ]
        self.log = []

    def execute_phase(self, phase_name, inputs, process, outputs):
        """
        Each phase is documented BEFORE execution.
        The success criteria cannot be changed after seeing results.
        """
        record = {
            'phase': phase_name,
            'inputs': inputs,
            'process': process,
            'expected_outputs': outputs,
            'status': 'pending'
        }
        self.log.append(record)
        return record
```

### Variable Isolation

When evaluating a system, change **one thing at a time**. If you modify the entry indicator, the stop loss, and the market filter simultaneously, you don't know which of the three caused the change in results.

```python
def isolated_test(base_system, variable_name, variable_values, data):
    """
    Test the effect of ONE variable while keeping everything else fixed.
    This is the scientific method: isolate to understand causality.
    """
    results = {}
    for value in variable_values:
        # Modify only the variable under study
        test_system = base_system.copy()
        test_system[variable_name] = value

        # Run backtest with everything else identical
        metrics = run_backtest(test_system, data)
        results[value] = {
            'profit_factor': metrics['profit_factor'],
            'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_drawdown'],
            'trades': metrics['total_trades']
        }

    return results

# Example: test different moving average periods,
# keeping the stop loss, take profit, and volume filter fixed
# results = isolated_test(my_system, 'ma_period', range(10, 60, 5), data)
```

### Standardized Tests

For comparisons to be valid, the test conditions must be identical:

- **Same data period** for all tests in a comparison
- **Same transaction costs** (commissions, slippage)
- **Same initial capital** and position sizing rules
- **Same seed** if there are random components

If you compare System A tested on 2020-2023 with System B tested on 2018-2024, the comparison is not valid. The market periods are different.

## From Hypothesis to System: The Complete Flow

### Phase 1: Hypothesis with Market Logic

Every system idea must start with a logical reason for why it should work. "I buy when the RSI crosses 30 because the backtest looks good" is not a hypothesis -- it's overfitting waiting to happen.

A valid hypothesis is based on an explainable market inefficiency:

```
GOOD: "Small cap stocks that open with a >10% gap on high volume
tend to continue in the gap direction for the first 30 minutes
because retail traders enter late chasing the move."

BAD: "If the 13 EMA crosses the 34 EMA when the RSI is between
42 and 58 and the MACD is positive, the price goes up."
```

The first one has a market reason (participant behavior). The second is an arbitrary combination of indicators that is probably noise.

### Phase 2: Data Splitting

Before touching a single parameter, split your data:

```python
def split_data(data, in_sample_pct=0.60, validation_pct=0.20):
    """
    Data split BEFORE any optimization.
    Once split, the boundaries are not moved.
    """
    n = len(data)
    is_end = int(n * in_sample_pct)
    val_end = int(n * (in_sample_pct + validation_pct))

    return {
        'in_sample': data[:is_end],           # for development and optimization
        'validation': data[is_end:val_end],    # for validating candidates
        'out_of_sample': data[val_end:]         # NEVER touched until the end
    }

# UNBREAKABLE RULE: the out-of-sample is used ONCE.
# If you use it to adjust and test again, it's no longer out-of-sample.
```

### Phase 3: In-Sample Optimization

You search for the best parameters using only the in-sample data. But "best" doesn't mean "most profitable" -- it means most **robust**.

A robust system maintains acceptable results across a wide range of parameters. If it only works with MA=17 but fails with MA=15 and MA=19, it's fragile.

```python
def evaluate_robustness(optimization_results, metric='profit_factor'):
    """
    A robust system has a "plateau" of good parameters,
    not an isolated peak. If the neighbors of the optimum also work,
    the signal is real. If only one point works, it's noise.
    """
    values = [r[metric] for r in optimization_results]
    peak_idx = values.index(max(values))

    # Verify that neighboring parameters are also good
    neighbors = []
    for offset in [-2, -1, 1, 2]:
        idx = peak_idx + offset
        if 0 <= idx < len(values):
            neighbors.append(values[idx])

    if not neighbors:
        return False, "Insufficient data to evaluate"

    peak_value = values[peak_idx]
    avg_neighbor = sum(neighbors) / len(neighbors)
    ratio = avg_neighbor / peak_value if peak_value > 0 else 0

    # If neighbors retain >70% of the peak value, it's robust
    return ratio > 0.70, f"Neighbor/peak ratio: {ratio:.2f}"
```

### Phase 4: Validation

The system with the selected parameters is tested on the validation data. Nothing is modified. If it passes, it advances. If it doesn't pass, it's discarded or you go back to the hypothesis.

**You do not re-optimize to pass validation.** That turns validation into in-sample.

### Phase 5: Out-of-Sample

The definitive test. One chance only. If the results are consistent with the in-sample (not identical -- consistent), the system is a candidate for paper trading.

### Phase 6: Paper Trading

Real execution without capital. You verify that real-time execution produces results compatible with the backtest. Expected differences: slippage, execution timing, data that differs slightly from historical.

### Phase 7: Live with Reduced Size

Real capital, but with the minimum possible position size. The goal is not to make money -- it's to validate that everything works in production.

## Errors the Protocol Prevents

| Error | Without protocol | With protocol |
|---|---|---|
| **Overfitting** | You optimize until the backtest is perfect | You validate on unseen data, evaluate parameter robustness |
| **Data snooping** | You try 500 combinations and pick the best | You define the hypothesis before testing, limit the variables |
| **Look-ahead bias** | You use future information without realizing it | The standardized procedure enforces temporal shift |
| **Survivorship bias** | You test with stocks that exist today | The protocol requires data with real historical composition |
| **Criteria shifting** | "This system doesn't pass my filter, but I'll use it anyway" | Criteria are defined before seeing results, and are not changed |

## When the Protocol Seems Excessive

At first, following a complete protocol for every idea seems slow. And it is -- but it saves you months of trading systems that don't work.

A legitimate shortcut for initial ideas: before the whole formal process, do a **quick test** with default parameters and no optimization. If the idea shows no signs of life with generic parameters, it's not worth dedicating a complete protocol to it.

But once you decide to move forward with an idea, the protocol is non-negotiable. "I'll skip validation because it takes time" is exactly how you end up trading overfitted systems with real money.
