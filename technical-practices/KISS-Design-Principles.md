> 🇪🇸 [Leer en Español](KISS-Design-Principles.es.md) | 🇺🇸 **English**

# KISS: System Design Principles

Keep It Simple, Stupid. In algorithmic trading, simplicity is not a limitation -- it's your main defense against overfitting.

## Why Simple Wins

Simplicity fosters robustness. And robustness is the single most important thing in a system, because it can't be guaranteed -- it can only be fostered during design.

A robust system maintains its edge on data it never saw. A complex system fits the past perfectly but fails in the future. The difference between the two, almost always, is the number of rules.

**Practical rule**: 1-2 entry rules, 1 filter at most, multiple exits. If you need more than that for it to work, you're probably capturing noise, not signal.

### How Many Rules Are Too Many

Every rule, every filter, every optimizable parameter you add gives the system more flexibility to fit to historical data. More complexity = more overfitting risk.

```
1-2 rules + 1 filter:  robust, hard to fit to noise
3-4 rules + 2 filters: gray zone, needs thorough validation
5+ rules + 3 filters:  almost certainly overfitting, no matter how perfect the backtest
```

If you can't explain your system in 2 minutes to someone who doesn't know trading, it's probably too complex.

## From Paper to Code: Pseudocode

Before coding, write the system in pseudocode. This is not optional -- it's the step that prevents logical errors and forces you to understand exactly what each part does.

**Pseudocode**: a mix between human language and programming language. It's not executable, but it's precise enough to translate into any language.

```
PSEUDOCODE for the Aberration system (Fitschen, 1986):

Variables:
  - average = simple moving average of N periods
  - upper_band = average + 2 x standard_deviation
  - lower_band = average - 2 x standard_deviation

Long entry:
  IF close > upper_band
    BUY at next bar open

Long exit:
  IF long AND close < average
    CLOSE long at next bar open

Short entry:
  IF close < lower_band
    SELL at next bar open

Short exit:
  IF short AND close > average
    CLOSE short at next bar open
```

**Advantages of pseudocode:**
- You catch logical errors before coding
- It translates easily to any language (Python, EasyLanguage, MQL, NinjaScript)
- It serves as system documentation
- If someone else reads it, they can verify the logic

## Longs and Shorts: Separate When Possible

Equities historically show asymmetry: they tend to rise gradually and fall quickly with higher volatility. Volatility is greater during declines. This means the optimal parameters for the long side are probably different from the short side.

**If you optimize long and short together**, the optimizer seeks a compromise that's not optimal for either. It typically biases toward the long side (there's more long data in bull markets).

**If you separate them**, each side has its own optimized parameters. The sum is usually better than the whole.

**But**: separating cuts the sample in half. With few trades, that reduces statistical significance and increases overfitting risk.

| Situation | Recommendation |
|---|---|
| Intraday system with many trades | Separate longs and shorts |
| Daily system with good history (10+ years) | Separate if there are enough trades per side |
| Weekly/monthly system | Trade together -- not enough sample to separate |
| Not trading short on equities | Totally valid. Many successful funds are long-only |

## Normalized ATR: Comparing Apples to Apples

Standard ATR measures volatility in points. But 100 points with the Nasdaq at 5,000 represents 2%, while 100 points with the Nasdaq at 18,000 is barely 0.55%. ATR alone doesn't allow you to compare relative volatility across assets with different prices or over time if the price has changed significantly.

**Solution**: normalize the ATR by dividing it by price.

```python
def atr_normalized(atr, high, low, close):
    """
    ATR as a percentage of price.
    Allows comparing volatility across assets and over time.
    """
    typical_price = (high + low + close) / 3
    return (atr / typical_price) * 100

# Comparison (daily, ~20 years):
# Nasdaq 100:  ATR% ~1.65%  -- high volatility
# S&P 500:     ATR% ~1.24%  -- moderate
# Gold (GLD):  ATR% ~0.97%  -- low
# Coffee:      ATR% ~2.24%  -- very high
# Crude Oil:   ATR% ~3.08%  -- extreme
```

**Uses of normalized ATR:**
- Compare volatility across assets to choose where to trade
- Adjust exposure: reduce contracts when volatility rises, increase when it falls (with limits)
- Size stops and TPs that adapt to the current volatility regime

## Choosing a Programming Language

| Language | Platform | Level | Best for |
|---|---|---|---|
| **EasyLanguage** | TradeStation, MultiCharts | Very high (almost pseudocode) | Beginners, rapid prototyping |
| **Python** | Independent | High | Flexibility, ML, data analysis |
| **NinjaScript** | NinjaTrader | High | NinjaTrader users |
| **MQL4/5** | MetaTrader | Medium-high | Forex, longer codebases |

**The dilemma**: choosing a language determines your platform, broker, and data.

- **EasyLanguage/TradeStation**: all-in-one (platform + data + broker + language). Ideal for getting started without connection complications
- **Python**: maximum flexibility but you have to solve data, broker, and execution connections separately

There's no "best" language. There's the one that best fits your profile and experience. If you already know Python, use Python. If you're starting from scratch, EasyLanguage has the shortest learning curve.

What matters is that you understand the code you're trading -- whether it's yours or someone else's. If you can't explain every line, don't trade it with real money.
