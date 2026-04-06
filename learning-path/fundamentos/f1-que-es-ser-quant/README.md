> 🇪🇸 [Leer en Español](README.es.md) | 🇺🇸 **English**

# F1: What Is Being a Quant?

**Fundamental Module 1 - Duration: 1-2 hours**

## Module Objectives

After completing this module you will be able to:
- Explain what a quant trader does vs. a traditional trader
- Understand why numbers work better than intuition
- Identify opportunities to apply quantitative methods
- Have clarity about your path to becoming a quant

## What Is a Quant Trader?

### Simple Definition
A **quant trader** is someone who uses **mathematics, statistics, and programming** to make trading decisions instead of intuition or "gut feeling."

### Real Example: Two Approaches

**Traditional Trader**:
```
"Apple is dropping a lot, I think it's going to bounce.
I'm going to buy because it 'feels' cheap."
```

**Quant Trader**:
```python
# Download 5 years of AAPL data
data = get_market_data('AAPL', '5y')

# Statistically calculate whether "strong drops" predict bounces
strong_drops = data[data['daily_change'] < -3%]
next_day_returns = strong_drops['next_day_change']

# Result: 67% of bounces the next day
# Historical probability > 50% -> Viable strategy
if probability > 0.6:
    place_order('BUY', 'AAPL', shares=100)
```

### Which Is Better?

**Traditional Trader**:
- Emotions affect decisions
- Cannot manage many stocks
- Hard to improve systematically
- Inconsistent results

**Quant Trader**:
- Data-driven decisions
- Can analyze 1000+ stocks
- Continuous improvement with more data
- Measurable and repeatable results

## Why Be a Quant?

### 1. **Scalability**
- An algorithm can monitor **the entire market** 24/7
- No human attention limit
- You can trade multiple strategies simultaneously

### 2. **Consistency**
- The same rules **always**
- No "bad days" where you decide differently
- Eliminates FOMO, fear, greed

### 3. **Continuous Improvement**
- Every day you have **more data** to improve
- You can test thousands of variations
- A/B testing of strategies

### 4. **Competitive Advantage**
- Most retail traders use intuition
- Access to **institutional tools** (free)
- You can automate while others work

## Examples of Quant Strategies

### 1. **Gap Trading** (Beginner)
```
RULE: If a stock opens 5%+ above the previous close
AND volume is 2x the average
THEN: Buy at the open, sell in 30 minutes

HISTORICAL RESULT: 62% win rate, 1.3 profit factor
```

### 2. **Mean Reversion** (Intermediate)
```
RULE: If price is 2 standard deviations below the 20-day mean
AND RSI < 30
THEN: Buy and hold until it returns to the mean

RESULT: 71% win rate in small caps
```

### 3. **Momentum + Volume** (Advanced)
```
RULE: If price breaks 52-week high
AND volume > 300% of average
AND sector is in an uptrend
THEN: Buy with dynamic stop loss

RESULT: 58% win rate, but winning trades are 3x larger
```

## Quant Tools

### **Software (All Free)**
- **Python**: Primary language
- **Pandas**: Data manipulation
- **YFinance**: Free market data
- **Matplotlib**: Charts
- **Jupyter**: Interactive analysis

### **Data (Many Free)**
- **Yahoo Finance**: Historical prices
- **Alpha Vantage**: Free API
- **Federal Reserve**: Economic data
- **SEC Filings**: Fundamental information

### **Brokers with APIs**
- **Interactive Brokers**: Professional API
- **Alpaca**: Commission-free with API
- **TD Ameritrade**: thinkorswim API

## Your First Quantitative Analysis

### Exercise 1.1: Gap Analysis
Let's test whether "gaps" actually work:

```python
# Don't worry if you don't understand all the code yet
# Just observe the METHODOLOGY

import yfinance as yf
import pandas as pd

# 1. HYPOTHESIS: Stocks that open +5% tend to keep rising
symbol = 'AAPL'
data = yf.download(symbol, start='2020-01-01', end='2024-01-01')

# 2. IDENTIFY gaps
data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100

# 3. FILTER large gaps (>5%)
big_gaps = data[data['Gap'] > 5]

# 4. MEASURE what happened next
big_gaps['Next_Hour_Return'] = (data['High'] - data['Open']) / data['Open'] * 100

# 5. STATISTICS
print(f"Total gaps >5%: {len(big_gaps)}")
print(f"Average first hour return: {big_gaps['Next_Hour_Return'].mean():.2f}%")
print(f"% of gaps that kept rising: {(big_gaps['Next_Hour_Return'] > 0).mean()*100:.1f}%")
```

**Your Task**: Run this code and reflect:
1. Do the numbers support the gap strategy?
2. What would you change to improve the strategy?
3. How is this different from "guessing" it will go up?

### Exercise 1.2: Define Your "Why"
Write 2-3 paragraphs answering:
1. **Why do you want to be a quant trader?**
2. **What attracts you most: programming, mathematics, or trading?**
3. **What is your goal in 6 months?**

## Types of Quant Traders

### 1. **Retail Quant** (Initial Goal)
- Manages own capital ($1K - $100K)
- Simple but effective strategies
- Basic automation
- Part-time job or lucrative hobby

### 2. **Hedge Fund Quant** (Long-term Goal)
- Manages institutional capital ($1M+)
- Complex strategies with ML/AI
- Advanced infrastructure
- Full-time job, salaries $200K+

### 3. **Prop Trading Quant**
- Trading with the firm's capital
- Focus on high-frequency or arbitrage
- Cutting-edge technology
- Profit sharing with the firm

### Which attracts you most?

## Common Obstacles (And How to Overcome Them)

### **"I don't know how to code"**
**Solution**: You only need basic Python. This course teaches you everything you need.

### **"I don't have capital"**
**Solution**: You can start with $100 in paper trading and build a track record.

### **"It's too complicated"**
**Solution**: Simple strategies work. You don't need to be Einstein.

### **"There's already too much competition"**
**Solution**: Retail quants compete against retail traders, not against Goldman Sachs.

### **"I don't have time"**
**Solution**: Once programmed, the system works while you sleep.

## Success Profiles

### **Jim Simons** (Renaissance Technologies)
- Mathematician turned trader
- Average annual returns: 35%+
- Uses complex statistical models
- Net worth: $25B+

### **Ray Dalio** (Bridgewater)
- Created "All Weather" portfolio using correlations
- Manages $150B+ using quantitative principles
- Focus on macroeconomic data

### **Andreas Clenow** (Retail Quant)
- Started as a retail trader
- Wrote books on quantitative trading
- Manages funds using "simple" strategies
- Proved you don't need a PhD to be successful

## Module Checkpoint

Before continuing to the next module, you should be able to:

### Knowledge
- [ ] Explain the difference between quant and traditional trader
- [ ] Give 3 advantages of quantitative trading
- [ ] Understand what "testing a hypothesis with data" means
- [ ] Identify what type of quant you want to be

### Mindset
- [ ] You are convinced that data > intuition
- [ ] You have clarity about your "why"
- [ ] You understand this is a gradual learning process
- [ ] You are excited to start coding

### Exercises
- [ ] You ran the gap analysis (even if you don't understand everything)
- [ ] You wrote your personal "why"
- [ ] You identified what type of quant you want to be

## Next Module

**F2: Basic Python Trading**
- Complete installation and setup
- Downloading real data
- Your first charts
- Calculating simple indicators

**Estimated time**: 3-4 hours

---

## Final Reflection

> **"The best time to plant a tree was 20 years ago. The second best time is now."**

Every day that passes without starting is one less day of historical data to analyze and improve. Markets generate new data every second.

**Your window of opportunity is NOW.**

**Ready for the next module?** -> [F2: Basic Python Trading](../f2-python-trading-basico/README.md)
