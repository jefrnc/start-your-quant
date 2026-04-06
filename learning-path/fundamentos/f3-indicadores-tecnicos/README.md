> 🇪🇸 [Leer en Español](README.es.md) | 🇺🇸 **English**

# F3: Technical Indicators

**Fundamental Module 3 - Duration: 2-3 hours**

## Module Objectives

After completing this module you will be able to:
- Understand what technical indicators are and what they are used for
- Calculate and interpret the most important indicators
- Combine multiple indicators for stronger signals
- Avoid common mistakes when using indicators

## What Are Technical Indicators?

**Technical indicators** are mathematical calculations based on price and volume that help to:
- **Identify trends** (is it going up or down?)
- **Detect reversals** (is it going to change direction?)
- **Measure momentum** (how strong is the movement?)
- **Find levels** (where to buy/sell?)

### Types of Indicators

| Type | Function | Examples |
|------|----------|----------|
| **Trend** | Identify direction | SMA, EMA, MACD |
| **Momentum** | Measure strength/speed | RSI, Stochastic |
| **Volatility** | Measure variability | Bollinger Bands, ATR |
| **Volume** | Confirm movements | OBV, Volume Profile |

## The 5 Essential Indicators

### 1. Simple Moving Average (SMA)

**What is it?** The average price over the last N days.

```python
# Code for Google Colab
!pip install yfinance matplotlib pandas

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Download data
symbol = 'AAPL'
data = yf.download(symbol, period='6mo')

# Calculate SMAs for different periods
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Visualize
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Price', linewidth=2)
plt.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.8)
plt.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.8)
plt.plot(data.index, data['SMA_200'], label='SMA 200', alpha=0.8)
plt.title(f'{symbol} - Simple Moving Averages')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Interpretation
current_price = data['Close'].iloc[-1]
sma20_current = data['SMA_20'].iloc[-1]

if current_price > sma20_current:
    print(f"Price ({current_price:.2f}) > SMA20 ({sma20_current:.2f}) = UPTREND")
else:
    print(f"Price ({current_price:.2f}) < SMA20 ({sma20_current:.2f}) = DOWNTREND")
```

**Interpretation:**
- Price > SMA = Uptrend
- Price < SMA = Downtrend
- Short SMA > Long SMA = Golden Cross (very bullish)
- Short SMA < Long SMA = Death Cross (very bearish)

### 2. RSI (Relative Strength Index)

**What is it?** Measures whether a stock is overbought or oversold (scale 0-100).

```python
def calculate_rsi(data, period=14):
    """Calculates the RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI
data['RSI'] = calculate_rsi(data)

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Price
ax1.plot(data.index, data['Close'], label='Price')
ax1.set_title(f'{symbol} - Price and RSI')
ax1.legend()
ax1.grid(True, alpha=0.3)

# RSI
ax2.plot(data.index, data['RSI'], color='purple', linewidth=2)
ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
ax2.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretation
current_rsi = data['RSI'].iloc[-1]
if current_rsi > 70:
    print(f"RSI = {current_rsi:.1f} - OVERBOUGHT (possible correction)")
elif current_rsi < 30:
    print(f"RSI = {current_rsi:.1f} - OVERSOLD (possible bounce)")
else:
    print(f"RSI = {current_rsi:.1f} - NEUTRAL ZONE")
```

**Interpretation:**
- RSI > 70 = Overbought (careful, may drop)
- RSI < 30 = Oversold (opportunity, may rise)
- RSI = 50 = Equilibrium
- Divergences = When price and RSI go in opposite directions

### 3. MACD (Moving Average Convergence Divergence)

**What is it?** Shows the relationship between two moving averages and momentum.

```python
def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculates MACD, Signal and Histogram"""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

# Calculate MACD
data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Price
ax1.plot(data.index, data['Close'], label='Price')
ax1.set_title(f'{symbol} - MACD Analysis')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MACD
ax2.plot(data.index, data['MACD'], label='MACD', linewidth=2)
ax2.plot(data.index, data['Signal'], label='Signal', linewidth=2)
ax2.bar(data.index, data['Histogram'], label='Histogram', alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretation
if data['MACD'].iloc[-1] > data['Signal'].iloc[-1]:
    print("MACD > Signal = BULLISH MOMENTUM")
    if data['Histogram'].iloc[-1] > data['Histogram'].iloc[-2]:
        print("   Histogram growing = Momentum accelerating")
else:
    print("MACD < Signal = BEARISH MOMENTUM")
    if data['Histogram'].iloc[-1] < data['Histogram'].iloc[-2]:
        print("   Histogram shrinking = Momentum weakening")
```

**Interpretation:**
- MACD crosses Signal upward = Buy signal
- MACD crosses Signal downward = Sell signal
- Positive and growing histogram = Strong bullish momentum
- Negative and shrinking histogram = Strong bearish momentum

### 4. Bollinger Bands

**What is it?** Shows a "normal" price range based on volatility.

```python
def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculates Bollinger Bands"""
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()

    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return upper_band, sma, lower_band

# Calculate Bands
data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)

# Calculate band width (volatility)
data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Price and Bands
ax1.plot(data.index, data['Close'], label='Price', linewidth=2, color='black')
ax1.plot(data.index, data['BB_Upper'], label='Upper Band', alpha=0.7)
ax1.plot(data.index, data['BB_Middle'], label='SMA 20', alpha=0.7)
ax1.plot(data.index, data['BB_Lower'], label='Lower Band', alpha=0.7)
ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1, color='gray')
ax1.set_title(f'{symbol} - Bollinger Bands')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Band Width (Volatility)
ax2.plot(data.index, data['BB_Width'], color='orange', linewidth=2)
ax2.set_ylabel('Band Width')
ax2.set_title('Volatility (Band Width)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretation
position = data['BB_Position'].iloc[-1]
if position > 1:
    print(f"Price ABOVE upper band ({position:.2f}) - Possible resistance")
elif position < 0:
    print(f"Price BELOW lower band ({position:.2f}) - Possible support")
elif position > 0.8:
    print(f"Price near upper band ({position:.2f}) - Strong trend")
elif position < 0.2:
    print(f"Price near lower band ({position:.2f}) - Bearish pressure")
else:
    print(f"Price in middle zone ({position:.2f}) - Normal range")
```

**Interpretation:**
- Price touches upper band = Possible resistance
- Price touches lower band = Possible support
- Bands narrowing = Low volatility (calm before the storm)
- Bands expanding = High volatility (strong move)

### 5. Volume (The Confirmer)

**What is it?** Shows how much interest there is in a price movement.

```python
# Volume Analysis
data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
data['Price_Change'] = data['Close'].pct_change()

# Visualize
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Price
ax1.plot(data.index, data['Close'], label='Price')
ax1.set_title(f'{symbol} - Volume Analysis')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Volume
colors = ['green' if c > 0 else 'red' for c in data['Price_Change']]
ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7, label='Volume')
ax2.plot(data.index, data['Volume_SMA'], color='blue', linewidth=2, label='20d Avg Volume')
ax2.set_ylabel('Volume')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Volume Ratio
ax3.bar(data.index, data['Volume_Ratio'], color='purple', alpha=0.7)
ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='High Volume (2x)')
ax3.set_ylabel('Volume Ratio')
ax3.set_ylim(0, 4)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretation
vol_ratio = data['Volume_Ratio'].iloc[-1]
price_change = data['Price_Change'].iloc[-1] * 100

if vol_ratio > 2 and price_change > 0:
    print(f"HIGH volume ({vol_ratio:.1f}x) + Price RISING = STRONG BUY SIGNAL")
elif vol_ratio > 2 and price_change < 0:
    print(f"HIGH volume ({vol_ratio:.1f}x) + Price FALLING = STRONG SELL SIGNAL")
elif vol_ratio < 0.5:
    print(f"LOW volume ({vol_ratio:.1f}x) = Weak movement, unreliable")
else:
    print(f"Normal volume ({vol_ratio:.1f}x)")
```

**Interpretation:**
- Price up + High volume = Strong, reliable movement
- Price up + Low volume = Weak movement, be careful
- Price down + High volume = Massive selling, stay away
- Volume spike = Something important is happening

## Combining Indicators (The Real Power)

### Multi-Indicator Signal System

```python
def generate_combined_signals(data):
    """Combines multiple indicators for more reliable signals"""

    signals = []

    # Calculate all indicators
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'], _ = calculate_macd(data)
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)
    data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

    # Scoring system
    score = 0

    # 1. Trend (SMA)
    if data['Close'].iloc[-1] > data['SMA_20'].iloc[-1]:
        score += 1
        signals.append("Price > SMA20")
    if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]:
        score += 1
        signals.append("SMA20 > SMA50")

    # 2. Momentum (RSI)
    if 30 < data['RSI'].iloc[-1] < 70:
        score += 1
        signals.append(f"RSI in neutral zone ({data['RSI'].iloc[-1]:.1f})")
    elif data['RSI'].iloc[-1] < 30:
        score += 2
        signals.append(f"RSI oversold ({data['RSI'].iloc[-1]:.1f})")

    # 3. MACD
    if data['MACD'].iloc[-1] > data['Signal'].iloc[-1]:
        score += 1
        signals.append("MACD > Signal")

    # 4. Bollinger Bands
    bb_pos = (data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1])
    if 0.2 < bb_pos < 0.8:
        score += 1
        signals.append(f"Price in normal BB range ({bb_pos:.2f})")

    # 5. Volume
    if data['Volume_Ratio'].iloc[-1] > 1.5:
        score += 1
        signals.append(f"High volume ({data['Volume_Ratio'].iloc[-1]:.1f}x)")

    return score, signals

# Analyze
score, signals = generate_combined_signals(data)

print(f"\n{'='*50}")
print(f"MULTI-INDICATOR ANALYSIS: {symbol}")
print(f"{'='*50}")
print(f"Total Score: {score}/7")
print("\nDetected Signals:")
for signal in signals:
    print(f"  {signal}")

print(f"\nRECOMMENDATION:")
if score >= 6:
    print("  STRONG BUY - Multiple confirmations")
elif score >= 4:
    print("  MODERATE BUY - Positive signals")
elif score >= 2:
    print("  NEUTRAL - Mixed signals")
else:
    print("  AVOID - Few positive signals")
```

## Common Mistakes to Avoid

### Mistake #1: Using a single indicator
**Problem**: Indicators fail, especially on their own.
**Solution**: Always combine 2-3 indicators of different types.

### Mistake #2: Ignoring market context
**Problem**: RSI can be overbought for weeks in a strong trend.
**Solution**: Consider the overall trend before acting on signals.

### Mistake #3: Not adjusting parameters
**Problem**: RSI(14) works differently in crypto vs. blue chips.
**Solution**: Test different periods and adjust per asset.

### Mistake #4: Reacting to every signal
**Problem**: Too many signals = overtrading.
**Solution**: Only act when multiple indicators agree.

### Mistake #5: Forgetting volume
**Problem**: Movements without volume are false.
**Solution**: ALWAYS confirm with volume.

## Final Project: Your Indicator Dashboard

```python
def create_complete_dashboard(symbol='AAPL'):
    """Creates a complete dashboard with all indicators"""

    # Download data
    data = yf.download(symbol, period='6mo')

    # Calculate all indicators
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(data)

    # Create visualization
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)

    # 1. Price and SMA
    axes[0].plot(data.index, data['Close'], label='Price', linewidth=2)
    axes[0].plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
    axes[0].set_title(f'{symbol} - Complete Indicator Dashboard')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. RSI
    axes[1].plot(data.index, data['RSI'], color='purple', linewidth=2)
    axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('RSI')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    # 3. MACD
    axes[2].plot(data.index, data['MACD'], label='MACD')
    axes[2].plot(data.index, data['Signal'], label='Signal')
    axes[2].bar(data.index, data['Histogram'], alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_ylabel('MACD')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 4. Bollinger Bands
    axes[3].plot(data.index, data['Close'], label='Price', color='black')
    axes[3].plot(data.index, data['BB_Upper'], 'r--', alpha=0.7)
    axes[3].plot(data.index, data['BB_Lower'], 'g--', alpha=0.7)
    axes[3].fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1)
    axes[3].set_ylabel('Bollinger')
    axes[3].grid(True, alpha=0.3)

    # 5. Volume
    colors = ['green' if c > o else 'red' for c, o in zip(data['Close'], data['Open'])]
    axes[4].bar(data.index, data['Volume'], color=colors, alpha=0.7)
    axes[4].set_ylabel('Volume')
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Final analysis
    score, signals = generate_combined_signals(data)
    return score, signals

# Run dashboard
score, signals = create_complete_dashboard('AAPL')
```

## Module Checkpoint

### Knowledge
- [ ] I understand the 5 main types of indicators
- [ ] I can calculate SMA, RSI, MACD, Bollinger Bands
- [ ] I can interpret signals from each indicator
- [ ] I understand why combining indicators is crucial

### Skills
- [ ] I can code any indicator from scratch
- [ ] I can create clear indicator visualizations
- [ ] I can combine multiple indicators for signals
- [ ] I avoid common beginner mistakes

### Project
- [ ] My multi-indicator dashboard works
- [ ] It generates combined signals correctly
- [ ] I can apply it to any stock

## Next Module

**F4: Your First Complete Strategy**
- Systematic strategy design
- Entry and exit rules
- Risk management
- Basic backtesting

**You now master the indicators! Time to create your first real strategy**

---

**Ready for your first strategy?** -> [F4: First Strategy](../f4-primera-estrategia/)
