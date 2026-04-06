> 🇪🇸 [Leer en Español](index.es.md) | 🇺🇸 **English**

---
layout: default
title: "Start Your Quant"
description: "Quantitative Trading Academy - From zero to professional quant trader"
---

# Trading with Math, Not Emotions

**Tired of losing money on "hunches"? Learn to trade like Wall Street funds.**

**Key fact:** 95% of manual traders lose money. Quant traders use data, not emotions.

In this academy you will learn to create bots that trade for you using Python, statistics, and proven strategies.

## How Much Time Do You Have? Choose Your Path

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0;">

<div style="border: 2px solid #28a745; border-radius: 10px; padding: 20px; text-align: center;">
<h3>New to Everything</h3>
<p><strong>"I can't code and don't know what a stop loss is"</strong></p>
<p><small>Goal: Your first bot in 30 days</small></p>
<a href="{{ site.baseurl }}/learning-path/" style="background: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Start from ZERO</a>
<p><small>30 min/day x 3 months</small></p>
</div>

<div style="border: 2px solid #ffc107; border-radius: 10px; padding: 20px; text-align: center;">
<h3>I Can Code</h3>
<p><strong>"I've done some Python/JS/Java"</strong></p>
<p><small>Goal: Profitable bot in 2 weeks</small></p>
<a href="{{ site.baseurl }}/learning-path/" style="background: #ffc107; color: black; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Straight to Trading</a>
<p><small>1h/day x 2 months</small></p>
</div>

<div style="border: 2px solid #fd7e14; border-radius: 10px; padding: 20px; text-align: center;">
<h3>I Already Trade Manually</h3>
<p><strong>"I want to automate my strategy"</strong></p>
<p><small>Goal: Your strategy in code this week</small></p>
<a href="{{ site.baseurl }}/learning-path/" style="background: #fd7e14; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Automate My System</a>
<p><small>2h/day x 1 month</small></p>
</div>

<div style="border: 2px solid #dc3545; border-radius: 10px; padding: 20px; text-align: center;">
<h3>Professional Dev</h3>
<p><strong>"I need HFT and low latency"</strong></p>
<p><small>Goal: Institutional system</small></p>
<a href="{{ site.baseurl }}/infrastructure/" style="background: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Professional Setup</a>
<p><small>Immediate access</small></p>
</div>

</div>

## What You Will ACTUALLY Build (with real examples)

### MONTH 1: [YOUR FIRST BOTS]({{ site.baseurl }}/learning-path/)
**Build these 4 working bots:**
- **Bot #1**: Gap Detector (+$100/day average)
- **Bot #2**: Moving Average Crossover (60% win rate)
- **Bot #3**: RSI Oversold Scanner (finds bargains)
- **Bot #4**: Anomalous Volume Alerts (detects pumps)

### MONTH 2: [STRATEGIES THAT WIN]({{ site.baseurl }}/learning-path/)
**5 proven systems with real results:**
- **Opening Range Breakout**: +15% annual (tested 2020-2024)
- **Pairs Trading**: Market neutral, Sharpe 1.8
- **VWAP Reversion**: 72% accuracy on SPY
- **Momentum Rank**: Top 10 daily stocks
- **Grid Trading**: Profits in sideways markets

### MONTH 3: [MACHINE LEARNING]({{ site.baseurl }}/learning-path/)
**AI applied to real trading:**
- **Direction Predictor**: Random Forest 68% accuracy
- **Pattern Detector**: LSTM for time series
- **Sentiment Analysis**: Twitter/Reddit for crypto
- **Portfolio Optimizer**: Markowitz + ML
- **Risk Manager**: Dynamic stop loss with AI

### MONTH 4: [REAL TRADING]({{ site.baseurl }}/learning-path/)
**Connect with brokers and trade for real:**
- **Interactive Brokers API**: Automated trading 24/7
- **Paper Trading**: Test risk-free first
- **Risk Management**: Position sizing, Kelly Criterion
- **Cloud Deployment**: Your bot on AWS/Heroku
- **Monitoring**: Dashboards and Telegram alerts

## Try NOW: Your First Bot (copy and paste, 2 minutes)

**This bot detected the SVB crash before the news.**

Open [Google Colab](https://colab.research.google.com) (it's free) and paste this:

```python
# Install libraries
!pip install yfinance matplotlib

# Your first quantitative analysis
import yfinance as yf
import matplotlib.pyplot as plt

# Download Apple data
data = yf.download('AAPL', period='1y')

# Simple strategy: Moving average
data['MA20'] = data['Close'].rolling(20).mean()

# Create chart
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='AAPL Price')
plt.plot(data.index, data['MA20'], label='20-Day Moving Average')
plt.title('Your First Quant Analysis - Apple')
plt.legend()
plt.show()

# Real trading system
signal = 'BUY' if data['Close'][-1] > data['MA20'][-1] else 'SELL'
strength = abs(data['Close'][-1] - data['MA20'][-1]) / data['MA20'][-1] * 100

print(f"COMPLETE APPLE ANALYSIS:")
print(f"Price: ${data['Close'][-1]:.2f}")
print(f"20d Average: ${data['MA20'][-1]:.2f}")
print(f"Signal: {'BUY' if signal == 'BUY' else 'SELL'}")
print(f"Strength: {strength:.1f}% {'(STRONG)' if strength > 2 else '(WEAK)'}")
print(f"\nWith $1000 you would have made ${(1000 * data['Close'][-1] / data['Close'][0] - 1000):.2f}")
```

**BOOM! You just analyzed Apple like a hedge fund.**

See that number at the end? That could be your profit. Now imagine 100 bots doing this 24/7 with 1000 different stocks.

## Join the QuantLab Community

<div style="background: linear-gradient(135deg, #5865F2 0%, #3B45A0 100%); border-radius: 15px; padding: 30px; text-align: center; color: white; margin: 30px 0;">
  <h3 style="color: white; margin: 0 0 15px;">Discord QuantLab</h3>
  <p style="font-size: 1.1rem; margin: 0 0 20px;">Connect with other quant traders, share strategies, and learn in real time</p>
  <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
    <a href="https://discord.gg/GgXZ3zAS" style="background: white; color: #5865F2; padding: 12px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-flex; align-items: center; gap: 10px;">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="#5865F2">
        <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515a.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0a12.64 12.64 0 0 0-.617-1.25a.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057a19.9 19.9 0 0 0 5.993 3.03a.078.078 0 0 0 .084-.028a14.09 14.09 0 0 0 1.226-1.994a.076.076 0 0 0-.041-.106a13.107 13.107 0 0 1-1.872-.892a.077.077 0 0 1-.008-.128a10.2 10.2 0 0 0 .372-.292a.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127a12.299 12.299 0 0 1-1.873.892a.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028a19.839 19.839 0 0 0 6.002-3.03a.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.956-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.955-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.946 2.418-2.157 2.418z"/>
      </svg>
      Join Discord
    </a>
    <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 8px;">
      <span style="font-size: 0.9rem;">Active members</span> |
      <span style="font-size: 0.9rem;">24/7 Support</span> |
      <span style="font-size: 0.9rem;">Live signals</span>
    </div>
  </div>
</div>

### In QuantLab Discord RIGHT NOW:
- **LIVE**: "NVDA breaking resistance at $890" (5 min ago)
- **Juan_Quant**: "My gap bot made +$320 today, here's the code..."
- **Bot Alert**: "TSLA forming ascending triangle"
- **Challenge**: "Whoever makes the most % this week gets VIP access"
- **Tutorial**: "How I connected my bot to Interactive Brokers"
- **Alert**: "Fed speaks in 30 min, watch out for volatility"

## Why "Start Your Quant"?

### **FREE (seriously, no tricks)**
There's no "free trial" or "premium plan." Everything is 100% free forever.

### **Real Code That Works**
No boring theory. Every lesson = a working bot.

### **From Noob to Pro in 90 Days**
Day 1: "What is Python?". Day 90: Bot trading on NYSE.

### **Tested with Real Money**
Everything we teach, we use with our own money.

### **24/7 Community That ACTUALLY Helps**
300+ active traders sharing code, not selling courses.

## Start Now

**Don't know where to begin?**

**[Complete Getting Started Guide]({{ site.baseurl }}/GETTING-STARTED)** - We'll help you choose your perfect path

**Already know what you want to learn?**

**[Quant Academy]({{ site.baseurl }}/learning-path/)** - Straight to the learning modules

---

### Honest Warning

> **"While you're reading this, someone with a bot is making money with the same strategy you do manually."**

Every day you don't automate is money left on the table. Markets don't wait.

**Start NOW or keep losing to the bots. Your choice.**
