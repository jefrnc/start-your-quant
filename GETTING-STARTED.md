> 🇪🇸 [Leer en Español](GETTING-STARTED.es.md) | 🇺🇸 **English**

# Getting Started - Start Your Quant

**Your adventure toward becoming a quant trader starts here.**

## What Will You Learn?

- **Quantitative Trading**: Use math and programming instead of intuition
- **Python for Finance**: Wall Street's #1 language
- **Data Analysis**: Find profitable patterns in the market
- **Automation**: Systems that trade while you sleep
- **Real Strategies**: Methods proven in real markets

## Start Without Installing Anything

### Option 1: Google Colab (Recommended)
**Perfect for beginners - No installation needed**

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Paste this code and run:

```python
# Install basic libraries
!pip install yfinance pandas matplotlib seaborn

# Your first quantitative analysis
import yfinance as yf
import matplotlib.pyplot as plt

# Download Apple data
data = yf.download('AAPL', period='1y')

# Create chart
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'])
plt.title('Apple - Your First Quant Analysis')
plt.show()

print(f"Current price: ${data['Close'][-1]:.2f}")
print("You're already a quant trader!")
```

### Option 2: Your Computer
**If you prefer to work locally**

1. **Install Python** (if you don't have it):
   - [python.org](https://python.org) -> Download latest version
   - Check "Add to PATH" on Windows

2. **Install libraries**:
   ```bash
   pip install yfinance pandas matplotlib seaborn numpy
   ```

3. **Verify**:
   ```bash
   python -c "import yfinance; print('All set!')"
   ```

## Where Should You Start?

### Quick Test: What's Your Level?

**Question 1: Have you programmed before?**
- A) Never -> Start at **F1**
- B) A little -> Start at **F2**
- C) Yes, but not Python -> Start at **F2**
- D) Yes, I know Python -> Start at **F3**

**Question 2: Have you traded before?**
- A) Never -> Start at **F1**
- B) A little manually -> Start at **F2**
- C) Yes, but manually -> Start at **F3**
- D) Yes, I know technical analysis -> Start at **E1**

### Your Personalized Plan (choose ONE):
| If you are... | Week 1 | Week 2 | Week 3 | Week 4 | GOAL |
|---------------|--------|---------|---------|---------|------|
| **Total Noob** | Python basics + Colab | Read Yahoo data | Your first indicator | Paper trading bot | Working bot |
| **Dev Without Trading** | Technical indicators | Backtesting basics | Optimization | Broker API | Real bot |
| **Manual Trader** | Python crash course | Automate YOUR strategy | Backtesting | Improvements | Your auto system |
| **Want It All NOW** | Complete setup | 3 strategies | Portfolio manager | Cloud deploy | Pro system |

## MINUTES 10-20: Ready-to-Copy Bots

### Bot #1: Morning Gap Detector (Profit from volatility)
```python
# Bot that detects gaps (openings with large differences)
import yfinance as yf
import pandas as pd

# List of volatile stocks
volatile = ['TSLA', 'NVDA', 'AMD', 'COIN', 'MARA']

print("SCANNING TODAY'S GAPS...\n")

for ticker in volatile:
    data = yf.download(ticker, period='5d', progress=False)

    # Gap = Today's open vs yesterday's close
    gap = ((data['Open'][-1] - data['Close'][-2]) / data['Close'][-2]) * 100

    if abs(gap) > 2:  # Gap greater than 2%
        direction = "UP" if gap > 0 else "DOWN"
        print(f"ALERT {ticker}: GAP {direction} of {gap:.2f}%")
        print(f"   Potential: Regression to ${data['Close'][-2]:.2f}")
        print(f"   Entry: ${data['Open'][-1]:.2f}")
        print(f"   Target: ${data['Close'][-2]:.2f}")
        print(f"   Potential gain: {abs(gap):.2f}%\n")
```

### Bot #2: Oversold Hunter (Buy the panic)
```python
# Bot that finds oversold stocks
import yfinance as yf
import pandas as pd

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

blue_chips = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']

print("LOOKING FOR BUY OPPORTUNITIES (RSI < 30)\n")

for ticker in blue_chips:
    data = yf.download(ticker, period='1mo', progress=False)
    data['RSI'] = calculate_rsi(data['Close'])

    current_rsi = data['RSI'][-1]

    if current_rsi < 35:  # Oversold
        print(f"{ticker} OVERSOLD!")
        print(f"   RSI: {current_rsi:.2f}")
        print(f"   Current price: ${data['Close'][-1]:.2f}")
        print(f"   30d average: ${data['Close'].mean():.2f}")
        print(f"   Bounce potential: {((data['Close'].mean() - data['Close'][-1]) / data['Close'][-1] * 100):.2f}%\n")
```

### Bot #3: Complete System with Stop Loss
```python
# Complete system with risk management
import yfinance as yf
import numpy as np

# Configuration
TICKER = 'SPY'  # S&P 500 ETF
INITIAL_CAPITAL = 10000
RISK_PER_TRADE = 0.02  # 2% maximum

data = yf.download(TICKER, period='6mo', progress=False)

# Indicators
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['Volatility'] = data['Close'].rolling(20).std()

# Signals
data['Signal'] = 0
data.loc[data['MA20'] > data['MA50'], 'Signal'] = 1  # Buy
data.loc[data['MA20'] < data['MA50'], 'Signal'] = -1  # Sell

# Dynamic stop loss (2x volatility)
data['StopLoss'] = data['Close'] - (2 * data['Volatility'])

# Simulate trades
capital = INITIAL_CAPITAL
trades = []

for i in range(1, len(data)):
    if data['Signal'].iloc[i] == 1 and data['Signal'].iloc[i-1] != 1:
        # Entry
        entry = data['Close'].iloc[i]
        stop = data['StopLoss'].iloc[i]
        size = (capital * RISK_PER_TRADE) / (entry - stop)

        trades.append({
            'Date': data.index[i].date(),
            'Type': 'BUY',
            'Price': entry,
            'Stop': stop,
            'Shares': int(size)
        })

# Results
print("COMPLETE BACKTEST - LAST 6 MONTHS\n")
print(f"Initial capital: ${INITIAL_CAPITAL:,}")
print(f"Total trades: {len(trades)}")
print(f"\nLAST 3 SIGNALS:")

for trade in trades[-3:]:
    print(f"\n{trade['Date']}: {trade['Type']} @ ${trade['Price']:.2f}")
    print(f"  Stop Loss: ${trade['Stop']:.2f}")
    print(f"  Position: {trade['Shares']} shares")
    print(f"  Risk: ${(trade['Shares'] * (trade['Price'] - trade['Stop'])):.2f}")
```

## MINUTES 20-30: Connect with a REAL Broker

### Option A: Paper Trading (No Risk)
```python
# Alpaca Paper Trading - FREE, no real money
# 1. Register at: alpaca.markets (2 min)
# 2. Get your API keys
# 3. Paste this code:

!pip install alpaca-py

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Your credentials (paper trading)
API_KEY = 'your_api_key'
SECRET_KEY = 'your_secret_key'

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Your account
account = client.get_account()
print(f"Balance: ${account.cash}")
print(f"Buying power: ${account.buying_power}")

# Make a trade
order_data = MarketOrderRequest(
    symbol="AAPL",
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

order = client.submit_order(order_data)
print(f"Order submitted: Buy 1 AAPL")
```

### Option B: Phone Notifications
```python
# Telegram Bot - Receive alerts on your phone
# 1. Search @BotFather on Telegram
# 2. Create a bot (/newbot)
# 3. Save the token

!pip install python-telegram-bot

import requests

def send_alert(message):
    token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={'chat_id': chat_id, 'text': message})

# Send alert from your bot
if signal == "BUY":
    send_alert(f"BUY {ticker} at ${price:.2f}")
```

## RESULT: Your Trading System in 30 Minutes

**You now have:**
- 3 working bots
- Opportunity scanner
- System with stop loss
- Broker connection (paper)
- Phone alerts

**Next step:**
- Join the Discord to share results
- Improve bots with the community
- Start the full course

**Useful Links:**
- [Discord QuantLab](https://discord.gg/GgXZ3zAS)
- [Complete Academy]({{ site.baseurl }}/learning-path/)
- [Code on GitHub](https://github.com/jefrnc/start-your-quant)

### Minutes 31-45: First Strategy
1. Copy the moving average code
2. Test it with different stocks
3. Modify the parameters

### Minutes 46-60: Personal Plan
1. Define your goal (why do you want to be a quant?)
2. Choose your entry path
3. Block 30 min daily on your calendar

## Common Problems

### "yfinance won't install"
```bash
pip install --upgrade pip
pip install yfinance
```

### "Charts don't appear"
```python
import matplotlib.pyplot as plt
plt.show()  # Add at the end of the code
```

### "Error downloading data"
- Check internet connection
- Try another symbol: 'MSFT', 'GOOGL'
- Use shorter period: period='1mo'

### "Python not recognized"
- Windows: Check "Add to PATH" when installing
- Mac: `brew install python`
- Linux: `sudo apt install python3`

## Staying Motivated

### Weekly Goals
- **Week 1**: Complete F1 and F2
- **Week 2**: Complete F3 and F4
- **Week 3**: Start strategies (E1)
- **Week 4**: First complete strategy

### Visible Progress
- Each module includes verifiable exercises
- Your code gradually improves
- Build a strategy portfolio
- Real performance metrics

### Community
- GitHub Issues for technical questions
- Discord for daily chat (coming soon)
- Share your progress with #StartYourQuant

## By the End You'll Have

### Technical Portfolio
- 5+ tested strategies
- Robust backtesting system
- Monitoring dashboard
- Reusable code

### Knowledge
- Python for trading
- Quantitative technical analysis
- Risk management
- Strategy optimization

### Skills
- Analyze any stock in minutes
- Test ideas quickly
- Automate trading decisions
- Evaluate performance objectively

## Start NOW!

**The best strategy is to start, even if it's imperfect.**

1. **Setup**: `python quick-start.py` (5 min)
2. **First lesson**: [F1 - What Is Being a Quant?](learning-path/fundamentos/f1-que-es-ser-quant/) (30 min)
3. **Commitment**: 30 min daily for 30 days

**In 30 days you'll have more quantitative knowledge than 95% of retail traders.**

---

**Ready to start?** -> Run `python quick-start.py` and begin your quant adventure!
