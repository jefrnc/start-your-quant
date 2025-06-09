# 📚 Quick Reference - Trading Algorítmico

> Referencia rápida de conceptos, métricas y código esencial para trading cuantitativo

## 🎯 Métricas Clave

### 📊 Performance Metrics

| Métrica | Fórmula | Interpretación | Valor Objetivo |
|---------|---------|----------------|----------------|
| **Sharpe Ratio** | `(Return - Risk_Free) / Volatility` | Rendimiento ajustado por riesgo | > 1.0 (>2.0 excelente) |
| **Calmar Ratio** | `Annual_Return / Max_Drawdown` | Retorno vs drawdown máximo | > 1.0 |
| **Win Rate** | `Winning_Trades / Total_Trades` | % de trades ganadores | > 50% (depende de R:R) |
| **Max Drawdown** | `Max(Peak - Trough) / Peak` | Pérdida máxima desde peak | < 20% |
| **Profit Factor** | `Gross_Profit / Gross_Loss` | Relación ganancia/pérdida | > 1.3 |

### 🎯 Position Sizing

```python
# Kelly Criterion
def kelly_position_size(win_rate, avg_win, avg_loss):
    return (win_rate - (1-win_rate) * (avg_loss/avg_win))

# Fixed Fractional
def fixed_fractional_size(capital, risk_per_trade=0.02):
    return capital * risk_per_trade

# Volatility-based
def volatility_position_size(capital, price, atr, multiplier=2):
    risk_amount = capital * 0.02  # 2% risk
    stop_distance = atr * multiplier
    return risk_amount / (stop_distance * price)
```

## 📈 Indicadores Esenciales

### 🔢 Moving Averages

```python
# Simple Moving Average
df['SMA_20'] = df['close'].rolling(20).mean()

# Exponential Moving Average  
df['EMA_20'] = df['close'].ewm(span=20).mean()

# Volume Weighted Average Price
df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
```

### 📊 Volatility Indicators

```python
# True Range
df['TR'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)

# Average True Range
df['ATR'] = df['TR'].rolling(14).mean()

# Bollinger Bands
df['BB_upper'] = df['SMA_20'] + (df['close'].rolling(20).std() * 2)
df['BB_lower'] = df['SMA_20'] - (df['close'].rolling(20).std() * 2)
```

### 📈 Volume Indicators

```python
# Relative Volume
df['RVOL'] = df['volume'] / df['volume'].rolling(20).mean()

# Volume Rate of Change
df['VROC'] = df['volume'].pct_change(periods=1)

# On Balance Volume
df['OBV'] = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, -1)).cumsum()
```

## 🎯 Estrategias Quick Setup

### 🚀 Gap and Go

```python
def gap_and_go_signal(df):
    # Condiciones
    gap_up = (df['open'] / df['close'].shift(1)) > 1.02  # 2% gap
    high_volume = df['volume'] > df['volume'].rolling(20).mean() * 2
    above_vwap = df['close'] > df['VWAP']
    
    # Señal de entrada
    entry_signal = gap_up & high_volume & above_vwap
    
    return entry_signal
```

### 📊 VWAP Reclaim

```python
def vwap_reclaim_signal(df):
    # Precio por debajo de VWAP
    below_vwap = df['close'] < df['VWAP']
    
    # Reclaim con volumen
    reclaim = (df['close'] > df['VWAP']) & below_vwap.shift(1)
    volume_confirmation = df['volume'] > df['volume'].rolling(10).mean()
    
    return reclaim & volume_confirmation
```

### 🔄 Mean Reversion

```python
def mean_reversion_signal(df):
    # RSI oversold
    df['RSI'] = calculate_rsi(df['close'])
    oversold = df['RSI'] < 30
    
    # Bollinger Band touch
    bb_touch = df['close'] <= df['BB_lower']
    
    # Volume spike
    volume_spike = df['volume'] > df['volume'].rolling(20).mean() * 1.5
    
    return oversold & bb_touch & volume_spike
```

## ⚖️ Risk Management Templates

### 🛡️ Stop Loss Types

```python
# ATR-based Stop
def atr_stop(entry_price, atr, multiplier=2, direction='long'):
    if direction == 'long':
        return entry_price - (atr * multiplier)
    else:
        return entry_price + (atr * multiplier)

# Percentage Stop
def percentage_stop(entry_price, percentage=0.05, direction='long'):
    if direction == 'long':
        return entry_price * (1 - percentage)
    else:
        return entry_price * (1 + percentage)

# Support/Resistance Stop
def technical_stop(entry_price, support_resistance_level, buffer=0.01):
    return support_resistance_level * (1 - buffer)
```

### 💰 Position Sizing Calculator

```python
class PositionSizer:
    def __init__(self, total_capital, max_risk_per_trade=0.02):
        self.capital = total_capital
        self.max_risk = max_risk_per_trade
    
    def calculate_shares(self, entry_price, stop_price):
        risk_per_share = abs(entry_price - stop_price)
        max_risk_amount = self.capital * self.max_risk
        max_shares = int(max_risk_amount / risk_per_share)
        
        # Máximo 50% del capital en una posición
        max_shares_by_capital = int((self.capital * 0.5) / entry_price)
        
        return min(max_shares, max_shares_by_capital)
```

## 🧪 Backtesting Quick Start

### 📊 Simple Backtest Framework

```python
class SimpleBacktester:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.positions = []
        self.trades = []
    
    def backtest_strategy(self, df, strategy_function):
        portfolio_value = [self.capital]
        
        for i in range(len(df)):
            current_data = df.iloc[:i+1]
            signal = strategy_function(current_data)
            
            if signal.iloc[-1] and not self.positions:
                # Enter position
                entry_price = df['close'].iloc[i]
                shares = self.calculate_position_size(entry_price)
                self.enter_position(entry_price, shares, i)
                
            elif self.positions:
                # Check exit conditions
                exit_signal = self.check_exit_conditions(df.iloc[i])
                if exit_signal:
                    self.exit_position(df['close'].iloc[i], i)
            
            # Update portfolio value
            current_value = self.calculate_portfolio_value(df['close'].iloc[i])
            portfolio_value.append(current_value)
        
        return portfolio_value
```

### 📈 Performance Calculator

```python
def calculate_performance_metrics(returns):
    total_return = (returns.iloc[-1] / returns.iloc[0]) - 1
    
    # Annualized return
    days = len(returns)
    annualized_return = (1 + total_return) ** (252/days) - 1
    
    # Volatility
    daily_returns = returns.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (assuming 2% risk-free rate)
    sharpe = (annualized_return - 0.02) / volatility
    
    # Max Drawdown
    peak = returns.expanding().max()
    drawdown = (returns - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}"
    }
```

## 🔧 Data Utilities

### 📊 Data Loading

```python
import yfinance as yf
import pandas as pd

# Get stock data
def get_stock_data(symbol, period='1y'):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    df.columns = df.columns.str.lower()
    return df

# Get multiple stocks
def get_multiple_stocks(symbols, period='1y'):
    data = {}
    for symbol in symbols:
        data[symbol] = get_stock_data(symbol, period)
    return data

# Basic data cleaning
def clean_data(df):
    # Remove weekends
    df = df[df.index.weekday < 5]
    
    # Remove outliers (more than 3 std from mean)
    for col in ['open', 'high', 'low', 'close']:
        mean = df[col].mean()
        std = df[col].std()
        df = df[abs(df[col] - mean) <= 3 * std]
    
    # Forward fill missing values
    df = df.fillna(method='ffill')
    
    return df
```

### 🔍 Market Screening

```python
# Screen for gap ups
def screen_gap_ups(symbols, min_gap=0.02):
    gap_stocks = []
    
    for symbol in symbols:
        df = get_stock_data(symbol, period='5d')
        if len(df) >= 2:
            gap = (df['open'].iloc[-1] / df['close'].iloc[-2]) - 1
            if gap >= min_gap:
                gap_stocks.append({
                    'symbol': symbol,
                    'gap_percentage': gap,
                    'volume': df['volume'].iloc[-1]
                })
    
    return sorted(gap_stocks, key=lambda x: x['gap_percentage'], reverse=True)
```

## 🎚️ Configuration Templates

### ⚙️ Strategy Parameters

```python
# Gap and Go Parameters
GAP_AND_GO_PARAMS = {
    'min_gap': 0.02,           # 2% minimum gap
    'min_volume_ratio': 2.0,   # 2x average volume
    'vwap_confirmation': True,  # Must be above VWAP
    'stop_loss_atr': 2.0,      # 2 ATR stop loss
    'take_profit_ratio': 2.0   # 2:1 reward:risk
}

# Mean Reversion Parameters  
MEAN_REVERSION_PARAMS = {
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'bb_periods': 20,
    'bb_std': 2.0,
    'volume_threshold': 1.5
}

# Risk Management
RISK_PARAMS = {
    'max_risk_per_trade': 0.02,  # 2% max risk
    'max_portfolio_risk': 0.06,  # 6% max total risk
    'max_position_size': 0.10,   # 10% max position
    'max_correlation': 0.7       # Max 70% correlation
}
```

### 📊 Timeframe Settings

```python
TIMEFRAMES = {
    'scalping': '1m',
    'day_trading': '5m',
    'swing_trading': '1h',
    'position_trading': '1d',
    'analysis': '1d'
}

MARKET_HOURS = {
    'premarket_start': '04:00',
    'market_open': '09:30',
    'market_close': '16:00',
    'after_hours_end': '20:00'
}
```

## 🚨 Error Handling

### 🛡️ Common Error Patterns

```python
def safe_execute_trade(trade_function, *args, **kwargs):
    try:
        return trade_function(*args, **kwargs)
    except ConnectionError:
        print("Connection lost - retrying...")
        time.sleep(5)
        return trade_function(*args, **kwargs)
    except InsufficientFundsError:
        print("Insufficient funds - reducing position size")
        kwargs['position_size'] *= 0.5
        return trade_function(*args, **kwargs)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Data validation
def validate_data(df):
    checks = {
        'has_data': len(df) > 0,
        'no_null_prices': df[['open', 'high', 'low', 'close']].isnull().sum().sum() == 0,
        'logical_prices': (df['high'] >= df['low']).all(),
        'positive_volume': (df['volume'] > 0).all()
    }
    
    if not all(checks.values()):
        raise ValueError(f"Data validation failed: {checks}")
    
    return True
```

## 📱 Cheat Codes

```python
# Quick profit/loss calculation
pnl = (exit_price - entry_price) * shares * (1 if direction == 'long' else -1)

# Quick percentage return
pct_return = (exit_price / entry_price - 1) * (1 if direction == 'long' else -1)

# Quick Sharpe ratio
sharpe = returns.mean() / returns.std() * np.sqrt(252)

# Quick max drawdown
max_dd = (portfolio_values / portfolio_values.cummax() - 1).min()

# Quick win rate
win_rate = (returns > 0).mean()
```

---

💡 **Pro Tip**: Bookmark esta página y úsala como referencia rápida mientras desarrollas y optimizas tus estrategias de trading.