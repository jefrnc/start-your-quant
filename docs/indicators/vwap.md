> 🇪🇸 [Leer en Español](vwap.es.md) | 🇺🇸 **English**

# VWAP and VWAP Reclaim

## What is VWAP?

Volume Weighted Average Price - the volume-weighted average price. It is essentially the "fair price" of the day based on where most volume was executed.

## Why It Matters in Small Caps

In small caps, VWAP is crucial because:
- Market makers use it as a reference
- Institutions measure their performance vs VWAP
- Acts as a magnet on high-volume days
- The "VWAP reclaim" is one of the most reliable setups

## Basic Calculation

```python
def calculate_vwap(df):
    """Calculate VWAP for intraday data"""
    # Typical price (more precise than just close)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Cumulative volume
    df['cum_volume'] = df['volume'].cumsum()
    
    # Cumulative price x volume
    df['cum_pv'] = (df['typical_price'] * df['volume']).cumsum()
    
    # VWAP
    df['vwap'] = df['cum_pv'] / df['cum_volume']
    
    return df
```

## VWAP with Standard Deviations

```python
def calculate_vwap_bands(df, num_std=2):
    """VWAP with standard deviation bands"""
    # First calculate VWAP
    df = calculate_vwap(df)
    
    # Calculate deviation
    df['vwap_variance'] = df['volume'] * (df['typical_price'] - df['vwap']) ** 2
    df['cum_variance'] = df['vwap_variance'].cumsum()
    df['vwap_std'] = np.sqrt(df['cum_variance'] / df['cum_volume'])
    
    # Bands
    df['vwap_upper'] = df['vwap'] + (num_std * df['vwap_std'])
    df['vwap_lower'] = df['vwap'] - (num_std * df['vwap_std'])
    
    return df
```

## VWAP Reclaim Setup

This is my favorite setup for small caps. When a stock loses VWAP and reclaims it with volume, it usually continues.

```python
def detect_vwap_reclaim(df, lookback=10, volume_threshold=1.5):
    """Detect VWAP reclaim setup"""
    # Calculate VWAP if it doesn't exist
    if 'vwap' not in df.columns:
        df = calculate_vwap(df)
    
    # Conditions for reclaim
    # 1. Was below VWAP
    df['below_vwap'] = df['low'] < df['vwap']
    
    # 2. Now above with volume
    df['above_vwap'] = df['close'] > df['vwap']
    df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * volume_threshold
    
    # 3. Reclaim = was below and now above with volume
    df['was_below'] = df['below_vwap'].rolling(lookback).max()
    df['vwap_reclaim'] = df['was_below'] & df['above_vwap'] & df['volume_spike']
    
    # Add reclaim strength
    df['reclaim_strength'] = np.where(
        df['vwap_reclaim'],
        (df['close'] - df['vwap']) / df['vwap'] * 100,
        0
    )
    
    return df
```

## Multi-Timeframe VWAP

```python
class MultiTimeframeVWAP:
    def __init__(self, df):
        self.df = df
        
    def add_daily_vwap(self):
        """Current day VWAP"""
        self.df['date'] = self.df.index.date
        daily_vwap = self.df.groupby('date').apply(calculate_vwap)
        self.df['daily_vwap'] = daily_vwap['vwap']
        
    def add_weekly_vwap(self):
        """Weekly VWAP"""
        self.df['week'] = self.df.index.isocalendar().week
        weekly_vwap = self.df.groupby('week').apply(calculate_vwap)
        self.df['weekly_vwap'] = weekly_vwap['vwap']
        
    def add_anchored_vwap(self, anchor_date):
        """Anchored VWAP from a specific date (e.g., from earnings)"""
        mask = self.df.index >= anchor_date
        anchored_data = self.df[mask].copy()
        anchored_data = calculate_vwap(anchored_data)
        self.df.loc[mask, 'anchored_vwap'] = anchored_data['vwap']
```

## VWAP for Gap Trading

```python
def vwap_gap_strategy(df, gap_threshold=10):
    """Strategy combining gaps and VWAP"""
    # Calculate gap
    df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # VWAP
    df = calculate_vwap(df)
    
    # Setup: Gap up + Hold above VWAP
    df['gap_up'] = df['gap_pct'] > gap_threshold
    df['holding_vwap'] = df['low'] > df['vwap']
    
    # Signal when gap up holds above VWAP
    df['signal'] = df['gap_up'] & df['holding_vwap']
    
    # Stop: Loss of VWAP
    df['stop_level'] = df['vwap'] * 0.99  # 1% below VWAP
    
    return df
```

## VWAP Magnets

On high-activity days, price tends to return to VWAP.

```python
def identify_vwap_magnet(df, distance_threshold=0.05):
    """Identify when price is very far from VWAP"""
    df = calculate_vwap(df)
    
    # Distance from VWAP
    df['distance_from_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Extremes
    df['extreme_above'] = df['distance_from_vwap'] > distance_threshold
    df['extreme_below'] = df['distance_from_vwap'] < -distance_threshold
    
    # Mean reversion signals
    df['short_signal'] = df['extreme_above'] & (df['volume'] > df['volume'].mean())
    df['long_signal'] = df['extreme_below'] & (df['volume'] > df['volume'].mean())
    
    return df
```

## VWAP Breaks with Volume Profile

```python
def vwap_volume_break(df, volume_multiplier=2):
    """VWAP break with volume confirmation"""
    df = calculate_vwap(df)
    
    # Volume profile
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['high_volume'] = df['volume'] > df['volume_ma'] * volume_multiplier
    
    # Breaks
    df['vwap_break_up'] = (df['close'] > df['vwap']) & (df['open'] < df['vwap'])
    df['vwap_break_down'] = (df['close'] < df['vwap']) & (df['open'] > df['vwap'])
    
    # Signals with volume
    df['bullish_break'] = df['vwap_break_up'] & df['high_volume']
    df['bearish_break'] = df['vwap_break_down'] & df['high_volume']
    
    # Add momentum
    df['break_momentum'] = np.where(
        df['bullish_break'],
        df['close'] - df['vwap'],
        np.where(df['bearish_break'], df['vwap'] - df['close'], 0)
    )
    
    return df
```

## VWAP for Risk Management

```python
class VWAPRiskManager:
    def __init__(self, position_type='long'):
        self.position_type = position_type
        
    def calculate_stop_loss(self, df, cushion=0.01):
        """VWAP-based stop loss"""
        if self.position_type == 'long':
            # Long: stop below VWAP
            df['stop_loss'] = df['vwap'] * (1 - cushion)
        else:
            # Short: stop above VWAP
            df['stop_loss'] = df['vwap'] * (1 + cushion)
            
        return df
    
    def position_health(self, df):
        """Evaluate position health vs VWAP"""
        if self.position_type == 'long':
            df['position_health'] = np.where(
                df['close'] > df['vwap'],
                'healthy',
                'warning'
            )
        else:
            df['position_health'] = np.where(
                df['close'] < df['vwap'],
                'healthy',
                'warning'
            )
            
        return df
```

## Backtesting VWAP Strategies

```python
def backtest_vwap_reclaim(df, initial_capital=10000):
    """Simple VWAP reclaim backtest"""
    df = detect_vwap_reclaim(df)
    
    # Trading logic
    position = 0
    cash = initial_capital
    trades = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Entry
        if row['vwap_reclaim'] and position == 0:
            shares = int(cash * 0.95 / row['close'])  # 95% of capital
            position = shares
            cash -= shares * row['close']
            
            trades.append({
                'date': row.name,
                'action': 'buy',
                'price': row['close'],
                'shares': shares,
                'reason': 'vwap_reclaim'
            })
        
        # Exit
        elif position > 0:
            # Stop loss: lost VWAP
            if row['close'] < row['vwap'] * 0.99:
                cash += position * row['close']
                trades.append({
                    'date': row.name,
                    'action': 'sell',
                    'price': row['close'],
                    'shares': position,
                    'reason': 'stop_loss'
                })
                position = 0
                
            # Take profit: 5% gain
            elif row['close'] > trades[-1]['price'] * 1.05:
                cash += position * row['close']
                trades.append({
                    'date': row.name,
                    'action': 'sell',
                    'price': row['close'],
                    'shares': position,
                    'reason': 'take_profit'
                })
                position = 0
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 1:
        wins = trades_df[trades_df['reason'].isin(['take_profit'])].shape[0]
        total_trades = len(trades_df) // 2  # Buy + Sell
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        final_value = cash + (position * df.iloc[-1]['close'] if position > 0 else 0)
        total_return = (final_value - initial_capital) / initial_capital
        
        return {
            'trades': trades_df,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_value': final_value
        }
```

## Real Trading Tips

### 1. Pre-Market VWAP
```python
def calculate_premarket_vwap(df):
    """Pre-market only VWAP for reference"""
    premarket = df.between_time('04:00', '09:29')
    return calculate_vwap(premarket)
```

### 2. VWAP Speed
```python
def vwap_acceleration(df, period=5):
    """How fast price moves relative to VWAP"""
    df['vwap_speed'] = (df['close'] - df['vwap']).diff(period)
    df['accelerating'] = df['vwap_speed'] > 0
    return df
```

### 3. Multi-Day VWAP Levels
```python
def key_vwap_levels(ticker, lookback_days=20):
    """Important VWAP levels from previous days"""
    levels = {}
    for i in range(lookback_days):
        date = pd.Timestamp.now() - pd.Timedelta(days=i)
        daily_data = get_intraday_data(ticker, date)
        vwap = calculate_vwap(daily_data)
        levels[date] = vwap.iloc[-1]
    
    return pd.Series(levels).sort_values()
```

## Real-Time Alerts

```python
class VWAPAlerts:
    def __init__(self, ticker):
        self.ticker = ticker
        self.alerted = set()
        
    def check_alerts(self, current_bar):
        alerts = []
        
        # VWAP reclaim
        if current_bar['vwap_reclaim'] and 'reclaim' not in self.alerted:
            alerts.append(f"{self.ticker}: VWAP RECLAIM @ ${current_bar['close']:.2f}")
            self.alerted.add('reclaim')
            
        # VWAP rejection
        if current_bar['high'] > current_bar['vwap'] and current_bar['close'] < current_bar['vwap']:
            alerts.append(f"{self.ticker}: VWAP REJECTION @ ${current_bar['vwap']:.2f}")
            
        return alerts
```

## Next Step

Let's continue with [Moving Averages](moving_averages.md) and how to combine them with VWAP.
