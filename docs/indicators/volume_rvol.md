> 🇪🇸 [Leer en Español](volume_rvol.es.md) | 🇺🇸 **English**

# Volume and RVol (Relative Volume)

## The Fuel Behind the Move

In small caps, volume is EVERYTHING. Without volume, there is no movement. With excessive volume, there is opportunity. RVol (Relative Volume) is my #1 indicator for filtering which stocks to watch.

## RVol Calculation

```python
def calculate_rvol(df, lookback=10, time_based=True):
    """Calculate Relative Volume"""
    if time_based:
        # Time-of-day RVol (more precise)
        df['time'] = df.index.time
        
        # Average volume for each minute over the last N days
        volume_by_time = {}
        for i in range(1, lookback + 1):
            date = df.index[-1].date() - pd.Timedelta(days=i)
            hist_data = df[df.index.date == date]
            for idx, row in hist_data.iterrows():
                time_key = idx.time()
                if time_key not in volume_by_time:
                    volume_by_time[time_key] = []
                volume_by_time[time_key].append(row['volume'])
        
        # Average by time
        avg_volume_by_time = {
            time: np.mean(vols) for time, vols in volume_by_time.items()
        }
        
        # Calculate RVol
        df['avg_volume_time'] = df.index.time.map(avg_volume_by_time)
        df['rvol'] = df['volume'] / df['avg_volume_time']
        
    else:
        # Simple RVol (less precise but faster)
        df['avg_volume'] = df['volume'].rolling(lookback).mean().shift(1)
        df['rvol'] = df['volume'] / df['avg_volume']
    
    # Fill NaN with 1 (normal volume)
    df['rvol'] = df['rvol'].fillna(1)
    
    return df
```

## RVol for Day Trading

```python
def rvol_day_trading_setup(df, rvol_threshold=2):
    """Identify setups based on RVol"""
    df = calculate_rvol(df, time_based=True)
    
    # Classify RVol levels
    df['rvol_level'] = pd.cut(
        df['rvol'],
        bins=[0, 1, 2, 3, 5, 10, np.inf],
        labels=['low', 'normal', 'high', 'very_high', 'extreme', 'explosive']
    )
    
    # Detect volume spikes
    df['volume_spike'] = df['rvol'] > rvol_threshold
    df['sustained_volume'] = df['volume_spike'].rolling(5).sum() >= 3  # 3 of 5 bars
    
    # Combination with price
    df['bullish_volume'] = df['volume_spike'] & (df['close'] > df['open'])
    df['bearish_volume'] = df['volume_spike'] & (df['close'] < df['open'])
    
    return df
```

## Volume Profile

Where the most volume was executed = key levels.

```python
def calculate_volume_profile(df, bins=50):
    """Create volume profile"""
    # Define price bins
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_bins = np.linspace(price_min, price_max, bins)
    
    # Accumulate volume by price level
    volume_profile = np.zeros(len(price_bins) - 1)
    
    for idx, row in df.iterrows():
        # Distribute volume between low and high
        for i in range(len(price_bins) - 1):
            if price_bins[i] <= row['high'] and price_bins[i+1] >= row['low']:
                # Portion of the range that falls in this bin
                overlap_low = max(row['low'], price_bins[i])
                overlap_high = min(row['high'], price_bins[i+1])
                overlap_pct = (overlap_high - overlap_low) / (row['high'] - row['low'])
                volume_profile[i] += row['volume'] * overlap_pct
    
    # Create DataFrame
    vp_df = pd.DataFrame({
        'price': (price_bins[:-1] + price_bins[1:]) / 2,
        'volume': volume_profile
    })
    
    # Identify POC (Point of Control)
    vp_df['poc'] = vp_df['volume'] == vp_df['volume'].max()
    
    # Value Area (70% of volume)
    total_volume = vp_df['volume'].sum()
    vp_df = vp_df.sort_values('volume', ascending=False)
    vp_df['cum_volume'] = vp_df['volume'].cumsum()
    vp_df['in_value_area'] = vp_df['cum_volume'] <= total_volume * 0.7
    
    return vp_df.sort_values('price')
```

## Volume Patterns

```python
def detect_volume_patterns(df, window=20):
    """Detect important volume patterns"""
    # Prepare data
    df['volume_ma'] = df['volume'].rolling(window).mean()
    df['volume_std'] = df['volume'].rolling(window).std()
    
    # 1. Climax Volume
    df['climax_volume'] = df['volume'] > (df['volume_ma'] + 3 * df['volume_std'])
    
    # 2. Dry Up (volume drying up)
    df['volume_dryup'] = df['volume'] < df['volume_ma'] * 0.5
    
    # 3. Volume Divergence
    df['price_up'] = df['close'] > df['close'].shift(5)
    df['volume_down'] = df['volume'] < df['volume'].shift(5)
    df['bearish_divergence'] = df['price_up'] & df['volume_down']
    
    # 4. Accumulation/Distribution
    df['ad_line'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['ad_cumulative'] = df['ad_line'].cumsum()
    
    # 5. Volume Rate of Change
    df['volume_roc'] = (df['volume'] - df['volume'].shift(window)) / df['volume'].shift(window) * 100
    
    return df
```

## Smart Money Volume

Detect institutional vs retail volume.

```python
def analyze_smart_money_volume(df, block_size=10000):
    """Identify institutional volume"""
    # Approximate average trade size
    df['avg_trade_size'] = df['volume'] / df['tick_count'] if 'tick_count' in df else 100
    
    # Large blocks
    df['large_block'] = df['avg_trade_size'] > block_size
    
    # Volume in first and last hour (institutional)
    df['hour'] = df.index.hour
    df['institutional_hours'] = df['hour'].isin([9, 10, 15])
    
    # Dark pool prints (if data available)
    if 'dark_pool_volume' in df.columns:
        df['dark_pool_ratio'] = df['dark_pool_volume'] / df['volume']
        df['high_dark_pool'] = df['dark_pool_ratio'] > 0.3
    
    # Net buying pressure
    df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['selling_pressure'] = 1 - df['buying_pressure']
    
    # Money Flow
    df['money_flow'] = df['typical_price'] * df['volume']
    df['positive_money_flow'] = np.where(df['typical_price'] > df['typical_price'].shift(1), 
                                         df['money_flow'], 0)
    df['negative_money_flow'] = np.where(df['typical_price'] < df['typical_price'].shift(1), 
                                         df['money_flow'], 0)
    
    return df
```

## Volume Breakouts

```python
def volume_breakout_signals(df, volume_multiplier=3, price_threshold=0.02):
    """Detect breakouts with volume"""
    df = calculate_rvol(df)
    
    # Prepare indicators
    df['resistance'] = df['high'].rolling(20).max()
    df['support'] = df['low'].rolling(20).min()
    
    # Bullish breakout
    df['breakout_up'] = (
        (df['close'] > df['resistance'].shift(1)) &  # Breaks resistance
        (df['rvol'] > volume_multiplier) &           # With high volume
        (df['close'] > df['open'])                   # Green candle
    )
    
    # Bearish breakout
    df['breakout_down'] = (
        (df['close'] < df['support'].shift(1)) &
        (df['rvol'] > volume_multiplier) &
        (df['close'] < df['open'])
    )
    
    # Breakout strength
    df['breakout_strength'] = np.where(
        df['breakout_up'],
        (df['close'] - df['resistance'].shift(1)) / df['resistance'].shift(1) * 100,
        np.where(
            df['breakout_down'],
            (df['support'].shift(1) - df['close']) / df['support'].shift(1) * 100,
            0
        )
    )
    
    return df
```

## OBV (On Balance Volume)

```python
def calculate_obv_signals(df):
    """On Balance Volume with signals"""
    # Classic OBV
    df['obv'] = (df['volume'] * np.sign(df['close'] - df['close'].shift(1))).cumsum()
    
    # Smoothed OBV
    df['obv_ema'] = df['obv'].ewm(span=20).mean()
    
    # Divergences
    df['price_high'] = df['close'].rolling(20).max()
    df['obv_high'] = df['obv'].rolling(20).max()
    
    # Bearish divergence: price makes new high, OBV does not
    df['bearish_obv_divergence'] = (
        (df['close'] == df['price_high']) & 
        (df['obv'] < df['obv_high'].shift(20))
    )
    
    # Confirmation signal
    df['obv_trend_up'] = df['obv'] > df['obv_ema']
    df['obv_breakout'] = df['obv'] > df['obv'].rolling(50).max().shift(1)
    
    return df
```

## Volume-Weighted Momentum

```python
def volume_weighted_momentum(df, period=14):
    """Volume-weighted momentum"""
    # Price momentum
    df['price_change'] = df['close'] - df['close'].shift(period)
    
    # Volume-weighted price change
    weights = df['volume'].rolling(period).apply(lambda x: x / x.sum())
    df['vw_momentum'] = df['price_change'] * weights
    
    # Normalize
    df['vw_momentum_normalized'] = df['vw_momentum'] / df['close'] * 100
    
    # Signals
    df['strong_bullish_momentum'] = (
        (df['vw_momentum_normalized'] > 5) & 
        (df['rvol'] > 2)
    )
    
    return df
```

## Pre-Market Volume Analysis

```python
def analyze_premarket_volume(ticker, date):
    """Analyze pre-market volume"""
    # Get pre-market data
    pm_start = pd.Timestamp(date).replace(hour=4, minute=0)
    pm_end = pd.Timestamp(date).replace(hour=9, minute=29)
    
    pm_data = get_intraday_data(ticker, start=pm_start, end=pm_end)
    
    # Metrics
    analysis = {
        'total_pm_volume': pm_data['volume'].sum(),
        'pm_vwap': calculate_vwap(pm_data)['vwap'].iloc[-1],
        'pm_high': pm_data['high'].max(),
        'pm_low': pm_data['low'].min(),
        'pm_range': (pm_data['high'].max() - pm_data['low'].min()) / pm_data['low'].min() * 100,
        'volume_profile': pm_data.groupby(pd.cut(pm_data['close'], bins=10))['volume'].sum()
    }
    
    # Compare with previous days
    historical_pm_volume = []
    for i in range(1, 11):
        hist_date = date - pd.Timedelta(days=i)
        hist_pm = get_intraday_data(ticker, start=hist_date.replace(hour=4), 
                                   end=hist_date.replace(hour=9, minute=29))
        historical_pm_volume.append(hist_pm['volume'].sum())
    
    analysis['pm_rvol'] = analysis['total_pm_volume'] / np.mean(historical_pm_volume)
    
    return analysis
```

## Volume Alerts

```python
class VolumeAlertSystem:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.alerted = set()
        
    def check_alerts(self, ticker, current_data):
        alerts = []
        
        # Extreme RVol
        if current_data['rvol'] > self.thresholds['rvol_extreme']:
            alert_key = f"{ticker}_rvol_{current_data.name}"
            if alert_key not in self.alerted:
                alerts.append(f"🚨 {ticker}: RVol {current_data['rvol']:.1f}x @ ${current_data['close']:.2f}")
                self.alerted.add(alert_key)
        
        # Climax volume
        if current_data.get('climax_volume', False):
            alerts.append(f"💥 {ticker}: CLIMAX VOLUME - Possible top/bottom")
        
        # Volume dry up at support
        if current_data.get('volume_dryup', False) and current_data['close'] > current_data['support']:
            alerts.append(f"📉 {ticker}: Volume dry up at support ${current_data['support']:.2f}")
        
        return alerts
```

## Practical Tips

### 1. Morning Volume Rate
```python
def morning_volume_rate(df):
    """First hour volume rate"""
    first_hour = df.between_time('09:30', '10:30')
    first_hour_vol = first_hour['volume'].sum()
    
    avg_daily_vol = df.groupby(df.index.date)['volume'].sum().mean()
    
    # If first hour > 30% of average daily volume = active day
    return first_hour_vol / avg_daily_vol
```

### 2. Volume Patterns by Hour
```python
def hourly_volume_pattern(df):
    """Typical volume pattern by hour"""
    df['hour'] = df.index.hour
    hourly_avg = df.groupby('hour')['volume'].mean()
    
    # Normalize (first hour = 100)
    hourly_pattern = hourly_avg / hourly_avg[9] * 100
    return hourly_pattern
```

## Next Step

Let's continue with [Spike HOD/LOD](spike_hod_lod.md), critical for timing in small caps.
