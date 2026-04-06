> 🇪🇸 [Leer en Español](bollinger_bands.es.md) | 🇺🇸 **English**

# Bollinger Bands - The Volatility Oscillator

## Definition

Bollinger Bands are a technical analysis tool for generating overbought or oversold signals. They consist of three lines: a simple moving average (middle band) and an upper and lower band at +/- 2 standard deviations.

## Indicator Philosophy

### Why Do They Work?
- **Mean Reversion**: Prices tend to return to the average after extreme deviations
- **Volatility Measurement**: Bands expand/contract based on market volatility
- **Dynamic Levels**: Unlike fixed support/resistance, they adapt to price

### Components
```
Upper_Band  = MA + (Standard_Deviation × Factor)
Middle_Band = Simple Moving Average
Lower_Band  = MA - (Standard_Deviation × Factor)
```

**Standard parameters**:
- **Period**: 20 (moving average)
- **Factor**: 2.0 (standard deviations)

## Reference Implementation

```python
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def Bollinger_Bands(df: pd.DataFrame, length: int = 20, std_deviation: float = 2.0, column: str = "Close") -> pd.DataFrame:
    """
    Bollinger Bands - Exact reference implementation
    
    Bollinger Bands identify overbought/oversold levels by measuring
    the deviation of price from its moving average.
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical asset data (must include Close column)
    length : int, default 20
        Window for moving average and standard deviation calculation
    std_deviation : float, default 2.0
        Number of standard deviations for upper and lower bands
    column : str, default "Close"
        Column to use in the calculation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: BB_Up, MA, BB_Lw
        
    Usage Example
    --------------
    >>> import yfinance as yf
    >>> df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
    >>> bb = Bollinger_Bands(df, length=20, std_deviation=2.0)
    >>> print(bb.head())
    """
    # Calculate using a copy to avoid modifying the original
    data = deepcopy(df)
    
    # Rolling window for mean and standard deviation
    rolling = data[column].rolling(window=length)
    
    # Moving average (middle band)
    data["MA"] = rolling.mean()
    
    # Standard deviation with ddof=0 (full population)
    std_bands = std_deviation * rolling.std(ddof=0)
    
    # Upper and lower bands
    data["BB_Up"] = data["MA"] + std_bands
    data["BB_Lw"] = data["MA"] - std_bands
    
    return data[["BB_Up", "MA", "BB_Lw"]]

def calculate_bollinger_signals(df: pd.DataFrame, bb_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals using Bollinger Bands
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical data with price
    bb_data : pd.DataFrame
        Bollinger Bands data (output of Bollinger_Bands)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with signals and statistics
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['bb_upper'] = bb_data['BB_Up']
    signals['bb_middle'] = bb_data['MA']
    signals['bb_lower'] = bb_data['BB_Lw']
    
    # Calculate relative position within the bands
    bb_width = bb_data['BB_Up'] - bb_data['BB_Lw']
    signals['bb_position'] = (df['Close'] - bb_data['BB_Lw']) / bb_width
    
    # Basic signals
    signals['oversold'] = df['Close'] < bb_data['BB_Lw']  # Price below lower band
    signals['overbought'] = df['Close'] > bb_data['BB_Up']  # Price above upper band
    signals['middle_cross'] = np.where(
        (df['Close'] > bb_data['MA']) & (df['Close'].shift(1) <= bb_data['MA'].shift(1)), 1,
        np.where((df['Close'] < bb_data['MA']) & (df['Close'].shift(1) >= bb_data['MA'].shift(1)), -1, 0)
    )
    
    # Squeeze detection (contracted bands = low volatility)
    signals['bb_width'] = bb_width
    signals['bb_squeeze'] = bb_width < bb_width.rolling(20).mean() * 0.8
    
    # Breakout signals
    signals['upper_breakout'] = (df['Close'] > bb_data['BB_Up']) & (df['Close'].shift(1) <= bb_data['BB_Up'].shift(1))
    signals['lower_breakout'] = (df['Close'] < bb_data['BB_Lw']) & (df['Close'].shift(1) >= bb_data['BB_Lw'].shift(1))
    
    return signals
```

## Trading Strategies with Bollinger Bands

### 1. Mean Reversion Strategy
```python
def bollinger_mean_reversion(df: pd.DataFrame, bb_period: int = 20, std_factor: float = 2.0):
    """
    Mean reversion strategy using Bollinger Bands
    """
    # Calculate bands
    bb = Bollinger_Bands(df, length=bb_period, std_deviation=std_factor)
    signals = calculate_bollinger_signals(df, bb)
    
    # Entry signals
    entry_signals = pd.Series(0, index=df.index)
    
    # Long when price touches lower band (oversold)
    long_entry = (
        signals['oversold'] & 
        (df['Volume'] > df['Volume'].rolling(20).mean()) &  # Confirm with volume
        (df['Close'] > df['Close'].shift(1))  # Price starting to recover
    )
    
    # Short when price touches upper band (overbought)
    short_entry = (
        signals['overbought'] &
        (df['Volume'] > df['Volume'].rolling(20).mean()) &
        (df['Close'] < df['Close'].shift(1))  # Price starting to fall
    )
    
    entry_signals[long_entry] = 1
    entry_signals[short_entry] = -1
    
    return {
        'signals': entry_signals,
        'bb_data': bb,
        'analysis': signals,
        'strategy_type': 'mean_reversion'
    }

def bollinger_breakout_strategy(df: pd.DataFrame, bb_period: int = 20, std_factor: float = 2.0):
    """
    Breakout strategy using Bollinger Bands
    """
    bb = Bollinger_Bands(df, length=bb_period, std_deviation=std_factor)
    signals = calculate_bollinger_signals(df, bb)
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Long on bullish breakout after squeeze
    long_breakout = (
        signals['upper_breakout'] &
        signals['bb_squeeze'].shift(1) &  # Previous squeeze present
        (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5)  # Strong volume
    )
    
    # Short on bearish breakout after squeeze
    short_breakout = (
        signals['lower_breakout'] &
        signals['bb_squeeze'].shift(1) &
        (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5)
    )
    
    entry_signals[long_breakout] = 1
    entry_signals[short_breakout] = -1
    
    return {
        'signals': entry_signals,
        'bb_data': bb,
        'analysis': signals,
        'strategy_type': 'breakout'
    }
```

### 2. Small Caps Specific Strategy
```python
def small_cap_bollinger_strategy(df: pd.DataFrame, volume_data: pd.DataFrame = None):
    """
    Small caps specific strategy using Bollinger Bands
    """
    # Parameters adjusted for small caps (higher volatility)
    bb = Bollinger_Bands(df, length=15, std_deviation=2.5)  # Wider bands
    signals = calculate_bollinger_signals(df, bb)
    
    # Add small cap specific filters
    if volume_data is not None:
        # RVOL filter
        avg_volume = df['Volume'].rolling(20).mean()
        rvol = df['Volume'] / avg_volume
        
        # Gap detection
        gap_up = (df['Open'] / df['Close'].shift(1)) > 1.02  # 2% gap up
        gap_down = (df['Open'] / df['Close'].shift(1)) < 0.98  # 2% gap down
        
        signals['rvol'] = rvol
        signals['gap_up'] = gap_up
        signals['gap_down'] = gap_down
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Long setup: Oversold + High RVOL + Gap fill potential
    long_setup = (
        signals['oversold'] &
        (signals.get('rvol', 1) > 2) &  # High volume
        signals.get('gap_down', False)  # Gap down being filled
    )
    
    # Short setup: Overbought + High RVOL + Gap fade
    short_setup = (
        signals['overbought'] &
        (signals.get('rvol', 1) > 2) &
        signals.get('gap_up', False)  # Gap up fading
    )
    
    entry_signals[long_setup] = 1
    entry_signals[short_setup] = -1
    
    return {
        'signals': entry_signals,
        'bb_data': bb,
        'analysis': signals,
        'strategy_type': 'small_cap_specialized'
    }
```

## Interpretation and Usage

### Main Signals

1. **Oversold**
   - Price touches or crosses lower band
   - Potential bullish reversal
   - ⚠️ Confirm with volume and momentum

2. **Overbought**
   - Price touches or crosses upper band  
   - Potential bearish reversal
   - ⚠️ In a strong trend it can extend

3. **Squeeze (Compression)**
   - Bands very close = low volatility
   - Precedes explosive moves
   - 🎯 Ideal setup for breakouts

### Powerful Combinations

```python
def bollinger_multi_timeframe(symbol: str, primary_tf: str = '1d', confirmation_tf: str = '1h'):
    """
    Multi-timeframe analysis with Bollinger Bands
    """
    import yfinance as yf
    
    # Data in multiple timeframes
    df_daily = yf.download(symbol, period="6mo", interval=primary_tf)
    df_hourly = yf.download(symbol, period="1mo", interval=confirmation_tf)
    
    # Bollinger on each timeframe
    bb_daily = Bollinger_Bands(df_daily, length=20, std_deviation=2.0)
    bb_hourly = Bollinger_Bands(df_hourly, length=20, std_deviation=2.0)
    
    # Combined signals
    daily_signals = calculate_bollinger_signals(df_daily, bb_daily)
    hourly_signals = calculate_bollinger_signals(df_hourly, bb_hourly)
    
    # Multi-timeframe setup
    current_daily = daily_signals.iloc[-1]
    current_hourly = hourly_signals.iloc[-1]
    
    analysis = {
        'daily_position': current_daily['bb_position'],
        'hourly_position': current_hourly['bb_position'],
        'daily_squeeze': current_daily['bb_squeeze'],
        'hourly_squeeze': current_hourly['bb_squeeze'],
        'confluence': None
    }
    
    # Detect confluence
    if current_daily['oversold'] and current_hourly['oversold']:
        analysis['confluence'] = 'STRONG_OVERSOLD'
    elif current_daily['overbought'] and current_hourly['overbought']:
        analysis['confluence'] = 'STRONG_OVERBOUGHT'
    elif current_daily['bb_squeeze'] and current_hourly['bb_squeeze']:
        analysis['confluence'] = 'MULTI_TF_SQUEEZE'
    
    return analysis
```

## Advanced Visualization

```python
def plot_bollinger_analysis(df: pd.DataFrame, bb_data: pd.DataFrame, signals: pd.DataFrame, title: str = "Bollinger Bands Analysis"):
    """
    Create complete Bollinger analysis chart
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), 
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Chart 1: Price + Bollinger Bands
    ax1.plot(df.index, df['Close'], 'k-', linewidth=2, label='Price')
    ax1.plot(bb_data.index, bb_data['BB_Up'], 'r--', label='Upper Band')
    ax1.plot(bb_data.index, bb_data['MA'], 'b-', label='Middle Band (MA)')
    ax1.plot(bb_data.index, bb_data['BB_Lw'], 'g--', label='Lower Band')
    ax1.fill_between(bb_data.index, bb_data['BB_Up'], bb_data['BB_Lw'], alpha=0.1, color='gray')
    
    # Mark signals
    oversold_points = df.index[signals['oversold']]
    overbought_points = df.index[signals['overbought']]
    
    ax1.scatter(oversold_points, df.loc[oversold_points, 'Close'], 
               color='green', marker='^', s=100, label='Oversold', zorder=5)
    ax1.scatter(overbought_points, df.loc[overbought_points, 'Close'], 
               color='red', marker='v', s=100, label='Overbought', zorder=5)
    
    ax1.set_title(f'{title} - Price & Bollinger Bands')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: BB Position (position within the bands)
    ax2.plot(signals.index, signals['bb_position'], 'purple', linewidth=2)
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Lower Band')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Upper Band')
    ax2.axhline(y=0.5, color='blue', linestyle='-', alpha=0.7, label='Middle')
    ax2.fill_between(signals.index, 0, 0.2, alpha=0.2, color='green', label='Oversold Zone')
    ax2.fill_between(signals.index, 0.8, 1, alpha=0.2, color='red', label='Overbought Zone')
    ax2.set_title('BB Position (0=Lower Band, 1=Upper Band)')
    ax2.set_ylabel('Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: BB Width (volatility)
    bb_width_norm = signals['bb_width'] / signals['bb_width'].rolling(50).mean()
    ax3.plot(signals.index, bb_width_norm, 'orange', linewidth=2)
    ax3.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Average')
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Squeeze Threshold')
    ax3.fill_between(signals.index, 0, 0.8, where=(bb_width_norm < 0.8), 
                    alpha=0.3, color='yellow', label='Squeeze Zone')
    ax3.set_title('BB Width (Normalized) - Volatility Measure')
    ax3.set_ylabel('Width Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Date formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

# Complete usage example
def bollinger_complete_example():
    """
    Complete analysis example with Bollinger Bands
    """
    import yfinance as yf
    
    # Get data
    ticker = "AAPL"
    df = yf.download(ticker, start="2023-01-01", end="2024-01-01", interval="1d")
    
    print(f"=== BOLLINGER BANDS ANALYSIS: {ticker} ===\n")
    
    # Calculate Bollinger Bands
    bb = Bollinger_Bands(df, length=20, std_deviation=2.0)
    signals = calculate_bollinger_signals(df, bb)
    
    # Statistics
    oversold_count = signals['oversold'].sum()
    overbought_count = signals['overbought'].sum()
    squeeze_count = signals['bb_squeeze'].sum()
    
    print(f"PERIOD STATISTICS:")
    print(f"   Oversold Signals: {oversold_count}")
    print(f"   Overbought Signals: {overbought_count}")
    print(f"   Days in Squeeze: {squeeze_count}")
    print(f"   % Time in Squeeze: {squeeze_count/len(signals)*100:.1f}%")
    
    # Current analysis
    current = signals.iloc[-1]
    print(f"\nCURRENT ANALYSIS:")
    print(f"   Price: ${current['price']:.2f}")
    print(f"   Upper Band: ${current['bb_upper']:.2f}")
    print(f"   Middle Band: ${current['bb_middle']:.2f}")
    print(f"   Lower Band: ${current['bb_lower']:.2f}")
    print(f"   BB Position: {current['bb_position']:.2f} (0=Lower, 1=Upper)")
    
    if current['bb_position'] < 0.2:
        print("   Status: OVERSOLD - Possible bounce")
    elif current['bb_position'] > 0.8:
        print("   Status: OVERBOUGHT - Possible correction")
    elif current['bb_squeeze']:
        print("   Status: SQUEEZE - Preparing for move")
    else:
        print("   Status: NEUTRAL")
    
    # Create chart
    plot_bollinger_analysis(df, bb, signals, f"Bollinger Analysis - {ticker}")
    
    return bb, signals

# Run example if executed directly
if __name__ == "__main__":
    bollinger_complete_example()
```

## Best Practices

### ✅ **Do's**

1. **Confirm with volume**: More reliable signals with elevated volume
2. **Use multiple timeframes**: Confluence increases probability
3. **Leverage squeeze**: Low volatility precedes high volatility
4. **Combine with trend**: In a strong trend, oversold/overbought are less reliable

### ❌ **Don'ts**

1. **Don't trade on touch alone**: Wait for reversal confirmation
2. **Don't ignore context**: A band is not absolute support/resistance
3. **Don't use fixed parameters**: Adjust based on asset volatility
4. **Don't trade against strong trends**: BB works best in ranges

### 🎯 **Parameters for Small Caps**

```python
SMALL_CAP_BB_PARAMS = {
    'gap_and_go': {'period': 15, 'std': 2.5},      # Wider for gaps
    'momentum': {'period': 20, 'std': 2.0},        # Standard
    'reversal': {'period': 25, 'std': 1.8},        # More sensitive
    'breakout': {'period': 10, 'std': 3.0}         # Very wide to avoid whipsaws
}
```

## Next Step

With Bollinger Bands mastered, let's continue with [Parabolic SAR](parabolic_sar.md) to identify trend changes.
