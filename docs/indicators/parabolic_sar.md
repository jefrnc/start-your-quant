> 🇪🇸 [Leer en Español](parabolic_sar.es.md) | 🇺🇸 **English**

# Parabolic SAR - The Stop and Reverse System

## Definition

The Parabolic SAR (Stop and Reverse) is an indicator that determines trend direction and potential reversals using a system of dots that follow price, functioning as dynamic trailing stops.

## Indicator Philosophy

### Why Does It Work?
- **Dynamic Trailing Stop**: Adjusts automatically based on momentum
- **Progressive Acceleration**: Factor that increases over time in a trend
- **Clear Signals**: Position change = trend change

### Key Concepts
- **SAR**: Current stop value
- **Acceleration Factor (AF)**: Progressive increment (0.02 by default)
- **Max Step**: Maximum AF limit (0.20 by default)

## Reference Implementation

```python
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def Parabolic_SAR(df: pd.DataFrame, increment: float = 0.02, max_step: float = 0.20) -> pd.DataFrame:
    """
    Parabolic SAR - Exact reference implementation
    
    The SAR indicator uses a stop and reverse system to identify
    entry and exit points based on price acceleration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical asset data (must include High, Low, Close)
    increment : float, default 0.02
        Initial increment of the acceleration factor (Alpha)
    max_step : float, default 0.20
        Maximum step the acceleration factor can reach
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: PSAR, UpTrend, DownTrend
        
    How to Trade It
    ---------------
    - BUY Signal: Dots change from above to below price
    - SELL Signal: Dots change from below to above price
    - Trailing Stop: Use PSAR as a dynamic stop loss
    
    Usage Example
    --------------
    >>> df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
    >>> psar = Parabolic_SAR(df, increment=0.02, max_step=0.20)
    >>> print(psar.head())
    """
    # Work with a copy to avoid modifying the original
    data = deepcopy(df)
    High, Low, Close = data["High"].values, data["Low"].values, data["Close"].values
    
    # Initialize arrays for trends
    psar_up = np.repeat(np.nan, Close.shape[0])
    psar_down = np.repeat(np.nan, Close.shape[0])
    
    # Initial state variables
    up_trend = True  # Start in uptrend
    up_trend_high = High[0]  # High during uptrend
    down_trend_low = Low[0]  # Low during downtrend
    acc_factor = increment  # Initial acceleration factor
    
    # Calculate PSAR for each point
    for i in range(2, Close.shape[0]):
        reversal = False
        max_high = High[i]
        min_low = Low[i]
        
        # === UPTREND ===
        if up_trend:
            # Calculate new PSAR for uptrend
            # PSAR = Previous_PSAR + AF * (EP - Previous_PSAR)
            Close[i] = Close[i - 1] + (acc_factor * (up_trend_high - Close[i - 1]))
            
            # Check if reversal occurs (price breaks PSAR)
            if min_low < Close[i]:
                reversal = True
                Close[i] = up_trend_high  # PSAR becomes the previous high
                down_trend_low = min_low  # New extreme point for downtrend
                acc_factor = increment  # Reset acceleration factor
            else:
                # Update high and accelerate if new high
                if max_high > up_trend_high:
                    up_trend_high = max_high
                    acc_factor = min(acc_factor + increment, max_step)
                
                # SAR rule: Cannot be higher than low of previous periods
                low1 = Low[i - 1]
                low2 = Low[i - 2]
                if low2 < Close[i]:
                    Close[i] = low2
                elif low1 < Close[i]:
                    Close[i] = low1
        
        # === DOWNTREND ===
        else:
            # Calculate new PSAR for downtrend
            Close[i] = Close[i - 1] - (acc_factor * (Close[i - 1] - down_trend_low))
            
            # Check if reversal occurs (price breaks PSAR)
            if max_high > Close[i]:
                reversal = True
                Close[i] = down_trend_low  # PSAR becomes the previous low
                up_trend_high = max_high  # New extreme point for uptrend
                acc_factor = increment  # Reset acceleration factor
            else:
                # Update low and accelerate if new low
                if min_low < down_trend_low:
                    down_trend_low = min_low
                    acc_factor = min(acc_factor + increment, max_step)
                
                # SAR rule: Cannot be lower than high of previous periods
                high1 = High[i - 1]
                high2 = High[i - 2]
                if high2 > Close[i]:
                    Close[i] = high2
                elif high1 > Close[i]:
                    Close[i] = high1
        
        # Update trend direction
        up_trend = up_trend != reversal  # XOR logic for state change
        
        # Assign dots based on trend
        if up_trend:
            psar_up[i] = Close[i]
        else:
            psar_down[i] = Close[i]
    
    # Create result DataFrame
    data["PSAR"] = Close
    data["UpTrend"] = psar_up
    data["DownTrend"] = psar_down
    
    return data[["PSAR", "UpTrend", "DownTrend"]]

def analyze_psar_signals(df: pd.DataFrame, psar_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze Parabolic SAR trading signals
    
    Parameters
    ----------
    df : pd.DataFrame
        Original historical data
    psar_data : pd.DataFrame
        PSAR data (output of Parabolic_SAR)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with signals and analysis
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['high'] = df['High']
    signals['low'] = df['Low']
    signals['psar'] = psar_data['PSAR']
    signals['up_trend'] = psar_data['UpTrend']
    signals['down_trend'] = psar_data['DownTrend']
    
    # Detect trend changes
    signals['trend'] = np.where(~pd.isna(psar_data['UpTrend']), 1, -1)
    signals['trend_change'] = signals['trend'].diff().fillna(0)
    
    # Entry signals
    signals['buy_signal'] = signals['trend_change'] == 2    # From bearish to bullish
    signals['sell_signal'] = signals['trend_change'] == -2  # From bullish to bearish
    
    # Distance from price to SAR (momentum indicator)
    signals['price_sar_distance'] = np.where(
        signals['trend'] == 1,
        (signals['price'] - signals['psar']) / signals['price'],  # Bullish: price above SAR
        (signals['psar'] - signals['price']) / signals['price']   # Bearish: SAR above price
    )
    
    # Trend strength (based on duration)
    trend_length = signals.groupby((signals['trend'] != signals['trend'].shift()).cumsum()).cumcount() + 1
    signals['trend_strength'] = trend_length
    
    # Signal quality
    signals['signal_quality'] = 'NONE'
    
    # High quality signals
    high_quality_buy = (
        signals['buy_signal'] &
        (signals['trend_strength'].shift(1) > 5) &  # Lasting previous bearish trend
        (df['Volume'] > df['Volume'].rolling(20).mean())  # Confirmatory volume
    )
    
    high_quality_sell = (
        signals['sell_signal'] &
        (signals['trend_strength'].shift(1) > 5) &  # Lasting previous bullish trend
        (df['Volume'] > df['Volume'].rolling(20).mean())
    )
    
    signals.loc[high_quality_buy, 'signal_quality'] = 'HIGH_QUALITY_BUY'
    signals.loc[high_quality_sell, 'signal_quality'] = 'HIGH_QUALITY_SELL'
    signals.loc[signals['buy_signal'] & ~high_quality_buy, 'signal_quality'] = 'MEDIUM_BUY'
    signals.loc[signals['sell_signal'] & ~high_quality_sell, 'signal_quality'] = 'MEDIUM_SELL'
    
    return signals
```

## Trading Strategies with PSAR

### 1. Trending Strategy
```python
def psar_trending_strategy(df: pd.DataFrame, af_increment: float = 0.02, af_max: float = 0.20):
    """
    Trend following strategy using PSAR
    """
    # Calculate PSAR
    psar = Parabolic_SAR(df, increment=af_increment, max_step=af_max)
    signals = analyze_psar_signals(df, psar)
    
    # Additional filters to improve signals
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    
    # Only trade in the direction of the larger trend
    bullish_context = sma_50 > sma_200
    bearish_context = sma_50 < sma_200
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Long entries
    long_entry = (
        signals['buy_signal'] &
        bullish_context &
        (signals['signal_quality'].isin(['HIGH_QUALITY_BUY', 'MEDIUM_BUY'])) &
        (df['Close'] > sma_50)  # Price above moving average
    )
    
    # Short entries  
    short_entry = (
        signals['sell_signal'] &
        bearish_context &
        (signals['signal_quality'].isin(['HIGH_QUALITY_SELL', 'MEDIUM_SELL'])) &
        (df['Close'] < sma_50)  # Price below moving average
    )
    
    entry_signals[long_entry] = 1
    entry_signals[short_entry] = -1
    
    return {
        'signals': entry_signals,
        'psar_data': psar,
        'analysis': signals,
        'strategy_type': 'trending'
    }

def psar_scalping_strategy(df: pd.DataFrame, timeframe: str = '15min'):
    """
    Scalping strategy using PSAR for small caps
    """
    # More sensitive PSAR for scalping
    psar = Parabolic_SAR(df, increment=0.01, max_step=0.10)  # More conservative
    signals = analyze_psar_signals(df, psar)
    
    # Specific filters for scalping
    atr = calculate_atr(df, period=14)
    bb = Bollinger_Bands(df, length=20, std_deviation=2.0)
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Long scalping setup
    long_scalp = (
        signals['buy_signal'] &
        (atr > atr.rolling(20).mean() * 1.2) &  # Elevated volatility
        (df['Close'] > bb['MA']) &  # Above Bollinger middle band
        (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5)  # Strong volume
    )
    
    # Short scalping setup
    short_scalp = (
        signals['sell_signal'] &
        (atr > atr.rolling(20).mean() * 1.2) &
        (df['Close'] < bb['MA']) &
        (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5)
    )
    
    entry_signals[long_scalp] = 1
    entry_signals[short_scalp] = -1
    
    return {
        'signals': entry_signals,
        'psar_data': psar,
        'analysis': signals,
        'strategy_type': 'scalping',
        'stop_loss': signals['psar']  # PSAR as dynamic stop
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Helper function to calculate ATR"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(period).mean()
```

### 2. Small Cap Gap & Go with PSAR
```python
def gap_and_go_psar_strategy(df: pd.DataFrame, gap_threshold: float = 0.05):
    """
    Combine Gap & Go with PSAR for small caps
    """
    # Detect gaps
    gap_up = (df['Open'] / df['Close'].shift(1) - 1) > gap_threshold
    gap_down = (df['Open'] / df['Close'].shift(1) - 1) < -gap_threshold
    
    # Adaptive PSAR for gaps
    psar = Parabolic_SAR(df, increment=0.03, max_step=0.25)  # More aggressive
    signals = analyze_psar_signals(df, psar)
    
    # Pre-market high/low simulation (using intraday data if available)
    premarket_high = df['High'].rolling(3).max()  # Approximation
    premarket_low = df['Low'].rolling(3).min()
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Gap up continuation with PSAR
    gap_up_continuation = (
        gap_up &
        signals['buy_signal'] &
        (df['Close'] > premarket_high) &  # Break of pre-market high
        (df['Volume'] > df['Volume'].rolling(20).mean() * 3)  # Heavy volume
    )
    
    # Gap down reversal with PSAR
    gap_down_reversal = (
        gap_down &
        signals['buy_signal'] &
        (df['Close'] > df['Open']) &  # Green candle after gap down
        (df['Volume'] > df['Volume'].rolling(20).mean() * 2)
    )
    
    # Gap fade with PSAR
    gap_fade = (
        gap_up &
        signals['sell_signal'] &
        (df['Close'] < df['Open']) &  # Red candle after gap up
        (signals['trend_strength'].shift(1) <= 3)  # Short uptrend before reversal
    )
    
    entry_signals[gap_up_continuation] = 1
    entry_signals[gap_down_reversal] = 1
    entry_signals[gap_fade] = -1
    
    return {
        'signals': entry_signals,
        'psar_data': psar,
        'analysis': signals,
        'gap_signals': {
            'gap_up': gap_up,
            'gap_down': gap_down,
            'gap_up_continuation': gap_up_continuation,
            'gap_down_reversal': gap_down_reversal,
            'gap_fade': gap_fade
        },
        'strategy_type': 'gap_and_go_psar'
    }
```

## Parameter Optimization

### Adaptive Parameters
```python
def adaptive_psar_parameters(df: pd.DataFrame, volatility_period: int = 20):
    """
    Calculate adaptive PSAR parameters based on volatility
    """
    # Measure current volatility
    returns = df['Close'].pct_change()
    rolling_vol = returns.rolling(volatility_period).std() * np.sqrt(252)
    current_vol = rolling_vol.iloc[-1]
    
    # Base parameters
    base_increment = 0.02
    base_max = 0.20
    
    # Adjust based on volatility
    if current_vol > 0.5:  # High volatility
        increment = base_increment * 0.5  # More conservative
        max_step = base_max * 0.75
        regime = "HIGH_VOLATILITY"
    elif current_vol < 0.2:  # Low volatility
        increment = base_increment * 1.5  # More aggressive
        max_step = base_max * 1.25
        regime = "LOW_VOLATILITY"
    else:  # Normal volatility
        increment = base_increment
        max_step = base_max
        regime = "NORMAL_VOLATILITY"
    
    return {
        'increment': increment,
        'max_step': max_step,
        'volatility': current_vol,
        'regime': regime
    }

def multi_timeframe_psar(symbol: str, primary_tf: str = '1d', secondary_tf: str = '1h'):
    """
    Multi-timeframe PSAR analysis
    """
    import yfinance as yf
    
    # Get data
    df_primary = yf.download(symbol, period="3mo", interval=primary_tf)
    df_secondary = yf.download(symbol, period="1mo", interval=secondary_tf)
    
    # PSAR on each timeframe
    psar_primary = Parabolic_SAR(df_primary)
    psar_secondary = Parabolic_SAR(df_secondary)
    
    signals_primary = analyze_psar_signals(df_primary, psar_primary)
    signals_secondary = analyze_psar_signals(df_secondary, psar_secondary)
    
    # Current state on both timeframes
    current_primary = signals_primary.iloc[-1]
    current_secondary = signals_secondary.iloc[-1]
    
    # Confluence analysis
    analysis = {
        'primary_trend': current_primary['trend'],
        'secondary_trend': current_secondary['trend'],
        'primary_strength': current_primary['trend_strength'],
        'secondary_strength': current_secondary['trend_strength'],
        'confluence': None
    }
    
    # Detect trend confluence
    if current_primary['trend'] == current_secondary['trend']:
        if current_primary['trend'] == 1:
            analysis['confluence'] = 'BULLISH_CONFLUENCE'
        else:
            analysis['confluence'] = 'BEARISH_CONFLUENCE'
    else:
        analysis['confluence'] = 'MIXED_SIGNALS'
    
    # Detect possible reversals
    if (current_primary['buy_signal'] and 
        current_secondary['trend'] == 1 and 
        current_secondary['trend_strength'] > 3):
        analysis['setup'] = 'STRONG_BUY_SETUP'
    elif (current_primary['sell_signal'] and 
          current_secondary['trend'] == -1 and 
          current_secondary['trend_strength'] > 3):
        analysis['setup'] = 'STRONG_SELL_SETUP'
    else:
        analysis['setup'] = 'NO_CLEAR_SETUP'
    
    return analysis
```

## Visualization and Analysis

```python
def plot_psar_analysis(df: pd.DataFrame, psar_data: pd.DataFrame, signals: pd.DataFrame, title: str = "Parabolic SAR Analysis"):
    """
    Create complete PSAR analysis chart
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12),
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Chart 1: Price + PSAR
    ax1.plot(df.index, df['Close'], 'k-', linewidth=2, label='Price', zorder=1)
    
    # PSAR points - different colors for up/down trend
    up_trend_mask = ~pd.isna(psar_data['UpTrend'])
    down_trend_mask = ~pd.isna(psar_data['DownTrend'])
    
    ax1.scatter(df.index[up_trend_mask], psar_data['UpTrend'][up_trend_mask], 
               color='green', s=20, label='PSAR (Bullish)', zorder=3)
    ax1.scatter(df.index[down_trend_mask], psar_data['DownTrend'][down_trend_mask], 
               color='red', s=20, label='PSAR (Bearish)', zorder=3)
    
    # Mark trend change signals
    buy_signals = df.index[signals['buy_signal']]
    sell_signals = df.index[signals['sell_signal']]
    
    ax1.scatter(buy_signals, df.loc[buy_signals, 'Close'], 
               color='lime', marker='^', s=100, label='Buy Signal', zorder=5)
    ax1.scatter(sell_signals, df.loc[sell_signals, 'Close'], 
               color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_title(f'{title} - Price & PSAR Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Trend Strength
    colors = ['red' if x == -1 else 'green' for x in signals['trend']]
    ax2.bar(signals.index, signals['trend_strength'], color=colors, alpha=0.7)
    ax2.set_title('Trend Strength (Days in Current Trend)')
    ax2.set_ylabel('Days')
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Price-SAR Distance (Momentum)
    ax3.plot(signals.index, signals['price_sar_distance'] * 100, 'purple', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.fill_between(signals.index, 0, signals['price_sar_distance'] * 100, 
                    where=(signals['price_sar_distance'] > 0), alpha=0.3, color='green')
    ax3.fill_between(signals.index, 0, signals['price_sar_distance'] * 100,
                    where=(signals['price_sar_distance'] < 0), alpha=0.3, color='red')
    ax3.set_title('Price-SAR Distance (Momentum %)')
    ax3.set_ylabel('Distance %')
    ax3.grid(True, alpha=0.3)
    
    # Date formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def psar_complete_example():
    """
    Complete analysis example with Parabolic SAR
    """
    import yfinance as yf
    
    # Get data
    ticker = "AAPL"
    df = yf.download(ticker, start="2023-01-01", end="2024-01-01", interval="1d")
    
    print(f"=== PARABOLIC SAR ANALYSIS: {ticker} ===\n")
    
    # Adaptive parameters
    adaptive_params = adaptive_psar_parameters(df)
    print(f"ADAPTIVE PARAMETERS:")
    print(f"   Volatility Regime: {adaptive_params['regime']}")
    print(f"   Current Volatility: {adaptive_params['volatility']:.1%}")
    print(f"   AF Increment: {adaptive_params['increment']:.3f}")
    print(f"   Max AF: {adaptive_params['max_step']:.2f}")
    
    # Calculate PSAR with adaptive parameters
    psar = Parabolic_SAR(df, 
                        increment=adaptive_params['increment'],
                        max_step=adaptive_params['max_step'])
    signals = analyze_psar_signals(df, psar)
    
    # Period statistics
    buy_signals_count = signals['buy_signal'].sum()
    sell_signals_count = signals['sell_signal'].sum()
    avg_trend_length = signals['trend_strength'].mean()
    
    print(f"\nPERIOD STATISTICS:")
    print(f"   Buy Signals: {buy_signals_count}")
    print(f"   Sell Signals: {sell_signals_count}")
    print(f"   Average Trend Duration: {avg_trend_length:.1f} days")
    
    # Current analysis
    current = signals.iloc[-1]
    print(f"\nCURRENT ANALYSIS:")
    print(f"   Price: ${current['price']:.2f}")
    print(f"   PSAR: ${current['psar']:.2f}")
    print(f"   Trend: {'BULLISH' if current['trend'] == 1 else 'BEARISH'}")
    print(f"   Trend Strength: {current['trend_strength']} days")
    print(f"   Price-SAR Distance: {current['price_sar_distance']:.2%}")
    
    if current['buy_signal']:
        print("   SIGNAL: BUY - Change to bullish trend")
    elif current['sell_signal']:
        print("   SIGNAL: SELL - Change to bearish trend")
    elif current['trend'] == 1:
        print(f"   HOLD LONG - SAR at ${current['psar']:.2f}")
    else:
        print(f"   HOLD SHORT - SAR at ${current['psar']:.2f}")
    
    # Multi-timeframe analysis
    mtf_analysis = multi_timeframe_psar(ticker)
    print(f"\nMULTI-TIMEFRAME ANALYSIS:")
    print(f"   Confluence: {mtf_analysis['confluence']}")
    print(f"   Setup: {mtf_analysis['setup']}")
    
    # Create chart
    plot_psar_analysis(df, psar, signals, f"Parabolic SAR Analysis - {ticker}")
    
    return psar, signals

# Run example
if __name__ == "__main__":
    psar_complete_example()
```

## Best Practices

### ✅ **Do's**

1. **Use as trailing stop**: PSAR excellent for protecting profits
2. **Combine with trend**: Confirm direction with MA or ADX
3. **Adjust parameters**: Adapt AF based on asset volatility
4. **Filter with volume**: More reliable signals with confirmatory volume

### ❌ **Don'ts**

1. **Don't use in sideways markets**: PSAR generates many false signals
2. **Don't ignore context**: A single signal is not enough to enter
3. **Don't use very high AF**: Can generate whipsaws in volatility
4. **Don't trade every signal**: Filter quality based on context

### 🎯 **Parameters for Different Styles**

```python
PSAR_PARAMETERS = {
    'conservative': {'increment': 0.01, 'max_step': 0.15},    # Fewer signals, more reliable
    'standard': {'increment': 0.02, 'max_step': 0.20},       # Classic parameters
    'aggressive': {'increment': 0.03, 'max_step': 0.25},     # More signals, higher risk
    'scalping': {'increment': 0.005, 'max_step': 0.10},      # Very sensitive for scalping
    'small_caps': {'increment': 0.025, 'max_step': 0.30}     # Adapted to high volatility
}
```

## Next Step

With Parabolic SAR mastered, let's continue with [SuperTrend](super_tendencia.md) for advanced trend analysis.
