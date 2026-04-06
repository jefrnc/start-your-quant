> 🇪🇸 [Leer en Español](super_tendencia.es.md) | 🇺🇸 **English**

# SuperTrend - The Ultimate Trend Follower

## Definition

SuperTrend is a trend-following indicator that draws a line on the candlestick chart. Depending on the color, it indicates a negative trend (red line above candles) or positive trend (green line below candles), functioning as dynamic support or resistance.

## Indicator Philosophy

### Why Does It Work?
- **ATR-Based**: Uses actual volatility to adjust distances
- **Dynamic Support/Resistance**: Adapts to market conditions
- **Clear Signals**: Green = bullish, red = bearish
- **Noise Filter**: Reduces false signals compared to simple MAs

### Key Components
```
ATR = Average True Range (volatility)
Mid_Price   = (High + Low) / 2
Upper_Band  = Mid_Price + (Factor × ATR)
Lower_Band  = Mid_Price - (Factor × ATR)
```

## Reference Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def SuperTrend(df: pd.DataFrame, length: int = 14, factor: float = 3.0) -> pd.DataFrame:
    """
    SuperTrend - Exact reference implementation
    
    A trend-following indicator that draws dynamic support/resistance
    lines based on ATR (Average True Range).
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical asset data (must include High, Low, Close)
    length : int, default 14
        Window for ATR calculation
    factor : float, default 3.0
        ATR multiplier for calculating the bands
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: FinalUpperB, FinalLowerB, SuperTrend
        
    How to Trade It
    ---------------
    - BUY: Green line (below price) - Bullish trend
    - SELL: Red line (above price) - Bearish trend
    - The line acts as dynamic support (green) or resistance (red)
    
    Usage Example
    --------------
    >>> df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
    >>> st = SuperTrend(df, length=14, factor=3.0)
    >>> print(st.head())
    """
    # Calculate True Range (TR)
    High, Low = df["High"], df["Low"]
    
    # True Range components
    H_minus_L = High - Low
    prev_close = df["Close"].shift(periods=1)
    H_minus_PC = abs(High - prev_close)
    L_minus_PC = abs(Low - prev_close)
    
    # True Range = max(H-L, |H-PC|, |L-PC|)
    TR = pd.Series(np.max([H_minus_L, H_minus_PC, L_minus_PC], axis=0), 
                   index=df.index, name="TR")
    
    # Calculate ATR using exponential smoothing
    ATR = TR.ewm(alpha=1 / length).mean()
    
    # Calculate mid price and basic bands
    mid = (High + Low) / 2
    FinalUpperB = mid + factor * ATR
    FinalLowerB = mid - factor * ATR
    
    # Initialize SuperTrend
    Supertrend = np.zeros(ATR.shape[0], dtype=bool)
    close = df["Close"]
    
    # Calculate SuperTrend point by point
    for i in range(1, ATR.shape[0]):
        # Determine trend direction
        if close[i] > FinalUpperB[i - 1]:
            # Price breaks upper band -> bullish trend
            Supertrend[i] = True
        elif close[i] < FinalLowerB[i - 1]:
            # Price breaks lower band -> bearish trend
            Supertrend[i] = False
        else:
            # Maintain previous trend
            Supertrend[i] = Supertrend[i - 1]
            
            # Adjust bands to prevent premature changes
            if Supertrend[i] == True and FinalLowerB[i] < FinalLowerB[i - 1]:
                # In bullish trend, lower band cannot decrease
                FinalLowerB[i] = FinalLowerB[i - 1]
            elif Supertrend[i] == False and FinalUpperB[i] > FinalUpperB[i - 1]:
                # In bearish trend, upper band cannot increase
                FinalUpperB[i] = FinalUpperB[i - 1]
        
        # Remove inactive band based on trend direction
        if Supertrend[i] == True:
            # Bullish trend: remove upper band
            FinalUpperB[i] = np.nan
        else:
            # Bearish trend: remove lower band
            FinalLowerB[i] = np.nan
    
    # Adjust first value
    if Supertrend[1] == False:
        FinalLowerB[0] = np.nan
    else:
        FinalUpperB[0] = np.nan
    
    # Prepare final data (remove warm-up period)
    FU = FinalUpperB[length - 1:]
    FL = FinalLowerB[length - 1:]
    
    # Create final SuperTrend combining both bands
    ST_array = np.nansum([FU, FL], axis=0)
    ST_array[0] = np.nan  # First value always NaN
    
    # Create result DataFrame
    ST_df = pd.concat([FU, FL], axis=1)
    ST_df["SuperTrend"] = ST_array
    ST_df.columns = ["FinalUpperB", "FinalLowerB", "SuperTrend"]
    
    return ST_df

def analyze_supertrend_signals(df: pd.DataFrame, st_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze SuperTrend trading signals
    
    Parameters
    ----------
    df : pd.DataFrame
        Original historical data
    st_data : pd.DataFrame
        SuperTrend data (output of SuperTrend)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with signals and analysis
    """
    # Align indices (ST has fewer data points due to warm-up period)
    aligned_df = df.loc[st_data.index]
    
    signals = pd.DataFrame(index=st_data.index)
    signals['price'] = aligned_df['Close']
    signals['high'] = aligned_df['High']
    signals['low'] = aligned_df['Low']
    signals['volume'] = aligned_df['Volume']
    signals['supertrend'] = st_data['SuperTrend']
    signals['upper_band'] = st_data['FinalUpperB']
    signals['lower_band'] = st_data['FinalLowerB']
    
    # Determine current trend
    signals['trend'] = np.where(~pd.isna(st_data['FinalLowerB']), 1, -1)  # 1=bullish, -1=bearish
    signals['trend_change'] = signals['trend'].diff().fillna(0)
    
    # Trend change signals
    signals['buy_signal'] = signals['trend_change'] == 2    # From bearish to bullish
    signals['sell_signal'] = signals['trend_change'] == -2  # From bullish to bearish
    
    # Distance from price to SuperTrend (trend strength)
    signals['price_st_distance'] = np.where(
        signals['trend'] == 1,
        (signals['price'] - signals['supertrend']) / signals['supertrend'],  # Bullish
        (signals['supertrend'] - signals['price']) / signals['supertrend']   # Bearish
    )
    
    # Trend duration
    trend_groups = (signals['trend'] != signals['trend'].shift()).cumsum()
    signals['trend_duration'] = signals.groupby(trend_groups).cumcount() + 1
    
    # Period volatility (using SuperTrend as proxy)
    st_changes = signals['supertrend'].pct_change().abs()
    signals['st_volatility'] = st_changes.rolling(10).mean()
    
    # Signal quality
    signals['signal_strength'] = 'NONE'
    
    # Strong signals (with confirmations)
    strong_buy = (
        signals['buy_signal'] &
        (signals['trend_duration'].shift(1) > 3) &  # Lasting bearish trend
        (signals['volume'] > signals['volume'].rolling(20).mean()) &  # Confirmatory volume
        (signals['price'] > signals['price'].shift(1))  # Bullish momentum
    )
    
    strong_sell = (
        signals['sell_signal'] &
        (signals['trend_duration'].shift(1) > 3) &  # Lasting bullish trend
        (signals['volume'] > signals['volume'].rolling(20).mean()) &
        (signals['price'] < signals['price'].shift(1))  # Bearish momentum
    )
    
    # Pullback signals (retracements within trend)
    pullback_buy = (
        (signals['trend'] == 1) &  # Bullish trend
        (signals['low'] <= signals['supertrend'] * 1.005) &  # Price near ST
        (signals['close'] > signals['supertrend']) &  # But closes above
        (signals['trend_duration'] > 5)  # Established trend
    )
    
    pullback_sell = (
        (signals['trend'] == -1) &  # Bearish trend
        (signals['high'] >= signals['supertrend'] * 0.995) &  # Price near ST
        (signals['close'] < signals['supertrend']) &  # But closes below
        (signals['trend_duration'] > 5)  # Established trend
    )
    
    # Assign quality levels
    signals.loc[strong_buy, 'signal_strength'] = 'STRONG_BUY'
    signals.loc[strong_sell, 'signal_strength'] = 'STRONG_SELL'
    signals.loc[pullback_buy, 'signal_strength'] = 'PULLBACK_BUY'
    signals.loc[pullback_sell, 'signal_strength'] = 'PULLBACK_SELL'
    signals.loc[signals['buy_signal'] & ~strong_buy, 'signal_strength'] = 'WEAK_BUY'
    signals.loc[signals['sell_signal'] & ~strong_sell, 'signal_strength'] = 'WEAK_SELL'
    
    return signals
```

## Trading Strategies with SuperTrend

### 1. Trend Following Strategy
```python
def supertrend_following_strategy(df: pd.DataFrame, st_length: int = 14, st_factor: float = 3.0):
    """
    Trend following strategy using SuperTrend
    """
    # Calculate SuperTrend
    st = SuperTrend(df, length=st_length, factor=st_factor)
    signals = analyze_supertrend_signals(df, st)
    
    # Additional filters
    sma_200 = df['Close'].rolling(200).mean()
    aligned_sma = sma_200.loc[signals.index]
    
    entry_signals = pd.Series(0, index=signals.index)
    
    # Long entries (only in overall bullish market)
    long_entry = (
        signals['buy_signal'] &
        (signals['signal_strength'].isin(['STRONG_BUY'])) &
        (signals['price'] > aligned_sma)  # Price above SMA 200
    )
    
    # Short entries (only in overall bearish market)
    short_entry = (
        signals['sell_signal'] &
        (signals['signal_strength'].isin(['STRONG_SELL'])) &
        (signals['price'] < aligned_sma)  # Price below SMA 200
    )
    
    # Pullback entries (entries on retracements)
    pullback_long = signals['signal_strength'] == 'PULLBACK_BUY'
    pullback_short = signals['signal_strength'] == 'PULLBACK_SELL'
    
    entry_signals[long_entry] = 1
    entry_signals[short_entry] = -1
    entry_signals[pullback_long] = 0.5   # Reduced position on pullbacks
    entry_signals[pullback_short] = -0.5
    
    return {
        'signals': entry_signals,
        'st_data': st,
        'analysis': signals,
        'strategy_type': 'trend_following'
    }

def supertrend_rsi_strategy(df: pd.DataFrame, st_length: int = 14, st_factor: float = 3.0):
    """
    Combine SuperTrend with RSI - strategy mentioned in reference
    """
    # SuperTrend
    st = SuperTrend(df, length=st_length, factor=st_factor)
    signals = analyze_supertrend_signals(df, st)
    
    # RSI
    rsi = calculate_rsi(df['Close'], period=14)
    aligned_rsi = rsi.loc[signals.index]
    
    entry_signals = pd.Series(0, index=signals.index)
    
    # Long setup: ST bullish + RSI crosses above 50
    long_setup = (
        signals['buy_signal'] &  # Change to bullish trend in ST
        (aligned_rsi > 50) &     # RSI above 50
        (aligned_rsi.shift(1) <= 50)  # RSI was below 50
    )
    
    # Short setup: ST bearish + RSI crosses below 50  
    short_setup = (
        signals['sell_signal'] &  # Change to bearish trend in ST
        (aligned_rsi < 50) &      # RSI below 50
        (aligned_rsi.shift(1) >= 50)  # RSI was above 50
    )
    
    # Additional pullback entries with RSI
    rsi_pullback_long = (
        (signals['trend'] == 1) &  # ST in bullish trend
        (aligned_rsi < 40) &       # RSI oversold
        (aligned_rsi > aligned_rsi.shift(1))  # RSI starting to recover
    )
    
    rsi_pullback_short = (
        (signals['trend'] == -1) &  # ST in bearish trend
        (aligned_rsi > 60) &        # RSI overbought
        (aligned_rsi < aligned_rsi.shift(1))  # RSI starting to fall
    )
    
    entry_signals[long_setup] = 1
    entry_signals[short_setup] = -1
    entry_signals[rsi_pullback_long] = 0.5
    entry_signals[rsi_pullback_short] = -0.5
    
    return {
        'signals': entry_signals,
        'st_data': st,
        'analysis': signals,
        'rsi': aligned_rsi,
        'strategy_type': 'st_rsi_combo'
    }

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Helper function to calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### 2. Small Cap Specific Strategy
```python
def small_cap_supertrend_strategy(df: pd.DataFrame, gap_threshold: float = 0.03):
    """
    SuperTrend strategy specific to small caps
    """
    # More sensitive SuperTrend for small caps
    st = SuperTrend(df, length=10, factor=2.5)  # More reactive
    signals = analyze_supertrend_signals(df, st)
    
    # Detect gaps
    gap_up = (df['Open'] / df['Close'].shift(1) - 1) > gap_threshold
    gap_down = (df['Open'] / df['Close'].shift(1) - 1) < -gap_threshold
    aligned_gap_up = gap_up.loc[signals.index]
    aligned_gap_down = gap_down.loc[signals.index]
    
    # RVOL (Relative Volume)
    avg_volume = df['Volume'].rolling(20).mean()
    rvol = df['Volume'] / avg_volume
    aligned_rvol = rvol.loc[signals.index]
    
    entry_signals = pd.Series(0, index=signals.index)
    
    # Gap & Go with SuperTrend
    gap_and_go_long = (
        aligned_gap_up &
        (signals['trend'] == 1) &  # ST confirms bullish
        (aligned_rvol > 3) &       # High relative volume
        (signals['price'] > signals['price'].shift(1))  # Momentum continues
    )
    
    # Gap fill with SuperTrend
    gap_fill_long = (
        aligned_gap_down &
        signals['buy_signal'] &    # ST changes to bullish
        (aligned_rvol > 2) &       # Confirmatory volume
        (signals['price'] > df['Open'].loc[signals.index])  # Price recovers above open
    )
    
    # Breakout with SuperTrend
    high_20 = df['High'].rolling(20).max()
    aligned_high_20 = high_20.loc[signals.index]
    
    breakout_long = (
        signals['buy_signal'] &
        (signals['high'] >= aligned_high_20.shift(1)) &  # New 20-day high
        (aligned_rvol > 2.5) &
        (signals['trend_duration'] <= 2)  # Recent trend change
    )
    
    # Short setups
    gap_fade_short = (
        aligned_gap_up &
        signals['sell_signal'] &   # ST changes to bearish
        (aligned_rvol > 2) &
        (signals['price'] < df['Open'].loc[signals.index])  # Price below open
    )
    
    entry_signals[gap_and_go_long] = 1
    entry_signals[gap_fill_long] = 1
    entry_signals[breakout_long] = 1
    entry_signals[gap_fade_short] = -1
    
    return {
        'signals': entry_signals,
        'st_data': st,
        'analysis': signals,
        'gap_signals': {
            'gap_up': aligned_gap_up,
            'gap_down': aligned_gap_down,
            'rvol': aligned_rvol
        },
        'strategy_type': 'small_cap_st'
    }

def adaptive_supertrend_parameters(df: pd.DataFrame, volatility_period: int = 20):
    """
    Adaptive parameters based on volatility
    """
    # Measure current volatility
    atr = calculate_atr_simple(df, period=14)
    current_atr = atr.iloc[-1]
    price = df['Close'].iloc[-1]
    atr_pct = current_atr / price
    
    # Base parameters
    base_length = 14
    base_factor = 3.0
    
    # Adjust based on volatility
    if atr_pct > 0.03:  # High volatility (>3%)
        length = base_length + 3  # More smoothing
        factor = base_factor * 1.2  # Wider bands
        regime = "HIGH_VOLATILITY"
    elif atr_pct < 0.015:  # Low volatility (<1.5%)
        length = base_length - 2  # More reactive
        factor = base_factor * 0.8  # Narrower bands
        regime = "LOW_VOLATILITY"
    else:
        length = base_length
        factor = base_factor
        regime = "NORMAL_VOLATILITY"
    
    return {
        'length': max(length, 5),  # Minimum 5 periods
        'factor': max(factor, 1.5),  # Minimum factor 1.5
        'atr_pct': atr_pct,
        'regime': regime
    }

def calculate_atr_simple(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Helper function to calculate simple ATR"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(period).mean()
```

## Optimization and Multi-Timeframe

```python
def multi_timeframe_supertrend(symbol: str, primary_tf: str = '1d', secondary_tf: str = '4h'):
    """
    Multi-timeframe SuperTrend analysis
    """
    import yfinance as yf
    
    # Get data
    df_primary = yf.download(symbol, period="6mo", interval=primary_tf)
    df_secondary = yf.download(symbol, period="2mo", interval=secondary_tf)
    
    # SuperTrend on each timeframe
    st_primary = SuperTrend(df_primary, length=14, factor=3.0)
    st_secondary = SuperTrend(df_secondary, length=14, factor=2.5)  # More sensitive on lower TF
    
    signals_primary = analyze_supertrend_signals(df_primary, st_primary)
    signals_secondary = analyze_supertrend_signals(df_secondary, st_secondary)
    
    # Current state
    current_primary = signals_primary.iloc[-1]
    current_secondary = signals_secondary.iloc[-1]
    
    analysis = {
        'primary_trend': current_primary['trend'],
        'secondary_trend': current_secondary['trend'],
        'primary_duration': current_primary['trend_duration'],
        'secondary_duration': current_secondary['trend_duration'],
        'primary_distance': current_primary['price_st_distance'],
        'secondary_distance': current_secondary['price_st_distance'],
        'confluence': None,
        'setup_quality': None
    }
    
    # Confluence analysis
    if current_primary['trend'] == current_secondary['trend']:
        if current_primary['trend'] == 1:
            analysis['confluence'] = 'BULLISH_ALIGNMENT'
            if (current_primary['trend_duration'] > 5 and 
                current_secondary['trend_duration'] > 3):
                analysis['setup_quality'] = 'STRONG_BULLISH'
        else:
            analysis['confluence'] = 'BEARISH_ALIGNMENT'
            if (current_primary['trend_duration'] > 5 and 
                current_secondary['trend_duration'] > 3):
                analysis['setup_quality'] = 'STRONG_BEARISH'
    else:
        analysis['confluence'] = 'MIXED_SIGNALS'
        analysis['setup_quality'] = 'CONFLICTED'
    
    # Detect high probability setups
    if (current_primary['buy_signal'] and 
        current_secondary['trend'] == 1 and 
        current_secondary['trend_duration'] > 2):
        analysis['setup_quality'] = 'HIGH_PROB_LONG'
    elif (current_primary['sell_signal'] and 
          current_secondary['trend'] == -1 and 
          current_secondary['trend_duration'] > 2):
        analysis['setup_quality'] = 'HIGH_PROB_SHORT'
    
    return analysis

def supertrend_parameter_optimization(df: pd.DataFrame, length_range: tuple = (10, 20), factor_range: tuple = (2.0, 4.0)):
    """
    Simple SuperTrend parameter optimization
    """
    import itertools
    
    # Parameter ranges to test
    lengths = range(length_range[0], length_range[1] + 1, 2)
    factors = np.arange(factor_range[0], factor_range[1] + 0.1, 0.5)
    
    results = []
    
    for length, factor in itertools.product(lengths, factors):
        try:
            # Calculate SuperTrend with parameters
            st = SuperTrend(df, length=length, factor=factor)
            signals = analyze_supertrend_signals(df, st)
            
            # Simple evaluation metrics
            total_signals = signals['buy_signal'].sum() + signals['sell_signal'].sum()
            if total_signals == 0:
                continue
                
            # Simulate simple returns
            position = 0
            returns = []
            
            for i in range(1, len(signals)):
                if signals['buy_signal'].iloc[i]:
                    position = 1
                elif signals['sell_signal'].iloc[i]:
                    position = -1
                
                if position != 0:
                    ret = position * (signals['price'].iloc[i] / signals['price'].iloc[i-1] - 1)
                    returns.append(ret)
            
            if len(returns) > 0:
                total_return = np.sum(returns)
                win_rate = np.mean([r > 0 for r in returns])
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                
                results.append({
                    'length': length,
                    'factor': factor,
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'sharpe': sharpe,
                    'total_signals': total_signals,
                    'score': sharpe * total_return  # Composite score
                })
        
        except Exception as e:
            continue
    
    # Find best parameters
    if results:
        best_result = max(results, key=lambda x: x['score'])
        return best_result, results
    else:
        return None, []
```

## Complete Visualization

```python
def plot_supertrend_analysis(df: pd.DataFrame, st_data: pd.DataFrame, signals: pd.DataFrame, title: str = "SuperTrend Analysis"):
    """
    Create complete SuperTrend analysis chart
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Align data
    aligned_df = df.loc[st_data.index]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12),
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Chart 1: Price + SuperTrend
    ax1.plot(aligned_df.index, aligned_df['Close'], 'k-', linewidth=2, label='Price', zorder=1)
    
    # SuperTrend lines
    bullish_mask = ~pd.isna(st_data['FinalLowerB'])
    bearish_mask = ~pd.isna(st_data['FinalUpperB'])
    
    ax1.plot(st_data.index[bullish_mask], st_data['FinalLowerB'][bullish_mask], 
             'g-', linewidth=3, label='SuperTrend (Bullish)', zorder=2)
    ax1.plot(st_data.index[bearish_mask], st_data['FinalUpperB'][bearish_mask], 
             'r-', linewidth=3, label='SuperTrend (Bearish)', zorder=2)
    
    # Mark signals
    buy_signals = signals.index[signals['buy_signal']]
    sell_signals = signals.index[signals['sell_signal']]
    
    ax1.scatter(buy_signals, signals.loc[buy_signals, 'price'], 
               color='lime', marker='^', s=150, label='Buy Signal', zorder=5)
    ax1.scatter(sell_signals, signals.loc[sell_signals, 'price'], 
               color='red', marker='v', s=150, label='Sell Signal', zorder=5)
    
    # Highlight strong signals
    strong_buys = signals.index[signals['signal_strength'] == 'STRONG_BUY']
    strong_sells = signals.index[signals['signal_strength'] == 'STRONG_SELL']
    
    ax1.scatter(strong_buys, signals.loc[strong_buys, 'price'], 
               color='darkgreen', marker='^', s=200, label='Strong Buy', zorder=6, edgecolors='white')
    ax1.scatter(strong_sells, signals.loc[strong_sells, 'price'], 
               color='darkred', marker='v', s=200, label='Strong Sell', zorder=6, edgecolors='white')
    
    ax1.set_title(f'{title} - Price & SuperTrend')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Trend Duration
    colors = ['red' if x == -1 else 'green' for x in signals['trend']]
    ax2.bar(signals.index, signals['trend_duration'], color=colors, alpha=0.7, width=1)
    ax2.set_title('Trend Duration (Days)')
    ax2.set_ylabel('Duration')
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Price-SuperTrend Distance
    ax3.plot(signals.index, signals['price_st_distance'] * 100, 'purple', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.fill_between(signals.index, 0, signals['price_st_distance'] * 100,
                    where=(signals['price_st_distance'] > 0), alpha=0.3, color='green')
    ax3.fill_between(signals.index, 0, signals['price_st_distance'] * 100,
                    where=(signals['price_st_distance'] < 0), alpha=0.3, color='red')
    ax3.set_title('Price-SuperTrend Distance (%)')
    ax3.set_ylabel('Distance %')
    ax3.grid(True, alpha=0.3)
    
    # Date formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def supertrend_complete_example():
    """
    Complete analysis example with SuperTrend
    """
    import yfinance as yf
    
    # Get data
    ticker = "AAPL"
    df = yf.download(ticker, start="2023-01-01", end="2024-01-01", interval="1d")
    
    print(f"=== SUPERTREND ANALYSIS: {ticker} ===\n")
    
    # Adaptive parameters
    adaptive_params = adaptive_supertrend_parameters(df)
    print(f"ADAPTIVE PARAMETERS:")
    print(f"   Volatility Regime: {adaptive_params['regime']}")
    print(f"   ATR %: {adaptive_params['atr_pct']:.2%}")
    print(f"   Length: {adaptive_params['length']}")
    print(f"   Factor: {adaptive_params['factor']:.1f}")
    
    # Calculate SuperTrend
    st = SuperTrend(df, 
                   length=adaptive_params['length'],
                   factor=adaptive_params['factor'])
    signals = analyze_supertrend_signals(df, st)
    
    # Statistics
    buy_signals_count = signals['buy_signal'].sum()
    sell_signals_count = signals['sell_signal'].sum()
    avg_trend_duration = signals['trend_duration'].mean()
    strong_signals = signals['signal_strength'].str.contains('STRONG').sum()
    
    print(f"\nPERIOD STATISTICS:")
    print(f"   Buy Signals: {buy_signals_count}")
    print(f"   Sell Signals: {sell_signals_count}")
    print(f"   Strong Signals: {strong_signals}")
    print(f"   Average Trend Duration: {avg_trend_duration:.1f} days")
    
    # Current analysis
    current = signals.iloc[-1]
    trend_name = "BULLISH" if current['trend'] == 1 else "BEARISH"
    
    print(f"\nCURRENT ANALYSIS:")
    print(f"   Price: ${current['price']:.2f}")
    print(f"   SuperTrend: ${current['supertrend']:.2f}")
    print(f"   Trend: {trend_name}")
    print(f"   Trend Duration: {current['trend_duration']} days")
    print(f"   Distance to ST: {current['price_st_distance']:.2%}")
    print(f"   Signal Strength: {current['signal_strength']}")
    
    if current['buy_signal']:
        print("   SIGNAL: BUY - Change to bullish trend")
    elif current['sell_signal']:
        print("   SIGNAL: SELL - Change to bearish trend")
    elif current['trend'] == 1:
        print(f"   HOLD LONG - Support at ${current['supertrend']:.2f}")
    else:
        print(f"   HOLD SHORT - Resistance at ${current['supertrend']:.2f}")
    
    # Multi-timeframe
    mtf_analysis = multi_timeframe_supertrend(ticker)
    print(f"\nMULTI-TIMEFRAME ANALYSIS:")
    print(f"   Confluence: {mtf_analysis['confluence']}")
    print(f"   Setup Quality: {mtf_analysis['setup_quality']}")
    
    # Optimization
    best_params, all_results = supertrend_parameter_optimization(df)
    if best_params:
        print(f"\nOPTIMAL PARAMETERS:")
        print(f"   Length: {best_params['length']}")
        print(f"   Factor: {best_params['factor']}")
        print(f"   Sharpe: {best_params['sharpe']:.2f}")
        print(f"   Win Rate: {best_params['win_rate']:.1%}")
    
    # Create chart
    plot_supertrend_analysis(df, st, signals, f"SuperTrend Analysis - {ticker}")
    
    return st, signals

# Run example
if __name__ == "__main__":
    supertrend_complete_example()
```

## Best Practices

### ✅ **Do's**

1. **Use as dynamic support/resistance**: ST excellent for trailing stops
2. **Combine with RSI**: The ST + RSI combination is very effective
3. **Adjust parameters based on volatility**: Higher factor for volatile assets
4. **Filter signals with volume**: Confirm trend changes with volume

### ❌ **Don'ts**

1. **Don't use in very choppy markets**: ST generates whipsaws in tight ranges
2. **Don't ignore the larger trend**: Confirm with higher timeframes
3. **Don't use a very low factor**: Can generate excessive signals
4. **Don't trade against context**: A green line doesn't guarantee a rally

### 🎯 **Recommended Parameters**

```python
SUPERTREND_SETTINGS = {
    'conservative': {'length': 21, 'factor': 4.0},    # Fewer signals, more reliable
    'standard': {'length': 14, 'factor': 3.0},       # Classic parameters
    'aggressive': {'length': 10, 'factor': 2.5},     # More signals, higher risk
    'small_caps': {'length': 12, 'factor': 2.8},     # Adapted to volatility
    'scalping': {'length': 7, 'factor': 2.0}         # For short timeframes
}
```

## Next Step
