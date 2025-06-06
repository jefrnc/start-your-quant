# Medias Móviles (EMA/SMA)

## Lo Básico que Funciona

Las medias móviles son simples pero poderosas. En small caps, uso principalmente:
- **9 EMA**: Para momentum intraday
- **20 SMA**: Soporte/resistencia key
- **50 SMA**: Tendencia intermedia
- **200 SMA**: La línea en la arena institucional

## SMA vs EMA

```python
def calculate_moving_averages(df):
    """Calcular SMA y EMA comunes"""
    # Simple Moving Average
    df['sma_9'] = df['close'].rolling(window=9).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Exponential Moving Average
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    return df
```

## EMA para Day Trading

La 9 EMA es mi go-to para entries en small caps.

```python
def ema_momentum_setup(df):
    """Setup usando 9 EMA para momentum"""
    # Calcular 9 EMA
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    
    # Precio respecto a EMA
    df['above_ema9'] = df['close'] > df['ema_9']
    df['ema9_distance'] = (df['close'] - df['ema_9']) / df['ema_9'] * 100
    
    # Detectar bounces
    df['ema9_touch'] = (df['low'] <= df['ema_9']) & (df['close'] > df['ema_9'])
    
    # Momentum: precio acelerando desde EMA
    df['ema9_momentum'] = df['ema9_distance'] - df['ema9_distance'].shift(1)
    df['bullish_momentum'] = (df['above_ema9'] & 
                             (df['ema9_momentum'] > 0) & 
                             (df['volume'] > df['volume'].rolling(20).mean()))
    
    return df
```

## Moving Average Ribbons

Múltiples EMAs para ver la "textura" del trend.

```python
def create_ema_ribbon(df, periods=[3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]):
    """Crear ribbon de EMAs"""
    for period in periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Calcular expansión/compresión del ribbon
    ema_cols = [f'ema_{p}' for p in periods]
    df['ribbon_max'] = df[ema_cols].max(axis=1)
    df['ribbon_min'] = df[ema_cols].min(axis=1)
    df['ribbon_width'] = (df['ribbon_max'] - df['ribbon_min']) / df['ribbon_min'] * 100
    
    # Trend strength
    df['ribbon_aligned'] = df[ema_cols].apply(
        lambda row: all(row[i] > row[i+1] for i in range(len(row)-1)), 
        axis=1
    )
    
    return df
```

## Hull Moving Average (HMA)

Menos lag, mejor para entries precisos.

```python
def calculate_hma(df, period=20):
    """Hull Moving Average - menos lag"""
    # WMA de período/2
    half_period = int(period / 2)
    wma_half = df['close'].rolling(half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
    )
    
    # WMA de período completo
    wma_full = df['close'].rolling(period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
    )
    
    # HMA
    raw_hma = 2 * wma_half - wma_full
    hma_period = int(np.sqrt(period))
    df['hma'] = raw_hma.rolling(hma_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
    )
    
    return df
```

## Crossover Systems

```python
def ma_crossover_signals(df, fast_period=9, slow_period=20, ma_type='ema'):
    """Sistema de cruces de medias"""
    # Calcular medias
    if ma_type == 'ema':
        df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    else:
        df['fast_ma'] = df['close'].rolling(fast_period).mean()
        df['slow_ma'] = df['close'].rolling(slow_period).mean()
    
    # Detectar cruces
    df['fast_above'] = df['fast_ma'] > df['slow_ma']
    df['golden_cross'] = (df['fast_above'] & ~df['fast_above'].shift(1))
    df['death_cross'] = (~df['fast_above'] & df['fast_above'].shift(1))
    
    # Fuerza del cruce
    df['cross_strength'] = abs(df['fast_ma'] - df['slow_ma']) / df['slow_ma'] * 100
    
    # Filtrar cruces débiles
    df['strong_golden'] = df['golden_cross'] & (df['cross_strength'] > 0.5)
    df['strong_death'] = df['death_cross'] & (df['cross_strength'] > 0.5)
    
    return df
```

## Dynamic Moving Averages

Adaptar el período según volatilidad.

```python
def adaptive_moving_average(df, base_period=20):
    """Media móvil que se adapta a la volatilidad"""
    # Calcular volatilidad
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['vol_rank'] = df['volatility'].rolling(100).rank(pct=True)
    
    # Ajustar período según volatilidad
    # Alta volatilidad = período más corto (más responsive)
    df['adaptive_period'] = base_period * (2 - df['vol_rank'])
    df['adaptive_period'] = df['adaptive_period'].clip(lower=5, upper=50).astype(int)
    
    # Calcular AMA
    df['ama'] = df['close'].copy()
    for i in range(20, len(df)):
        period = int(df['adaptive_period'].iloc[i])
        df.loc[df.index[i], 'ama'] = df['close'].iloc[max(0, i-period):i].mean()
    
    return df
```

## MA Support/Resistance Levels

```python
def identify_ma_levels(df):
    """Identificar MAs actuando como soporte/resistencia"""
    # Calcular todas las MAs
    ma_periods = [9, 20, 50, 200]
    for period in ma_periods:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Detectar toques
    tolerance = 0.002  # 0.2%
    
    for period in ma_periods:
        # Soporte: low toca MA pero cierra arriba
        df[f'sma_{period}_support'] = (
            (df['low'] <= df[f'sma_{period}'] * (1 + tolerance)) &
            (df['low'] >= df[f'sma_{period}'] * (1 - tolerance)) &
            (df['close'] > df[f'sma_{period}'])
        )
        
        # Resistencia: high toca MA pero cierra abajo
        df[f'sma_{period}_resistance'] = (
            (df['high'] >= df[f'sma_{period}'] * (1 - tolerance)) &
            (df['high'] <= df[f'sma_{period}'] * (1 + tolerance)) &
            (df['close'] < df[f'sma_{period}'])
        )
    
    return df
```

## MA Slope Analysis

La dirección importa tanto como el precio.

```python
def ma_slope_analysis(df, period=20):
    """Analizar pendiente de las medias móviles"""
    # Calcular MA
    df[f'sma_{period}'] = df['close'].rolling(period).mean()
    
    # Slope (cambio porcentual)
    lookback = 5
    df[f'sma_{period}_slope'] = (
        df[f'sma_{period}'] - df[f'sma_{period}'].shift(lookback)
    ) / df[f'sma_{period}'].shift(lookback) * 100
    
    # Categorizar slope
    df['trend_strength'] = pd.cut(
        df[f'sma_{period}_slope'],
        bins=[-np.inf, -2, -0.5, 0.5, 2, np.inf],
        labels=['strong_down', 'down', 'flat', 'up', 'strong_up']
    )
    
    # Aceleración
    df[f'sma_{period}_acceleration'] = df[f'sma_{period}_slope'].diff()
    
    return df
```

## Combo: MA + Volume

```python
def ma_volume_confirmation(df):
    """Combinar señales de MA con volumen"""
    # MAs básicas
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['high_volume'] = df['volume'] > df['volume_sma'] * 1.5
    
    # Señales confirmadas
    df['confirmed_break_9ema'] = (
        (df['close'] > df['ema_9']) & 
        (df['open'] < df['ema_9']) & 
        df['high_volume']
    )
    
    df['confirmed_support_20sma'] = (
        (df['low'] <= df['sma_20']) & 
        (df['close'] > df['sma_20']) & 
        df['high_volume']
    )
    
    return df
```

## MA para Risk Management

```python
class MABasedStops:
    def __init__(self, ma_type='ema', period=9):
        self.ma_type = ma_type
        self.period = period
        
    def calculate_stop(self, df, buffer=0.02):
        """Stop dinámico basado en MA"""
        # Calcular MA
        if self.ma_type == 'ema':
            df['stop_ma'] = df['close'].ewm(span=self.period, adjust=False).mean()
        else:
            df['stop_ma'] = df['close'].rolling(self.period).mean()
        
        # Stop con buffer
        df['long_stop'] = df['stop_ma'] * (1 - buffer)
        df['short_stop'] = df['stop_ma'] * (1 + buffer)
        
        # Trailing stop que solo sube (para longs)
        df['trailing_stop'] = df['long_stop'].cummax()
        
        return df
```

## Backtesting MA Strategy

```python
def backtest_ma_strategy(df, initial_capital=10000):
    """Backtest de estrategia simple con MAs"""
    # Setup
    df = ma_crossover_signals(df, fast_period=9, slow_period=20)
    
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(20, len(df)):  # Empezar después de tener MAs
        row = df.iloc[i]
        
        # Entry: Golden cross
        if row['strong_golden'] and position == 0:
            shares = int(capital * 0.95 / row['close'])
            position = shares
            capital -= shares * row['close']
            trades.append({
                'date': row.name,
                'type': 'buy',
                'price': row['close'],
                'shares': shares
            })
        
        # Exit: Death cross o stop loss
        elif position > 0:
            if row['strong_death'] or row['close'] < row['slow_ma'] * 0.98:
                capital += position * row['close']
                trades.append({
                    'date': row.name,
                    'type': 'sell',
                    'price': row['close'],
                    'shares': position
                })
                position = 0
    
    return pd.DataFrame(trades)
```

## Tips Prácticos

### 1. MA Confluence
```python
def find_ma_confluence(df, tolerance=0.01):
    """Encontrar donde múltiples MAs convergen"""
    mas = ['sma_20', 'sma_50', 'ema_9', 'ema_20']
    
    # Calcular distancia entre MAs
    for i, ma1 in enumerate(mas):
        for ma2 in mas[i+1:]:
            distance = abs(df[ma1] - df[ma2]) / df[ma1]
            if distance < tolerance:
                df[f'{ma1}_{ma2}_confluence'] = True
```

### 2. MA Fans
```python
def guppy_multiple_ma(df):
    """Guppy Multiple Moving Average"""
    # Corto plazo
    short_periods = [3, 5, 8, 10, 12, 15]
    # Largo plazo
    long_periods = [30, 35, 40, 45, 50, 60]
    
    for p in short_periods:
        df[f'ema_short_{p}'] = df['close'].ewm(span=p).mean()
    for p in long_periods:
        df[f'ema_long_{p}'] = df['close'].ewm(span=p).mean()
```

## Alertas

```python
def ma_alerts(df, ticker):
    """Generar alertas basadas en MAs"""
    alerts = []
    
    latest = df.iloc[-1]
    
    # Alerta de cruce
    if latest['golden_cross']:
        alerts.append(f"{ticker}: GOLDEN CROSS {latest['fast_ma']:.2f} > {latest['slow_ma']:.2f}")
    
    # Alerta de soporte
    if latest['sma_20_support']:
        alerts.append(f"{ticker}: BOUNCE en 20 SMA @ ${latest['sma_20']:.2f}")
    
    return alerts
```

## Siguiente Paso

Continuemos con [Volumen y RVol](volume_rvol.md), el combustible de los movimientos.