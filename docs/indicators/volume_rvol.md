# Volumen y RVol (Relative Volume)

## El Combustible del Movimiento

En small caps, el volumen es TODO. Sin volumen, no hay movimiento. Con volumen excesivo, hay oportunidad. RVol (Relative Volume) es mi indicador #1 para filtrar qué stocks mirar.

## Cálculo de RVol

```python
def calculate_rvol(df, lookback=10, time_based=True):
    """Calcular Relative Volume"""
    if time_based:
        # RVol por tiempo del día (más preciso)
        df['time'] = df.index.time
        
        # Volumen promedio para cada minuto de los últimos N días
        volume_by_time = {}
        for i in range(1, lookback + 1):
            date = df.index[-1].date() - pd.Timedelta(days=i)
            hist_data = df[df.index.date == date]
            for idx, row in hist_data.iterrows():
                time_key = idx.time()
                if time_key not in volume_by_time:
                    volume_by_time[time_key] = []
                volume_by_time[time_key].append(row['volume'])
        
        # Promedio por tiempo
        avg_volume_by_time = {
            time: np.mean(vols) for time, vols in volume_by_time.items()
        }
        
        # Calcular RVol
        df['avg_volume_time'] = df.index.time.map(avg_volume_by_time)
        df['rvol'] = df['volume'] / df['avg_volume_time']
        
    else:
        # RVol simple (menos preciso pero más rápido)
        df['avg_volume'] = df['volume'].rolling(lookback).mean().shift(1)
        df['rvol'] = df['volume'] / df['avg_volume']
    
    # Llenar NaN con 1 (volumen normal)
    df['rvol'] = df['rvol'].fillna(1)
    
    return df
```

## RVol para Day Trading

```python
def rvol_day_trading_setup(df, rvol_threshold=2):
    """Identificar setups basados en RVol"""
    df = calculate_rvol(df, time_based=True)
    
    # Clasificar niveles de RVol
    df['rvol_level'] = pd.cut(
        df['rvol'],
        bins=[0, 1, 2, 3, 5, 10, np.inf],
        labels=['low', 'normal', 'high', 'very_high', 'extreme', 'explosive']
    )
    
    # Detectar spikes de volumen
    df['volume_spike'] = df['rvol'] > rvol_threshold
    df['sustained_volume'] = df['volume_spike'].rolling(5).sum() >= 3  # 3 de 5 barras
    
    # Combinación con precio
    df['bullish_volume'] = df['volume_spike'] & (df['close'] > df['open'])
    df['bearish_volume'] = df['volume_spike'] & (df['close'] < df['open'])
    
    return df
```

## Volume Profile

Dónde se ejecutó más volumen = niveles clave.

```python
def calculate_volume_profile(df, bins=50):
    """Crear perfil de volumen"""
    # Definir bins de precio
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_bins = np.linspace(price_min, price_max, bins)
    
    # Acumular volumen por nivel de precio
    volume_profile = np.zeros(len(price_bins) - 1)
    
    for idx, row in df.iterrows():
        # Distribuir volumen entre low y high
        for i in range(len(price_bins) - 1):
            if price_bins[i] <= row['high'] and price_bins[i+1] >= row['low']:
                # Porción del rango que cae en este bin
                overlap_low = max(row['low'], price_bins[i])
                overlap_high = min(row['high'], price_bins[i+1])
                overlap_pct = (overlap_high - overlap_low) / (row['high'] - row['low'])
                volume_profile[i] += row['volume'] * overlap_pct
    
    # Crear DataFrame
    vp_df = pd.DataFrame({
        'price': (price_bins[:-1] + price_bins[1:]) / 2,
        'volume': volume_profile
    })
    
    # Identificar POC (Point of Control)
    vp_df['poc'] = vp_df['volume'] == vp_df['volume'].max()
    
    # Value Area (70% del volumen)
    total_volume = vp_df['volume'].sum()
    vp_df = vp_df.sort_values('volume', ascending=False)
    vp_df['cum_volume'] = vp_df['volume'].cumsum()
    vp_df['in_value_area'] = vp_df['cum_volume'] <= total_volume * 0.7
    
    return vp_df.sort_values('price')
```

## Volume Patterns

```python
def detect_volume_patterns(df, window=20):
    """Detectar patrones de volumen importantes"""
    # Preparar datos
    df['volume_ma'] = df['volume'].rolling(window).mean()
    df['volume_std'] = df['volume'].rolling(window).std()
    
    # 1. Climax Volume
    df['climax_volume'] = df['volume'] > (df['volume_ma'] + 3 * df['volume_std'])
    
    # 2. Dry Up (secado de volumen)
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

Detectar volumen institucional vs retail.

```python
def analyze_smart_money_volume(df, block_size=10000):
    """Identificar volumen institucional"""
    # Tamaño promedio de trade (aproximado)
    df['avg_trade_size'] = df['volume'] / df['tick_count'] if 'tick_count' in df else 100
    
    # Large blocks
    df['large_block'] = df['avg_trade_size'] > block_size
    
    # Volume en primera y última hora (institucionales)
    df['hour'] = df.index.hour
    df['institutional_hours'] = df['hour'].isin([9, 10, 15])
    
    # Dark pool prints (si tienes data)
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
    """Detectar breakouts con volumen"""
    df = calculate_rvol(df)
    
    # Preparar indicadores
    df['resistance'] = df['high'].rolling(20).max()
    df['support'] = df['low'].rolling(20).min()
    
    # Breakout alcista
    df['breakout_up'] = (
        (df['close'] > df['resistance'].shift(1)) &  # Rompe resistencia
        (df['rvol'] > volume_multiplier) &           # Con volumen alto
        (df['close'] > df['open'])                   # Vela verde
    )
    
    # Breakout bajista
    df['breakout_down'] = (
        (df['close'] < df['support'].shift(1)) &
        (df['rvol'] > volume_multiplier) &
        (df['close'] < df['open'])
    )
    
    # Fuerza del breakout
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
    """On Balance Volume con señales"""
    # OBV clásico
    df['obv'] = (df['volume'] * np.sign(df['close'] - df['close'].shift(1))).cumsum()
    
    # OBV suavizado
    df['obv_ema'] = df['obv'].ewm(span=20).mean()
    
    # Divergencias
    df['price_high'] = df['close'].rolling(20).max()
    df['obv_high'] = df['obv'].rolling(20).max()
    
    # Divergencia bajista: precio hace nuevo high, OBV no
    df['bearish_obv_divergence'] = (
        (df['close'] == df['price_high']) & 
        (df['obv'] < df['obv_high'].shift(20))
    )
    
    # Señal de confirmación
    df['obv_trend_up'] = df['obv'] > df['obv_ema']
    df['obv_breakout'] = df['obv'] > df['obv'].rolling(50).max().shift(1)
    
    return df
```

## Volume-Weighted Momentum

```python
def volume_weighted_momentum(df, period=14):
    """Momentum ponderado por volumen"""
    # Price momentum
    df['price_change'] = df['close'] - df['close'].shift(period)
    
    # Volume-weighted price change
    weights = df['volume'].rolling(period).apply(lambda x: x / x.sum())
    df['vw_momentum'] = df['price_change'] * weights
    
    # Normalizar
    df['vw_momentum_normalized'] = df['vw_momentum'] / df['close'] * 100
    
    # Señales
    df['strong_bullish_momentum'] = (
        (df['vw_momentum_normalized'] > 5) & 
        (df['rvol'] > 2)
    )
    
    return df
```

## Pre-Market Volume Analysis

```python
def analyze_premarket_volume(ticker, date):
    """Analizar volumen pre-market"""
    # Obtener data pre-market
    pm_start = pd.Timestamp(date).replace(hour=4, minute=0)
    pm_end = pd.Timestamp(date).replace(hour=9, minute=29)
    
    pm_data = get_intraday_data(ticker, start=pm_start, end=pm_end)
    
    # Métricas
    analysis = {
        'total_pm_volume': pm_data['volume'].sum(),
        'pm_vwap': calculate_vwap(pm_data)['vwap'].iloc[-1],
        'pm_high': pm_data['high'].max(),
        'pm_low': pm_data['low'].min(),
        'pm_range': (pm_data['high'].max() - pm_data['low'].min()) / pm_data['low'].min() * 100,
        'volume_profile': pm_data.groupby(pd.cut(pm_data['close'], bins=10))['volume'].sum()
    }
    
    # Comparar con días anteriores
    historical_pm_volume = []
    for i in range(1, 11):
        hist_date = date - pd.Timedelta(days=i)
        hist_pm = get_intraday_data(ticker, start=hist_date.replace(hour=4), 
                                   end=hist_date.replace(hour=9, minute=29))
        historical_pm_volume.append(hist_pm['volume'].sum())
    
    analysis['pm_rvol'] = analysis['total_pm_volume'] / np.mean(historical_pm_volume)
    
    return analysis
```

## Alertas de Volumen

```python
class VolumeAlertSystem:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.alerted = set()
        
    def check_alerts(self, ticker, current_data):
        alerts = []
        
        # RVol extremo
        if current_data['rvol'] > self.thresholds['rvol_extreme']:
            alert_key = f"{ticker}_rvol_{current_data.name}"
            if alert_key not in self.alerted:
                alerts.append(f"🚨 {ticker}: RVol {current_data['rvol']:.1f}x @ ${current_data['close']:.2f}")
                self.alerted.add(alert_key)
        
        # Climax volume
        if current_data.get('climax_volume', False):
            alerts.append(f"💥 {ticker}: CLIMAX VOLUME - Posible top/bottom")
        
        # Volume dry up en soporte
        if current_data.get('volume_dryup', False) and current_data['close'] > current_data['support']:
            alerts.append(f"📉 {ticker}: Volume dry up en soporte ${current_data['support']:.2f}")
        
        return alerts
```

## Tips Prácticos

### 1. Morning Volume Rate
```python
def morning_volume_rate(df):
    """Tasa de volumen en primera hora"""
    first_hour = df.between_time('09:30', '10:30')
    first_hour_vol = first_hour['volume'].sum()
    
    avg_daily_vol = df.groupby(df.index.date)['volume'].sum().mean()
    
    # Si primera hora > 30% del volumen diario promedio = día activo
    return first_hour_vol / avg_daily_vol
```

### 2. Volume Patterns por Hora
```python
def hourly_volume_pattern(df):
    """Patrón típico de volumen por hora"""
    df['hour'] = df.index.hour
    hourly_avg = df.groupby('hour')['volume'].mean()
    
    # Normalizar (primera hora = 100)
    hourly_pattern = hourly_avg / hourly_avg[9] * 100
    return hourly_pattern
```

## Siguiente Paso

Continuemos con [Spike HOD/LOD](spike_hod_lod.md), crítico para timing en small caps.