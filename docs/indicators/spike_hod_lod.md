# Spike HOD/LOD (High/Low of Day)

## El Arte del Timing

En small caps, los spikes de HOD (High of Day) y LOD (Low of Day) son momentos decisivos. Un break del HOD con volumen puede significar el inicio de un runner. Un break del LOD puede ser capitulación.

## Tracking HOD/LOD en Tiempo Real

```python
def track_hod_lod(df):
    """Trackear HOD/LOD durante el día"""
    # Agrupar por fecha
    df['date'] = df.index.date
    
    # HOD/LOD running
    df['hod'] = df.groupby('date')['high'].cummax()
    df['lod'] = df.groupby('date')['low'].cummin()
    
    # Distancia de HOD/LOD
    df['distance_from_hod'] = (df['hod'] - df['close']) / df['close'] * 100
    df['distance_from_lod'] = (df['close'] - df['lod']) / df['lod'] * 100
    
    # Tiempo desde último HOD/LOD
    df['is_new_hod'] = df['high'] >= df['hod']
    df['is_new_lod'] = df['low'] <= df['lod']
    
    # Minutes since HOD/LOD
    df['minutes_since_hod'] = 0
    df['minutes_since_lod'] = 0
    
    for date in df['date'].unique():
        mask = df['date'] == date
        daily_data = df[mask].copy()
        
        last_hod_idx = 0
        last_lod_idx = 0
        
        for i, (idx, row) in enumerate(daily_data.iterrows()):
            if row['is_new_hod']:
                last_hod_idx = i
            if row['is_new_lod']:
                last_lod_idx = i
                
            df.loc[idx, 'minutes_since_hod'] = i - last_hod_idx
            df.loc[idx, 'minutes_since_lod'] = i - last_lod_idx
    
    return df
```

## HOD Spike Setup

```python
def hod_spike_setup(df, volume_threshold=2, strength_threshold=0.02):
    """Detectar spike de HOD con confirmación"""
    df = track_hod_lod(df)
    df = calculate_rvol(df)
    
    # Condiciones para HOD spike
    df['hod_break'] = (
        df['is_new_hod'] &  # Nuevo HOD
        (df['rvol'] > volume_threshold) &  # Volumen alto
        (df['close'] > df['open'])  # Vela verde
    )
    
    # Fuerza del break
    df['hod_break_strength'] = np.where(
        df['hod_break'],
        (df['close'] - df['hod'].shift(1)) / df['hod'].shift(1) * 100,
        0
    )
    
    # Clasificar calidad del break
    df['hod_quality'] = pd.cut(
        df['hod_break_strength'],
        bins=[0, 0.5, 2, 5, np.inf],
        labels=['weak', 'decent', 'strong', 'explosive']
    )
    
    # Continuación del spike
    df['hod_continuation'] = (
        df['hod_break'] &
        (df['hod_break_strength'] > strength_threshold) &
        (df['close'] > df['hod'].shift(1) * 1.01)  # 1% sobre HOD anterior
    )
    
    return df
```

## LOD Bounce Detection

```python
def lod_bounce_setup(df, bounce_threshold=0.03):
    """Detectar bounces del LOD"""
    df = track_hod_lod(df)
    
    # Test del LOD
    df['lod_test'] = (
        (df['low'] <= df['lod'] * 1.001) &  # Cerca del LOD
        (df['close'] > df['lod'] * 1.01)    # Cierra 1% arriba del LOD
    )
    
    # Fuerza del bounce
    df['bounce_strength'] = np.where(
        df['lod_test'],
        (df['close'] - df['low']) / df['low'] * 100,
        0
    )
    
    # Double bottom
    df['double_bottom'] = (
        df['lod_test'] &
        (df['minutes_since_lod'] > 30) &  # Al menos 30 min desde último LOD
        (abs(df['low'] - df['lod']) / df['lod'] < 0.005)  # Dentro del 0.5%
    )
    
    # Hammer/Doji en LOD
    df['hammer_at_lod'] = (
        df['lod_test'] &
        ((df['close'] - df['low']) / (df['high'] - df['low']) > 0.7) &  # Close en top 30%
        ((df['high'] - df['low']) / df['open'] > 0.02)  # Rango mínimo 2%
    )
    
    return df
```

## Multi-Day HOD/LOD Levels

```python
def multi_day_levels(ticker, lookback_days=5):
    """Obtener niveles HOD/LOD de múltiples días"""
    levels = {}
    
    for i in range(lookback_days):
        date = pd.Timestamp.now().date() - pd.Timedelta(days=i)
        
        try:
            daily_data = get_intraday_data(ticker, date)
            levels[date] = {
                'hod': daily_data['high'].max(),
                'lod': daily_data['low'].min(),
                'volume': daily_data['volume'].sum(),
                'range_pct': (daily_data['high'].max() - daily_data['low'].min()) / daily_data['low'].min() * 100
            }
        except:
            continue
    
    # Crear DataFrame de niveles
    levels_df = pd.DataFrame(levels).T
    
    # Identificar niveles clave
    levels_df['key_resistance'] = levels_df['hod'] > levels_df['hod'].quantile(0.8)
    levels_df['key_support'] = levels_df['lod'] < levels_df['lod'].quantile(0.2)
    
    return levels_df
```

## Time-Based HOD/LOD Analysis

```python
def hod_lod_by_time(df):
    """Analizar cuándo ocurren típicamente HOD/LOD"""
    df = track_hod_lod(df)
    
    # Agregar timestamp info
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_of_day'] = df.index.time
    
    # Frecuencia de HOD por hora
    hod_times = df[df['is_new_hod']]['hour'].value_counts().sort_index()
    lod_times = df[df['is_new_lod']]['hour'].value_counts().sort_index()
    
    # Probabilidad de HOD/LOD por período
    time_periods = {
        'opening': (9, 10),
        'morning': (10, 12),
        'midday': (12, 14),
        'afternoon': (14, 16)
    }
    
    hod_probabilities = {}
    lod_probabilities = {}
    
    for period, (start, end) in time_periods.items():
        period_mask = (df['hour'] >= start) & (df['hour'] < end)
        
        hod_in_period = df[period_mask & df['is_new_hod']].shape[0]
        lod_in_period = df[period_mask & df['is_new_lod']].shape[0]
        total_in_period = df[period_mask].shape[0]
        
        hod_probabilities[period] = hod_in_period / total_in_period
        lod_probabilities[period] = lod_in_period / total_in_period
    
    return {
        'hod_by_hour': hod_times,
        'lod_by_hour': lod_times,
        'hod_probabilities': hod_probabilities,
        'lod_probabilities': lod_probabilities
    }
```

## Failed Break Analysis

```python
def analyze_failed_breaks(df, failure_threshold=0.005):
    """Analizar breaks fallidos de HOD/LOD"""
    df = track_hod_lod(df)
    
    # Identificar breaks iniciales
    df['hod_break_attempt'] = df['high'] > df['hod'].shift(1)
    df['lod_break_attempt'] = df['low'] < df['lod'].shift(1)
    
    # Failed breaks
    df['failed_hod_break'] = (
        df['hod_break_attempt'] &
        (df['close'] < df['hod'].shift(1) * (1 + failure_threshold))
    )
    
    df['failed_lod_break'] = (
        df['lod_break_attempt'] &
        (df['close'] > df['lod'].shift(1) * (1 - failure_threshold))
    )
    
    # Strength of rejection
    df['hod_rejection_strength'] = np.where(
        df['failed_hod_break'],
        (df['hod'].shift(1) - df['close']) / df['close'] * 100,
        0
    )
    
    df['lod_rejection_strength'] = np.where(
        df['failed_lod_break'],
        (df['close'] - df['lod'].shift(1)) / df['lod'].shift(1) * 100,
        0
    )
    
    return df
```

## Progressive HOD/LOD Strategy

```python
def progressive_hod_strategy(df, position_sizes=[0.25, 0.25, 0.5]):
    """Estrategia de entries progresivos en HOD breaks"""
    df = hod_spike_setup(df)
    
    # Diferentes niveles de confirmación
    df['hod_level_1'] = df['hod_break']  # Break inicial
    df['hod_level_2'] = (  # Break con volumen
        df['hod_break'] & 
        (df['rvol'] > 3)
    )
    df['hod_level_3'] = (  # Break fuerte con continuación
        df['hod_continuation'] &
        (df['hod_break_strength'] > 2)
    )
    
    # Backtesting con entries progresivos
    signals = []
    
    for i, row in df.iterrows():
        if row['hod_level_1']:
            signals.append({
                'timestamp': i,
                'entry_level': 1,
                'price': row['close'],
                'size': position_sizes[0],
                'stop': row['hod'] * 0.98
            })
        
        if row['hod_level_2']:
            signals.append({
                'timestamp': i,
                'entry_level': 2,
                'price': row['close'],
                'size': position_sizes[1],
                'stop': row['hod'] * 0.99
            })
        
        if row['hod_level_3']:
            signals.append({
                'timestamp': i,
                'entry_level': 3,
                'price': row['close'],
                'size': position_sizes[2],
                'stop': row['lod']  # Wider stop for strongest signal
            })
    
    return pd.DataFrame(signals)
```

## Gap and HOD Combination

```python
def gap_hod_combo(df, gap_threshold=10):
    """Combinar gap analysis con HOD breaks"""
    # Calcular gap
    df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    df = track_hod_lod(df)
    df = hod_spike_setup(df)
    
    # Gap up + holding gains + HOD break
    df['gap_hod_setup'] = (
        (df['gap_pct'] > gap_threshold) &  # Gap up significativo
        (df['close'] > df['open']) &       # Manteniendo gains
        df['hod_break']                    # Breaking HOD
    )
    
    # Classify setup strength
    df['setup_strength'] = 0
    df.loc[df['gap_hod_setup'] & (df['gap_pct'] > 20), 'setup_strength'] = 3
    df.loc[df['gap_hod_setup'] & (df['gap_pct'] > 15), 'setup_strength'] = 2
    df.loc[df['gap_hod_setup'], 'setup_strength'] = 1
    
    return df
```

## Real-Time Monitoring

```python
class HODLODMonitor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.daily_hod = 0
        self.daily_lod = float('inf')
        self.hod_breaks = []
        self.lod_tests = []
        
    def update(self, new_bar):
        """Update con nueva barra"""
        # Actualizar HOD/LOD
        if new_bar['high'] > self.daily_hod:
            self.daily_hod = new_bar['high']
            self.hod_breaks.append({
                'time': new_bar.name,
                'price': new_bar['high'],
                'volume': new_bar['volume']
            })
        
        if new_bar['low'] < self.daily_lod:
            self.daily_lod = new_bar['low']
            self.lod_tests.append({
                'time': new_bar.name,
                'price': new_bar['low'],
                'volume': new_bar['volume']
            })
    
    def get_status(self):
        """Estado actual"""
        return {
            'ticker': self.ticker,
            'hod': self.daily_hod,
            'lod': self.daily_lod,
            'hod_breaks_count': len(self.hod_breaks),
            'lod_tests_count': len(self.lod_tests),
            'latest_hod_break': self.hod_breaks[-1] if self.hod_breaks else None,
            'latest_lod_test': self.lod_tests[-1] if self.lod_tests else None
        }
```

## Alertas HOD/LOD

```python
def hod_lod_alerts(df, ticker):
    """Generar alertas para HOD/LOD"""
    alerts = []
    latest = df.iloc[-1]
    
    # HOD break
    if latest.get('hod_break', False):
        alerts.append(f"🚀 {ticker}: NEW HOD @ ${latest['hod']:.2f} (RVol: {latest['rvol']:.1f}x)")
    
    # Strong HOD continuation
    if latest.get('hod_continuation', False):
        alerts.append(f"💪 {ticker}: HOD CONTINUATION - Strength: {latest['hod_break_strength']:.1f}%")
    
    # LOD bounce
    if latest.get('lod_test', False):
        alerts.append(f"⚡ {ticker}: LOD BOUNCE @ ${latest['lod']:.2f}")
    
    # Failed break (reversal)
    if latest.get('failed_hod_break', False):
        alerts.append(f"⚠️ {ticker}: FAILED HOD BREAK - Rejection at ${latest['hod']:.2f}")
    
    return alerts
```

## Siguiente Paso

Finalizemos los indicadores con [Gap % y Float](gap_float.md), fundamentales para el screening de small caps.