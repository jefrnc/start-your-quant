# Limpieza y Estructuración de Datos

## Por Qué Importa

Garbage in, garbage out. Puedes tener la mejor estrategia del mundo, pero si tus datos están mal, tus resultados serán ficción. He perdido días debuggeando "estrategias rotas" que en realidad tenían datos corruptos.

## Problemas Comunes en Trading Data

### 1. Missing Data (Huecos)
```python
# Problema: Faltan barras durante horario de mercado
datetime              close   volume
2024-01-15 09:30:00  175.00  500000
2024-01-15 09:31:00  175.20  450000
2024-01-15 09:32:00  NaN     NaN      # <-- Missing!
2024-01-15 09:33:00  175.45  380000
```

**Solución:**
```python
def fill_missing_bars(df, market_hours_only=True):
    # Crear índice completo
    if market_hours_only:
        full_index = pd.date_range(
            start=df.index[0].replace(hour=9, minute=30),
            end=df.index[-1].replace(hour=16, minute=0),
            freq='1min'
        )
        # Filtrar solo horario de mercado
        full_index = full_index[(full_index.time >= pd.Timestamp('09:30').time()) & 
                               (full_index.time <= pd.Timestamp('16:00').time())]
    
    # Reindexar y forward fill
    df = df.reindex(full_index)
    df['close'] = df['close'].fillna(method='ffill')
    df['volume'] = df['volume'].fillna(0)  # Volumen 0 si no hubo trades
    
    # OHLC: forward fill desde el close anterior
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df['close'])
    df['low'] = df['low'].fillna(df['close'])
    
    return df
```

### 2. Outliers y Fat Fingers
```python
# Problema: Trades erróneos que distorsionan el análisis
def detect_outliers(df, method='iqr', threshold=3):
    if method == 'iqr':
        # Método IQR (Interquartile Range)
        Q1 = df['close'].quantile(0.25)
        Q3 = df['close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
    elif method == 'zscore':
        # Método Z-score
        mean = df['close'].mean()
        std = df['close'].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    
    # Marcar outliers
    df['is_outlier'] = (df['close'] < lower_bound) | (df['close'] > upper_bound)
    
    # Opción 1: Eliminar
    # df = df[~df['is_outlier']]
    
    # Opción 2: Capear (mejor para trading)
    df.loc[df['close'] > upper_bound, 'close'] = upper_bound
    df.loc[df['close'] < lower_bound, 'close'] = lower_bound
    
    return df
```

### 3. Ajustes por Splits y Dividendos
```python
# Problema: AAPL hizo split 4:1, datos históricos incorrectos
def adjust_for_splits(df, splits_df):
    """
    splits_df debe tener columnas: date, ratio
    """
    df = df.copy()
    
    for _, split in splits_df.iterrows():
        split_date = split['date']
        ratio = split['ratio']
        
        # Ajustar precios anteriores al split
        mask = df.index < split_date
        df.loc[mask, ['open', 'high', 'low', 'close']] /= ratio
        df.loc[mask, 'volume'] *= ratio
    
    return df

# Usando yfinance (ya viene ajustado)
ticker = yf.Ticker('AAPL')
# auto_adjust=True ajusta automáticamente por splits y dividendos
df = ticker.history(period='2y', auto_adjust=True)
```

### 4. Timezone Hell
```python
# Problema: Mezclar timezones = desastre
def standardize_timezone(df, target_tz='America/New_York'):
    """
    Convierte todo a Eastern Time (NYSE)
    """
    if df.index.tz is None:
        # Asumir UTC si no tiene timezone
        df.index = df.index.tz_localize('UTC')
    
    # Convertir a Eastern
    df.index = df.index.tz_convert(target_tz)
    
    # Opcional: remover timezone para cálculos más rápidos
    # df.index = df.index.tz_localize(None)
    
    return df
```

### 5. Duplicados
```python
# Problema: El mismo trade reportado múltiples veces
def remove_duplicates(df):
    # Método 1: Duplicados exactos
    df = df[~df.index.duplicated(keep='first')]
    
    # Método 2: Duplicados en ventana temporal (más agresivo)
    df = df.sort_index()
    time_diff = df.index.to_series().diff()
    
    # Eliminar si hay otro trade en < 1 segundo
    mask = time_diff > pd.Timedelta('1s')
    mask.iloc[0] = True  # Mantener el primer trade
    
    return df[mask]
```

## Pipeline de Limpieza Completo

```python
class DataCleaner:
    def __init__(self, config=None):
        self.config = config or self.default_config()
    
    def default_config(self):
        return {
            'remove_outliers': True,
            'outlier_method': 'iqr',
            'outlier_threshold': 3,
            'fill_missing': True,
            'adjust_splits': True,
            'timezone': 'America/New_York',
            'validate_prices': True
        }
    
    def clean(self, df, ticker=None):
        """Pipeline completo de limpieza"""
        original_len = len(df)
        
        # 1. Validación básica
        df = self.validate_basic(df)
        
        # 2. Remover duplicados
        df = df[~df.index.duplicated(keep='first')]
        
        # 3. Timezone
        if self.config['timezone']:
            df = standardize_timezone(df, self.config['timezone'])
        
        # 4. Fill missing
        if self.config['fill_missing']:
            df = fill_missing_bars(df)
        
        # 5. Outliers
        if self.config['remove_outliers']:
            df = detect_outliers(
                df, 
                method=self.config['outlier_method'],
                threshold=self.config['outlier_threshold']
            )
        
        # 6. Validar OHLC relationships
        if self.config['validate_prices']:
            df = self.validate_ohlc(df)
        
        # 7. Ajustar por splits si tenemos la info
        if self.config['adjust_splits'] and ticker:
            df = self.auto_adjust_splits(df, ticker)
        
        print(f"Cleaned {ticker}: {original_len} -> {len(df)} rows")
        return df
    
    def validate_basic(self, df):
        """Validaciones básicas de sanidad"""
        # Remover filas con todos NaN
        df = df.dropna(how='all')
        
        # Remover volumen negativo
        df = df[df['volume'] >= 0]
        
        # Remover precios negativos o cero
        price_cols = ['open', 'high', 'low', 'close']
        df = df[(df[price_cols] > 0).all(axis=1)]
        
        return df
    
    def validate_ohlc(self, df):
        """Asegurar que high >= low, etc."""
        # High debe ser >= que open, close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        # Low debe ser <= que open, close
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
    
    def auto_adjust_splits(self, df, ticker):
        """Auto-detectar y ajustar splits grandes"""
        # Detectar cambios de precio > 40% overnight
        df['prev_close'] = df['close'].shift(1)
        df['overnight_change'] = df['open'] / df['prev_close']
        
        potential_splits = df[
            (df['overnight_change'] < 0.6) |  # Posible split
            (df['overnight_change'] > 1.4)     # Posible reverse split
        ]
        
        if len(potential_splits) > 0:
            print(f"Warning: Potential splits detected for {ticker}")
            print(potential_splits[['close', 'prev_close', 'overnight_change']])
        
        df = df.drop(['prev_close', 'overnight_change'], axis=1)
        return df
```

## Estructuración para Analysis

### Agregar Features Comunes
```python
def add_technical_features(df):
    """Agregar indicadores comunes para análisis"""
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatilidad
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Rango
    df['range'] = (df['high'] - df['low']) / df['low'] * 100
    df['range_avg'] = df['range'].rolling(20).mean()
    
    # Volumen
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Gaps
    df['gap'] = df['open'] / df['close'].shift(1) - 1
    
    return df
```

### Estructura para Backtesting
```python
class BacktestData:
    def __init__(self, data, universe=None):
        self.data = data
        self.universe = universe or list(data.keys())
        self.current_index = 0
        
    def prepare_for_backtest(self):
        """Preparar data para backtesting eficiente"""
        for ticker in self.universe:
            df = self.data[ticker]
            
            # Pre-calcular indicadores
            df = add_technical_features(df)
            
            # Crear columnas para señales
            df['signal'] = 0
            df['position'] = 0
            df['pnl'] = 0
            
            # Indexar por fecha para lookups rápidos
            df = df.sort_index()
            
            self.data[ticker] = df
    
    def get_snapshot(self, date, lookback=20):
        """Obtener vista de los datos hasta cierta fecha"""
        snapshot = {}
        for ticker in self.universe:
            df = self.data[ticker]
            # Datos hasta la fecha, incluyendo lookback
            mask = df.index <= date
            historical = df[mask].tail(lookback)
            snapshot[ticker] = historical
        
        return snapshot
```

## Validación Final

```python
def validate_dataset(data_dict, start_date, end_date):
    """Validación completa del dataset antes de backtest"""
    report = {
        'tickers': len(data_dict),
        'issues': [],
        'stats': {}
    }
    
    for ticker, df in data_dict.items():
        # Verificar cobertura temporal
        actual_start = df.index[0]
        actual_end = df.index[-1]
        
        if actual_start > pd.Timestamp(start_date):
            report['issues'].append(f"{ticker}: Data starts late ({actual_start})")
        
        if actual_end < pd.Timestamp(end_date):
            report['issues'].append(f"{ticker}: Data ends early ({actual_end})")
        
        # Stats por ticker
        report['stats'][ticker] = {
            'rows': len(df),
            'missing_closes': df['close'].isna().sum(),
            'zero_volume_bars': (df['volume'] == 0).sum(),
            'date_range': f"{actual_start} to {actual_end}"
        }
    
    return report

# Uso
report = validate_dataset(cleaned_data, '2023-01-01', '2024-01-01')
if report['issues']:
    print("WARNING: Data issues found:")
    for issue in report['issues']:
        print(f"  - {issue}")
```

## Mi Checklist Personal

```python
# clean_all.py
def my_cleaning_pipeline(raw_data):
    steps = [
        ('Remove duplicates', lambda df: df[~df.index.duplicated()]),
        ('Fix timezone', lambda df: standardize_timezone(df)),
        ('Validate OHLC', lambda df: validate_ohlc(df)),
        ('Fill missing', lambda df: fill_missing_bars(df)),
        ('Remove outliers', lambda df: detect_outliers(df, method='iqr')),
        ('Add features', lambda df: add_technical_features(df)),
        ('Final validation', lambda df: validate_basic(df))
    ]
    
    for step_name, step_func in steps:
        print(f"Running: {step_name}")
        raw_data = step_func(raw_data)
    
    return raw_data
```

## Storage Best Practices

```python
# Guardar datos limpios
def save_clean_data(df, ticker, format='parquet'):
    date_str = pd.Timestamp.now().strftime('%Y%m%d')
    
    if format == 'parquet':
        # Más eficiente para lectura
        df.to_parquet(f'data/clean/{ticker}_{date_str}.parquet')
    elif format == 'hdf':
        # Mejor para datasets muy grandes
        df.to_hdf(f'data/clean/{ticker}_{date_str}.h5', key='data')
    elif format == 'csv':
        # Compatible pero menos eficiente
        df.to_csv(f'data/clean/{ticker}_{date_str}.csv')
```

## Red Flags en tus Datos

1. **Demasiados gaps**: Proveedor poco confiable
2. **Volumen errático**: Posibles ajustes incorrectos
3. **Precios perfectamente redondos**: Data sintética
4. **Patrones repetitivos**: Posible data duplicada
5. **Volatilidad irreal**: Outliers no detectados

## Siguiente Paso

Con datos limpios, ahora podemos crear [Datasets para Backtesting](backtesting_datasets.md) optimizados.