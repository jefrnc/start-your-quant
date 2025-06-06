# Tipos de Datos: EOD, Intradía y Tick

## End-of-Day (EOD) Data

### ¿Qué es?
Datos con un solo punto por día: Open, High, Low, Close, Volume (OHLCV).

```python
# Ejemplo EOD data
date         open    high    low     close   volume
2024-01-15   175.00  178.50  174.25  177.80  45000000
2024-01-16   177.80  179.00  176.00  178.25  42000000
```

### Cuándo Usar
- Swing trading (holding días/semanas)
- Análisis de tendencias largas
- Screening inicial de ideas
- Backtesting de estrategias position

### Pros y Contras
✅ **Pros:**
- Gratis o barato
- Fácil de manejar
- Menos ruido
- Backtests rápidos

❌ **Contras:**
- No sirve para day trading
- Pierde información intradía
- No puedes optimizar entries/exits

### Código Ejemplo
```python
import yfinance as yf
import pandas as pd

# Obtener EOD data
ticker = 'AAPL'
eod_data = yf.download(ticker, start='2023-01-01', end='2024-01-01')

# Calcular métricas simples
eod_data['SMA20'] = eod_data['Close'].rolling(20).mean()
eod_data['Daily_Range'] = ((eod_data['High'] - eod_data['Low']) / eod_data['Low'] * 100)
eod_data['Gap'] = (eod_data['Open'] / eod_data['Close'].shift(1) - 1) * 100
```

## Intraday Data (Barras de Minutos)

### ¿Qué es?
OHLCV para intervalos específicos: 1min, 5min, 15min, etc.

```python
# Ejemplo 5-min bars
datetime              open    high    low     close   volume
2024-01-15 09:30:00  175.00  175.50  174.95  175.20  500000
2024-01-15 09:35:00  175.20  175.80  175.10  175.75  450000
2024-01-15 09:40:00  175.75  176.00  175.50  175.55  380000
```

### Cuándo Usar
- Day trading
- Entries/exits precisos
- Patrones intradía (VWAP, breakouts)
- Gestión de riesgo intradía

### Resoluciones Comunes
```python
RESOLUTIONS = {
    'scalping': '1min',
    'day_trading': '5min',
    'swing_entries': '15min',
    'trend_confirmation': '60min'
}
```

### Manejo de Datos
```python
# Con Polygon.io
from polygon import RESTClient
client = RESTClient("YOUR_API_KEY")

# 5-minute bars
bars = client.get_aggs(
    ticker="AAPL",
    multiplier=5,
    timespan="minute",
    from_="2024-01-15",
    to="2024-01-15"
)

# Convertir a DataFrame
df = pd.DataFrame(bars)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)

# Calcular VWAP
df['cum_vol'] = df['volume'].cumsum()
df['cum_vol_price'] = (df['close'] * df['volume']).cumsum()
df['vwap'] = df['cum_vol_price'] / df['cum_vol']
```

## Tick Data

### ¿Qué es?
Cada transacción individual con timestamp exacto.

```python
# Ejemplo tick data
timestamp              price   size  exchange  conditions
2024-01-15 09:30:00.123  175.00  100   NYSE     ['regular']
2024-01-15 09:30:00.125  175.01  500   NASDAQ   ['regular']
2024-01-15 09:30:00.127  175.00  200   ARCA     ['odd_lot']
```

### Cuándo Usar
- High frequency trading
- Análisis de microestructura
- Detección de blocks/dark pools
- Slippage analysis exacto

### Consideraciones
- **Tamaño**: 1GB+ por día para stocks líquidos
- **Procesamiento**: Necesitas código optimizado
- **Costo**: $100-500+/mes para data quality

### Trabajando con Tick Data
```python
# Ejemplo con Polygon tick data
trades = client.list_trades(
    ticker="AAPL",
    timestamp="2024-01-15",
    limit=50000
)

# Procesar para análisis
tick_df = pd.DataFrame(trades)
tick_df['timestamp'] = pd.to_datetime(tick_df['sip_timestamp'], unit='ns')

# Detectar prints grandes
large_prints = tick_df[tick_df['size'] >= 10000]

# Analizar por exchange
exchange_volume = tick_df.groupby('exchange')['size'].sum()

# Crear barras de tiempo desde ticks
def create_time_bars(ticks, bar_size='5T'):
    ticks.set_index('timestamp', inplace=True)
    bars = ticks.resample(bar_size).agg({
        'price': ['first', 'max', 'min', 'last'],
        'size': 'sum'
    })
    bars.columns = ['open', 'high', 'low', 'close', 'volume']
    return bars
```

## Comparación Práctica

| Tipo | Tamaño/Día | Costo | Use Case | Latencia |
|------|------------|-------|----------|----------|
| EOD | 1 línea | $0 | Swing/Position | N/A |
| 1-min | 390 líneas | $20-50 | Day trading | 1 min |
| Tick | 100k-1M líneas | $100+ | HFT/Analysis | Real-time |

## Agregación de Datos

### De Tick a Minuto
```python
def aggregate_ticks_to_bars(ticks, bar_type='time', bar_size=60):
    if bar_type == 'time':
        # Barras de tiempo (cada 60 segundos)
        bars = ticks.resample(f'{bar_size}S').agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum'
        })
    
    elif bar_type == 'volume':
        # Barras de volumen (cada N shares)
        bars = aggregate_volume_bars(ticks, bar_size)
    
    elif bar_type == 'dollar':
        # Barras de dólares (cada $N traded)
        ticks['dollar_vol'] = ticks['price'] * ticks['size']
        bars = aggregate_dollar_bars(ticks, bar_size)
    
    return bars
```

### Volume Bars (Avanzado)
```python
def create_volume_bars(ticks, volume_per_bar=100000):
    bars = []
    current_bar = {'volume': 0, 'high': 0, 'low': float('inf')}
    
    for _, tick in ticks.iterrows():
        current_bar['volume'] += tick['size']
        current_bar['high'] = max(current_bar['high'], tick['price'])
        current_bar['low'] = min(current_bar['low'], tick['price'])
        
        if current_bar['volume'] >= volume_per_bar:
            bars.append(current_bar)
            current_bar = {'volume': 0, 'high': 0, 'low': float('inf')}
    
    return pd.DataFrame(bars)
```

## Calidad de Datos

### Checklist de Validación
```python
def validate_intraday_data(df):
    issues = []
    
    # 1. Gaps temporales
    expected_bars = pd.date_range(
        start=df.index[0].replace(hour=9, minute=30),
        end=df.index[0].replace(hour=16, minute=0),
        freq='1min'
    )
    missing = expected_bars.difference(df.index)
    if len(missing) > 0:
        issues.append(f"Missing {len(missing)} bars")
    
    # 2. Precios negativos o cero
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        issues.append("Zero or negative prices found")
    
    # 3. High/Low consistency
    invalid_hl = df['high'] < df['low']
    if invalid_hl.any():
        issues.append(f"{invalid_hl.sum()} bars with high < low")
    
    # 4. Volumen sospechoso
    if (df['volume'] == 0).sum() > len(df) * 0.1:
        issues.append("Too many zero volume bars")
    
    return issues
```

## Mi Approach Personal

```python
# Uso diferentes tipos según la estrategia
DATA_CONFIG = {
    'gap_scanner': {
        'type': 'EOD',
        'source': 'yahoo',
        'reason': 'Solo necesito gap % overnight'
    },
    'vwap_trading': {
        'type': '1min',
        'source': 'polygon',
        'reason': 'VWAP accuracy + entry timing'
    },
    'tape_reading': {
        'type': 'tick',
        'source': 'polygon_websocket',
        'reason': 'Ver order flow en tiempo real'
    }
}
```

## Tips Prácticos

1. **Empieza con EOD**, es gratis y suficiente para aprender
2. **Upgrade a 5-min** cuando hagas day trading
3. **Tick data** solo si haces HFT o analysis profundo
4. **Guarda localmente** data que uses frecuentemente
5. **Timestamp timezone** siempre en Eastern (NYSE time)

## Storage Eficiente

```python
# Guardar eficientemente
df.to_parquet('data/AAPL_2024_1min.parquet')  # Mejor que CSV
df.to_hdf('data/ticks.h5', key='AAPL')  # Para datasets grandes

# Leer eficientemente
df = pd.read_parquet('data/AAPL_2024_1min.parquet')
df = pd.read_hdf('data/ticks.h5', key='AAPL', 
                 where='timestamp >= "2024-01-15" & timestamp < "2024-01-16"')
```

## Siguiente Paso

Ahora que entiendes los tipos de datos, vamos a [Limpieza de Datos](data_cleaning.md) para asegurar que tu data sea confiable.