# Fuentes de Datos Históricos

## Overview

La calidad de tus datos determina la calidad de tu backtesting. Aquí están las fuentes que uso actualmente, cada una con sus pros y contras.

## Yahoo Finance

### Características
- **Gratis** para datos EOD
- Cobertura global de stocks
- Datos desde 1960s para muchos tickers
- API Python sencilla con `yfinance`

### Instalación y Uso
```python
pip install yfinance
```

```python
import yfinance as yf
import pandas as pd

# Datos diarios
ticker = yf.Ticker("AAPL")
data = ticker.history(start="2023-01-01", end="2023-12-31")

# Múltiples tickers
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2023-01-01", end="2023-12-31")

# Datos intraday (últimos 60 días max)
intraday = ticker.history(period="1d", interval="1m")
```

### Limitaciones
- Solo datos EOD confiables
- Intraday limitado a 60 días
- No incluye dark pool o odd lots
- Splits y dividendos a veces incorrectos

### Cuándo Usar
- Backtesting inicial de estrategias swing
- Screening rápido de ideas
- Datos históricos largos

## Polygon.io

### Características
- **API profesional** con datos tick-by-tick
- WebSocket para datos real-time
- Histórico completo desde 2003
- Incluye dark pools, odd lots, condiciones

### Setup
```python
pip install polygon-api-client
```

```python
from polygon import RESTClient
import os

client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))

# Aggregates (barras)
aggs = client.get_aggs(
    ticker="AAPL",
    multiplier=5,
    timespan="minute",
    from_="2023-01-01",
    to="2023-12-31"
)

# Trades (tick data)
trades = client.list_trades(
    ticker="AAPL",
    timestamp="2023-06-01"
)

# Quote data (NBBO)
quotes = client.list_quotes(
    ticker="AAPL",
    timestamp="2023-06-01"
)
```

### Planes y Precios
- **Starter**: $29/mes - Datos EOD + 2 años historical
- **Developer**: $99/mes - 5 años historical + websocket
- **Advanced**: $249/mes - Full historical + todas las features

### Cuándo Usar
- Estrategias intraday que necesitan precisión
- Análisis de microestructura
- Backtesting con fills realistas

## TWS de Interactive Brokers

### Características
- Datos real-time incluidos con cuenta
- API robusta para automatización
- Histórico limitado pero gratis
- Conexión directa para live trading

### Configuración con ib_insync
```python
pip install ib_insync
```

```python
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Paper: 7497, Live: 7496

# Contract
contract = Stock('AAPL', 'SMART', 'USD')

# Datos históricos
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='30 D',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=True
)

# Convertir a DataFrame
df = util.df(bars)

# Real-time bars
def onBarUpdate(bars, hasNewBar):
    print(bars[-1])

bars = ib.reqRealTimeBars(contract, 5, 'TRADES', True)
bars.updateEvent += onBarUpdate
```

### Limitaciones
- Máximo 1 año de datos minute
- Rate limits estrictos
- Requiere TWS abierto

### Cuándo Usar
- Trading automatizado en producción
- Paper trading con datos reales
- Verificación de otros data sources

## DAS Trader

### Características
- Plataforma profesional de day trading
- Level 2 completo
- Hotkeys y automatización
- Integración con múltiples brokers

### Exportar Datos
```python
# DAS guarda logs en formato CSV
import pandas as pd
import glob

# Leer trades ejecutados
trades_files = glob.glob('C:/DAS/Trades/*.csv')
trades = pd.concat([pd.read_csv(f) for f in trades_files])

# Procesar para análisis
trades['Time'] = pd.to_datetime(trades['Time'])
trades['PnL'] = trades['ExitPrice'] - trades['EntryPrice']
```

### Integración con Python
```python
# Usar DAS API (requiere DAS Trader Pro)
import win32com.client

das = win32com.client.Dispatch("DAS.Application")
das.SendOrder("BUY", "AAPL", 100, "MARKET")
```

### Cuándo Usar
- Ejecución manual con análisis post-trade
- Combinar ejecución manual con análisis quant
- Testing de estrategias antes de automatizar

## QuantConnect

### Características
- **Plataforma cloud completa**
- Datos incluidos (equities, options, futures, forex, crypto)
- Motor de backtesting profesional
- Deploy directo a live trading

### Ejemplo de Algoritmo
```python
class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Agregar securities
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        
        # Indicadores
        self.sma = self.SMA("SPY", 20, Resolution.Daily)
        
    def OnData(self, data):
        if not self.sma.IsReady:
            return
            
        if data["SPY"].Price > self.sma.Current.Value:
            self.SetHoldings("SPY", 1.0)
        else:
            self.Liquidate("SPY")
```

### Ventajas
- No necesitas gestionar infraestructura
- Datos limpios y ajustados
- Community y ejemplos extensos

### Cuándo Usar
- Estrategias multi-asset complejas
- Cuando no quieres gestionar datos
- Transición rápida de backtest a live

## Flash Research

### Características
- Enfocado en **market microstructure**
- Análisis de tape reading
- Identificación de footprints institucionales
- Datos de options flow

### Casos de Uso
```python
# Ejemplo conceptual - Flash Research provee insights, no raw data
insights = {
    'institutional_accumulation': ['AAPL', 'MSFT'],
    'unusual_options_activity': [
        {'ticker': 'NVDA', 'strike': 500, 'volume': 10000}
    ],
    'dark_pool_prints': [
        {'ticker': 'TSLA', 'size': 500000, 'price': 250.50}
    ]
}

# Usar estos insights para filtrar universo
universe = screen_stocks(insights['institutional_accumulation'])
```

### Cuándo Usar
- Confirmación de señales técnicas
- Identificar acumulación institucional
- Options flow para direccionalidad

## Mi Stack Actual

```python
# config.py
DATA_SOURCES = {
    'historical': 'polygon',      # Para backtesting preciso
    'realtime': 'ibkr_tws',      # Para ejecución
    'screening': 'yahoo',         # Para ideas rápidas
    'research': 'quantconnect',   # Para estrategias complejas
    'insights': 'flash_research'  # Para confirmación
}

# data_manager.py
class DataManager:
    def __init__(self):
        self.polygon = PolygonClient()
        self.yahoo = YahooClient()
        self.ibkr = IBKRClient()
        
    def get_data(self, ticker, source='auto'):
        if source == 'auto':
            # Lógica para elegir mejor fuente
            if self.need_intraday:
                return self.polygon.get_data(ticker)
            else:
                return self.yahoo.get_data(ticker)
```

## Costos Mensuales

| Fuente | Costo | Lo que obtienes |
|--------|-------|-----------------|
| Yahoo Finance | $0 | EOD data, screening básico |
| Polygon.io | $99 | 5 años intraday, websocket |
| IBKR | $10 + comms | Real-time, ejecución |
| DAS Trader | $150 | Plataforma pro + data |
| QuantConnect | $0-$200 | Backtest + live deployment |
| Flash Research | Variable | Market intelligence |

**Total**: ~$300-400/mes para setup profesional

## Tips para Empezar

1. **Empieza gratis**: Yahoo Finance + paper trading IBKR
2. **Primer upgrade**: Polygon.io Developer ($99)
3. **Cuando seas consistente**: Agrega DAS o similar
4. **Para escalar**: QuantConnect para múltiples estrategias

## Data Quality Checklist

```python
def validate_data(df):
    checks = {
        'no_gaps': df.index.is_monotonic_increasing,
        'no_nulls': not df.isnull().any().any(),
        'volume_positive': (df['Volume'] >= 0).all(),
        'prices_positive': (df[['Open', 'High', 'Low', 'Close']] > 0).all().all(),
        'high_low_valid': (df['High'] >= df['Low']).all(),
        'ohlc_valid': (
            (df['High'] >= df[['Open', 'Close']].max(axis=1)).all() &
            (df['Low'] <= df[['Open', 'Close']].min(axis=1)).all()
        )
    }
    
    return pd.Series(checks)
```

## Siguiente Paso

Continúa con [Tipos de Datos](data_types.md) para entender las diferencias entre EOD, intraday y tick data.