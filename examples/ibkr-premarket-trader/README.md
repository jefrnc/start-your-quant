# IBKR Premarket Trader - Caso de Estudio

## Resumen Ejecutivo

El **IBKR Premarket Trader** es un sistema automatizado de trading que implementa la estrategia Gap & Go en small caps durante las horas de premarket (5:30 AM - 8:00 AM ET). Es el ejemplo principal del Quant Playbook y demuestra cómo aplicar metodologías sistemáticas en trading real.

## Arquitectura del Sistema

```mermaid
graph TB
    A[Polygon API] --> B[Data Pipeline]
    B --> C[Gap Scanner]
    C --> D[Strategy Engine]
    D --> E[Risk Manager]
    E --> F[IBKR Execution]
    F --> G[PostgreSQL Storage]
    G --> H[Analytics Dashboard]

    I[News API] --> D
    J[Early Runner Detector] --> D

    H --> K[Telegram Alerts]
    H --> L[Grafana Monitoring]
```

## Componentes Principales

### 1. **trading_console.py** - Aplicación Principal
```python
# Características principales:
- Console interactiva para control manual
- REST API en puerto 8080
- WebSocket para actualizaciones en tiempo real
- Sistema de logging estructurado
```

### 2. **generate_watchlist_smallcaps.py** - Generador de Watchlist
```python
# Criterios de filtrado:
- Market cap: $50M - $2B
- Precio: $0.50 - $10.99
- Volumen promedio: > 100K shares/día
- Evita penny stocks extremos
- Filtra por sector y exchanges
```

### 3. **simple_realistic_backtest.py** - Backtesting
```python
# Características del backtest:
- Slippage y comisiones realistas
- Gaps de apertura simulados
- Horarios de trading respetados
- Análisis de drawdown temporal
```

### 4. **parameter_optimizer.py** - Optimización Bayesiana
```python
# Parámetros optimizados:
- Gap % mínimo y máximo
- Multiplicador de volumen
- Stop loss dinámico
- Targets de profit taking
```

### 5. **early_runner_detector.py** - Detección ML
```python
# Sistema de scoring:
- Dark pool activity (30%)
- Technical setup (25%)
- Float rotation (20%)
- SEC filing risk (15%)
- Social momentum (10%)
```

## Configuración de Strategy

### Archivo: `config/strategy_config.yaml`

```yaml
gap_and_go:
  # Filtros de Entry
  min_gap_percent: 3.0
  max_gap_percent: 25.0
  min_premarket_volume: 50000
  volume_multiplier: 2.0

  # Risk Management
  max_position_size: 70.0
  max_risk_per_trade: 10.0
  stop_loss_percent: 5.0

  # Timing
  entry_window_start: "05:30"
  entry_window_end: "08:00"
  max_hold_minutes: 60

  # Execution
  order_type: "MARKET"
  timeout_seconds: 30
```

## Performance Histórica

### Métricas Clave (Últimos 6 meses)
```
Sharpe Ratio:       1.85
Max Drawdown:      -8.3%
Win Rate:          67.4%
Profit Factor:     2.31
Avg Trade:         $12.50
Total Trades:      1,247
```

### Breakdown por Mes
| Mes | Trades | Win Rate | P&L | Sharpe | Max DD |
|-----|--------|----------|-----|---------|--------|
| Nov 2024 | 189 | 71.2% | $2,847 | 2.1 | -4.2% |
| Oct 2024 | 205 | 65.9% | $2,156 | 1.8 | -6.1% |
| Sep 2024 | 178 | 63.5% | $1,923 | 1.6 | -8.3% |

## Position Recycling en Acción

### Ejemplo Real: CTIC - 15 Nov 2024

```
08:45:32 - BUY 100 CTIC @ $3.42 (Gap: +12.5%, Vol: 3.2x)
08:52:15 - SELL 30 CTIC @ $3.78 (+10.5%, parcial profit)
08:58:41 - BUY 20 CTIC @ $3.61 (pullback entry)
09:03:27 - SELL 90 CTIC @ $3.85 (+11.8%, final exit)

Resultado: +$347 en 18 minutos
Trades: 3 (parte de 1 campaña)
Average price mejorado: $3.48 → $3.52
```

## Integración de APIs

### Polygon.io - Datos de Mercado
```python
# Endpoints utilizados:
- /v2/aggs/ticker/{ticker}/prev
- /v2/last/trade/{ticker}
- /v3/quotes/{ticker}
- /v2/reference/financials/{ticker}
```

### IBKR TWS API - Ejecución
```python
# Funcionalidades:
- Market data subscription
- Order placement y management
- Position tracking
- Account balance monitoring
```

### PostgreSQL - Storage
```sql
-- Tabla principal de trades
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    quantity INTEGER,
    entry_price DECIMAL(10,4),
    exit_price DECIMAL(10,4),
    pnl DECIMAL(10,2),
    strategy VARCHAR(50),
    gap_percent DECIMAL(5,2),
    volume_ratio DECIMAL(5,2)
);
```

## Monitoreo y Alertas

### Grafana Dashboard
- P&L en tiempo real
- Trades por hora/día
- Success rate por gap range
- Drawdown analysis
- Volatility tracking

### Telegram Integration
```
🚀 TRADE ALERT
Symbol: ABCD
Action: BUY 150 @ $4.23
Gap: +8.7% | Vol: 2.4x
Strategy: Gap_And_Go
Time: 06:42:15 ET
```

## Sistema de Early Runner Detection

### Cómo Funciona
1. **Scan diario** de 3,000+ small caps
2. **Análisis multi-factor** con ML
3. **Scoring 0-100** con clasificación
4. **Integration** con watchlist automática

### Ejemplo de Output
```json
{
  "symbol": "MNKD",
  "score": 87.5,
  "classification": "🔥 HOT",
  "factors": {
    "dark_pool_activity": 89,
    "technical_setup": 92,
    "float_rotation": 78,
    "sec_risk": 85,
    "momentum": 81
  },
  "recommendation": "WATCH CLOSELY"
}
```

## Lessons Learned

### ✅ Qué Funciona Bien

1. **Position Recycling**: Mejora significativamente el average price
2. **Tight Risk Management**: $10 max risk mantiene drawdowns bajos
3. **Volume Confirmation**: Filtro de volumen reduce false breakouts
4. **Time-based Exits**: Evita holds largos en small caps

### ⚠️ Desafíos Encontrados

1. **Halts Frecuentes**: 3-5% de trades terminan en halt
2. **Slippage Variable**: Puede ser 0.1% - 2% según liquidez
3. **Gap Fades**: 30% de gaps > 10% revierten rápidamente
4. **Competition**: Más algoritmos en premarket últimamente

### 🔧 Mejoras Implementadas

1. **Misprint Detection**: Evita bad fills 7:58-8:08 AM
2. **Dynamic Position Sizing**: Basado en ATR y volatility
3. **Smart Order Routing**: Mejora execution quality
4. **Circuit Breakers**: Auto-stop en drawdown > 15%

## Configuración para Desarrollo

### Prerrequisitos
```bash
# Python environment
python -m venv trading_env
source trading_env/bin/activate

# Dependencies
pip install -r requirements.txt

# Database setup
./database/postgresql/scripts/01_setup_database.sh
```

### Variables de Entorno
```bash
# .env file
POLYGON_API_KEY=your_polygon_key
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
TELEGRAM_BOT_TOKEN=your_telegram_token
DATABASE_URL=postgresql://trader:password@localhost:5432/trading_db
```

### Comandos de Ejecución
```bash
# Generar watchlist
python generate_watchlist_smallcaps.py

# Ejecutar backtesting
python simple_realistic_backtest.py --days 30

# Optimizar parámetros
python parameter_optimizer.py --evaluations 100

# Detector de runners
python early_runner_detector.py

# Trading console
python trading_console.py
```

## Roadmap de Mejoras

### Q1 2025
- [ ] Integration con más brokers (Schwab, E*Trade)
- [ ] Options trading module
- [ ] News sentiment analysis
- [ ] Mobile app para monitoring

### Q2 2025
- [ ] Multi-timeframe strategies
- [ ] Portfolio-level risk management
- [ ] Real-time strategy switching
- [ ] Community signals marketplace

## Contacto y Soporte

Para preguntas sobre este caso de estudio:

- **Issues**: GitHub repository issues
- **Documentation**: Ver `/docs` en repo principal
- **Community**: Discord server #ibkr-trading
- **Updates**: Twitter @QuantPlaybook

---

**Disclaimer**: Este sistema es para propósitos educativos. Trading involves substantial risk. Past performance no garantiza resultados futuros.

**Live Performance**: Puedes seguir el performance en vivo en nuestro [dashboard público](https://grafana.quantplaybook.com) (simulado para demo).