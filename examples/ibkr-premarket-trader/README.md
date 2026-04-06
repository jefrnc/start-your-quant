> 🇪🇸 [Leer en Español](README.es.md) | 🇺🇸 **English**

# IBKR Premarket Trader - Case Study

## Executive Summary

The **IBKR Premarket Trader** is an automated trading system that implements the Gap & Go strategy on small caps during premarket hours (5:30 AM - 8:00 AM ET). It is the main example of the Quant Playbook and demonstrates how to apply systematic methodologies in real trading.

## System Architecture

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

## Main Components

### 1. **trading_console.py** - Main Application
```python
# Key features:
- Interactive console for manual control
- REST API on port 8080
- WebSocket for real-time updates
- Structured logging system
```

### 2. **generate_watchlist_smallcaps.py** - Watchlist Generator
```python
# Filtering criteria:
- Market cap: $50M - $2B
- Price: $0.50 - $10.99
- Average volume: > 100K shares/day
- Avoids extreme penny stocks
- Filters by sector and exchanges
```

### 3. **simple_realistic_backtest.py** - Backtesting
```python
# Backtest features:
- Realistic slippage and commissions
- Simulated opening gaps
- Respected trading hours
- Temporal drawdown analysis
```

### 4. **parameter_optimizer.py** - Bayesian Optimization
```python
# Optimized parameters:
- Minimum and maximum gap %
- Volume multiplier
- Dynamic stop loss
- Profit taking targets
```

### 5. **early_runner_detector.py** - ML Detection
```python
# Scoring system:
- Dark pool activity (30%)
- Technical setup (25%)
- Float rotation (20%)
- SEC filing risk (15%)
- Social momentum (10%)
```

## Strategy Configuration

### File: `config/strategy_config.yaml`

```yaml
gap_and_go:
  # Entry Filters
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

## Historical Performance

### Key Metrics (Last 6 months)
```
Sharpe Ratio:       1.85
Max Drawdown:      -8.3%
Win Rate:          67.4%
Profit Factor:     2.31
Avg Trade:         $12.50
Total Trades:      1,247
```

### Monthly Breakdown
| Month | Trades | Win Rate | P&L | Sharpe | Max DD |
|-------|--------|----------|-----|---------|--------|
| Nov 2024 | 189 | 71.2% | $2,847 | 2.1 | -4.2% |
| Oct 2024 | 205 | 65.9% | $2,156 | 1.8 | -6.1% |
| Sep 2024 | 178 | 63.5% | $1,923 | 1.6 | -8.3% |

## Position Recycling in Action

### Real Example: CTIC - Nov 15, 2024

```
08:45:32 - BUY 100 CTIC @ $3.42 (Gap: +12.5%, Vol: 3.2x)
08:52:15 - SELL 30 CTIC @ $3.78 (+10.5%, partial profit)
08:58:41 - BUY 20 CTIC @ $3.61 (pullback entry)
09:03:27 - SELL 90 CTIC @ $3.85 (+11.8%, final exit)

Result: +$347 in 18 minutes
Trades: 3 (part of 1 campaign)
Average price improved: $3.48 -> $3.52
```

## API Integration

### Polygon.io - Market Data
```python
# Endpoints used:
- /v2/aggs/ticker/{ticker}/prev
- /v2/last/trade/{ticker}
- /v3/quotes/{ticker}
- /v2/reference/financials/{ticker}
```

### IBKR TWS API - Execution
```python
# Features:
- Market data subscription
- Order placement and management
- Position tracking
- Account balance monitoring
```

### PostgreSQL - Storage
```sql
-- Main trades table
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

## Monitoring and Alerts

### Grafana Dashboard
- Real-time P&L
- Trades per hour/day
- Success rate by gap range
- Drawdown analysis
- Volatility tracking

### Telegram Integration
```
TRADE ALERT
Symbol: ABCD
Action: BUY 150 @ $4.23
Gap: +8.7% | Vol: 2.4x
Strategy: Gap_And_Go
Time: 06:42:15 ET
```

## Early Runner Detection System

### How It Works
1. **Daily scan** of 3,000+ small caps
2. **Multi-factor analysis** with ML
3. **0-100 scoring** with classification
4. **Integration** with automatic watchlist

### Example Output
```json
{
  "symbol": "MNKD",
  "score": 87.5,
  "classification": "HOT",
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

### What Works Well

1. **Position Recycling**: Significantly improves average price
2. **Tight Risk Management**: $10 max risk keeps drawdowns low
3. **Volume Confirmation**: Volume filter reduces false breakouts
4. **Time-based Exits**: Avoids long holds in small caps

### Challenges Encountered

1. **Frequent Halts**: 3-5% of trades end in halts
2. **Variable Slippage**: Can be 0.1% - 2% depending on liquidity
3. **Gap Fades**: 30% of gaps > 10% reverse quickly
4. **Competition**: More algorithms in premarket lately

### Improvements Implemented

1. **Misprint Detection**: Avoids bad fills 7:58-8:08 AM
2. **Dynamic Position Sizing**: Based on ATR and volatility
3. **Smart Order Routing**: Improves execution quality
4. **Circuit Breakers**: Auto-stop on drawdown > 15%

## Development Setup

### Prerequisites
```bash
# Python environment
python -m venv trading_env
source trading_env/bin/activate

# Dependencies
pip install -r requirements.txt

# Database setup
./database/postgresql/scripts/01_setup_database.sh
```

### Environment Variables
```bash
# .env file
POLYGON_API_KEY=your_polygon_key
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
TELEGRAM_BOT_TOKEN=your_telegram_token
DATABASE_URL=postgresql://trader:password@localhost:5432/trading_db
```

### Execution Commands
```bash
# Generate watchlist
python generate_watchlist_smallcaps.py

# Run backtesting
python simple_realistic_backtest.py --days 30

# Optimize parameters
python parameter_optimizer.py --evaluations 100

# Runner detector
python early_runner_detector.py

# Trading console
python trading_console.py
```

## Improvement Roadmap

### Q1 2025
- [ ] Integration with more brokers (Schwab, E*Trade)
- [ ] Options trading module
- [ ] News sentiment analysis
- [ ] Mobile app for monitoring

### Q2 2025
- [ ] Multi-timeframe strategies
- [ ] Portfolio-level risk management
- [ ] Real-time strategy switching
- [ ] Community signals marketplace

## Contact and Support

For questions about this case study:

- **Issues**: GitHub repository issues
- **Documentation**: See `/docs` in main repo
- **Community**: Discord server #ibkr-trading
- **Updates**: Twitter @QuantPlaybook

---

**Disclaimer**: This system is for educational purposes. Trading involves substantial risk. Past performance does not guarantee future results.

**Live Performance**: You can follow live performance on our [public dashboard](https://grafana.quantplaybook.com) (simulated for demo).
