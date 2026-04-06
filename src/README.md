> 🇪🇸 [Leer en Español](README.es.md) | 🇺🇸 **English**

# Start Your Quant - Source Code

This folder contains reference Python implementations that complement the Start Your Quant project documentation.

## Structure

```
src/
├── README.md                          # This file
├── indicators/                        # Technical indicators
│   ├── moving_averages.py            # Moving averages (SMA, EMA, WMA)
│   └── vwap.py                       # VWAP and bands
├── strategies/                        # Trading strategies
│   └── gap_and_go.py                 # Gap and Go strategy
├── backtesting/                       # Backtesting engine
│   ├── simple_engine.py              # Simple backtesting engine
│   └── trade_reporting.py            # CSV export for TraderVue
├── risk/                             # Risk management
│   └── position_sizing.py            # Sizing models
├── data/                             # Data sources
│   └── data_sources.py               # APIs and data management
└── examples/                         # Complete examples
    └── complete_strategy_example.py  # Integrated example
```

## Main Components

### Technical Indicators (`indicators/`)

- **Moving Averages**: SMA, EMA, WMA and crossover signals
- **VWAP**: Volume Weighted Average Price with bands

### Strategies (`strategies/`)

- **Gap and Go**: Opening gap strategy with volume confirmation

### Backtesting (`backtesting/`)

- **Simple Engine**: Basic backtesting engine with portfolio management
- **Trade Reporting**: Trade export to CSV for TraderVue, TradesViz, and personal analysis

### Risk Management (`risk/`)

- **Position Sizing**: Multiple models (fixed, Kelly, ATR, risk parity)

### Data (`data/`)

- **Data Sources**: Interfaces for Yahoo Finance, Alpha Vantage, and others

### Examples (`examples/`)

- **Complete Strategy**: Example integrating all components

## Installation and Usage

### Requirements

```bash
pip install pandas numpy matplotlib requests
```

### Basic Usage

```python
# Example using indicators
from indicators.moving_averages import MovingAverages
import pandas as pd

# Create sample data
prices = pd.Series([100, 101, 99, 102, 104, 103, 105])

# Calculate moving averages
ma = MovingAverages()
sma_5 = ma.sma(prices, 5)
ema_5 = ma.ema(prices, 5)
```

### Run Complete Example

```bash
cd src/examples
python complete_strategy_example.py
```

## Relationship with Documentation

Each code module is designed to specifically complement the documentation in `docs/`:

| Code | Related Documentation |
|------|----------------------|
| `indicators/moving_averages.py` | `docs/indicators/moving_averages.md` |
| `indicators/vwap.py` | `docs/indicators/vwap.md` |
| `strategies/gap_and_go.py` | `docs/strategies/gap_and_go.md` |
| `backtesting/simple_engine.py` | `docs/backtesting/simple_engine.md` |
| `backtesting/trade_reporting.py` | `docs/backtesting/simple_engine.md` |
| `risk/position_sizing.py` | `docs/risk/position_sizing.md` |
| `data/data_sources.py` | `docs/data/data_sources.md` |

## Code Features

### Modular Design
- Each component is independent and reusable
- Clear interfaces between modules
- Easy extension and customization

### Educational Code
- Detailed comments in Spanish and English
- Usage examples in each module
- Step-by-step explanatory implementations

### Quality Standards
- Type hints for better documentation
- Appropriate error handling
- Input data validation

### Flexibility
- Customizable configuration
- Multiple algorithms per category
- Adjustable parameters

## Code Development

This code has been developed following quantitative trading best practices and implements advanced concepts:

- **Model Evaluation**: Robust backtesting and validation metrics
- **AI in Trading**: Concepts applied in pattern analysis and signals
- **Risk Management**: Quantitative models for sizing and control

## Next Steps

1. **Expand Indicators**: Bollinger Bands, RSI, MACD
2. **More Strategies**: Mean reversion, momentum, arbitrage
3. **ML Integration**: Implement transformers for finance
4. **Live Trading**: Connect with real brokers
5. **Advanced Risk**: VaR, stress testing, correlation analysis

## Contributing

To contribute to the code:

1. Maintain the educational style and clear comments
2. Include usage examples in each module
3. Ensure it complements existing documentation
4. Add unit tests when appropriate

## Important Notes

**Disclaimer**: This code is for educational purposes. It does not constitute financial advice. Always do your own research before investing real money.

**Learning**: Use this code alongside the documentation in `docs/` for a complete quantitative trading learning experience.

**Customization**: Modify the parameters and algorithms according to your specific trading needs.
