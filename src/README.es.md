> 🇺🇸 [Read in English](README.md) | 🇪🇸 **Español**

# Start Your Quant - Source Code

Esta carpeta contiene implementaciones de referencia en Python que complementan la documentación del proyecto Start Your Quant.

## Estructura

```
src/
├── README.md                          # Este archivo
├── indicators/                        # Indicadores técnicos
│   ├── moving_averages.py            # Medias móviles (SMA, EMA, WMA)
│   └── vwap.py                       # VWAP y bandas
├── strategies/                        # Estrategias de trading
│   └── gap_and_go.py                 # Estrategia Gap and Go
├── backtesting/                       # Motor de backtesting
│   ├── simple_engine.py              # Motor simple de backtesting
│   └── trade_reporting.py            # Exportación CSV para TraderVue
├── risk/                             # Gestión de riesgo
│   └── position_sizing.py            # Modelos de dimensionamiento
├── data/                             # Fuentes de datos
│   └── data_sources.py               # APIs y gestión de datos
└── examples/                         # Ejemplos completos
    └── complete_strategy_example.py  # Ejemplo integral
```

## Componentes Principales

### 📊 Indicadores Técnicos (`indicators/`)

- **Moving Averages**: SMA, EMA, WMA y señales de cruce
- **VWAP**: Volume Weighted Average Price con bandas

### 🎯 Estrategias (`strategies/`)

- **Gap and Go**: Estrategia para gaps de apertura con confirmación de volumen

### 🔄 Backtesting (`backtesting/`)

- **Simple Engine**: Motor básico de backtesting con gestión de portafolio
- **Trade Reporting**: Exportación de trades a CSV para TraderVue, TradesViz y análisis personal

### ⚖️ Gestión de Riesgo (`risk/`)

- **Position Sizing**: Múltiples modelos (fijo, Kelly, ATR, paridad de riesgo)

### 📈 Datos (`data/`)

- **Data Sources**: Interfaces para Yahoo Finance, Alpha Vantage y otros

### 🚀 Ejemplos (`examples/`)

- **Complete Strategy**: Ejemplo que integra todos los componentes

## Instalación y Uso

### Requisitos

```bash
pip install pandas numpy matplotlib requests
```

### Uso Básico

```python
# Ejemplo de uso de indicadores
from indicators.moving_averages import MovingAverages
import pandas as pd

# Crear datos de ejemplo
prices = pd.Series([100, 101, 99, 102, 104, 103, 105])

# Calcular medias móviles
ma = MovingAverages()
sma_5 = ma.sma(prices, 5)
ema_5 = ma.ema(prices, 5)
```

### Ejecutar Ejemplo Completo

```bash
cd src/examples
python complete_strategy_example.py
```

## Relación con Documentación

Cada módulo de código está diseñado para complementar específicamente la documentación en `docs/`:

| Código | Documentación Relacionada |
|--------|---------------------------|
| `indicators/moving_averages.py` | `docs/indicators/moving_averages.md` |
| `indicators/vwap.py` | `docs/indicators/vwap.md` |
| `strategies/gap_and_go.py` | `docs/strategies/gap_and_go.md` |
| `backtesting/simple_engine.py` | `docs/backtesting/simple_engine.md` |
| `backtesting/trade_reporting.py` | `docs/backtesting/simple_engine.md` |
| `risk/position_sizing.py` | `docs/risk/position_sizing.md` |
| `data/data_sources.py` | `docs/data/data_sources.md` |

## Características del Código

### ✅ Diseño Modular
- Cada componente es independiente y reutilizable
- Interfaces claras entre módulos
- Fácil extensión y personalización

### ✅ Código Educativo
- Comentarios detallados en español e inglés
- Ejemplos de uso en cada módulo
- Implementaciones explicativas paso a paso

### ✅ Estándares de Calidad
- Type hints para mejor documentación
- Manejo de errores apropiado
- Validación de datos de entrada

### ✅ Flexibilidad
- Configuración personalizable
- Múltiples algoritmos por categoría
- Parámetros ajustables

## Desarrollo del Código

Este código ha sido desarrollado siguiendo las mejores prácticas de trading cuantitativo e implementa conceptos avanzados:

- **Evaluación de Modelos**: Métricas robustas de backtesting y validación
- **IA en Trading**: Conceptos aplicados en análisis de patrones y señales
- **Gestión de Riesgo**: Modelos cuantitativos para dimensionamiento y control

## Próximos Pasos

1. **Expandir Indicadores**: Bollinger Bands, RSI, MACD
2. **Más Estrategias**: Mean reversion, momentum, arbitraje
3. **ML Integration**: Implementar transformers para finanzas
4. **Live Trading**: Conexión con brokers reales
5. **Advanced Risk**: VaR, stress testing, correlation analysis

## Contribuir

Para contribuir al código:

1. Mantén el estilo educativo y comentarios claros
2. Incluye ejemplos de uso en cada módulo
3. Asegúrate de que complementa la documentación existente
4. Añade tests unitarios cuando sea apropiado

## Notas Importantes

⚠️ **Disclaimer**: Este código es para fines educativos. No constituye asesoramiento financiero. Siempre realiza tu propia investigación antes de invertir dinero real.

📚 **Aprendizaje**: Use este código junto con la documentación en `docs/` para un aprendizaje completo de trading cuantitativo.

🔧 **Personalización**: Modifica los parámetros y algoritmos según tus necesidades específicas de trading.