# Start Your Quant 🚀

Guía completa para iniciarte en el trading cuantitativo, con enfoque especial en small caps y estrategias algorítmicas. Documentación profesional con ejemplos prácticos usando Yahoo Finance, Polygon.io, IBKR TWS, DAS Trader, QuantConnect y Flash Research.

## 📚 Índice de Documentación

### 🚀 [0. Setup Inicial](docs/setup/)
- [Getting Started - Configuración inicial](docs/setup/getting_started.md)
- [Integración con Brokers](docs/setup/broker_integration.md)
- [DAS Trader Integration](docs/setup/das_trader_integration.md) 🔥 **Nuevo**
- [Configuración de Data Providers](docs/setup/data_providers.md)

### 🧠 [1. Fundamentos del Trading Cuantitativo](docs/fundamentals/)
- [¿Qué es el trading cuantitativo?](docs/fundamentals/what_is_quant.md)
- [Diferencias entre discretionary y quant](docs/fundamentals/discretionary_vs_quant.md)
- [Por qué usar código](docs/fundamentals/why_code.md)
- [Tipos de estrategias cuantitativas](docs/fundamentals/strategy_types.md)

### 💾 [2. Obtención y Manejo de Datos](docs/data/)
- [Fuentes de datos (Yahoo, Polygon, IBKR, etc.)](docs/data/data_sources.md)
- [Tipos de datos: EOD, intradía y tick](docs/data/data_types.md)
- [Limpieza y estructuración](docs/data/data_cleaning.md)
- [Creación de datasets para backtesting](docs/data/backtesting_datasets.md)

### 📊 [3. Indicadores Técnicos para Small Caps](docs/indicators/)
- [VWAP y VWAP Reclaim](docs/indicators/vwap.md)
- [Medias móviles (EMA/SMA)](docs/indicators/moving_averages.md)
- [Volumen y RVol](docs/indicators/volume_rvol.md)
- [Spike HOD/LOD](docs/indicators/spike_hod_lod.md)
- [Gap % y Float](docs/indicators/gap_float.md)
- [Bollinger Bands](docs/indicators/bollinger_bands.md) 🔥 **Nuevo**
- [Parabolic SAR](docs/indicators/parabolic_sar.md) 🔥 **Nuevo**
- [SuperTrend](docs/indicators/super_tendencia.md) 🔥 **Nuevo**

### 🧪 [4. Backtesting](docs/backtesting/)
- [¿Qué es un backtest?](docs/backtesting/what_is_backtest.md)
- [Motor de backtest simple](docs/backtesting/simple_engine.md)
- [Métricas clave](docs/backtesting/metrics.md)
- [Cómo evitar overfitting](docs/backtesting/overfitting.md)
- [Análisis avanzado con ML](docs/backtesting/advanced_analysis.md)

### 🛡️ [5. Gestión de Riesgo Cuantitativa](docs/risk/)
- [Tamaño de posición con riesgo fijo](docs/risk/position_sizing.md)
- [Límites de riesgo diario](docs/risk/risk_limits.md)
- [Stop loss y trailing stops](docs/risk/stops.md)
- [Riesgo asimétrico](docs/risk/asymmetric_risk.md)

### 🎯 [6. Estrategias Aplicadas a Small Caps](docs/strategies/)
- [Gap & Go](docs/strategies/gap_and_go.md)
- [VWAP Reclaim](docs/strategies/vwap_reclaim.md)
- [First Green/Red Day](docs/strategies/first_green_red_day.md)
- [Low Float Runners](docs/strategies/low_float_runners.md)
- [Parabolic Reversal](docs/strategies/parabolic_reversal.md)
- [Short Selling Avanzado](docs/strategies/short_selling_advanced.md)

### 🛠️ [7. Herramientas y Librerías](docs/tools/)
- [Infraestructura Trading Avanzada](docs/tools/advanced_trading_infrastructure.md)
- [Configuración Flash Research](docs/tools/flash_research_config.md)
- [Librerías Esenciales Python](docs/tools/essential_libraries.md)
- [Workflow de Desarrollo](docs/tools/development_workflow.md)

### 📈 [8. Análisis y Validación](docs/analysis/)
- [Validación de Trades](docs/analysis/trade_validation.md)
- [Tracking de Performance](docs/analysis/performance_tracking.md)
- [Market Microstructure y Tape Reading](docs/analysis/market_microstructure.md)
- [Machine Learning para Trading](docs/analysis/machine_learning.md) 🔥 **Nuevo**
- [Análisis de Sentimiento](docs/analysis/sentiment_analysis.md) 🔥 **Nuevo**
- [Análisis Fundamental](docs/analysis/fundamental_analysis.md) 🔥 **Nuevo**

### 🤖 [9. Automatización](docs/automation/)
- [Arquitectura del Sistema](docs/automation/system_architecture.md)
- [Estrategias Automatizadas](docs/automation/automation_strategies.md)

### 💡 [10. Ejemplos Prácticos](docs/examples/)
- [Integración Multi-Plataforma](docs/examples/platform_integration.md)

## 🚀 Quick Start

1. **Revisa los requisitos iniciales**
   - Consulta [Getting Started](docs/setup/getting_started.md) para configuración completa
   - Instala Python 3.8+ y las librerías requeridas

2. **Configura tus fuentes de datos**
   - [Yahoo Finance](docs/setup/data_providers.md) (gratuito para empezar)
   - [Polygon.io](docs/setup/data_providers.md) (API key requerida)
   - [IBKR TWS](docs/setup/broker_integration.md) (para trading institucional)
   - [DAS Trader](docs/setup/das_trader_integration.md) (para small caps/day trading)

3. **Explora las estrategias**
   - Empieza con [Gap & Go](docs/strategies/gap_and_go.md)
   - Implementa [VWAP Reclaim](docs/strategies/vwap_reclaim.md)

4. **Ejecuta tu primer backtest**
   - Usa el [Motor Simple](docs/backtesting/simple_engine.md)
   - Analiza las [Métricas](docs/backtesting/metrics.md)

## 📖 Cómo usar esta guía

- Si eres **principiante**: Empieza por [Setup Inicial](docs/setup/) y [Fundamentos](docs/fundamentals/)
- Si ya tienes **experiencia trading**: Ve directo a [Estrategias](docs/strategies/) y [Indicadores](docs/indicators/)
- Si quieres **automatizar**: Revisa [Herramientas](docs/tools/) y [Automatización](docs/automation/)
- Si buscas **integración multi-plataforma**: Consulta [Ejemplos Prácticos](docs/examples/)

## 🎯 Características Principales

- ✅ **Estrategias probadas** específicas para small caps
- ✅ **Código funcional** con ejemplos reales
- ✅ **Integración completa** con brokers y data providers populares
- ✅ **Sistema de backtesting** robusto con métricas avanzadas
- ✅ **Gestión de riesgo** cuantitativa integrada
- ✅ **Arquitectura escalable** para automatización
- ✅ **Documentación en español** con tono personal

## 🛠️ Stack Tecnológico

- **Python** como lenguaje principal
- **Pandas/NumPy** para manipulación de datos
- **Plotly/Matplotlib** para visualización
- **yfinance/Polygon** para datos de mercado
- **IBKR TWS/DAS Trader/Alpaca** para ejecución
- **QuantConnect** para backtesting en la nube
- **Streamlit** para dashboards
- **Docker/Kubernetes** para deployment

## 📝 Notas Importantes

- Esta documentación está enfocada en **small caps** (market cap < $2B) y **day trading**
- El contenido asume conocimiento básico de **trading** y **Python**
- Todas las estrategias incluyen **advertencias de riesgo** apropiadas
- Los ejemplos usan **datos reales** de las plataformas mencionadas
- **⚠️ Small caps son extremadamente volátiles** - requieren experiencia y gestión de riesgo estricta

## 🤝 Contribución y Desarrollo

Esta documentación es un proyecto en constante evolución, desarrollado con un enfoque riguroso en la validación técnica y optimización continua. El objetivo es mantener:

- **Precisión técnica** en todas las fórmulas e implementaciones
- **Código optimizado** y siguiendo mejores prácticas
- **Metodologías actualizadas** con los últimos desarrollos del sector
- **Contenido verificado** a través de backtesting y análisis cuantitativo

### Contribuciones Bienvenidas

Cualquier **mejora, corrección u observación es enormemente agradecida**:

- 🐛 **Reporta errores** en implementaciones o documentación
- 💡 **Sugiere mejoras** en estrategias o metodologías
- 📚 **Propón nuevo contenido** basado en tu experiencia
- 🔧 **Optimiza código** existente o agrega nuevas funcionalidades

**Formas de contribuir:**
- Abre un issue en GitHub con tu sugerencia
- Propón cambios vía pull request
- Comparte tu experiencia implementando las estrategias
- Reporta resultados de backtesting en diferentes condiciones de mercado

Tu experiencia práctica y feedback son fundamentales para mantener esta documentación actualizada y útil para la comunidad de traders cuantitativos.

## 📚 Recursos Adicionales Recomendados

### 🎓 Cursos y Material Educativo

**🔥 Altamente Recomendado:**
- **[Trading Cuantitativo en Python](https://github.com/AxelMunguiaQuintero/Trading-Cuantitativo-en-Python)** - Curso completo con 15+ módulos prácticos que incluye:
  - Integración con brokers (OANDA, FXCM, Interactive Brokers)
  - Machine Learning aplicado a trading (HMM, XGBoost, Redes Neuronales)
  - Análisis de sentimiento con VADER y web scraping
  - Sistema de trading completo y unificado
  - Optimización de portafolios con métodos cuantitativos
  - **Material base que inspiró muchas implementaciones de este repositorio**

### 🛠️ Herramientas y Plataformas
- **[QuantConnect](https://www.quantconnect.com/)** - Backtesting en la nube con datos institucionales
- **[Zipline](https://github.com/quantopian/zipline)** - Motor de backtesting open source
- **[Backtrader](https://github.com/mementum/backtrader)** - Framework de backtesting en Python
- **[TradingView](https://www.tradingview.com/)** - Análisis técnico y alertas

### 📖 Libros Fundamentales
- **"Quantitative Trading" - Ernest Chan** - Fundamentos del trading algorítmico
- **"Algorithmic Trading" - Ernie Chan** - Estrategias prácticas
- **"The Little Book of Common Sense Investing" - John Bogle** - Fundamentos de inversión
- **"A Man for All Markets" - Edward Thorp** - Historia del trading cuantitativo

### 🌐 Comunidades y Blogs
- **[QuantStart](https://www.quantstart.com/)** - Artículos técnicos de trading cuantitativo
- **[r/algotrading](https://reddit.com/r/algotrading)** - Comunidad activa de traders algorítmicos
- **[Quantitative Finance Stack Exchange](https://quant.stackexchange.com/)** - Preguntas técnicas especializadas

### 💡 APIs y Proveedores de Datos
- **[Alpha Vantage](https://www.alphavantage.co/)** - API gratuita con límites
- **[IEX Cloud](https://iexcloud.io/)** - Datos de mercado económicos
- **[Quandl](https://www.quandl.com/)** - Datos económicos y alternativos
- **[Polygon.io](https://polygon.io/)** - Datos en tiempo real y históricos

**Nota**: El repositorio de **Trading Cuantitativo en Python** es especialmente valioso ya que proporciona implementaciones detalladas de muchos conceptos avanzados y sirvió como referencia técnica para el desarrollo de varias secciones de esta documentación.
