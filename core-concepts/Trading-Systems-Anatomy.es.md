> 🇺🇸 [Read in English](Trading-Systems-Anatomy.md) | 🇪🇸 **Español**

# Anatomía de un Sistema de Trading

Un sistema de trading es un conjunto de reglas objetivas, ordenadas y programables que generan señales de entrada y salida. Si no podés escribirlo en código, no es un sistema — es una opinión.

## Las 4 Etapas del Trader

Casi todo trader pasa por estas fases evolutivas — un concepto ampliamente discutido en la literatura de trading sistemático. Saber en cuál estás te ahorra meses de frustración.

### Etapa 1: Trader Discrecional

Opera por intuición, tips de Twitter, y "lo que siente". Depende de opiniones externas. Busca la señal mágica que siempre funcione.

**Síntoma principal**: no puede explicar su estrategia en 3 reglas claras.

La mayoría de traders pierde dinero en esta etapa y nunca sale de ella.

### Etapa 2: Trader Técnico

Empieza a usar indicadores (RSI, MACD, medias móviles). Busca la combinación perfecta de indicadores — el "santo grial".

**Síntoma principal**: cambia de indicadores cada semana buscando el que "funcione siempre".

El avance real viene cuando acepta que ningún indicador es mágico y empieza a pensar en **reglas fijas**.

### Etapa 3: Trader de Sistemas

Opera con reglas objetivas. Backtestea. Mide. Tiene expectativas numéricas de su sistema (win rate, profit factor, drawdown esperado).

**Síntoma principal**: puede ver 10 trades perdedores seguidos y no cambiar el sistema, porque sabe que es estadísticamente esperado.

```python
# Este es el mindset de la etapa 3:
# "Mi sistema tiene 40% de acierto y ratio 2.5:1.
#  Una racha de 10 pérdidas tiene probabilidad del 0.6%.
#  Es raro pero posible. No cambio nada."

def expected_value(win_rate, avg_win, avg_loss):
    """Expectancy por trade — si es positivo, el sistema es viable."""
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

# 40% win rate, gana $250 promedio, pierde $100 promedio
ev = expected_value(0.40, 250, 100)
print(f"Expectancy por trade: ${ev:.2f}")  # $40.00
```

### Etapa 4: Gestor de Portfolio

Deja de pensar en "el sistema" y empieza a pensar en **el portfolio de sistemas**. Múltiples estrategias, múltiples mercados, múltiples timeframes. Gestión de liquidez y correlación entre sistemas.

**Síntoma principal**: le importa más la correlación entre sus sistemas que el Sharpe de cualquiera individual.

**La transición clave**: pasar de optimizar UN sistema a optimizar la COMBINACIÓN de sistemas. Un portfolio de 5 sistemas mediocres pero descorrelacionados supera a un solo sistema "perfecto".

## Clasificación de Sistemas por Estrategia

### Tendenciales (Momentum/Trend Following)

Compran caro para vender más caro. Siguen la tendencia hasta que se agota.

| Característica | Valor típico |
|---|---|
| Win rate | 30-45% |
| Ratio ganancia/pérdida | 2:1 a 5:1 |
| Mejor mercado | Trending fuerte |
| Peor mercado | Lateral/choppy |
| Psicología | Difícil — muchas pérdidas pequeñas |

```python
def trend_following_signal(prices, fast=20, slow=50):
    """
    Señal tendencial clásica: cruce de medias.
    Pocas señales, muchas falsas en mercados laterales,
    pero captura los movimientos grandes.
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()

    signal = 0
    if fast_ma.iloc[-1] > slow_ma.iloc[-1] and fast_ma.iloc[-2] <= slow_ma.iloc[-2]:
        signal = 1   # compra
    elif fast_ma.iloc[-1] < slow_ma.iloc[-1] and fast_ma.iloc[-2] >= slow_ma.iloc[-2]:
        signal = -1  # venta
    return signal
```

### Anti-Tendenciales (Mean Reversion)

Compran barato cerca de soportes, venden caro cerca de resistencias. Asumen que el precio vuelve a su media.

| Característica | Valor típico |
|---|---|
| Win rate | 55-70% |
| Ratio ganancia/pérdida | 0.5:1 a 1.5:1 |
| Mejor mercado | Lateral/rango |
| Peor mercado | Trending fuerte |
| Psicología | Más llevadero — muchos aciertos |

```python
def mean_reversion_signal(prices, lookback=20, z_threshold=2.0):
    """
    Señal de reversión a la media usando z-score.
    Compra cuando el precio está 2 desviaciones por debajo de la media.
    """
    mean = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    z_score = (prices.iloc[-1] - mean.iloc[-1]) / std.iloc[-1]

    if z_score < -z_threshold:
        return 1   # sobreventa → compra
    elif z_score > z_threshold:
        return -1  # sobrecompra → venta
    return 0
```

### Volatility Breakout / ORB

Entran en la ruptura de un rango (apertura, rango previo, etc.) y cierran rápido. Híbridos entre tendenciales y anti-tendenciales.

```python
def orb_signal(open_price, high_first_15min, low_first_15min, current_price):
    """
    Opening Range Breakout: entra si el precio rompe
    el rango de los primeros 15 minutos.
    """
    range_size = high_first_15min - low_first_15min

    if current_price > high_first_15min:
        return 1, high_first_15min - range_size  # largo, stop debajo del rango
    elif current_price < low_first_15min:
        return -1, low_first_15min + range_size   # corto, stop arriba del rango
    return 0, None
```

### Otros tipos que vale la pena conocer

| Tipo | Idea central | Complejidad |
|---|---|---|
| **Rotacional** | Rota capital entre activos según fuerza relativa | Media |
| **Market Making** | Provee liquidez comprando bid / vendiendo ask | Alta |
| **Pairs Trading** | Largo un activo + corto otro correlacionado | Media-Alta |
| **Arbitraje estadístico** | Explota discrepancias temporales de precio | Alta |
| **Estacional** | Patrones que se repiten en fechas específicas | Baja |

## No Existe el Santo Grial (Individual)

Un sistema con 40% de aciertos puede ser más rentable que uno con 70%. Lo que importa es la **expectancy** (valor esperado por trade) y cómo se comporta en combinación con otros sistemas.

### El verdadero santo grial: diversificación de sistemas

```python
import numpy as np

def portfolio_sharpe(returns_matrix, weights):
    """
    El Sharpe de un portfolio de sistemas descorrelacionados
    es mayor que el de cualquier sistema individual.
    """
    portfolio_return = np.dot(weights, returns_matrix.mean(axis=0)) * 252
    portfolio_vol = np.sqrt(
        np.dot(weights, np.dot(returns_matrix.cov() * 252, weights))
    )
    return portfolio_return / portfolio_vol

# Ejemplo: 3 sistemas con Sharpe individual de ~1.0
# pero baja correlación entre sí → Sharpe del portfolio > 1.5
```

La clave no es encontrar el sistema perfecto. Es construir un portfolio donde:
- Los sistemas sean **rentables individualmente** (expectancy positiva)
- Tengan **baja correlación entre sí** (no todos pierden al mismo tiempo)
- Operen en **mercados o timeframes distintos** (acciones + futuros, intradiario + swing)

## Algo Trading: Mitos vs Realidad

### Mito: "Algo trading = alta frecuencia"

**Realidad**: un algoritmo puede operar en gráficos mensuales. Lo que lo hace algorítmico es que las reglas están codificadas, no la velocidad de ejecución.

### Mito: "Los mercados son aleatorios, no se puede ganar sistemáticamente"

La hipótesis del mercado eficiente (popularizada por Malkiel en *A Random Walk Down Wall Street*, 1973) sostiene que los precios reflejan toda la información disponible y que no se puede superar al mercado de forma sistemática y consistente después de costos. Pero:

- El mercado no tiene distribución perfectamente normal — hay fat tails, clusters de volatilidad, y asimetrías (las caídas son rápidas y con volatilidad alta, las subidas son graduales)
- La información no es perfecta ni instantánea para todos los participantes
- Existen ineficiencias explotables, especialmente en instrumentos de menor capitalización

No necesitás predecir el futuro. Necesitás encontrar patrones con **edge estadístico** y gestionar el riesgo para que ese edge se materialice en miles de trades.

### Mito: "Necesitás infraestructura de Wall Street"

**Realidad**: el gap tecnológico entre firmas institucionales y traders individuales nunca fue tan chico. Con Python, un broker con API, y datos de mercado asequibles, podés construir y operar sistemas rentables. Se estima que en US más del 60% del volumen es algorítmico (incluyendo market makers y HFT), pero una gran porción de ese volumen proviene de sistemas relativamente simples bien ejecutados, no de infraestructura de nanosegundos.

## Trading Algorítmico vs Discrecional

| Dimensión | Algorítmico | Discrecional |
|---|---|---|
| **Emociones** | Minimizadas — el código ejecuta | Presentes en cada trade |
| **Expectativas** | Estimables vía backtest (con limitaciones) | Estimativas, subjetivas |
| **Disciplina** | Inherente al código | Requiere fuerza de voluntad |
| **Adaptación a cambios** | Requiere re-desarrollo | Inmediata (en traders expertos) |
| **Diversificación** | Fácil — correr N sistemas en paralelo | Difícil — un cerebro, un mercado |
| **Drawdown** | Esperado y cuantificado | Sorpresivo y emocionalmente duro |
| **Escalabilidad** | Alta | Limitada por el trader |

El trading algorítmico no es "mejor" — es **diferente**. Un trader discrecional experto puede superar a muchos algoritmos. Pero la curva de aprendizaje del discrecional es más larga, el desgaste emocional es mayor, y la consistencia es más difícil de mantener.

La ventaja real del algorítmico: podés saber de antemano, con datos históricos, qué tan mal puede ir. Saber que tu sistema puede tener 12 pérdidas consecutivas y que eso es normal es infinitamente más manejable que descubrirlo en vivo sin contexto.
