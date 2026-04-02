# Estructura de un Sistema: Entradas, Salidas y Stops

Un sistema de trading se descompone en tres partes: **cómo entrás**, **cómo salís con pérdida**, y **cómo salís con ganancia**. Separarlos no es solo una formalidad — es lo que te permite evaluarlos con el [método científico](./Scientific-Method-System-Development.md), aislar qué funciona y qué no, y optimizar sin contaminar los resultados.

## Principio Fundamental: Simplicidad

Si no podés explicar en dos minutos a cualquier persona por qué tu sistema compra y por qué vende, probablemente sea demasiado complejo. No necesitás machine learning ni miles de líneas de código. Las cosas simples funcionan en el mercado algorítmico — y son más robustas.

Una entrada, un filtro como máximo. En salidas sí podés (y conviene) usar múltiples métodos combinados.

## Tipos de Órdenes para Entradas

### Stop (para Entrar en Tendencia)

Se activa cuando el precio supera un nivel por arriba (compra) o cae por debajo (venta). Es la entrada natural de sistemas de ruptura y seguimiento de tendencia.

```python
def entry_stop_order(current_price, stop_level, side='long'):
    """
    Entrada por stop: compra si supera un nivel, vende si lo pierde.
    Siempre stop market — queremos garantizar la ejecución.
    """
    if side == 'long' and current_price >= stop_level:
        return 'BUY_MARKET'  # ejecuta al mejor precio disponible
    elif side == 'short' and current_price <= stop_level:
        return 'SELL_MARKET'
    return None

# Ventaja: garantiza la entrada, el filtro va implícito
# Desventaja: desliza (slippage) porque entrás a favor del movimiento
# y puede haber muchas órdenes saltando en el mismo nivel
```

**Siempre stop market, nunca stop limit para entradas.** Si usás stop es porque querés garantizar la ejecución. Un stop limit puede no ejecutar si el precio salta rápido.

### Limit (para Entrar Contra Tendencia)

Comprás cuando el precio cae hacia tu precio. Vendés cuando sube hacia tu precio. La entrada natural de sistemas de soporte/resistencia y reversión a la media.

```python
def entry_limit_order(current_price, limit_level, side='long'):
    """
    Entrada por limit: compra si el precio cae a mi nivel o mejor.
    No desliza negativamente, pero puede no ejecutar.
    """
    if side == 'long' and current_price <= limit_level:
        return 'BUY_LIMIT'
    elif side == 'short' and current_price >= limit_level:
        return 'SELL_LIMIT'
    return None

# Ventaja: sin slippage negativo — comprás al precio que querés o mejor
# Desventaja: puede no ejecutar si no llega al precio exacto
```

**Problema para backtesting**: que el precio haya tocado tu nivel en el histórico no garantiza que hubieras ejecutado. Puede haber habido miles de órdenes delante en la cola. Y los trades que "no ejecutan" suelen ser positivos, lo que sobreestima tu backtest.

### A Mercado (Next Bar Open)

Ejecuta al mejor precio disponible inmediatamente. Usada cuando la señal no tiene un precio implícito (ej: un oscilador cruza un umbral).

```python
# Patrón típico: señal al cierre de la barra → entrada en la apertura siguiente
if signal_at_close:
    order = 'BUY_MARKET_NEXT_BAR'
    # El sistema marca el precio teórico como el open de la siguiente barra
    # En real, ejecutarás un poco arriba o abajo (slippage bidireccional)
```

### Cuándo Usar Cada Tipo

| Modelo de entrada | Orden recomendada |
|---|---|
| Ruptura de máximos/mínimos | Stop |
| Soporte/resistencia, pivots | Limit |
| Cruce de medias, ruptura de canal | Stop |
| Oscilador (RSI, estocástico) sin precio implícito | A mercado (next bar) |
| Reversión a la media | Limit |

## Setups de Entrada Clásicos

### Canales de Donchian (Ruptura de N barras)

Comprar cuando el precio supera el máximo de N barras. Vender cuando pierde el mínimo. Creado en los años 60, base del sistema de las Tortugas de Richard Dennis. Sigue funcionando en activos tendenciales.

```python
def donchian_entry(highs, lows, close, period=20):
    upper = highs.rolling(period).max()
    lower = lows.rolling(period).min()

    if close.iloc[-1] > upper.iloc[-2]:  # supera máximo de ayer
        return 'LONG'
    elif close.iloc[-1] < lower.iloc[-2]:
        return 'SHORT'
    return None
```

### Medias Móviles

El indicador más usado. Tres variantes de entrada:
- **Cruce de una media**: precio cruza la media → señal
- **Cruce de dos medias**: media rápida cruza la lenta
- **Pendiente de la media**: cambio de dirección → señal

No hay evidencia clara de que una media (simple, exponencial, ponderada) sea consistentemente mejor que otra. Probá en tu sistema específico.

### Bandas de Bollinger

Media de 20 períodos ± 2 desviaciones estándar. Dos usos opuestos:

- **Tendencial**: comprar cuando rompe la banda superior (expansión de volatilidad). Entra tarde, con stops amplios, pero captura movimientos grandes
- **Reversión**: comprar cuando recupera la banda inferior después de haberla perdido. Entrada más precisa, stop más corto

### Indicadores de Momentum

**RSI** (Wilder): mide la magnitud de ganancias vs pérdidas en cierres sucesivos, normalizado de 0 a 100. Por encima de 50 = presión compradora dominante. También se usa como oscilador de sobrecompra/sobreventa.

**MACD** (Gerald Appel): diferencia entre dos EMAs. Indicador de aceleración — mide si las medias se separan o convergen.

**Estocástico** (George Lane): normaliza el precio respecto al rango del período, de 0 a 100. Más suave que el RSI por incluir suavizado. Típicamente para sobrecompra/sobreventa, pero explorá su uso tendencial con valores de período altos.

**ATR** (Wilder): no es un indicador de señal sino de volatilidad. Mide el rango promedio incluyendo gaps. Fundamental para dimensionar stops y profits.

> No te quedes con el uso típico de los indicadores. Un RSI no es solo para sobrecompra/sobreventa — puede funcionar como filtro de tendencia. Un canal de Donchian no es solo para ruptura — puede usarse para reversión comprando en la banda inferior. Cuestioná todo y probá.

## Setups de Salida

### La Asimetría Entrada-Salida

Para entradas: un setup simple, sin muchas reglas. Para salidas: múltiples métodos combinados. Podés (y conviene) salir por stop, por take profit, por señal contraria Y por tiempo, todo en el mismo sistema.

### Salida por Take Profit

Tres formas de calcularlo:

```python
def take_profit(entry_price, method='volatility', **kwargs):
    """Calcular nivel de take profit."""
    if method == 'fixed':
        return entry_price + kwargs['amount']

    elif method == 'percentage':
        return entry_price * (1 + kwargs['pct'])

    elif method == 'volatility':
        # ATR ajusta el TP a la volatilidad actual del mercado
        return entry_price + kwargs['atr'] * kwargs['multiplier']
```

**Recomendación**: ajustar por volatilidad. Un TP fijo de $1,000 no significa lo mismo cuando el activo se mueve 3% al día que cuando se mueve 0.5%.

### Salida por Stop Loss

El stop loss **no es para ganar dinero — es para protegerlo**. Si un stop loss aumenta el beneficio de tu sistema, es mala señal: probablemente hay overfitting.

**El stop loss no es gratis.** Casi siempre, un sistema gana menos con stop que sin stop. Lo que el stop te da es protección contra eventos aberrantes.

```python
def stop_loss(entry_price, method='volatility', **kwargs):
    if method == 'fixed':
        return entry_price - kwargs['amount']
    elif method == 'percentage':
        return entry_price * (1 - kwargs['pct'])
    elif method == 'volatility':
        return entry_price - kwargs['atr'] * kwargs['multiplier']
```

### Salida por Señal Contraria

En casi todos los sistemas, la señal opuesta a la entrada te saca del mercado. Si compraste cuando la media cruzó al alza y ahora cruza a la baja, cerrás.

Puede ser el mismo indicador de la entrada u otro diferente, aunque agregar indicadores diferentes aumenta la complejidad y el riesgo de overfitting.

### Salida Temporal

Suena raro pero funciona: salir después de N barras si el trade no se movió lo suficiente. La lógica es simple — estar invertido es un riesgo. Si lográs el mismo beneficio estando menos tiempo en el mercado, tu ratio riesgo/retorno mejora.

```python
def temporal_exit(bars_in_trade, max_bars, current_pnl=None):
    """
    Salida temporal: si después de N barras el trade no fue a ningún lado,
    salí. Reducís riesgo sin perder beneficio.
    """
    if bars_in_trade >= max_bars:
        return True
    # Variante: si está en pérdida después de N barras, salir antes
    if current_pnl is not None and current_pnl < 0 and bars_in_trade >= max_bars // 2:
        return True
    return False
```

También podés usar ciclos estacionales: no operar los lunes, cerrar antes de noticias macro, evitar determinados meses.

### Trailing Stop

Acompaña al precio a favor de tu posición. Suena ideal en teoría — protegés el beneficio acumulado. En la práctica, **son difíciles de calibrar correctamente** y tienden a acoplarse (overfitting) a los datos del backtest.

El problema principal en sistemas tendenciales: el trailing te puede sacar de los trades con mayor recorrido. El precio retrocede un poco (normal en cualquier tendencia), salta el trailing, y el precio sigue subiendo sin vos. En mercados con retrocesos suaves, pueden funcionar mejor.

Si los usás, ajustá por volatilidad o porcentaje — nunca por valor absoluto, porque un trailing fijo de $500 no tiene el mismo significado con precios altos que con precios bajos.

### Stop Catastrófico

Un stop que idealmente nunca se activa. Cubre cisnes negros — eventos que no están en tus datos históricos.

No todo está en el backtest. Un gap del 9% en el DAX por el Brexit fue tres veces mayor que cualquier gap anterior en los datos. Ningún backtest lo hubiera capturado.

**Consideraciones**:
- Los circuit breakers (7%, 14%, 20% en US) existen pero no garantizan ejecución — en pánico puede no haber contrapartida
- Un stop catastrófico no mejora el beneficio del sistema — está ahí para que sobrevivas al evento que tu modelo no previó
- En un portfolio de 15+ sistemas, algunos pueden no tener stop explícito si salen por señal contraria rápidamente. Con 1-2 sistemas, el stop es imprescindible

## Money Management Básico desde el Inicio

Antes de evaluar cualquier cosa, incorporá un money management básico para ecualizar los resultados a lo largo del tiempo.

```python
def equalized_position_size(account_value, price, atr, risk_per_trade_pct=0.01):
    """
    Position sizing ajustado por volatilidad para que los resultados
    sean comparables a lo largo de todo el histórico.

    Sin esto, los trades de 2024 (Nasdaq a 18,000) dominan el backtest
    vs los de 2009 (Nasdaq a 1,500). Eso sesga la optimización.
    """
    risk_dollars = account_value * risk_per_trade_pct
    shares = int(risk_dollars / atr) if atr > 0 else 0
    return shares
```

**No es lo mismo $1,000 con el Nasdaq a 5,000 que con el Nasdaq a 15,000.** Si no ecualizás, las operaciones recientes (con precios más altos) dominan el análisis y sesgan la optimización. Este money management es solo para ecualizar datos — el algoritmo final de gestión monetaria se elige al final del proceso.

## Evaluación Aislada: Entradas y Salidas por Separado

El método científico exige aislar variables. Pero no podés evaluar una entrada sin una salida (necesitás trades completos para medir).

**Para evaluar entradas**: poné salidas estandarizadas (stop y TP fijos ajustados por ATR) y no las toques. Compará diferentes entradas con las mismas salidas.

**Para evaluar salidas**: usá las entradas ya validadas, o mejor aún, entradas aleatorias. Si tu salida no supera las entradas aleatorias, no tiene edge.

```python
import random

def random_entry(data, probability=0.05):
    """
    Entrada aleatoria: en cada barra, 5% de probabilidad de entrar.
    Cualquier método de entrada debería superar esto.
    Si no lo supera, tu entrada no tiene ventaja.
    """
    return random.random() < probability
```

## El Proceso Completo

1. Definí tu perfil → qué tipo de sistema buscás
2. Investigá ideas → libros, plataformas, experiencia propia
3. Diseñá la entrada → un setup simple, un solo indicador principal
4. Evaluá la entrada con salidas estandarizadas
5. Diseñá las salidas → múltiples: TP, stop, señal contraria, temporal
6. Evaluá cada salida aisladamente
7. Combiná todo → entrada + salidas + money management básico
8. Optimizá con protocolo (in-sample, validación, out-of-sample)
9. Evaluá stop loss al final → no al principio, para no sesgar la evaluación de la ventaja pura
