> 🇺🇸 [Read in English](Expected-Growth-Metrics-Hierarchy.md) | 🇪🇸 **Español**

# Expected Growth: La Métrica que Casi Nadie Usa

La mayoría de traders evalúan sus sistemas con Sharpe ratio, profit factor, o win rate. Estas métricas son útiles pero incompletas — ninguna captura lo que realmente importa: **cuánto crece tu cuenta trade a trade cuando reinvertís las ganancias**.

El Expected Growth (EG) sí lo captura. Y revela una verdad contraintuitiva: dos sistemas con la misma expectancy aritmética pueden tener resultados radicalmente distintos cuando se componen.

## La Jerarquía de Métricas

```
Expected Growth (EG)  >  Expectancy (Edge)  >  Win Rate  >  Profit Ratio
       ↑                      ↑                    ↑              ↑
   Crecimiento real      Ganancia promedio     Frecuencia    Tamaño relativo
   con compounding       por trade             de aciertos   de ganancias
```

Cada métrica de la derecha alimenta a la siguiente de la izquierda, pero **no la determina**. Podés tener excelente profit ratio y win rate, buena expectancy, y aún así crecer poco geométricamente. La razón es que el compounding no es lineal.

## Las Métricas Paso a Paso

Antes de llegar al EG, necesitás entender las tres métricas que lo alimentan. Cada una mide algo distinto, y cada una sola es insuficiente.

### Win Rate (Tasa de Acierto)

**Qué mide**: el porcentaje de trades que terminan en ganancia.

**Cómo se calcula**:

```python
win_rate = trades_ganadores / trades_totales
# 75 ganadores de 100 trades → win_rate = 0.75 (75%)
```

**Cómo se interpreta**: indica la frecuencia con la que el sistema acierta. Un win rate del 60% significa que de cada 10 trades, aproximadamente 6 son ganadores y 4 son perdedores.

**Valores típicos por tipo de sistema**:
- Sistemas tendenciales (trend following): 30-45%
- Sistemas de reversión a la media (mean reversion): 55-70%
- Sistemas de breakout / volatilidad: 40-55%

**Por qué sola no alcanza**: un sistema con 90% de acierto puede perder dinero si las 10% de pérdidas son catastróficas (ej: gana $1 nueve veces, pierde $20 una vez → pierde $11 neto). Un sistema con 30% puede ser muy rentable si las ganadoras son enormes. El win rate no dice nada sobre el **tamaño** de las ganancias y pérdidas.

### Profit Ratio (Ratio de Ganancia/Pérdida)

**Qué mide**: la relación entre lo que ganás cuando acertás y lo que perdés cuando errás. También se lo conoce como **reward-to-risk ratio** o **payoff ratio**.

**Cómo se calcula**:

```python
profit_ratio = ganancia_promedio / perdida_promedio
# Si en promedio ganás $200 y perdés $100 → profit_ratio = 2.0 (2:1)
```

**Cómo se interpreta**: un ratio de 2:1 significa que cada trade ganador recupera lo que dos trades perdedores sacaron. Un ratio de 0.5:1 significa que necesitás ganar el doble de veces que perdés solo para quedar en cero.

**Valores típicos**:
- Sistemas tendenciales: 2:1 a 5:1 (pocas ganadoras pero grandes)
- Sistemas de reversión: 0.5:1 a 1.5:1 (muchas ganadoras pero chicas)
- Breakout: 1:1 a 3:1

**Por qué solo no alcanza**: un ratio de 10:1 suena espectacular, pero si tu win rate es 5%, perdés 95 de cada 100 trades. 95 × $100 perdidos = $9,500 de pérdida, 5 × $1,000 ganados = $5,000. Ratio increíble, resultado desastroso.

### Expectancy / Edge (Esperanza Matemática)

**Qué mide**: la ganancia promedio esperada por cada dólar que arriesgás. Es la primera métrica que combina win rate y profit ratio en un solo número. También se la conoce como **esperanza matemática**, **edge**, o simplemente **expectancy**.

**Cómo se calcula**:

```python
def expectancy(win_rate, profit_ratio):
    """
    Ganancia esperada por dólar arriesgado.
    Positivo = ventaja. Negativo = el mercado te come.
    Cero = juego neutral (como un casino sin ventaja de la casa).
    """
    return win_rate * profit_ratio - (1 - win_rate)

# Ejemplos:
expectancy(0.50, 2.0)  # = 0.50 → ganás $0.50 por cada $1 arriesgado
expectancy(0.75, 1.0)  # = 0.50 → ganás $0.50 por cada $1 arriesgado
expectancy(0.40, 1.0)  # = -0.20 → PERDÉS $0.20 por cada $1 (no operar)
expectancy(0.30, 5.0)  # = 0.80 → ganás $0.80 por cada $1 arriesgado
```

**Cómo se interpreta**:
- **Positivo**: el sistema tiene ventaja estadística. A largo plazo, ganás dinero
- **Cero**: juego neutral. Las comisiones te van a hacer perder
- **Negativo**: el sistema pierde dinero sistemáticamente. No hay position sizing ni gestión de riesgo que lo salve

**Umbral práctico**: una expectancy de al menos 0.10-0.20 (10-20 centavos por dólar arriesgado) es necesaria para cubrir costos de transacción y slippage. Por debajo de eso, los costos reales se comen la ventaja.

**Por qué sola no alcanza**: la expectancy es una media aritmética. Asume que apostás siempre la misma cantidad fija. Pero si reinvertís ganancias (compounding), el tamaño de tus posiciones crece con tu cuenta. Y cuando componés, la frecuencia de las ganancias (win rate) importa de una manera que la expectancy aritmética no captura. Eso es exactamente lo que mide el Expected Growth.

### Expected Growth (EG)

**Qué mide**: la tasa de crecimiento geométrico esperada por trade cuando el tamaño de posición se ajusta al capital disponible. Es la métrica que captura el **crecimiento real de tu cuenta con compounding**.

**Por qué es diferente a la expectancy**: la expectancy te dice "en promedio ganás X por trade". El EG te dice "tu cuenta crece X% por trade cuando reinvertís". La diferencia es enorme porque el compounding no es lineal — una pérdida del 50% requiere una ganancia del 100% para recuperar.

**Cómo se calcula**: usa la fracción de Kelly (la proporción óptima de capital a arriesgar por trade) y la aplica a la fórmula de crecimiento geométrico.

```python
def expected_growth(win_rate, profit_ratio):
    """
    Crecimiento geométrico esperado por trade con Kelly sizing.
    
    Fórmula: EG = (1 + f*R)^p * (1 - f)^(1-p) - 1
    donde f = Kelly fraction = expectancy / profit_ratio
    """
    p = win_rate
    R = profit_ratio
    edge = p * R - (1 - p)
    
    if edge <= 0:
        return 0  # sin ventaja, no hay crecimiento
    
    f = edge / R  # fracción de Kelly
    eg = (1 + f * R) ** p * (1 - f) ** (1 - p) - 1
    return eg
```

## El Ejemplo que Cambia Todo

Dos sistemas con **exactamente la misma expectancy aritmética** (0.50 por dólar arriesgado):

```python
# Sistema 1: pocas ganadoras pero grandes
eg1 = expected_growth(win_rate=0.50, profit_ratio=2.0)

# Sistema 2: muchas ganadoras pero chicas  
eg2 = expected_growth(win_rate=0.75, profit_ratio=1.0)
```

| Métrica | Sistema 1 | Sistema 2 |
|---|---|---|
| Win Rate | 50% | 75% |
| Profit Ratio | 2:1 | 1:1 |
| Expectancy | 0.50 | 0.50 |
| Kelly fraction | 25% | 50% |
| **Expected Growth** | **6.1%** | **14.0%** |

Misma expectancy, pero el Sistema 2 crece **2.3 veces más rápido**.

### Por Qué Pasa Esto

El win rate entra en la fórmula de EG como **exponente**, no como multiplicador:

```
EG = (1 + f*R)^p * (1 - f)^(1-p) - 1
                 ↑              ↑
              exponente       exponente
```

Cuando el win rate es alto (75%), el primer término `(1 + f*R)^p` domina — ganás frecuentemente y cada ganancia se compone sobre la anterior. El segundo término `(1 - f)^(1-p)` tiene poco impacto porque las pérdidas son infrecuentes.

Cuando el win rate es bajo (50%), aunque las ganancias individuales son mayores (ratio 2:1), las pérdidas intermedias frenan el compounding. Cada pérdida reduce la base sobre la cual la siguiente ganancia se calcula.

**En compounding, la frecuencia de las ganancias importa más que su tamaño.**

### Las Rachas Ganadoras como Motor del Compounding

El win rate como exponente tiene una consecuencia directa: determina la probabilidad de rachas ganadoras consecutivas. Y las rachas son el motor del crecimiento geométrico.

| Win Rate | P(10 wins consecutivos) | Efecto en compounding |
|---|---|---|
| 50% | 0.1% — casi nunca pasa | Plano, crece lento |
| 75% | 5.6% — pasa regularmente | Fuerte, curva exponencial |
| 83% | ~15% — pasa seguido | Explosivo |

Con 75% de WR, una racha de 10 ganadoras consecutivas ocurre 1 de cada ~18 secuencias de 10 trades. Cada una de esas rachas es un "boost" de compounding donde el capital crece sin interrupciones de pérdidas. Con 50% de WR, esas rachas prácticamente no existen.

### La Trampa: EG por Trade vs EG por Día

Un sistema con 14% de EG por trade que opera 0.5 veces por día produce menos crecimiento real que uno con 6% de EG que opera 3 veces por día:

```python
# Crecimiento diario real = (1 + EG_per_trade) ^ trades_per_day - 1
daily_growth_A = (1 + 0.14) ** 0.5 - 1   # ≈ 6.8% diario
daily_growth_B = (1 + 0.06) ** 3 - 1     # ≈ 19.1% diario
# El sistema B crece ~3x más rápido a pesar de tener menos EG por trade
```

**EG por trade × frecuencia de trades = crecimiento real del portfolio.** Cuando comparás sistemas, no mires solo el EG — multiplicalo por la frecuencia de operación.

## Simulación: 1000 Trades

```python
import numpy as np

def simulate_system(win_rate, profit_ratio, trades=1000, simulations=10000):
    """
    Simula el crecimiento de cuenta con Kelly sizing.
    Muestra la distribución real, no solo el promedio.
    """
    edge = win_rate * profit_ratio - (1 - win_rate)
    kelly_fraction = edge / profit_ratio
    
    # Usar half-Kelly (más conservador, estándar en la práctica)
    f = kelly_fraction * 0.5
    
    final_values = []
    for _ in range(simulations):
        capital = 1.0
        outcomes = np.random.random(trades) < win_rate
        for win in outcomes:
            if win:
                capital *= (1 + f * profit_ratio)
            else:
                capital *= (1 - f)
        final_values.append(capital)
    
    return {
        'median_growth': np.median(final_values),
        'mean_growth': np.mean(final_values),
        'pct_profitable': np.mean(np.array(final_values) > 1.0) * 100,
        'worst_5pct': np.percentile(final_values, 5),
    }

# Con half-Kelly (más conservador que Kelly completo):
# Sistema 1 (50% WR, 2:1): crecimiento mediano moderado
# Sistema 2 (75% WR, 1:1): crecimiento mediano significativamente mayor
```

La simulación confirma lo que la fórmula predice: el Sistema 2 no solo crece más rápido en promedio — tiene menor varianza y mayor probabilidad de ser rentable en cualquier ventana de N trades.

## Implicaciones Prácticas

### 1. No Descartes Sistemas de Win Rate Alto con Ratio Bajo

La sabiduría convencional dice "buscá ratio 2:1 o más". Pero un sistema con 70% de acierto y ratio 1:1 puede ser superior a uno con 40% de acierto y ratio 3:1, incluso si la expectancy aritmética es similar. El EG lo revela.

### 2. Ojo con los Sistemas "Espectaculares" de Bajo Win Rate

Un sistema tendencial con 30% de acierto y ratio 5:1 tiene buena expectancy (0.80). Pero el EG puede ser modesto porque las rachas de pérdidas frenan el compounding. Necesitás sobrevivir 7-10 pérdidas consecutivas antes de que llegue la ganadora grande — y cada pérdida reduce tu base de capital.

```python
# Tendencial agresivo: 30% WR, 5:1 ratio
eg_trend = expected_growth(0.30, 5.0)  # Edge=0.80, EG≈7.8%

# Mean reversion conservador: 65% WR, 1.2:1 ratio
eg_mr = expected_growth(0.65, 1.2)     # Edge=0.43, EG≈6.6%

# El tendencial tiene CASI DOBLE de expectancy (0.80 vs 0.43)
# pero solo 18% más de EG (7.8% vs 6.6%)
# El compounding "castiga" el bajo win rate
```

### 3. El EG como Criterio de Selección de Portfolio

Cuando tenés que elegir entre sistemas para tu portfolio, el EG es mejor criterio que el Sharpe ratio o el profit factor:

- **Sharpe ratio**: penaliza la volatilidad al alza (una ganancia enorme baja el Sharpe, lo cual es absurdo)
- **Profit factor**: no distingue entre frecuencia y tamaño de trades
- **EG**: captura exactamente lo que querés maximizar — el crecimiento geométrico de tu cuenta

## Kelly Criterion: Cuánto Arriesgar por Trade

El EG depende de cuánto arriesgás por trade. Arriesgar muy poco desaprovecha la ventaja. Arriesgar demasiado la destruye. El **Kelly Criterion** (John Kelly, 1956, Bell Labs) te da la fracción óptima de capital a arriesgar para maximizar el crecimiento geométrico a largo plazo.

### La Fórmula Básica

```python
def kelly_fraction(win_rate, profit_ratio):
    """
    Fracción óptima del capital a arriesgar por trade.
    Maximiza el crecimiento geométrico a largo plazo.
    """
    edge = win_rate * profit_ratio - (1 - win_rate)
    if edge <= 0:
        return 0  # sin ventaja, no arriesgar nada
    return edge / profit_ratio

# Sistema con 60% WR y ratio 1.5:1
f = kelly_fraction(0.60, 1.5)
# Edge = 0.60*1.5 - 0.40 = 0.50
# Kelly = 0.50 / 1.5 = 0.333 → arriesgar 33% del capital por trade
```

**Qué significa**: si tu sistema tiene 60% de acierto con ratio 1.5:1, Kelly te dice que arriesgues el 33% de tu capital en cada trade para crecer lo más rápido posible.

33% suena enorme. Y lo es. Esa es exactamente la trampa de Kelly completo.

### Por Qué Kelly Completo es Peligroso

La relación entre tamaño de posición y volatilidad **no es lineal — es exponencial**. Duplicar el tamaño no duplica la volatilidad; la cuadruplica o más.

```python
def eg_at_fraction(win_rate, profit_ratio, fraction):
    """EG para cualquier fracción de capital (no solo Kelly óptimo)."""
    p = win_rate
    R = profit_ratio
    f = fraction
    if f <= 0 or f >= 1:
        return 0
    return (1 + f * R) ** p * (1 - f) ** (1 - p) - 1

# Sistema: 60% WR, 1.5:1 ratio, Kelly óptimo = 33%
wr, ratio = 0.60, 1.5
kelly = kelly_fraction(wr, ratio)  # 0.333

fractions = [0.05, 0.10, 0.167, 0.25, 0.333, 0.50, 0.667]
for f in fractions:
    eg = eg_at_fraction(wr, ratio, f)
    label = ""
    if abs(f - kelly) < 0.01: label = " ← KELLY ÓPTIMO"
    if abs(f - kelly*0.5) < 0.01: label = " ← HALF KELLY"
    if abs(f - kelly*1.5) < 0.02: label = " ← 1.5x KELLY"
    print(f"  f={f:.1%}: EG={eg*100:.2f}%{label}")

# Resultado:
#   f=5.0%:  EG=2.31%
#   f=10.0%: EG=4.26%
#   f=16.7%: EG=6.29%  ← HALF KELLY
#   f=25.0%: EG=7.90%
#   f=33.3%: EG=8.45%  ← KELLY ÓPTIMO
#   f=50.0%: EG=6.03%  ← 1.5x KELLY (¡SIMILAR EG que Half Kelly!)
#   f=66.7%: EG=-2.33% ← 2x KELLY (¡PÉRDIDA!)
```

Mirá lo que pasa:

| Fracción | Relación a Kelly | EG | Observación |
|---|---|---|---|
| 5% | 0.15x Kelly | 2.3% | Muy conservador, crece lento |
| 16.7% | **Half Kelly** | 6.3% | **~74% del EG óptimo, volatilidad manejable** |
| 33.3% | **Kelly completo** | 8.5% | Máximo teórico, volatilidad extrema |
| 50% | 1.5x Kelly | 6.0% | **Similar EG a Half Kelly, pero con volatilidad masiva** |
| 66.7% | 2x Kelly | -2.3% | **Perdés dinero.** Oversizing destruye el edge |

### Los Tres Insights Clave

**1. A 1.5x Kelly obtenés el mismo retorno que a 0.5x Kelly, pero con volatilidad brutal.**

Esto es la asimetría mortal del sizing. Pasarte de Kelly es mucho peor que quedarte corto. Si errás por debajo, crecés más lento. Si errás por arriba, podés destruir la cuenta.

**2. La volatilidad escala exponencialmente con el tamaño.**

No es que duplicar la posición duplique el riesgo. Lo cuadruplica. Por eso un pequeño error en la estimación de tus parámetros (win rate, ratio) puede ser catastrófico con Kelly completo — si tu win rate real es 55% en vez de 60%, pasaste de estar en Kelly óptimo a estar sobredimensionado.

**3. El max drawdown esperado es aproximadamente igual al porcentaje de Kelly.**

Si usás half Kelly (16.7% del capital por trade), esperá drawdowns de hasta ~16-17%. Si usás Kelly completo (33%), esperá drawdowns de ~33%. Esta es una regla empírica, no exacta, pero es útil para calibrar expectativas.

### Half Kelly: El Estándar de la Industria

La mayoría de practitioners usan **half Kelly** (la mitad de la fracción óptima). La math justifica por qué:

- Obtenés **~74% del crecimiento** del Kelly óptimo (varía según el sistema, pero consistentemente entre 70-80%)
- Con **volatilidad significativamente menor**
- El drawdown esperado se reduce a la mitad
- Tenés margen de error: si tus estimaciones de win rate o ratio están off, seguís del lado seguro de la curva

```python
def practical_kelly(win_rate, profit_ratio, fraction_of_kelly=0.5):
    """
    Kelly ajustado para uso real.
    fraction_of_kelly=0.50 → half Kelly (estándar)
    fraction_of_kelly=0.25 → quarter Kelly (para datos limitados)
    """
    edge = win_rate * profit_ratio - (1 - win_rate)
    if edge <= 0:
        return 0
    full_kelly = edge / profit_ratio
    return full_kelly * fraction_of_kelly

# Half Kelly para un sistema 60% WR, 1.5:1
f = practical_kelly(0.60, 1.5, fraction_of_kelly=0.50)
# = 0.333 * 0.5 = 0.167 → arriesgar 16.7% por trade
```

### Quarter Kelly: Para Cuando No Estás Seguro

Si tenés datos limitados (pocas operaciones en el backtest), parámetros estimados con incertidumbre, o un sistema nuevo que todavía no validaste en vivo, **quarter Kelly** (25% del óptimo) es más prudente:

- Crecés más lento (~56% del EG óptimo)
- Pero sobrevivís a errores de estimación mucho mayores
- Ideal para los primeros 6-12 meses de un sistema nuevo en producción

### Kelly con Stop Loss: Ajustar por el Riesgo Real

El Kelly básico asume que perdés el 100% de lo arriesgado en cada trade perdedor. Pero si usás stop loss, tu pérdida real es menor. Eso permite posiciones más grandes:

```python
def kelly_stop_adjusted(win_rate, profit_ratio, stop_loss_pct):
    """
    Kelly ajustado por stop loss.
    Si tu stop es del 2% del precio, podés tener posiciones más grandes
    que si arriesgás el 100%.
    
    Parámetros:
    - stop_loss_pct: pérdida máxima por trade como fracción (0.02 = 2%)
    """
    edge = win_rate * profit_ratio - (1 - win_rate)
    if edge <= 0:
        return 0
    
    kelly_base = edge / profit_ratio
    kelly_adjusted = kelly_base / stop_loss_pct
    return kelly_adjusted

# Sistema 60% WR, 1.5:1 ratio, stop del 2%
position = kelly_stop_adjusted(0.60, 1.5, stop_loss_pct=0.02)
# Kelly base = 33.3%
# Ajustado = 33.3% / 2% = 16.67x del capital
# Es decir: con stops del 2%, podés apalancar hasta ~16x

# En la práctica, con half Kelly ajustado:
position_half = position * 0.5  # ~8x
```

**La lógica**: un stop más ajustado limita la pérdida por trade, lo que permite posiciones más grandes para la misma cantidad de riesgo en dólares. Esto te mantiene más frecuentemente expuesto cuando el trade va a favor.

**Precaución**: esto asume que el stop siempre se ejecuta al precio exacto. En la realidad, hay slippage, gaps overnight, y mercados que se saltan tu stop. Nunca dimensiones asumiendo ejecución perfecta del stop.

### Escalamiento por Tamaño de Cuenta

Kelly teórico no considera restricciones de mercado. En la práctica, el tamaño de la cuenta limita cuánto Kelly podés usar:

| Tamaño de cuenta | Kelly práctico | Por qué |
|---|---|---|
| < $25K | Hasta 50% de Kelly | Poca diversificación, cada trade pesa mucho |
| $25K - $100K | 33-50% | Empezás a tener margen para diversificar |
| $100K - $200K | 25-33% | Slippage empieza a importar en small caps |
| $200K - $500K | 12.5-25% | Fill probability baja, movés precio al entrar |
| $500K+ | Stake fijo o < 12.5% | En small caps, tu orden ES el mercado |

La razón: slippage, probabilidad de ejecución y restricciones de liquidez escalan con el tamaño de posición. Una posición de $500K en una small cap de $2 va a mover el precio significativamente al entrar y al salir. Kelly teórico no sabe esto.

### Error Común: Ajustar Kelly por "Calidad del Setup"

"Uso 50% Kelly en setups A+ y 25% en setups B." Esto es incorrecto.

Kelly **ya incorpora la calidad del setup** a través del win rate y el profit ratio. Un setup A+ naturalmente tiene mejor WR y/o mejor ratio, lo que produce un Kelly fraction más alto. Un setup B tiene peores métricas, lo que produce un Kelly más bajo.

Si ajustás manualmente encima de eso, estás sobreescribiendo la matemática con tu opinión. La única razón válida para reducir Kelly es incertidumbre en los parámetros (pocos datos, sistema nuevo) — y para eso están half Kelly y quarter Kelly.

### Alternativa: Optimal-f de Ralph Vince

Kelly asume distribución binaria (ganás R o perdés 1). **Optimal-f** de Ralph Vince usa la distribución completa de retornos históricos para encontrar la fracción óptima. Es conceptualmente superior porque no simplifica la distribución, pero es computacionalmente más costoso y requiere suficientes trades históricos para que la distribución empírica sea representativa.

En la práctica, Kelly con half/quarter adjustment es suficiente para la mayoría de los sistemas. Optimal-f es relevante si operás con distribuciones muy asimétricas (shorts en small caps, por ejemplo).

## Cómo Incorporar EG y Kelly en tu Proceso

1. **Calculá EG para todos tus sistemas** y compará contra su expectancy aritmética. Vas a encontrar sorpresas — sistemas que parecían equivalentes por expectancy no lo son por EG

2. **Usá EG como función objetivo en optimización** en lugar de net profit o Sharpe. El optimizador buscará parámetros que maximicen el crecimiento geométrico real

3. **Compará sistemas con EG antes de armar un portfolio**. Un portfolio de sistemas con EG alto individualmente, y baja correlación entre sí, es la combinación más potente

4. **Recordá que EG asume Kelly sizing**. Si usás position sizing fijo (siempre el mismo monto), la expectancy aritmética es suficiente. El EG importa cuando componés — y si no estás componiendo, estás dejando crecimiento sobre la mesa

## Anti-Scalping: La Matemática de No Cortar Ganadores

La mayoría de los traders (y muchos algos) cortan las ganancias demasiado temprano. El instinto dice "asegurá la ganancia". La matemática dice lo contrario.

### Halfway Probability: Probabilidades Condicionales en Acción

Si tu sistema tiene un win rate del 80% y un trade ya está en +5%, ¿cuál es la probabilidad de que llegue a +10%?

La intuición dice "ya gané 5%, mejor cierro". Pero la matemática dice lo contrario. Cada tick a tu favor es **evidencia bayesiana** de que la tesis del trade es correcta. La probabilidad condicional (dado que ya estás en ganancia) de llegar al target **aumenta** a medida que el trade avanza:

Ejemplo con datos empíricos de un sistema con ~80% WR en small caps:

| Tu ganancia actual | P(duplicar al siguiente nivel) | Implicación |
|---|---|---|
| +5% | ~94% de llegar a +10% | Cubrir es tirar dinero |
| +7.5% | ~80% de llegar a +15% | Todavía extremadamente probable |
| +10% | ~70% de llegar a +20% | Mantener |
| +15% | La probabilidad se estabiliza | Home runs — HOLD |

*Estos valores son específicos de un sistema particular. Calculá los tuyos con la función de simulación condicional más abajo.*

Esto es **Bayesian updating** aplicado a trading: cada movimiento a favor actualiza tu estimación de la probabilidad de éxito hacia arriba.

Estos valores vienen de simulaciones con datos reales de sistemas con ~80% de WR. No hay una fórmula cerrada simple que los reproduzca — dependen de la distribución específica de retornos del sistema. La forma de calcularlos para tu sistema es con **simulación condicional**:

```python
import numpy as np

def estimate_conditional_probability(trade_returns, current_pct, target_pct, n_sims=50000):
    """
    Estima la probabilidad de alcanzar target_pct dado que
    ya estás en current_pct, usando la distribución real de trades.
    
    Simula trayectorias que empiezan en current_pct y cuenta
    cuántas alcanzan target_pct antes de volver a 0%.
    """
    reached_target = 0
    for _ in range(n_sims):
        pnl = current_pct
        for _ in range(50):  # máximo 50 trades para llegar
            trade = np.random.choice(trade_returns) * 100  # a porcentaje
            pnl += trade
            if pnl >= target_pct:
                reached_target += 1
                break
            if pnl <= 0:
                break
    return reached_target / n_sims

# Uso: estimate_conditional_probability(mis_trades, 5.0, 10.0)
# Con tus datos reales, vas a obtener las probabilidades específicas
# de tu sistema — no una aproximación genérica.
```

### El Sesgo de Cortar Ganadores

Pensá en 100 trades de tu sistema. Algunos van a ser perdedores grandes (max loss). Otros van a ser ganadores grandes (home runs). La distribución natural del sistema produce ambos.

Si absorbés los max losses completos (porque el stop se ejecuta y no podés evitarlos) pero cortás los home runs prematuramente (porque "asegurás ganancia"), estás haciendo algo muy específico: **sesgando la distribución de resultados en tu contra**.

```
Distribución natural del sistema:
[pérdida grande] [pérdida chica] [ganancia chica] [ganancia grande]
      ← los absorbés completos →      ← los cortás temprano →

Resultado: tu sistema real tiene peores métricas que el backtest
porque eliminaste las colas positivas pero mantuviste las negativas.
```

### La Probabilidad Acumulada

La probabilidad de no tener ni un solo home run en N trades es `(1 - P_homerun)^N`. Esto cae exponencialmente:

```python
def prob_at_least_one_homerun(p_homerun_per_trade, n_trades):
    """P(al menos 1 home run en N trades)"""
    return 1 - (1 - p_homerun_per_trade) ** n_trades

# Si cada trade tiene 20% de probabilidad de ser home run:
for n in [5, 10, 20, 50]:
    p = prob_at_least_one_homerun(0.20, n)
    print(f"  {n} trades: {p:.0%} de probabilidad de al menos 1 home run")

# 5 trades: 67%
# 10 trades: 89%
# 20 trades: 99%
# 50 trades: 99.99%
```

En 10 trades, tenés ~90% de probabilidad de al menos un home run. Pero si cortás todos los trades a +5% en vez de dejarlos llegar a +10%, ese home run nunca se materializa en tu cuenta.

### Implicaciones para Trailing Stops y Targets

1. **Trailing stops muy ajustados matan los home runs.** Si tu trailing protege el +3% pero el sistema produce trades de +15% regularmente, el trailing te saca antes de que la cola positiva se materialice

2. **Targets fijos limitan el upside.** Un TP en 2:1 cuando el sistema naturalmente produce trades de 5:1 está regalando la diferencia

3. **La solución no es "no usar stops/targets"** — es calibrarlos con la distribución real de tu sistema. Si el backtest muestra que el 15% de tus trades producen ganancias > 3R, tu trailing o target no debería cortar en 2R

4. **Evaluá el costo de oportunidad**: ¿cuánto EG estás sacrificando por la "tranquilidad" de asegurar ganancias temprano? Calculalo con la fórmula de EG usando el profit ratio real vs el profit ratio cortado

### Scalping Produce Expectativa Negativa

El dato más duro contra el scalping: en simulaciones con datos reales, cubrir sistemáticamente al +5% produce **Sim-EG negativo**. Literalmente perdés dinero a largo plazo haciendo scalping en un sistema que es rentable si lo dejás correr.

| Estrategia de salida | Resultado |
|---|---|
| Cubrir todo al +5% | Sim-EG **negativo** — perdés dinero |
| Cubrir todo al +10% | ~2% profit, Sim-EG ~0.4% — marginal |
| Set and forget (mantener al cierre) | Mejor resultado posible |

La conclusión empírica es consistente: **no se encontró una estrategia de cobertura parcial que supere al full-day hold**. El costo de asegurar ganancias temprano supera al beneficio de evitar los retrocesos.

Esto no significa que nunca debas cerrar un trade antes del target. Significa que si tu sistema tiene edge, la decisión por defecto debería ser mantener, y la carga de la prueba está en demostrar que cerrar antes mejora el Sim-EG — no al revés.

## Sim-EG: Monte Carlo como Métrica

La fórmula cerrada de EG asume distribución binaria (ganás R o perdés 1). Tu sistema real tiene una distribución continua de resultados — trades que ganan poco, trades que ganan mucho, trades que pierden distinto cada vez. Para capturar esto, usamos **simulación de Monte Carlo no como validación, sino como la métrica misma**.

### El Proceso

```python
import numpy as np

def sim_eg(trade_returns, n_simulated_trades=10000, n_runs=3):
    """
    Simulated Expected Growth: estima el EG real del sistema
    usando la distribución empírica de trades (no la teórica).
    
    Más robusto que la fórmula cerrada cuando:
    - La distribución no es binaria (la mayoría de los casos)
    - Hay asimetría (skew) en los retornos
    - Hay fat tails
    
    Parámetros:
    - trade_returns: array de retornos por trade del backtest
      (ej: [0.02, -0.01, 0.05, -0.008, ...])
    - n_simulated_trades: trades a simular por corrida (10K es estándar)
    - n_runs: corridas para promediar (3 es suficiente con 10K trades)
    """
    eg_estimates = []

    for _ in range(n_runs):
        # Resamplear con reemplazo de la distribución real
        sampled = np.random.choice(trade_returns, size=n_simulated_trades, replace=True)

        # Calcular crecimiento geométrico
        # Cada trade multiplica el capital por (1 + retorno)
        growth_factors = 1 + sampled
        final_value = np.prod(growth_factors)

        # EG = crecimiento por trade = raíz N-ésima del valor final - 1
        eg = final_value ** (1 / n_simulated_trades) - 1
        eg_estimates.append(eg)

    return {
        'sim_eg': np.mean(eg_estimates),
        'eg_std': np.std(eg_estimates),
        'eg_runs': eg_estimates,
    }

# Ejemplo de uso:
# trades = np.array([resultados de tu backtest])
# result = sim_eg(trades)
# print(f"Sim-EG: {result['sim_eg']*100:.2f}% por trade")
```

### Por Qué 10,000 Trades

Con 1,000-2,000 trades simulados, los resultados varían bastante entre corridas (ej: 3.8%, 5.1%, 4.2%). Con 10,000, convergen (ej: 5.0%, 5.2%, 5.07%). Tres corridas de 10K dan 30K trades efectivos, suficiente para que el estimador sea estable.

### Por Qué Sim-EG > Fórmula Cerrada

La fórmula cerrada de EG asume que cada trade gana exactamente R o pierde exactamente 1. En la realidad:

- Un trade puede ganar 0.5R, 1R, 2R, o 5R
- Un trade puede perder 0.3R, 0.7R, o 1R (si tiene stop)
- La distribución puede tener skew positivo (cola derecha más larga)
- Puede haber fat tails que la fórmula binaria no captura

El Sim-EG usa la **distribución empírica real** de tus trades. Es esencialmente un **bootstrap resampling** aplicado al crecimiento compuesto. No hace supuestos sobre la forma de la distribución — usa directamente lo que tu sistema produjo.

### Sim-EG como Quality Gate

Usá Sim-EG como filtro mínimo de calidad: **si Sim-EG < 2% en 10K trades, el edge es demasiado frágil para operar.** Los costos reales (slippage, comisiones, errores de ejecución) van a consumir un edge tan fino.

### Simulaciones del Peor Caso

El poder real del Sim-EG es explorar los extremos. Con miles de corridas de bootstrap, podés encontrar el peor escenario posible con tu distribución de trades:

```python
def sim_eg_worst_case(trade_returns, n_trades=100, n_simulations=10000):
    """
    ¿Cuál es el peor escenario plausible para N trades?
    Busca entre miles de simulaciones la peor trayectoria.
    """
    final_values = []
    for _ in range(n_simulations):
        sampled = np.random.choice(trade_returns, size=n_trades, replace=True)
        capital = np.prod(1 + sampled)
        final_values.append(capital)

    return {
        'median': np.median(final_values),
        'worst_5pct': np.percentile(final_values, 5),
        'worst_found': min(final_values),
        'best_found': max(final_values),
        'pct_profitable': np.mean(np.array(final_values) > 1.0) * 100,
    }

# Si en 10,000 simulaciones de 100 trades, el PEOR caso
# todavía duplica el capital, tenés convicción real.
# Si el peor caso pierde dinero, el edge es frágil.
```

Esto da un nivel de confianza que ninguna otra métrica provee: "incluso en el peor escenario del bootstrap, ¿sobrevivo?"

### Block Bootstrap: Preservar Rachas

El bootstrap estándar (i.i.d.) asume que cada trade es independiente del anterior. Pero en la realidad puede haber **autocorrelación temporal** — rachas ganadoras o perdedoras que dependen del régimen de mercado.

El **block bootstrap** resuelve esto: en vez de resamplear trades individuales, resamplea bloques de trades consecutivos (ej: bloques de 5-10 trades). Esto preserva la estructura temporal.

```python
def sim_eg_block_bootstrap(trade_returns, block_size=5, n_trades=10000, n_runs=3):
    """
    Block bootstrap: resamplea bloques consecutivos de trades
    para preservar autocorrelación temporal.
    
    Si hay regime dependency (el mercado alterna fases buenas/malas),
    el bootstrap iid lo esconde. El block bootstrap lo preserva.
    """
    n = len(trade_returns)
    eg_estimates = []

    for _ in range(n_runs):
        sampled = []
        while len(sampled) < n_trades:
            start = np.random.randint(0, n - block_size)
            sampled.extend(trade_returns[start:start + block_size])
        sampled = np.array(sampled[:n_trades])

        final_value = np.prod(1 + sampled)
        eg = final_value ** (1 / n_trades) - 1
        eg_estimates.append(eg)

    return np.mean(eg_estimates)

# Si Sim-EG iid ≈ Sim-EG block → no hay autocorrelación significativa
# Si Sim-EG block << Sim-EG iid → hay regime dependency que inflaba el resultado
```

Si la diferencia entre Sim-EG estándar y block bootstrap es grande, tu sistema probablemente depende de un régimen de mercado específico y va a sufrir cuando el régimen cambie.

### Cuándo Usar Cada Uno

| Situación | Usar |
|---|---|
| Comparar ideas rápidamente | Fórmula cerrada de EG |
| Evaluar un sistema con backtest completo | Sim-EG |
| Pocos trades (< 100) | Fórmula cerrada (Sim-EG no tiene suficientes datos para resamplear) |
| Distribución con fat tails o skew | Sim-EG (captura la forma real) |
| Optimización (miles de evaluaciones) | Fórmula cerrada (más rápida) |

### Integración Práctica

Agregá Sim-EG como una columna más en tu evaluación de sistemas, junto con Sharpe, profit factor, y max drawdown:

```python
def full_system_evaluation(trade_returns):
    """Evaluación completa de un sistema."""
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    wr = len(wins) / len(trade_returns)
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    profit_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    edge = wr * profit_ratio - (1 - wr)
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('inf')

    # Sim-EG captura lo que las otras métricas no pueden
    seg = sim_eg(trade_returns)

    return {
        'trades': len(trade_returns),
        'win_rate': f"{wr:.1%}",
        'profit_ratio': f"{profit_ratio:.2f}",
        'expectancy': f"{edge:.3f}",
        'profit_factor': f"{pf:.2f}",
        'sim_eg': f"{seg['sim_eg']*100:.2f}% por trade",
    }
```

## Limitaciones del EG y Sim-EG

- **Fórmula cerrada de EG**: asume distribución binaria, conocimiento perfecto de parámetros, independencia entre trades. Útil para comparar rápido, no para decisiones finales
- **Sim-EG**: más robusto pero necesita suficientes trades históricos (mínimo ~200 para que el resampleo sea representativo). No captura cambios de régimen de mercado
- **Ambos**: no capturan correlación temporal entre trades (rachas), ni el impacto de costos variables, ni eventos que no están en los datos históricos (cisnes negros)

Aún con estas limitaciones, EG y Sim-EG son métricas más informativas que expectancy aritmética, Sharpe ratio o profit factor para cualquier sistema que opere con position sizing proporcional al capital. La razón es simple: son las únicas que miden lo que realmente querés maximizar — el crecimiento geométrico de tu cuenta.
