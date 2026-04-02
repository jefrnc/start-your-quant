# Filtros, Selección de Activos y Ejemplos Prácticos

## Filtros: Poder y Peligro

Un filtro es una regla que elimina trades. El objetivo es quitar los perdedores sin tocar los ganadores. Suena perfecto — y por eso es peligroso.

### Por Qué los Filtros Acoplan

Cada filtro que agregás reduce la muestra. Con menos trades, tu evaluación estadística pierde significancia. Y lo peor: es tremendamente fácil mejorar un backtest con filtros.

```
Sin filtro:  100 trades, 45% win rate, PF 1.3
+ Quitar lunes: 82 trades, 48% win rate, PF 1.5
+ Quitar octubre: 74 trades, 51% win rate, PF 1.7
+ Si máximo ayer < máximo anteayer: 58 trades, 55% win rate, PF 2.1
```

Cada filtro "mejora" el sistema en el backtest. Pero lo que estás haciendo es acoplar el sistema a los datos históricos. En real, esos patrones probablemente no se repitan.

### Reglas para Filtrar sin Destruir

1. **Un filtro por sistema, como máximo.** Si necesitás dos filtros para que funcione, la señal de entrada probablemente no tiene edge
2. **Debe tener lógica de mercado.** "No opero los lunes" necesita una razón (ej: menor liquidez en apertura semanal), no solo que el backtest mejore
3. **Evaluar en todo el histórico.** El filtro debe actuar en suficientes casos a lo largo del tiempo, no solo en un período
4. **El sistema debe funcionar razonablemente SIN el filtro.** Si sin filtro el sistema es un desastre, el filtro está maquillando un sistema roto

## Cada Activo Tiene Personalidad

Un sistema que funciona en el Nasdaq no necesariamente funciona en soja, y uno que funciona en gráfico diario puede fallar en 5 minutos.

### Regla General por Timeframe

| Timeframe | Ruido | Extrapolabilidad | Trades | Mejor para |
|---|---|---|---|---|
| 1-5 min | Muy alto | Baja — específico por activo | Muchos | Un solo activo, bien calibrado |
| 15-60 min | Alto | Moderada | Moderados | Grupo de activos similares |
| Diario | Moderado | Alta | Pocos-moderados | Múltiples activos, ideas universales |
| Semanal/Mensual | Bajo | Muy alta | Muy pocos | Cestas de 50-100 activos para tener significancia |

**A menor timeframe, más ruido, más difícil de extrapolar a otros activos, y más rápido se degrada el edge.** Las ideas que funcionan en gráficos diarios y semanales tienden a ser más universales y robustas.

### Cómo Medir Tendencialidad y Volatilidad

Antes de aplicar un sistema a un activo, medí si ese activo tiene las características que tu sistema necesita.

**ADX (Average Directional Index)**: mide si hay tendencia, sin importar la dirección. Por encima de ~20 se considera que hay tendencia. No te dice si es alcista o bajista — para eso están las líneas DI+ (presión compradora) y DI- (presión vendedora) que lo acompañan.

**ATR normalizado (ATR%)**: el ATR dividido por el precio, expresado en porcentaje. Permite comparar volatilidad entre activos con precios muy distintos. Ver [KISS: Principios de Diseño](./KISS-Design-Principles.md) para la implementación completa.

Valores orientativos comparando activos en gráficos diarios de largo plazo:

| Activo | ADX medio | ATR% medio | Perfil |
|---|---|---|---|
| Nasdaq 100 | ~23 | ~1.6% | Volátil, tendencia moderada |
| S&P 500 | ~25 | ~1.2% | Menos volátil, más tendencial |
| Oro (GLD) | ~23 | ~1.0% | Baja volatilidad, tendencia moderada |
| Soja | ~24 | ~2.2% | Alta volatilidad, buena tendencia |
| Petróleo | ~26 | ~3.0% | Muy volátil, muy tendencial |

*Estos valores son orientativos y varían según el período analizado. Siempre verificá con tus propios datos.*

### Patrones por Tipo de Activo

**Índices bursátiles masivos** (S&P 500, Nasdaq): en intradiario son muy reversivos — cuesta capturar tendencia. En diario y semanal sí muestran tendencias claras. La volatilidad es asimétrica: aumenta en caídas, baja en subidas.

**Materias primas menos operadas** (soja, gas natural, café): tienden a ser más tendenciales. Menos arbitraje algorítmico, movimientos más direccionales.

**Renta fija** (bonos, TLT): tendencialidad similar a acciones pero menor volatilidad. Sistemas tendenciales pueden funcionar bien.

**Regla general**: a más masivo y líquido el activo, menos tendencia en intradiario y más reversión a la media. A menos operado, más tendencia.

## Cuatro Ejemplos que Enseñan

Estos no son sistemas para operar — son para entender la estructura de entrada/salida y cómo diferentes reglas producen diferentes perfiles de riesgo.

### 1. Buy and Hold: La Referencia

El setup más simple posible. Comprar y mantener. Sin salida, sin stop, sin nada.

```python
# Setup de entrada: comprar en la primera barra
# Setup de salida: ninguno (hasta el final del backtest)
# Resultado en SPY (~25 años): ~7.5% anual (sin dividendos)
# Drawdown: severo (50%+ en 2008)
# Tiempo invertido: 100%
```

**Para qué sirve**: es tu benchmark. Cualquier sistema que no supere al buy & hold ajustado por riesgo no tiene razón de existir.

### 2. Media de 200: El Tendencial Clásico

Comprar cuando el cierre está por encima de la media de 200 días. Vender cuando cae por debajo.

```python
def sma200_signal(close, sma200):
    """El tendencial más básico que existe."""
    if close > sma200:
        return 'BUY'
    elif close < sma200:
        return 'SELL'
    return 'HOLD'

# Resultado en SPY:
# Win rate: ~30% — la mayoría de señales son falsas
# Profit factor: alto — las pocas ganadoras son enormes
# Operaciones: ~105 en todo el histórico
# Tiempo invertido: ~71%
# Drawdown: menor que buy & hold
```

**Lección**: un sistema puede acertar solo el 30% de las veces y ser rentable. Lo que importa es el tamaño de las ganadoras vs las perdedoras. Este sistema falla constantemente en mercados laterales pero captura las tendencias grandes.

### 3. Golden Cross: Menos Ruido, Menos Trades

Media de 50 cruza por encima de media de 200 → comprar. Cruza por debajo → vender.

```python
def golden_cross_signal(sma50, sma50_prev, sma200, sma200_prev):
    """Más lento que la media de 200 sola, pero mucho más limpio."""
    if sma50 > sma200 and sma50_prev <= sma200_prev:
        return 'BUY'   # golden cross
    elif sma50 < sma200 and sma50_prev >= sma200_prev:
        return 'SELL'  # death cross
    return 'HOLD'

# Resultado en SPY:
# Win rate: ~84% — muy pocos fallos
# Operaciones: ~13 en todo el histórico
# Retorno: ~4.5% anual
# Problema: tan pocas operaciones que no tiene significancia estadística
```

**Lección**: un win rate del 84% impresiona, pero con solo 13 trades no podés concluir nada estadísticamente. Un sistema necesita cientos de trades para ser evaluable. Este ejemplo muestra por qué los sistemas de largo plazo necesitan testearse en múltiples activos.

### 4. TPS de Connors: Promediar a la Baja con Método

TPS significa **Time Price Scale** — un sistema publicado por Larry Connors que escala posiciones comprando más a medida que el precio corrige dentro de una tendencia alcista. Combina filtro (media 200), señal de entrada (RSI de 2 períodos), y escalado de posición.

**Lado largo:**
- **Filtro**: precio por encima de media de 200 (mercado alcista)
- **Nivel 1**: RSI(2) < 25 durante 2 días seguidos → comprar 10%
- **Nivel 2**: si el precio cae desde la entrada anterior → agregar 20%
- **Nivel 3**: si cae más → agregar 30%
- **Nivel 4**: si cae más → agregar 40% (total: 100%)
- **Salida**: RSI(2) cierra por encima de 70

**Lado corto**: simétrico (por debajo de media 200, RSI(2) > 75)

```python
def tps_connors_signal(close, sma200, rsi2, rsi2_prev, position_level):
    """
    Lógica del TPS de Connors (simplificada).
    Compra correcciones dentro de tendencia alcista.
    Escala la posición a medida que la corrección se profundiza.
    """
    levels = {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.40}

    # Filtro: solo largo si está por encima de 200
    if close <= sma200:
        return None, 0

    # Entrada: RSI(2) por debajo de 25, dos días seguidos
    if position_level == 0:
        if rsi2 < 25 and rsi2_prev < 25:
            return 'BUY', levels[0]  # 10%
        return None, 0

    # Escalado: si el precio cayó desde la última entrada
    if position_level < 4:
        # (simplificado — en real se compara contra el close de la última entrada)
        return 'ADD', levels.get(position_level, 0)

    return None, 0

def tps_exit(rsi2):
    """Sale cuando RSI(2) supera 70."""
    return rsi2 > 70
```

**Resultados típicos en SPY:**
- Win rate largo: ~71%
- Tiempo invertido: ~18% (solo largo)
- Profit factor: alto
- Sin stop loss explícito

**Lecciones:**

1. **Promediar a la baja puede funcionar en ETFs diversificados dentro de tendencia alcista** — rompe la regla clásica de "cortar pérdidas rápido", pero tiene lógica: un ETF amplio como SPY históricamente recupera correcciones dentro de una tendencia alcista confirmada por la media de 200. Fuera de tendencia alcista (precio debajo de la media 200), esta lógica NO aplica

2. **La baja exposición es una ventaja**. Solo el 18% del tiempo invertido significa que el 82% del capital está libre para otros sistemas. Esto habilita apalancamiento o un portfolio de estrategias

3. **El riesgo existe aunque no tenga stop.** Sin stop, un cisne negro puede generar un drawdown mayor a cualquier cosa en el histórico. Connors lo reconoce y recomienda operar TPS solo en ETFs diversificados (como SPY), nunca en acciones individuales, porque un ETF amplio tiene mucha menor probabilidad de colapso que una acción sola

4. **Lado corto: más rentable por trade pero más riesgoso.** Mayor profit factor pero drawdown ~2x vs el lado largo. La volatilidad es asimétrica — las caídas son rápidas y violentas

## Pragmatismo sobre Purismo

El método científico dice evaluar entradas y salidas por separado. Y es lo ideal. Pero si estás empezando y encontraste un sistema en un libro que querés probar completo — probalo completo. No dejes que el perfeccionismo te paralice.

Lo importante es:

- **Entender qué hace cada parte.** Aunque no las evalúes por separado, sabé cuál es tu entrada, cuál es tu salida, y por qué
- **Ser crítico.** Si un libro dice "el RSI se usa para sobrecompra/sobreventa", probá usarlo como filtro de tendencia. Si dicen "nunca promedies a la baja", probá con datos. Cuestioná todo
- **Ser pragmático.** Si los ciclos lunares te funcionan y podés demostrarlo estadísticamente, adelante. Lo que importa es el edge demostrable, no la elegancia teórica

Con más experiencia, serás menos categórico. "Todo depende" es la frase más honesta en trading algorítmico.
