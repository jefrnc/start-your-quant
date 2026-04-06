> 🇺🇸 [Read in English](History-of-Algo-Trading.md) | 🇪🇸 **Español**

# Historia del Trading Algorítmico: Lo que Importa para Vos Hoy

La historia del algo trading no es solo trivia — cada etapa dejó herramientas, lecciones y estrategias que siguen vigentes. Entender de dónde viene este campo te ayuda a separar lo esencial de lo que es moda.

## 1949-1970: Las Reglas Antes que las Máquinas

### Richard Donchian y el Nacimiento del Trading Sistemático

En 1949, Richard Donchian lanzó el primer fondo que operaba con **reglas estrictas y objetivas** — sin discreción, sin "feeling". Su herramienta principal: los canales de 4 semanas (comprar cuando el precio rompe el máximo de 20 días, vender cuando rompe el mínimo).

```python
def donchian_channel(highs, lows, period=20):
    """
    La estrategia de Donchian de 1949. Sí, sigue funcionando.
    Muchos CTAs (Commodity Trading Advisors) operan variaciones
    de esto con miles de millones bajo gestión.
    """
    upper = highs.rolling(period).max()
    lower = lows.rolling(period).min()
    middle = (upper + lower) / 2
    return upper, lower, middle

def donchian_signal(close, upper, lower):
    if close > upper:
        return 1   # breakout alcista → largo
    elif close < lower:
        return -1  # breakout bajista → corto
    return 0       # dentro del canal → sin señal
```

**Lección vigente**: no necesitás complejidad para tener edge. Los canales de Donchian, con 75 años de historia, siguen siendo la base de muchos fondos trend-following. La sofisticación está en la gestión de riesgo y portfolio, no en la señal de entrada.

### Markowitz y la Teoría de Portfolios (1952)

Harry Markowitz formalizó algo que hoy parece obvio: la diversificación reduce el riesgo sin reducir proporcionalmente el retorno. Su frontera eficiente demostró matemáticamente que **un portfolio bien construido supera a cualquier activo individual ajustado por riesgo**.

A partir de los años 60, las ideas de Markowitz empezaron a aplicarse computacionalmente en universidades e instituciones financieras, sentando las bases para el arbitraje y la optimización de portfolios asistida por computadora.

**Lección vigente**: la diversificación entre sistemas descorrelacionados sigue siendo el concepto más potente en gestión de portfolios algorítmicos. No es glamoroso, pero funciona.

## 1978-1998: La Infraestructura que Hizo Todo Posible

### Los Cimientos del Mercado Electrónico

| Año | Evento | Por qué importa |
|---|---|---|
| 1978 | Primer sistema de negociación intermercado (Nasdaq) | Los mercados empiezan a conectarse electrónicamente |
| 1981 | Fundación de Bloomberg | Terminal de referencia institucional — acceso a datos en tiempo real |
| 1982 | Jim Simons funda Renaissance Technologies | Empieza como firma de investigación, no como fondo quant aún |
| 1991 | World Wide Web | Información financiera accesible globalmente por primera vez |
| 1993 | Interactive Brokers se lanza como broker online | Democratiza el acceso a mercados — antes necesitabas llamar a un broker por teléfono |
| 1998 | SEC regula mercados electrónicos | Nacimiento oficial del trading algorítmico moderno |

### Renaissance Technologies: Referente, No Requisito

Jim Simons armó el equipo más extraordinario de la historia del trading: matemáticos, físicos, criptógrafos. Su fondo Medallion tiene retornos anualizados extraordinarios (reportados en torno al 60-70% bruto antes de fees, según diversas fuentes) desde finales de los 80.

Pero Medallion opera con ventajas que un trader individual no puede replicar:
- Infraestructura de datos y ejecución de miles de millones de dólares
- Equipos de 300+ PhD dedicados full-time
- Acceso a datos y mercados que no están disponibles para retail

**Lección vigente**: Renaissance demuestra que el mercado tiene ineficiencias explotables con métodos cuantitativos. Pero no necesitás su nivel de sofisticación. Hay un espacio enorme entre "opero por intuición" y "tengo 300 PhDs" donde estrategias relativamente simples, bien ejecutadas, son rentables.

## 2000-2010: La Explosión

### Decimalización: El Cambio que Nadie Menciona

Entre 2000 y 2001, los mercados US completaron la transición de cotizar en fracciones (1/16 de dólar = $0.0625) a cotizar en centavos ($0.01). Esto parece menor, pero fue revolucionario:

- **Antes**: el spread mínimo era $0.0625. Una estrategia que ganara menos que eso por trade era inviable
- **Después**: el spread se comprimió a $0.01. Estrategias de alta frecuencia y scalping se volvieron posibles

El volumen algorítmico pasó de ~5% a ~50% en esta década. No fue por "mejores algoritmos" — fue porque la microestructura del mercado finalmente permitía que funcionaran.

**Lección vigente**: cuando evaluás una estrategia histórica, considerá los costos de transacción de la época. Un backtest que empieza en 1995 con spreads de $0.01 está mintiendo — los spreads reales eran 6x más grandes. Siempre modelá costos realistas.

```python
def realistic_transaction_costs(year, is_small_cap=False):
    """
    Costos aproximados de transacción por era.
    Tu backtest debería usar estos, no un costo fijo.
    """
    if year < 2001:
        spread = 0.0625  # pre-decimalización
        commission = 0.01  # por acción
    elif year < 2010:
        spread = 0.02 if not is_small_cap else 0.05
        commission = 0.005
    else:
        spread = 0.01 if not is_small_cap else 0.03
        commission = 0.005  # IBKR-style

    # Para small caps el spread puede ser mucho mayor
    return {
        'spread_per_share': spread,
        'commission_per_share': commission,
        'total_roundtrip': (spread + commission) * 2
    }
```

### El Flash Crash de 2010

El 6 de mayo de 2010, el Dow Jones cayó ~1000 puntos en minutos. La causa: un algoritmo ejecutó una orden de venta masiva de futuros E-mini S&P 500 sin límite de precio, creando una cascada donde otros algoritmos reaccionaron vendiendo, que a su vez disparó más ventas algorítmicas.

Consecuencias directas:
- Se implementaron **circuit breakers** (HALT) que pausan el mercado en caídas del 7%, 14% y 20%
- Se crearon reglas de "limit up/limit down" para acciones individuales
- Mayor escrutinio regulatorio sobre el trading algorítmico

**Lección vigente**: tu sistema debe contemplar halts y condiciones extremas de mercado. Un backtest que ignora halts va a sobreestimar la capacidad de salida en crashes. También: usar market orders en momentos de pánico es peligroso — tu order puede llenar a precios absurdos.

```python
def is_market_halted(price_change_pct, level_1=-7, level_2=-14, level_3=-20):
    """
    Circuit breakers del mercado US (post-2010).
    Tu sistema debe saber que no puede operar durante halts.
    """
    if price_change_pct <= level_3:
        return "HALT_LEVEL_3 — mercado cerrado por el día"
    elif price_change_pct <= level_2:
        return "HALT_LEVEL_2 — pausa 15 min (solo antes de 15:25 ET)"
    elif price_change_pct <= level_1:
        return "HALT_LEVEL_1 — pausa 15 min (solo antes de 15:25 ET)"
    return None
```

## 2010-Presente: HFT, IA, y la Democratización

### High Frequency Trading: El Extremo del Espectro

El HFT opera en microsegundos y nanosegundos. Requiere:
- Colocación física de servidores junto al exchange (colocation)
- Conexiones de fibra óptica dedicadas (o incluso microondas)
- Hardware FPGA personalizado
- Inversión de millones en infraestructura

El HFT captura ineficiencias que duran fracciones de segundo. Como trader individual, **no estás compitiendo contra HFT** — estás operando en timeframes completamente diferentes. Un sistema que opera en gráficos de 15 minutos o diarios no compite por las mismas ineficiencias que uno que opera en nanosegundos.

### IA y Machine Learning en Trading

La narrativa actual es que "la IA va a revolucionar el trading". La realidad es más matizada:

- **Lo que funciona**: ML para procesamiento de datos alternativos (NLP en noticias, sentiment de redes sociales), detección de regímenes de mercado, optimización de ejecución
- **Lo que es difícil**: predecir dirección de precios con ML puro. Los mercados financieros tienen una relación señal/ruido extremadamente baja comparado con otros dominios de ML
- **Lo que no necesitás**: no hace falta usar deep learning para ser rentable. Un sistema de cruce de medias con buena gestión de riesgo puede superar a un modelo LSTM mal implementado

### La Era Actual: Tu Ventaja como Trader Individual

Nunca en la historia fue tan accesible hacer trading algorítmico:

| Antes (pre-2000) | Ahora |
|---|---|
| Datos históricos costaban miles de dólares | Yahoo Finance, Polygon.io gratis o baratos |
| Ejecutar una orden requería llamar al broker | APIs que ejecutan en milisegundos |
| Backtesting requería infraestructura propia | Python + pandas en tu laptop |
| Información del mercado era privilegio institucional | Fluye en tiempo real para todos |

En US, el 60-70% del volumen es algorítmico. Pero la mayoría de ese volumen son estrategias institucionales que operan en timeframes y capitales completamente diferentes al tuyo. Las ineficiencias en small caps, en premarket, en eventos específicos — esas siguen ahí para quien las busque con disciplina y método.

## Timeline Visual

```
1949  Donchian: primeras reglas sistemáticas
  │
1952  Markowitz: teoría moderna de portfolios
  │
1960  Primera operación de arbitraje computacional
  │
1978  Primer sistema de negociación electrónica (Nasdaq)
  │
1981  Bloomberg — datos institucionales en tiempo real
  │
1982  Renaissance Technologies fundada
  │
1991  World Wide Web — información globalizada
  │
1993  Interactive Brokers — trading online democratizado
  │
1998  SEC regula mercados electrónicos → nace el algo trading moderno
  │
2001  Decimalización → explosión de estrategias de corto plazo
  │
2005  ~35% del volumen US es algorítmico
  │
2010  Flash Crash → circuit breakers → más regulación
  │
2010  ~50% del volumen US es algorítmico
  │
2015  HFT en nanosegundos, colocación, FPGA
  │
2020  ~65% del volumen US es algorítmico
  │        
HOY   IA/ML, datos alternativos, acceso democratizado
```

## Qué Llevarte de Todo Esto

1. **Las estrategias simples sobreviven décadas**. Los canales de Donchian tienen 75 años y siguen funcionando. No subestimes lo básico.

2. **La infraestructura importa más que el algoritmo**. Cada salto grande en algo trading vino de cambios en infraestructura (electrónica, internet, decimalización), no de algoritmos más inteligentes.

3. **No competís contra Renaissance ni contra HFT**. Operás en timeframes y mercados donde tus ventajas (flexibilidad, costos bajos, nichos específicos) son reales.

4. **El mercado genera regulación reactiva**. Cada crisis genera nuevas reglas. Tu sistema debe ser adaptable a cambios regulatorios, no depender de una mecánica específica.

5. **La tecnología es accesible como nunca**. La brecha entre institucional y retail se achicó dramáticamente. Lo que te diferencia hoy no es la tecnología — es la disciplina, la gestión de riesgo, y la paciencia para dejar que el edge se materialice.
