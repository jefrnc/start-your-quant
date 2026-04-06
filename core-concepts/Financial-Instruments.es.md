> 🇺🇸 [Read in English](Financial-Instruments.md) | 🇪🇸 **Español**

# Instrumentos Financieros para Quant Traders

Antes de construir un sistema algorítmico, necesitás entender **qué estás tradeando**. Cada instrumento tiene reglas distintas que afectan directamente cómo diseñás, backtesteas y ejecutás tu estrategia.

## Acciones (Renta Variable)

Una acción representa propiedad parcial de una empresa. Cuando comprás una acción, sos socio — con derechos a dividendos y voto en juntas.

Se llama "renta variable" porque el rendimiento total (precio + dividendos) es incierto — a diferencia de un bono donde el cupón está fijado de antemano.

### Lo que importa para tu sistema

- **Representación gráfica**: se grafica por el *last* (último precio al que se ejecutó una operación)
- **Horarios**: mercado regular 9:30-16:00 ET, premarket desde 4:00 AM, afterhours hasta 20:00
- **Liquidez**: varía enormemente — una large cap como AAPL vs una small cap de $2 son mundos distintos
- **Short selling**: requiere localizar acciones prestadas (locate), no siempre disponible en small caps
- **Sin vencimiento**: podés mantener una posición indefinidamente

### Por qué el mercado americano

El mercado US concentra el mayor volumen, la mayor cantidad de instrumentos listados, y las APIs más maduras para trading algorítmico. Si tu cuenta es chica, las comisiones de brokers como IBKR ($0.005/acción) son manejables. Empezar con acciones españolas o europeas limita tus oportunidades y tus herramientas.

```python
import yfinance as yf

# Datos de una acción — así de simple es acceder
data = yf.download('AAPL', period='1mo', interval='1d')
print(f"Último precio: ${data['Close'].iloc[-1]:.2f}")
print(f"Volumen promedio: {data['Volume'].mean():,.0f} acciones/día")
```

## Bonos (Renta Fija)

Instrumentos de deuda: le prestás dinero a una empresa o gobierno y te devuelven el principal más un cupón (interés fijo). Se llama "renta fija" porque el cupón está predeterminado.

### Lo que importa para tu sistema

- **El cupón no cambia, pero el precio del bono sí**: existe un mercado secundario donde los bonos cotizan a descuento o prima
- **Relación inversa con tasas de interés**: cuando las tasas suben, los precios de bonos bajan — esto es tradeable algorítmicamente
- **Mayor cupón = mayor riesgo**: un bono de Argentina que paga 15% es más riesgoso que un Treasury US al 4%
- **Vencimiento fijo**: si esperás al vencimiento, cobrás exactamente lo pactado (salvo default)

### Aplicación quant

Los bonos son fundamentales para estrategias de **curva de tasas**, **spread trading** (ej: largo bonos corporativos / corto treasuries), y como indicador macro para sistemas de acciones. Un sistema de momentum en acciones que ignora el mercado de bonos está operando con información incompleta.

## Futuros

Contratos estandarizados para comprar/vender un activo a un precio pactado en una fecha futura. Cotizan en mercados regulados (CME, EUREX, etc).

### Lo que importa para tu sistema

- **Tienen vencimiento**: si no cerrás antes, te pueden entregar el activo físico (materias primas) o liquidar por diferencia (financieros)
- **Rollover**: para mantener exposición continua, hay que "rolear" del contrato próximo al siguiente — tu backtest DEBE considerar esto
- **Representación gráfica**: se grafica por el *last*, igual que acciones
- **Margen**: no pagás el 100% del valor — operás con margen, lo que amplifica ganancias y pérdidas
- **Posición corta nativa**: podés vender futuros sin restricciones de locate

### Trampas comunes en backtesting

Para un tratamiento completo del ajuste de datos en futuros, ver [Calidad de Datos y Ajustes](../technical-practices/Data-Quality-Adjustments.md).

```python
# MAL: usar un gráfico continuo sin ajustar por rollover
# Los gaps entre contratos generan señales falsas

# BIEN: ajustar backward por diferencia al momento del rollover
# Concepto simplificado — en la práctica se usan herramientas
# especializadas o las funciones de la plataforma de trading.
#
# La idea: en la fecha de rolo, calcular la diferencia de precio
# entre el contrato nuevo y el viejo, y restar esa diferencia
# a todo el histórico anterior para eliminar el gap artificial.
```

### Especificaciones que debés conocer antes de operar

Cada futuro tiene especificaciones únicas. Antes de incluir cualquier futuro en tu sistema, verificá:

| Especificación | Por qué importa |
|---|---|
| Tamaño del contrato | Define cuánto capital necesitás realmente |
| Tick mínimo y valor | Afecta tu stop loss mínimo en dólares |
| Último día de trading | Si tu sistema no cierra antes, el broker lo hará (con penalidad) |
| Entregable vs. liquidación cash | Nunca quieras que te lleguen 1000 barriles de petróleo |
| Horario de trading | Algunos futuros operan casi 24h, otros no |

## Opciones

Le dan al comprador un **derecho** (no obligación) a comprar o vender un activo a un precio determinado (strike). El vendedor tiene la **obligación** si el comprador ejerce.

- **Call**: derecho a comprar
- **Put**: derecho a vender
- El comprador paga una **prima** por ese derecho (como un seguro)

### Las 4 posiciones básicas

| Posición | Visión | Riesgo máximo | Ganancia máxima |
|---|---|---|---|
| Comprar Call | Alcista | La prima pagada | Ilimitada |
| Vender Call | Bajista/Neutral | Ilimitada | La prima cobrada |
| Comprar Put | Bajista | La prima pagada | Strike - Prima |
| Vender Put | Alcista/Neutral | Strike - Prima | La prima cobrada |

### Por qué las opciones son difíciles de algoritmizar

Las opciones dependen de múltiples variables simultáneamente (las "griegas"):

- **Delta**: sensibilidad al precio del subyacente
- **Theta**: decaimiento temporal (la opción pierde valor cada día)
- **Vega**: sensibilidad a la volatilidad implícita
- **Gamma**: aceleración del delta

Además, cada subyacente tiene decenas de strikes y vencimientos activos simultáneamente. Esto multiplica la complejidad de datos, backtesting y ejecución. Si estás empezando en trading algorítmico, las opciones **no** son el mejor punto de entrada.

### Opciones americanas vs europeas

- **Americanas**: se pueden ejercer en cualquier momento → más flexibilidad pero más variables para modelar
- **Europeas**: solo al vencimiento → más simples de modelar algorítmicamente

## CFDs (Contratos por Diferencia)

Producto derivado puramente especulativo. Comprás/vendés la diferencia de precio sin poseer el activo. **No cotizan en mercados regulados** — son productos OTC negociados directamente con tu broker.

### Lo que importa para tu sistema

- **Tu broker es tu contraparte**: él "crea" el mercado, lo que genera un conflicto de interés estructural
- **No hay precio único**: cada broker puede tener spreads y precios distintos para el mismo CFD
- **Swaps overnight**: mantener posiciones abiertas de un día al otro tiene costo — esto destruye estrategias de swing trading lentas
- **Sin vencimiento**: a diferencia de futuros, no expiran
- **Spread variable**: en momentos de volatilidad alta, el broker puede ampliar el spread significativamente o hasta cerrar posiciones

### Representación gráfica

Los CFDs (y forex) típicamente se negocian y grafican a partir del **bid o ask** (o un mid-price calculado), no del last como en exchanges centralizados. Esto significa que:

```
# Un backtest de CFD que usa datos de "close" sin distinguir bid/ask
# está ignorando el spread real del broker.
#
# Si tu estrategia tiene un profit promedio de 5 pips y el spread
# es de 2 pips, el spread se come el 40% de tu ganancia.
# En backtest no lo ves. En real, sí.
```

### Cuándo tiene sentido usar CFDs

Con cuentas muy chicas (< $5,000) donde no podés acceder a futuros por margen, los CFDs dan acceso a mercados con tamaños de posición pequeños. Pero si podés operar futuros, preferí futuros: están regulados, tienen cámara de compensación, y el precio es transparente.

## Forex (Mercado de Divisas)

Mercado descentralizado donde se negocian pares de divisas. Es el mercado más grande del mundo (~USD 6-7 trillones diarios de volumen) y opera 24 horas en días laborables.

### Lo que importa para tu sistema

- **Pares**: siempre se tradea una divisa contra otra (EUR/USD, GBP/JPY). Si EUR/USD sube, el euro se fortalece vs. el dólar
- **OTC como los CFDs**: no hay exchange central, cada proveedor de liquidez puede tener precio distinto
- **Sesiones**: se mueve por zonas horarias — Tokio → Londres → Nueva York. La liquidez y volatilidad cambian según la sesión
- **Representación por bid/ask**: igual que CFDs, tu backtest debe contemplar el spread real
- **Sin vencimiento**: similar a CFDs, las posiciones se mantienen indefinidamente (con swaps)

### Ventaja para sistemas algorítmicos

Forex es uno de los mercados más amigables para algoritmos por su liquidez masiva, operación continua, y volatilidad predecible por sesión. Muchas firmas quant operan forex como su primer mercado.

## Tabla Comparativa para Decidir

| Criterio | Acciones | Futuros | Opciones | CFDs | Forex |
|---|---|---|---|---|---|
| **Capital mínimo práctico** | $500+ | $5,000+ | $2,000+ | $200+ | $200+ |
| **Regulación** | Alta | Alta | Alta | Baja | Baja |
| **Complejidad algorítmica** | Media | Media | Alta | Baja | Media |
| **Datos para backtest** | Fácil | Media | Difícil | Difícil | Media |
| **Datos por** | Last | Last | Last | Bid/Ask | Bid/Ask |
| **Vencimiento** | No | Sí | Sí | No | No |
| **Short selling** | Limitado | Nativo | Vía puts | Nativo | Nativo |
| **Mejor para empezar algo** | Sí | Con capital | No | Con cuenta chica | Sí |

## Implicaciones para tu Sistema

### Si estás empezando

Arrancá con **acciones US** o **forex majors**. Datos abundantes, brokers accesibles, y la complejidad es manejable. Construí tu primer sistema funcional antes de agregar instrumentos más complejos.

### Si ya tenés un sistema rentable

Diversificar por instrumento es tan importante como diversificar por estrategia. Un portfolio con sistemas en acciones + futuros + forex tiene correlaciones más bajas que uno solo en acciones. La descorrelación entre instrumentos es el verdadero "santo grial" de la gestión de portfolio.

### Errores comunes

1. **Backtestear CFDs con datos de exchange**: los precios no son los mismos — tu broker tiene su propio feed
2. **Ignorar el costo de rollover en futuros**: un sistema que rola 12 veces al año tiene 12 eventos de slippage extra
3. **Asumir que podés shortear cualquier acción**: en small caps, los locates son caros o inexistentes
4. **Usar el mismo framework de backtesting para opciones que para acciones**: las opciones necesitan modelar decaimiento temporal, volatilidad implícita, y múltiples strikes simultáneos
