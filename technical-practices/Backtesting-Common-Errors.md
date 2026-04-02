# Errores Comunes en Backtesting

Tu backtest solo es útil si es reproducible en real. Para los tres niveles de validación (clásico, forward, walk-forward), ver [Backtesting: Del Clásico al Walk-Forward](./Backtesting-Three-Levels.md). Para problemas específicos de datos, ver [Calidad de Datos y Ajustes](./Data-Quality-Adjustments.md). Un sistema que parece espectacular en el histórico pero tiene un error de configuración, un activo no operable, o una lectura a futuro, no vale nada. Estos son los errores más frecuentes y cómo detectarlos.

## 1. Operar un Activo No Operable

Los índices (VIX, S&P 500, IBEX 35, Nasdaq) son activos sintéticos — no se pueden comprar ni vender directamente. Para invertir necesitás un derivado: futuro, ETF, opción o CFD.

**El caso del VIX**: un sistema de reversión a la media en el índice VIX puede verse espectacular (profit factor 5+, curva casi perfecta). Pero el índice VIX no es operable. Cuando pasás el mismo sistema al futuro del VIX, puede dejar de funcionar o requerir ajustes significativos.

**Por qué**: el futuro del VIX suele estar en **contango** fuerte — cada vencimiento sucesivo cotiza más caro que el actual (típicamente entre 2% y 8% entre meses, dependiendo de las condiciones de mercado). Esto crea una caída artificial persistente en el gráfico continuo que destruye la ciclicidad limpia que ves en el índice.

**Regla**: antes de backtestear, verificá que el activo sea operable. Si es un índice, usá el futuro o ETF correspondiente.

### Contango y Backwardation

- **Contango** (lo habitual): vencimientos futuros más caros que el spot. Refleja el costo del tiempo: tipos de interés, almacenamiento (en commodities), incertidumbre
- **Backwardation**: vencimientos futuros más baratos que el spot. En commodities indica escasez actual o demanda inmediata fuerte. En el VIX, pasar a backwardation indica pánico de mercado (alta demanda de protección inmediata)

El contango afecta a todos los futuros, pero en la mayoría es pequeño (~1% entre trimestres en índices de bolsa). En el VIX es extremo y destruye estrategias que funcionan en el índice puro.

## 2. Look-Ahead Bias (Lectura a Futuro)

Usar información que no estaría disponible en el momento de la decisión.

**Ejemplos comunes:**

```python
# MAL: comprar en la apertura de hoy basándote en el cierre de hoy
# Al cierre ya no podés comprar en la apertura — ya pasó
if close_today > sma200_today:
    buy_at(open_today)  # IMPOSIBLE — open_today ya pasó cuando tenés close_today

# BIEN: señal al cierre de hoy → comprar en la apertura de mañana
if close_today > sma200_today:
    buy_at(open_tomorrow)  # CORRECTO — decisión hoy, ejecución mañana
```

**Datos externos**: los COT (Commitment of Traders) se publican los viernes pero están fechados el martes anterior. Si tu sistema usa la fecha del dato (martes) en vez de la fecha de publicación (viernes), estás leyendo a futuro.

Lo mismo con resultados empresariales, datos macro (PIB, empleo), o cualquier dato que tenga una fecha de referencia distinta a la fecha de disponibilidad.

**Regla**: toda información que uses debe estar 100% disponible en el momento en que se evalúa. Si hay duda, usá la fecha de publicación, no la fecha del dato.

## 3. Look Inside Bar (Orden de Ejecución dentro de la Vela)

De una vela histórica solo conocés 4 datos: open, high, low, close. No sabés en qué orden se movió el precio internamente.

**El problema**: si tenés un stop en 100 y un take profit en 105, y la vela tiene low=99 y high=106, ¿cuál saltó primero? Si bajó primero → stop. Si subió primero → TP. El resultado es opuesto según el orden.

```
Escenario A: baja → sube    →  stop loss salta primero  →  pérdida
Escenario B: sube → baja    →  take profit salta primero →  ganancia
```

El motor de backtest **deduce** el orden, pero puede equivocarse.

**Solución**: activar Look Inside Bar (TradeStation) o Bar Magnifier (MultiCharts), que carga un timeframe inferior (1 minuto) para simular cómo se formó cada vela.

**Trampa**: incluso con 1 minuto, si tus stops son muy cercanos, pueden saltar dentro de una vela de 1 minuto y el problema se repite. Verificá que no haya órdenes que salten en la misma vela del timeframe inferior.

**Caso real**: un sistema con trailing stop muy ajustado mostraba una curva espectacular. Al activar Look Inside Bar, se convirtió en pérdidas. El trailing saltaba dentro de la vela antes de que el precio llegara al TP.

## 4. Normas del Mercado que No Conocés

Cada mercado tiene reglas específicas que pueden hacer que tu sistema sea inoperable.

| Regla | Ejemplo |
|---|---|
| **Tipos de órdenes restringidos** | El VIX no acepta market orders ni stops en horario extendido — solo limit |
| **Horarios de trading** | El Globex no acepta stops en premarket |
| **Circuit breakers** | Acciones US se pausan en caídas del 7%, 14%, 20% |
| **Position limits** | Cada futuro tiene un máximo de contratos permitido |
| **Liquidez por horario** | Un activo puede tener spreads de 1 tick a las 10 AM y de 20 ticks a las 3 AM |
| **Short selling** | Algunas acciones son Hard-To-Borrow (HTB) — no podés shortearlas o es caro |

**Dónde encontrar esta info**: páginas de los mercados (CME, CBOE, NYSE). Buscar "contract specifications" para futuros. El CME tiene cursos gratuitos excelentes.

**Regla**: antes de operar cualquier activo nuevo, leé las especificaciones del contrato. Es una vez — después ya lo sabés.

## 5. Órdenes Limit en Backtest

Las órdenes limit tienen un problema que los stops y market orders no tienen: **puede que no ejecuten aunque el precio toque tu nivel**.

En real, cuando ponés una limit a 100.00 para comprar, hay una cola de órdenes. Si hay 500 órdenes delante y solo se ejecutan 400, te quedás afuera. Pero el backtest marca la operación como ejecutada porque el precio tocó 100.00.

**El sesgo**: los trades que "no ejecutan" casi siempre hubieran sido ganadores (el precio tocó tu nivel y rebotó). Esto infla el backtest.

**Solución conservadora**: configurar el backtest para que ejecute limits solo cuando el precio **excede** tu nivel (un tick más allá), no cuando lo toca. Es más pesimista pero más realista.

## 6. Slippage y Comisiones

El slippage es la diferencia entre el precio teórico del sistema y el precio real de ejecución. Las comisiones son lo que cobra el broker.

**Regla general**: 1-2 ticks de slippage por operación en condiciones normales. Más en:
- Momentos de alta volatilidad
- Activos poco líquidos
- Operativa en noticias o rupturas (muchos stops saltando al mismo tiempo)

Si tu sistema tiene un profit promedio de 3 ticks por trade y el slippage + comisiones suman 2 ticks, te queda 1 tick — cualquier variación te pone en pérdidas. Los sistemas viables necesitan margen holgado sobre los costos de transacción.

## Checklist de Evaluación Preliminar

Antes de pasar un sistema a optimización formal, verificá:

- [ ] El activo es operable (no es un índice puro)
- [ ] Conozco las especificaciones del contrato (horarios, órdenes permitidas, vencimientos, limits)
- [ ] No hay lectura a futuro en el código (toda info disponible al momento de la decisión)
- [ ] Los datos externos usan fecha de publicación, no fecha del dato
- [ ] Look Inside Bar activado si uso stops/TPs cercanos
- [ ] Limits configurados con ejecución en "excede" no en "toca"
- [ ] Slippage y comisiones incluidos (aunque sea estimación)
- [ ] He mirado el gráfico en distintos momentos: alta/baja volatilidad, tendencia, lateral, crash
- [ ] Las señales son coherentes visualmente con lo que el sistema debería hacer
- [ ] Todo es 100% reproducible en operativa real
