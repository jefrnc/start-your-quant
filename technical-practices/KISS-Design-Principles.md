# KISS: Principios de Diseño de Sistemas

Keep It Simple, Stupid. En trading algorítmico, la simplicidad no es una limitación — es tu principal defensa contra el overfitting.

## Por Qué Simple Gana

La simplicidad fomenta la robustez. Y la robustez es lo más importante de todo en un sistema, porque no se puede garantizar — solo se puede fomentar durante el diseño.

Un sistema robusto mantiene su edge en datos que nunca vio. Un sistema complejo se ajusta perfectamente al pasado pero falla en el futuro. La diferencia entre ambos, casi siempre, es la cantidad de reglas.

**Regla práctica**: 1-2 reglas de entrada, 1 filtro como máximo, múltiples salidas. Si necesitás más de eso para que funcione, probablemente estés capturando ruido, no señal.

### Cuántas Reglas Son Demasiadas

Cada regla, cada filtro, cada parámetro optimizable que agregás le da al sistema más flexibilidad para ajustarse a los datos históricos. Más complejidad = más riesgo de overfitting.

```
1-2 reglas + 1 filtro:  robusto, difícil de ajustar al ruido
3-4 reglas + 2 filtros: zona gris, hay que validar muy bien
5+ reglas + 3 filtros:  casi seguro overfitting, por más que el backtest sea perfecto
```

Si no podés explicar tu sistema en 2 minutos a alguien que no sabe de trading, probablemente sea demasiado complejo.

## Del Papel al Código: Pseudocódigo

Antes de programar, escribí el sistema en pseudocódigo. No es opcional — es el paso que previene errores lógicos y te fuerza a entender exactamente qué hace cada parte.

**Pseudocódigo**: mezcla entre lenguaje humano y lenguaje de programación. No es ejecutable, pero es suficientemente preciso para traducirlo a cualquier lenguaje.

```
PSEUDOCÓDIGO del sistema Aberration (Fitschen, 1986):

Variables:
  - media = media simple de N períodos
  - banda_superior = media + 2 × desviación_estándar
  - banda_inferior = media - 2 × desviación_estándar

Entrada largo:
  SI cierre > banda_superior
    COMPRAR en apertura siguiente barra

Salida largo:
  SI estoy largo Y cierre < media
    CERRAR largo en apertura siguiente barra

Entrada corto:
  SI cierre < banda_inferior
    VENDER en apertura siguiente barra

Salida corto:
  SI estoy corto Y cierre > media
    CERRAR corto en apertura siguiente barra
```

**Ventajas del pseudocódigo:**
- Detectás errores lógicos antes de programar
- Se traduce fácilmente a cualquier lenguaje (Python, EasyLanguage, MQL, NinjaScript)
- Sirve como documentación del sistema
- Si otra persona lo lee, puede verificar la lógica

## Largos y Cortos: Separar Cuando se Pueda

La renta variable históricamente muestra asimetría: tiende a subir de forma gradual y a caer de forma rápida y con mayor volatilidad. La volatilidad es mayor en caídas. Esto significa que los parámetros óptimos para el lado largo probablemente sean distintos a los del lado corto.

**Si optimizás largo y corto juntos**, el optimizador busca un compromiso que no es óptimo para ninguno de los dos. Normalmente se sesga hacia el lado largo (hay más datos largos en mercados alcistas).

**Si los separás**, cada lado tiene sus propios parámetros optimizados. La suma suele ser mejor que el conjunto.

**Pero**: separar divide la muestra a la mitad. Con pocos trades, eso reduce la significancia estadística y aumenta el riesgo de overfitting.

| Situación | Recomendación |
|---|---|
| Sistema intradía con muchos trades | Separar largos y cortos |
| Sistema diario con buen histórico (10+ años) | Separar si hay suficientes trades por lado |
| Sistema semanal/mensual | Operar conjunto — no hay suficiente muestra para separar |
| No operar cortos en renta variable | Totalmente válido. Muchos fondos exitosos son solo largo |

## ATR Normalizado: Comparar Manzanas con Manzanas

El ATR estándar mide volatilidad en puntos. Pero 100 puntos con el Nasdaq a 5,000 representan un 2%, mientras que 100 puntos con el Nasdaq a 18,000 son apenas un 0.55%. El ATR por sí solo no permite comparar volatilidad relativa entre activos con precios distintos ni a lo largo del tiempo si el precio cambió mucho.

**Solución**: normalizar el ATR dividiéndolo por el precio.

```python
def atr_normalized(atr, high, low, close):
    """
    ATR como porcentaje del precio.
    Permite comparar volatilidad entre activos y a lo largo del tiempo.
    """
    typical_price = (high + low + close) / 3
    return (atr / typical_price) * 100

# Comparación (diario, ~20 años):
# Nasdaq 100:  ATR% ~1.65%  — alta volatilidad
# S&P 500:     ATR% ~1.24%  — moderada
# Oro (GLD):   ATR% ~0.97%  — baja
# Café:        ATR% ~2.24%  — muy alta
# Petróleo:    ATR% ~3.08%  — extrema
```

**Usos del ATR normalizado:**
- Comparar volatilidad entre activos para elegir dónde operar
- Ajustar la exposición: reducir contratos cuando la volatilidad sube, aumentar cuando baja (con límites)
- Dimensionar stops y TPs que se adapten al régimen de volatilidad actual

## Elegir Lenguaje de Programación

| Lenguaje | Plataforma | Nivel | Ideal para |
|---|---|---|---|
| **EasyLanguage** | TradeStation, MultiCharts | Muy alto (casi pseudocódigo) | Principiantes, prototipado rápido |
| **Python** | Independiente | Alto | Flexibilidad, ML, análisis de datos |
| **NinjaScript** | NinjaTrader | Alto | Usuarios de NinjaTrader |
| **MQL4/5** | MetaTrader | Medio-alto | Forex, códigos más largos |

**El dilema**: elegir un lenguaje condiciona plataforma, broker y datos.

- **EasyLanguage/TradeStation**: todo-en-uno (plataforma + datos + broker + lenguaje). Ideal para empezar sin complicaciones de conexión
- **Python**: máxima flexibilidad pero hay que resolver conexiones a datos, broker y ejecución por separado

No existe el lenguaje "mejor". Existe el que mejor se adapta a tu perfil y experiencia. Si ya sabés Python, usá Python. Si empezás de cero, EasyLanguage tiene la curva de aprendizaje más corta.

Lo que importa es que entiendas el código que operás — sea tuyo o de terceros. Si no podés explicar cada línea, no lo operes con dinero real.
