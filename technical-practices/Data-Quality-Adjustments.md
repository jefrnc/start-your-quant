# Calidad de Datos y Ajustes para Backtesting

Los datos son tu materia prima. Si tu base de datos no representa fielmente lo que pasó en el mercado, tu backtest no vale nada — por más sofisticado que sea tu algoritmo. Este es probablemente el tema más subestimado en trading algorítmico.

## El Problema del Rolo en Futuros

Los futuros vencen. El Mini S&P 500 vence trimestralmente (tercer viernes de marzo, junio, septiembre, diciembre). Para construir un gráfico histórico mayor a un trimestre, necesitás **unir vencimientos** en lo que se llama un gráfico continuo.

El problema: entre vencimientos casi siempre hay un **gap de precio**. Este gap no es producto de oferta y demanda real — es un artefacto del enlace entre dos contratos diferentes. Si no lo tratás correctamente, ese gap contamina tus indicadores y distorsiona tu backtest.

### De dónde sale el gap

Cuando un futuro empieza a cotizar (ej: septiembre), cotiza **por encima** del contrato que está por vencer (ej: junio). La diferencia se explica por:

- **Tipos de interés**: el valor temporal del dinero. Dinero hoy vale más que dinero en 3 meses
- **Dividendos**: en futuros de índices, los dividendos que pagarán las acciones entre hoy y el vencimiento
- **Costo de almacenamiento**: en materias primas (petróleo, granos), el costo de guardar el activo físico

En índices americanos con tipos de interés altos, esta diferencia puede ser de 40-45 puntos. En bonos (ej: Bund), los gaps son enormes.

### Cuándo rolar

El momento óptimo para cambiar de contrato en tu gráfico continuo es **cuando el nuevo vencimiento tiene más volumen que el antiguo**:

```python
def detect_roll_date(front_volume, next_volume):
    """
    Detecta el día en que el próximo vencimiento supera al actual en volumen.
    Ese es el día óptimo para el enlace en el gráfico continuo.
    """
    for date in front_volume.index:
        if date in next_volume.index:
            if next_volume[date] > front_volume[date]:
                return date
    return None

# IMPORTANTE: el timing varía por mercado
# - Índices US (S&P, Nasdaq, Dow): 5-7 días antes del vencimiento
# - DAX y futuros europeos: varía, puede ser pocos días antes o incluso cerca del vencimiento
# - Bonos: varía según el mercado
# Conocer las fechas de cada futuro que operás es tu responsabilidad
```

**Nota clave**: el momento del rolo operativo (cerrar una posición y abrir otra en el nuevo contrato) NO tiene que coincidir exactamente con el momento del enlace en el gráfico continuo. Son dos cosas distintas.

## Tres Métodos de Ajuste

### 1. Sin ajustar

Simplemente enlazás los contratos tal como cotizan. El gap queda ahí.

**Ventaja**: los precios históricos son reales — lo que cotizó, cotizó.

**Problema**: el gap artificial contamina cualquier indicador que use datos de más de un día. Y no es solo el día del rolo — una EMA de 50 períodos puede estar afectada durante semanas porque su fórmula recursiva arrastra el error.

```python
# Indicadores especialmente sensibles a gaps sin ajustar:
# - EMA (media exponencial): la fórmula recursiva propaga el error
# - ATR: usa el true range que incluye el gap entre cierres
# - ADX: derivado del ATR
# - RSI: usa cambios de precio que incluyen el gap falso
# - Estocástico: compara precio actual con rango, alterado por el gap
#
# Pueden estar alterados por MÁS barras que su período de cálculo
# debido a las fórmulas recursivas que usan datos anteriores.
```

### 2. Ajuste por valor absoluto (puntos)

Restás (o sumás) la diferencia del gap a todo el histórico previo.

```python
def adjust_absolute(data, roll_gaps):
    """
    Ajuste backward por valor absoluto.
    Mantiene el contrato actual al precio real y modifica el pasado.
    """
    adjusted = data.copy()
    cumulative_adjustment = 0

    # Procesamos de más reciente a más antiguo
    for roll_date, gap in sorted(roll_gaps.items(), reverse=True):
        cumulative_adjustment += gap
        mask = adjusted.index < roll_date
        for col in ['open', 'high', 'low', 'close']:
            adjusted.loc[mask, col] -= cumulative_adjustment

    return adjusted
```

**Ventaja**: mantiene el tick mínimo del activo. Si el Mini S&P se mueve de 0.25 en 0.25, los precios ajustados también lo hacen.

**Desventaja**: en históricos largos, la relación proporcional de precios se distorsiona. 40 puntos de ajuste no significan lo mismo cuando el S&P estaba a 2,000 que cuando está a 5,000.

### 3. Ajuste por ratio (porcentaje) — Recomendado

Ajustás proporcionalmente, preservando la relación porcentual de los precios.

```python
def adjust_ratio(data, roll_dates_and_prices):
    """
    Ajuste backward por ratio.
    Preserva relaciones porcentuales — el método más correcto
    para históricos largos.
    """
    adjusted = data.copy()
    cumulative_ratio = 1.0

    for roll_date, old_close, new_close in sorted(
        roll_dates_and_prices, reverse=True
    ):
        ratio = new_close / old_close
        cumulative_ratio *= ratio
        mask = adjusted.index < roll_date
        for col in ['open', 'high', 'low', 'close']:
            adjusted.loc[mask, col] /= cumulative_ratio

    return adjusted
```

**Ventaja**: un movimiento del 1% en 2003 se representa igual que un 1% en 2024. Esto es correcto porque los mercados se mueven en porcentajes, no en puntos absolutos.

**Desventaja**: los precios pierden el tick mínimo (aparecen decimales que no existen en el mercado real). Solución: redondear las órdenes al tick real antes de enviarlas.

### Cuál usar

| Situación | Recomendación |
|---|---|
| Backtest con indicadores en porcentaje (ATR%, ROC) | Ratio |
| Backtest con indicadores en puntos absolutos | Valor absoluto funciona |
| Histórico corto (< 2 años) | Cualquiera, la diferencia es mínima |
| Histórico largo (> 5 años) | Ratio, sin duda |
| Quiero ver precios históricos reales | Sin ajustar (solo para visualización, no para backtest) |

**La combinación ideal**: ajustar por ratio y trabajar con indicadores en porcentaje.

```python
# EN VEZ DE:
atr = calculate_atr(data, 14)  # ATR en puntos — inconsistente en el tiempo

# USAR:
atr_pct = calculate_atr(data, 14) / data['close'] * 100  # ATR en % — consistente
```

## Dividendos en Acciones: El Mismo Problema

Cuando una acción paga dividendo, el precio se descuenta automáticamente por el monto del dividendo. Tu posición patrimonial no cambia (tenés la acción que vale menos + el efectivo del dividendo), pero tu gráfico muestra una caída que no fue producto de oferta y demanda.

```
Ejemplo: acción cotiza a $100, paga dividendo de $2
- Antes: 1 acción × $100 = $100
- Después: 1 acción × $98 + $2 efectivo = $100
- El gráfico muestra una caída de 2% que NO es pérdida real
```

Para backtesting, es mejor **ajustar por dividendos** (backward, por ratio) para que tus indicadores y señales no se contaminen con caídas artificiales.

### Índices Total Return

Los índices "Total Return" (en España les dicen "con dividendos") ya incorporan este ajuste. Reinvierten los dividendos en el índice, mostrando el rendimiento real de un inversor que reinvierte.

La diferencia puede ser enorme: el IBEX 35 normal puede parecer lejos de sus máximos históricos nominales, mientras que el IBEX con dividendos puede haberlos superado ampliamente — la brecha se acumula año tras año. En acciones europeas que pagan dividendos altos, el efecto es muy pronunciado. En el Nasdaq, donde los dividendos son menores, el efecto existe pero es menos dramático.

**Para backtesting de estrategias sobre índices**: usá el Total Return o el ETF ajustado por dividendos (como SPY ajustado backward). Son la representación más fiel del rendimiento real.

## Errores Frecuentes con Datos

### 1. Asumir que todos los datos son iguales

Diferentes proveedores pueden tener datos distintos para el mismo activo. Comparar bases de datos antes de confiar ciegamente en una.

Incluso proveedores de referencia tienen problemas con datos muy antiguos. Si tu proveedor tiene datos del oro desde 1975, verificá la calidad de esos primeros años — la tecnología de captura de datos era rudimentaria.

### 2. No verificar el delay en datos de tiempo real

Diferentes fuentes de datos en tiempo real pueden no estar sincronizadas. Si tu señal usa datos de un feed y tu orden va a otro, un desfase de segundos puede importar.

### 3. Usar datos de backtest y tiempo real de calidades diferentes

Si tu backtest usa datos limpios de alta calidad pero tu ejecución en vivo usa un feed inferior, los resultados van a divergir. Las propiedades de ambos datasets deben ser análogas.

### 4. Ignorar el survivorship bias en acciones

Si backtestás una estrategia rotacional del S&P 500 desde 2010, necesitás la composición del índice **en cada momento histórico**, incluyendo las empresas que fueron removidas. Las que sobrevivieron hasta hoy tienen un sesgo positivo inherente.

## Checklist Antes de Backtestear

- [ ] ¿Mis datos de futuros están ajustados? ¿Por ratio o por valor absoluto?
- [ ] ¿Conozco las fechas de vencimiento y el patrón de rolo de cada futuro?
- [ ] ¿Mis datos de acciones están ajustados por dividendos y splits?
- [ ] ¿La calidad de mis datos históricos es comparable a la de mis datos en tiempo real?
- [ ] ¿He verificado la calidad del proveedor, especialmente en datos antiguos?
- [ ] ¿Mis indicadores usan porcentajes en vez de valores absolutos cuando es posible?
- [ ] ¿Estoy consciente del survivorship bias si uso universos de acciones?
