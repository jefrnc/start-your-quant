# Crecimiento Compuesto y Los 10 Riesgos Reales del Trading

## El Motor del Crecimiento: Interés Compuesto en Trading

El concepto más poderoso en finanzas no es ningún indicador — es el interés compuesto. En trading, equivale a **reinvertir las ganancias en tu sistema** para que el tamaño de tus posiciones crezca con tu cuenta.

### Simple vs Compuesto: La Diferencia Real

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_growth(initial_capital, monthly_return, months):
    """Compara crecimiento simple vs compuesto."""
    simple = np.zeros(months + 1)
    compound = np.zeros(months + 1)
    simple[0] = compound[0] = initial_capital

    monthly_gain = initial_capital * monthly_return

    for m in range(1, months + 1):
        simple[m] = simple[m-1] + monthly_gain              # siempre gana lo mismo
        compound[m] = compound[m-1] * (1 + monthly_return)   # gana sobre lo acumulado

    return simple, compound

capital = 10_000
tasa = 0.03  # 3% mensual
meses = 60   # 5 años

simple, compuesto = simulate_growth(capital, tasa, meses)

print(f"Capital inicial: ${capital:,.0f}")
print(f"Después de {meses} meses al {tasa*100}% mensual:")
print(f"  Simple:    ${simple[-1]:,.0f} (+{(simple[-1]/capital - 1)*100:.0f}%)")
print(f"  Compuesto: ${compuesto[-1]:,.0f} (+{(compuesto[-1]/capital - 1)*100:.0f}%)")
# Simple:    $28,000 (+180%)
# Compuesto: $58,916 (+489%)
```

La diferencia se amplifica con el tiempo. En 5 años al 3% mensual, el compuesto genera **casi 3 veces más** que el simple. Esto es por qué los primeros meses "no se siente" el progreso, pero después de un año la curva se separa dramáticamente.

### Aplicación Práctica: Position Sizing que Crece con tu Cuenta

```python
def dynamic_position_size(account_balance, risk_per_trade_pct, entry_price, stop_price):
    """
    El position sizing se recalcula con cada trade basándose
    en el balance actual — no en el balance inicial.
    Esto ES interés compuesto aplicado a trading.
    """
    risk_amount = account_balance * risk_per_trade_pct
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share == 0:
        return 0

    shares = int(risk_amount / risk_per_share)
    return shares

# Mes 1: cuenta de $10,000
shares_m1 = dynamic_position_size(10_000, 0.01, 5.00, 4.80)
# Mes 12: cuenta creció a $14,000
shares_m12 = dynamic_position_size(14_000, 0.01, 5.00, 4.80)

print(f"Mes 1:  {shares_m1} acciones (riesgo ${10_000 * 0.01:.0f})")
print(f"Mes 12: {shares_m12} acciones (riesgo ${14_000 * 0.01:.0f})")
# Mismo riesgo porcentual, más capital trabajando.
```

### La Trampa: El Compuesto También Funciona en Contra

El interés compuesto amplifica pérdidas igual que ganancias. Si perdés 20% de tu cuenta, necesitás ganar 25% para volver al punto de partida. Si perdés 50%, necesitás ganar 100%.

```python
def recovery_needed(drawdown_pct):
    """Cuánto necesitás ganar para recuperar un drawdown."""
    return (1 / (1 - drawdown_pct) - 1) * 100

for dd in [0.10, 0.20, 0.30, 0.50, 0.70]:
    recovery = recovery_needed(dd)
    print(f"Pérdida {dd*100:.0f}% → necesitás ganar {recovery:.0f}% para recuperar")

# Pérdida 10% → necesitás ganar 11%
# Pérdida 20% → necesitás ganar 25%
# Pérdida 30% → necesitás ganar 43%
# Pérdida 50% → necesitás ganar 100%
# Pérdida 70% → necesitás ganar 233%
```

**Conclusión**: proteger el capital no es conservadurismo — es matemática. Un drawdown del 50% te pone en una posición donde necesitás duplicar tu cuenta solo para volver a cero.

## Los 10 Riesgos Reales del Trading Algorítmico

La mayoría de traders solo piensa en el riesgo de mercado. Pero hay al menos 10 tipos de riesgo, y los menos obvios son los que más daño hacen.

### 1. Riesgo de Mercado

El más evidente: el precio se mueve en contra. Gaps overnight, flash crashes, eventos macro.

**Mitigación**: stop losses, position sizing, diversificación temporal (no todo el capital al mismo tiempo).

### 2. Riesgo de Diseño

Tu algoritmo tiene un bug o una lógica defectuosa. Backtestea bien pero por razones equivocadas (lookahead bias, overfitting, survivorship bias).

**Mitigación**: walk-forward analysis, out-of-sample testing, revisión de código por pares, paper trading antes de capital real.

### 3. Riesgo de Liquidez

No hay suficiente volumen para entrar o salir al precio que querés. El slippage se come tu edge. Especialmente relevante en small caps.

**Mitigación**: filtrar por volumen mínimo, limitar el tamaño de posición como porcentaje del volumen diario, usar limit orders en vez de market orders.

```python
def max_position_by_liquidity(avg_daily_volume, max_pct_of_volume=0.01):
    """
    Nunca operar más del 1% del volumen diario promedio.
    En small caps, incluso 1% puede mover el precio.
    """
    return int(avg_daily_volume * max_pct_of_volume)

# Acción con 500k de volumen diario → máximo 5,000 acciones
max_shares = max_position_by_liquidity(500_000)
print(f"Posición máxima por liquidez: {max_shares:,} acciones")
```

### 4. Riesgo de Rotura

Tu sistema deja de funcionar. El mercado cambió de régimen y el edge que explotaba ya no existe. Todo sistema tiene vida útil.

**Mitigación**: monitorear métricas rolling (Sharpe de 30/60/90 días), definir condiciones de desactivación antes de lanzar el sistema, no depender de un solo sistema.

### 5. Riesgo Operativo

Fallos técnicos: se corta internet en medio de una posición abierta, el servidor se cae, la API del broker no responde, un deploy sale mal.

**Mitigación**: UPS para electricidad, conexión de backup, alertas de monitoreo, capacidad de cerrar posiciones desde el celular, stops en el servidor del broker (no solo locales).

### 6. Riesgo de Crédito

No poder cumplir con las obligaciones financieras frente al broker — por ejemplo, un margin call que no podés cubrir porque las pérdidas superaron tu capital disponible. Distinto del riesgo de mercado (que el precio se mueva en contra): el riesgo de crédito es que no tengas los fondos para responder.

**Mitigación**: mantener margen holgado, evitar posiciones overnight en instrumentos con alto riesgo de gap, y dimensionar posiciones para que el peor escenario razonable no comprometa la cuenta.

### 7. Riesgo de Contraparte

Que tu broker no pueda responder. Quiebra del broker, bloqueo de fondos, incapacidad de ejecutar órdenes en un crash.

**Mitigación**: usar brokers regulados (SIPC en US), no tener todo el capital en un solo broker, verificar la solvencia y regulación del broker.

### 8. Riesgo Regulatorio

Cambios en regulación que afectan tu operativa. Nuevas reglas de margen, impuestos, restricciones a short selling (como la prohibición temporal en 2008 y 2020), cambios en la PDT rule.

**Mitigación**: mantenerse informado, diseñar sistemas que no dependan de una sola mecánica regulatoria.

### 9. Riesgo Legal

Demandas, problemas de propiedad intelectual si usás código de terceros, violaciones involuntarias de regulación.

**Mitigación**: entender las reglas de tu jurisdicción, tener licencias adecuadas si gestionás capital de terceros.

### 10. Riesgo Reputacional

Relevante si gestionás capital externo o publicás resultados. Un drawdown público puede destruir tu capacidad de levantar capital futuro.

**Mitigación**: ser transparente con los riesgos, no prometer retornos, documentar tu track record incluyendo los malos períodos.

### Riesgo por Tipo de Operativa

No todos los riesgos pesan igual según cómo operés:

| Riesgo | Intradiario | Swing/Diario |
|---|---|---|
| Mercado | Menor (sin overnight) | Mayor (gaps) |
| Liquidez | Mayor (necesitás entrar/salir rápido) | Menor |
| Rotura | Mayor (más sensible a ruido) | Menor (señales más claras) |
| Operativo | Mayor (dependés del uptime constante) | Menor |
| Contraparte | Mayor (más interacción con broker) | Menor |
| Crédito | Menor (posiciones cortas) | Mayor (margin calls overnight) |

## El Framework Completo: Crecimiento Sostenible

Juntando todo — el crecimiento compuesto es tu motor, pero los riesgos son los frenos. Un sistema rentable que ignora los riesgos eventualmente explota. Un sistema ultra-conservador que ignora el compounding nunca crece.

El balance:

Para implementar esto como parte de tu operativa, ver [Plan de Trading](./Trading-Plan-Framework.md).

1. **Expectancy positiva**: tu sistema debe ganar más de lo que pierde en promedio
2. **Position sizing dinámico**: que crezca con tu cuenta (compounding)
3. **Protección de capital**: drawdown máximo definido donde se reduce tamaño o se para
4. **Diversificación**: múltiples sistemas, múltiples instrumentos, múltiples timeframes
5. **Monitoreo continuo**: métricas rolling que detecten degradación antes de que sea catastrófica

```python
def should_reduce_risk(rolling_sharpe_30d, rolling_sharpe_90d, max_drawdown_current):
    """
    Framework simple de control: si las métricas se degradan,
    reducir exposición antes de que el drawdown sea irrecuperable.
    """
    if max_drawdown_current > 0.15:  # drawdown > 15%
        return "STOP — pausar sistema, revisar"
    if rolling_sharpe_30d < 0 and rolling_sharpe_90d > 0:
        return "REDUCIR — bajar tamaño de posición al 50%"
    if rolling_sharpe_30d < 0 and rolling_sharpe_90d < 0:
        return "STOP — el edge puede haber desaparecido"
    return "NORMAL — operar a tamaño completo"
```

Controlar el riesgo no es opcional — es lo que te mantiene en el juego el tiempo suficiente para que el compounding haga su magia.
