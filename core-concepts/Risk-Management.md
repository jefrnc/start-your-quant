# Gestión de Riesgo Sistemática

## Filosofía: Risk-First Design

En el trading cuantitativo, **la gestión de riesgo no es un add-on, es el fundamento**. Cada decisión de trading debe empezar con la pregunta: "¿Cuánto puedo permitirme perder en esta operación?"

### Principios Fundamentales

1. **Capital Preservation > Profit Maximization**
2. **Riesgo Predefinido**: Nunca entrar sin saber el exit
3. **Position Sizing Matemático**: Basado en probabilidades, no intuición
4. **Diversificación Temporal**: No todo el capital al mismo tiempo
5. **Circuit Breakers**: Límites automáticos para prevenir catástrofes

## Framework de Riesgo para Small Caps

### Parámetros Base del Sistema

```yaml
risk_parameters:
  # Riesgo por Trade
  max_risk_per_trade: 10.0        # $10 máximo por operación
  max_position_size: 70.0         # $70 máximo por posición

  # Riesgo Diario/Semanal
  max_daily_loss: 50.0            # $50 pérdida máxima diaria
  max_weekly_loss: 150.0          # $150 pérdida máxima semanal

  # Drawdown Limits
  max_account_drawdown: 0.15      # 15% drawdown máximo
  emergency_stop_drawdown: 0.25   # 25% emergency stop

  # Concentración
  max_positions_concurrent: 3     # Máximo 3 posiciones simultáneas
  max_exposure_per_symbol: 0.05   # 5% del capital por símbolo
```

## Position Sizing Metodologías

### 1. **Fixed Dollar Risk (Nuestro Método Principal)**

```python
def calculate_position_size_fixed_risk(
    entry_price: float,
    stop_loss_price: float,
    max_risk_dollars: float = 10.0
) -> int:
    """
    Calcula shares basado en riesgo fijo en dólares

    Ejemplo:
    - Entry: $5.00
    - Stop: $4.75
    - Risk: $10
    - Position: $10 / ($5.00 - $4.75) = 40 shares
    """
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share <= 0:
        raise ValueError("Stop loss debe ser diferente al precio de entrada")

    shares = int(max_risk_dollars / risk_per_share)

    # Verificar límite de posición máxima
    max_shares_by_position = int(70.0 / entry_price)

    return min(shares, max_shares_by_position)
```

### 2. **Percentage Risk Method**

```python
def calculate_position_size_percentage(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
    risk_percentage: float = 0.01  # 1% del account
) -> int:
    """
    Position sizing basado en % del account
    Útil para accounts más grandes
    """
    max_risk_dollars = account_balance * risk_percentage
    return calculate_position_size_fixed_risk(
        entry_price, stop_loss_price, max_risk_dollars
    )
```

### 3. **ATR-Based Position Sizing**

```python
def calculate_position_size_atr(
    entry_price: float,
    atr: float,
    atr_multiplier: float = 2.0,
    max_risk_dollars: float = 10.0
) -> int:
    """
    Position sizing basado en volatilidad (ATR)
    Stop loss = entry_price - (ATR * multiplier)
    """
    stop_loss_price = entry_price - (atr * atr_multiplier)

    return calculate_position_size_fixed_risk(
        entry_price, stop_loss_price, max_risk_dollars
    )
```

## Stop Loss Strategies

### 1. **Fixed Percentage Stop**
```python
# Ejemplo: 5% stop loss
stop_price = entry_price * 0.95  # Para long positions
```
**Pros**: Simple, predecible
**Cons**: No considera volatilidad del instrumento

### 2. **ATR-Based Stop**
```python
# Ejemplo: 2x ATR stop
stop_price = entry_price - (atr * 2.0)
```
**Pros**: Se adapta a volatilidad
**Cons**: Puede ser demasiado amplio en small caps

### 3. **Technical Level Stop**
```python
# Stop bajo soporte técnico
support_level = identify_support_level(price_data)
stop_price = support_level * 0.99  # 1% buffer
```
**Pros**: Lógica de mercado
**Cons**: Subjetivo, puede cambiar

### 4. **Time-Based Stop**
```python
# Exit después de X minutos sin movimiento favorable
if minutes_since_entry > 30 and pnl < 0:
    exit_position()
```
**Pros**: Evita holds largos
**Cons**: Puede salir prematuramente

## Diversificación y Correlación

### Análisis de Correlación entre Posiciones

```python
import pandas as pd
import numpy as np

def calculate_portfolio_correlation(positions: dict) -> pd.DataFrame:
    """
    Calcula correlación entre posiciones actuales

    Args:
        positions: {symbol: quantity} dict

    Returns:
        Correlation matrix
    """
    symbols = list(positions.keys())

    # Obtener returns históricos
    returns_data = {}
    for symbol in symbols:
        returns_data[symbol] = get_historical_returns(symbol, days=30)

    df = pd.DataFrame(returns_data)
    correlation_matrix = df.corr()

    return correlation_matrix

def check_correlation_risk(positions: dict, max_correlation: float = 0.7):
    """
    Verifica si las posiciones están muy correlacionadas
    """
    corr_matrix = calculate_portfolio_correlation(positions)

    # Buscar correlaciones altas
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > max_correlation:
                high_corr_pairs.append({
                    'symbol1': corr_matrix.columns[i],
                    'symbol2': corr_matrix.columns[j],
                    'correlation': corr
                })

    return high_corr_pairs
```

### Reglas de Diversificación

1. **Sector Limits**: Máximo 50% del capital en un sector
2. **Market Cap Limits**: No más de 3 micro caps simultáneamente
3. **Geographic Limits**: Para international trading
4. **Time Diversification**: Escalonar entradas en el tiempo

## Drawdown Management

### Tipos de Drawdown

1. **Account Drawdown**: Pérdida desde equity peak
2. **Strategy Drawdown**: Pérdida de una estrategia específica
3. **Daily Drawdown**: Pérdida intradiaria
4. **Monthly Drawdown**: Pérdida mensual

### Sistema de Alertas por Drawdown

```python
class DrawdownMonitor:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_balance = initial_balance

        # Límites de alerta
        self.warning_drawdown = 0.10    # 10% warning
        self.danger_drawdown = 0.15     # 15% reduce size
        self.emergency_drawdown = 0.25  # 25% stop trading

    def update_balance(self, new_balance: float):
        self.current_balance = new_balance

        # Actualizar peak si corresponde
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # Calcular drawdown actual
        current_drawdown = (self.peak_balance - new_balance) / self.peak_balance

        # Generar alertas
        if current_drawdown >= self.emergency_drawdown:
            return "EMERGENCY_STOP"
        elif current_drawdown >= self.danger_drawdown:
            return "REDUCE_SIZE"
        elif current_drawdown >= self.warning_drawdown:
            return "WARNING"
        else:
            return "NORMAL"

    def get_drawdown_stats(self) -> dict:
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance

        return {
            'current_drawdown': current_drawdown,
            'peak_balance': self.peak_balance,
            'current_balance': self.current_balance,
            'dollars_from_peak': self.peak_balance - self.current_balance
        }
```

## Position Recycling Risk Management

### Gestión de Riesgo en Múltiples Entradas

Nuestro enfoque de "position recycling" requiere gestión de riesgo especial:

```python
class PositionRecyclingRisk:
    def __init__(self, symbol: str, max_total_risk: float = 15.0):
        self.symbol = symbol
        self.max_total_risk = max_total_risk
        self.positions = []  # Lista de {quantity, entry_price, timestamp}
        self.total_quantity = 0
        self.weighted_avg_price = 0.0

    def can_add_position(self, new_quantity: int, new_price: float) -> bool:
        """
        Verifica si podemos añadir una nueva posición sin exceder riesgo
        """
        # Calcular nueva posición total
        new_total_quantity = self.total_quantity + new_quantity
        new_total_value = (self.weighted_avg_price * self.total_quantity +
                          new_price * new_quantity)
        new_avg_price = new_total_value / new_total_quantity

        # Calcular riesgo con stop loss a 5%
        potential_loss = new_total_quantity * new_avg_price * 0.05

        return potential_loss <= self.max_total_risk

    def add_position(self, quantity: int, price: float):
        """Añade nueva posición y actualiza métricas"""
        if not self.can_add_position(quantity, price):
            raise ValueError("Excede límite de riesgo total")

        # Actualizar weighted average
        total_value = self.weighted_avg_price * self.total_quantity + price * quantity
        self.total_quantity += quantity
        self.weighted_avg_price = total_value / self.total_quantity

        # Registrar posición
        self.positions.append({
            'quantity': quantity,
            'price': price,
            'timestamp': pd.Timestamp.now()
        })
```

## Risk Metrics y Monitoring

### Métricas Clave a Trackear

1. **Risk-Adjusted Returns**
   - Sharpe Ratio
   - Sortino Ratio
   - Calmar Ratio

2. **Drawdown Metrics**
   - Maximum Drawdown
   - Average Drawdown
   - Drawdown Duration

3. **Risk Concentration**
   - Position Concentration
   - Sector Concentration
   - Time Concentration

### Dashboard de Risk Monitoring

```python
def generate_risk_report(trades_df: pd.DataFrame) -> dict:
    """
    Genera reporte completo de riesgo
    """
    # Calcular equity curve
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()

    # Drawdown analysis
    equity_curve = trades_df['cumulative_pnl']
    running_max = equity_curve.expanding().max()
    drawdown = equity_curve - running_max

    # Risk metrics
    daily_returns = trades_df.groupby(trades_df['date'].dt.date)['pnl'].sum()

    return {
        'max_drawdown': drawdown.min(),
        'current_drawdown': drawdown.iloc[-1],
        'avg_daily_pnl': daily_returns.mean(),
        'daily_volatility': daily_returns.std(),
        'sharpe_ratio': daily_returns.mean() / daily_returns.std() * np.sqrt(252),
        'win_rate': (trades_df['pnl'] > 0).mean(),
        'largest_loss': trades_df['pnl'].min(),
        'largest_win': trades_df['pnl'].max(),
        'total_trades': len(trades_df)
    }
```

## Escenarios de Crisis y Contingencias

### Plan de Contingencia por Escenario

#### 1. **Market Flash Crash**
```python
# Auto-liquidar todas las posiciones si:
if market_drop_5min > 0.05:  # 5% drop en 5 minutos
    liquidate_all_positions()
    suspend_new_entries(hours=2)
```

#### 2. **Individual Stock Halt**
```python
# Si una posición es suspendida:
if stock_halted:
    # No panic - es normal en small caps
    # Revisar razón del halt
    # Preparar exit plan cuando reanude
    monitor_halt_reason()
```

#### 3. **System Failure**
```python
# Backup manual procedures
emergency_contacts = [
    "Broker phone number",
    "Alternative execution platform",
    "Manual position tracking sheet"
]
```

#### 4. **Account Breach**
```python
if account_equity < stop_loss_level:
    # 1. Stop all automated trading
    # 2. Review all positions
    # 3. Liquidate if necessary
    # 4. Analyze what went wrong
    # 5. Adjust parameters before resuming
    emergency_stop_protocol()
```

## Psicología del Risk Management

### Common Risk Management Mistakes

1. **"Just this once" mentality**
   - Exceder position size "porque es muy buena oportunidad"
   - Solution: Automatización, sin overrides manuales

2. **Revenge trading**
   - Aumentar size después de pérdidas para "recuperar"
   - Solution: Circuit breakers automáticos

3. **Fear of missing out (FOMO)**
   - Entrar sin stop loss definido
   - Solution: No entry sin exit plan

4. **Overconfidence after wins**
   - Relajar risk management después de streak ganador
   - Solution: Risk parameters constantes

### Mental Framework para Risk Management

```
Antes de cada trade, preguntarse:

1. ¿Cuál es mi máxima pérdida aceptable?
2. ¿Dónde está mi stop loss?
3. ¿Cómo afecta esta posición a mi risk total?
4. ¿Qué hago si el trade va contra mí?
5. ¿Estoy emocionalmente preparado para la pérdida?

Si no puedes responder todas estas preguntas claramente,
NO HAGAS EL TRADE.
```

## Implementación Práctica

### Checklist Pre-Trade
- [ ] Position size calculado según riesgo fijo
- [ ] Stop loss definido y programado
- [ ] Verificar correlación con posiciones existentes
- [ ] Confirmar que no excede límites diarios/semanales
- [ ] Drawdown actual dentro de parámetros
- [ ] Plan de exit (tanto ganancia como pérdida)

### Monitoreo Intradiario
- [ ] P&L actual vs límites diarios
- [ ] Posiciones cerca de stop loss
- [ ] Nuevas noticias que afecten posiciones
- [ ] Correlaciones inesperadas entre posiciones

### Review Diario
- [ ] Análisis de todos los trades del día
- [ ] Actualización de métricas de riesgo
- [ ] Verificación de que no se violaron reglas
- [ ] Planning para siguiente día

---

**Remember**: En trading, no es cuánto ganas lo que importa, sino cuánto no pierdes. Un trader que preserve capital consistentemente siempre tendrá otra oportunidad de ganar.

**Next Steps**:
- Leer [Performance Metrics](./Performance-Metrics.md) para métricas de evaluación
- Implementar [Position Sizing Calculator](../scripts/strategy-metrics/position-sizing/)
- Estudiar [Backtesting](../technical-practices/Backtesting.md) para validar risk management