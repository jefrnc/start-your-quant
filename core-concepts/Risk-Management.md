> 🇪🇸 [Leer en Español](Risk-Management.es.md) | 🇺🇸 **English**

# Systematic Risk Management

## Philosophy: Risk-First Design

In quantitative trading, **risk management is not an add-on, it's the foundation**. Every trading decision must start with the question: "How much can I afford to lose on this trade?"

### Fundamental Principles

1. **Capital Preservation > Profit Maximization**
2. **Predefined Risk**: Never enter without knowing the exit
3. **Mathematical Position Sizing**: Based on probabilities, not intuition
4. **Temporal Diversification**: Not all capital at the same time
5. **Circuit Breakers**: Automatic limits to prevent catastrophes

## Risk Framework for Small Caps

### System Base Parameters

```yaml
risk_parameters:
  # Risk per Trade
  max_risk_per_trade: 10.0        # $10 maximum per trade
  max_position_size: 70.0         # $70 maximum per position

  # Daily/Weekly Risk
  max_daily_loss: 50.0            # $50 maximum daily loss
  max_weekly_loss: 150.0          # $150 maximum weekly loss

  # Drawdown Limits
  max_account_drawdown: 0.15      # 15% maximum drawdown
  emergency_stop_drawdown: 0.25   # 25% emergency stop

  # Concentration
  max_positions_concurrent: 3     # Maximum 3 simultaneous positions
  max_exposure_per_symbol: 0.05   # 5% of capital per symbol
```

## Position Sizing Methodologies

### 1. **Fixed Dollar Risk (Our Primary Method)**

```python
def calculate_position_size_fixed_risk(
    entry_price: float,
    stop_loss_price: float,
    max_risk_dollars: float = 10.0
) -> int:
    """
    Calculates shares based on fixed dollar risk

    Example:
    - Entry: $5.00
    - Stop: $4.75
    - Risk: $10
    - Position: $10 / ($5.00 - $4.75) = 40 shares
    """
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share <= 0:
        raise ValueError("Stop loss must be different from entry price")

    shares = int(max_risk_dollars / risk_per_share)

    # Check maximum position limit
    max_shares_by_position = int(70.0 / entry_price)

    return min(shares, max_shares_by_position)
```

### 2. **Percentage Risk Method**

```python
def calculate_position_size_percentage(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
    risk_percentage: float = 0.01  # 1% of account
) -> int:
    """
    Position sizing based on % of account
    Useful for larger accounts
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
    Position sizing based on volatility (ATR)
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
# Example: 5% stop loss
stop_price = entry_price * 0.95  # For long positions
```
**Pros**: Simple, predictable
**Cons**: Doesn't consider instrument volatility

### 2. **ATR-Based Stop**
```python
# Example: 2x ATR stop
stop_price = entry_price - (atr * 2.0)
```
**Pros**: Adapts to volatility
**Cons**: Can be too wide in small caps

### 3. **Technical Level Stop**
```python
# Stop below technical support
support_level = identify_support_level(price_data)
stop_price = support_level * 0.99  # 1% buffer
```
**Pros**: Market logic
**Cons**: Subjective, can change

### 4. **Time-Based Stop**
```python
# Exit after X minutes without favorable movement
if minutes_since_entry > 30 and pnl < 0:
    exit_position()
```
**Pros**: Avoids long holds
**Cons**: May exit prematurely

## Diversification and Correlation

### Correlation Analysis Between Positions

```python
import pandas as pd
import numpy as np

def calculate_portfolio_correlation(positions: dict) -> pd.DataFrame:
    """
    Calculates correlation between current positions

    Args:
        positions: {symbol: quantity} dict

    Returns:
        Correlation matrix
    """
    symbols = list(positions.keys())

    # Get historical returns
    returns_data = {}
    for symbol in symbols:
        returns_data[symbol] = get_historical_returns(symbol, days=30)

    df = pd.DataFrame(returns_data)
    correlation_matrix = df.corr()

    return correlation_matrix

def check_correlation_risk(positions: dict, max_correlation: float = 0.7):
    """
    Checks if positions are too correlated
    """
    corr_matrix = calculate_portfolio_correlation(positions)

    # Find high correlations
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

### Diversification Rules

1. **Sector Limits**: Maximum 50% of capital in one sector
2. **Market Cap Limits**: No more than 3 micro caps simultaneously
3. **Geographic Limits**: For international trading
4. **Time Diversification**: Stagger entries over time

## Drawdown Management

### Types of Drawdown

1. **Account Drawdown**: Loss from equity peak
2. **Strategy Drawdown**: Loss from a specific strategy
3. **Daily Drawdown**: Intraday loss
4. **Monthly Drawdown**: Monthly loss

### Drawdown Alert System

```python
class DrawdownMonitor:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_balance = initial_balance

        # Alert limits
        self.warning_drawdown = 0.10    # 10% warning
        self.danger_drawdown = 0.15     # 15% reduce size
        self.emergency_drawdown = 0.25  # 25% stop trading

    def update_balance(self, new_balance: float):
        self.current_balance = new_balance

        # Update peak if applicable
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # Calculate current drawdown
        current_drawdown = (self.peak_balance - new_balance) / self.peak_balance

        # Generate alerts
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

### Risk Management for Multiple Entries

Our "position recycling" approach requires special risk management:

```python
class PositionRecyclingRisk:
    def __init__(self, symbol: str, max_total_risk: float = 15.0):
        self.symbol = symbol
        self.max_total_risk = max_total_risk
        self.positions = []  # List of {quantity, entry_price, timestamp}
        self.total_quantity = 0
        self.weighted_avg_price = 0.0

    def can_add_position(self, new_quantity: int, new_price: float) -> bool:
        """
        Checks if we can add a new position without exceeding risk
        """
        # Calculate new total position
        new_total_quantity = self.total_quantity + new_quantity
        new_total_value = (self.weighted_avg_price * self.total_quantity +
                          new_price * new_quantity)
        new_avg_price = new_total_value / new_total_quantity

        # Calculate risk with 5% stop loss
        potential_loss = new_total_quantity * new_avg_price * 0.05

        return potential_loss <= self.max_total_risk

    def add_position(self, quantity: int, price: float):
        """Adds new position and updates metrics"""
        if not self.can_add_position(quantity, price):
            raise ValueError("Exceeds total risk limit")

        # Update weighted average
        total_value = self.weighted_avg_price * self.total_quantity + price * quantity
        self.total_quantity += quantity
        self.weighted_avg_price = total_value / self.total_quantity

        # Record position
        self.positions.append({
            'quantity': quantity,
            'price': price,
            'timestamp': pd.Timestamp.now()
        })
```

## Risk Metrics and Monitoring

### Key Metrics to Track

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

### Risk Monitoring Dashboard

```python
def generate_risk_report(trades_df: pd.DataFrame) -> dict:
    """
    Generates a complete risk report
    """
    # Calculate equity curve
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

## Crisis Scenarios and Contingencies

### Contingency Plan by Scenario

#### 1. **Market Flash Crash**
```python
# Auto-liquidate all positions if:
if market_drop_5min > 0.05:  # 5% drop in 5 minutes
    liquidate_all_positions()
    suspend_new_entries(hours=2)
```

#### 2. **Individual Stock Halt**
```python
# If a position is halted:
if stock_halted:
    # Don't panic - it's normal in small caps
    # Review halt reason
    # Prepare exit plan for when it resumes
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

## Psychology of Risk Management

### Common Risk Management Mistakes

1. **"Just this once" mentality**
   - Exceeding position size "because it's a great opportunity"
   - Solution: Automation, no manual overrides

2. **Revenge trading**
   - Increasing size after losses to "recover"
   - Solution: Automatic circuit breakers

3. **Fear of missing out (FOMO)**
   - Entering without a defined stop loss
   - Solution: No entry without an exit plan

4. **Overconfidence after wins**
   - Relaxing risk management after a winning streak
   - Solution: Constant risk parameters

### Mental Framework for Risk Management

```
Before every trade, ask yourself:

1. What is my maximum acceptable loss?
2. Where is my stop loss?
3. How does this position affect my total risk?
4. What do I do if the trade goes against me?
5. Am I emotionally prepared for the loss?

If you can't answer all of these questions clearly,
DON'T TAKE THE TRADE.
```

## Practical Implementation

### Pre-Trade Checklist
- [ ] Position size calculated based on fixed risk
- [ ] Stop loss defined and programmed
- [ ] Verify correlation with existing positions
- [ ] Confirm it doesn't exceed daily/weekly limits
- [ ] Current drawdown within parameters
- [ ] Exit plan (both profit and loss)

### Intraday Monitoring
- [ ] Current P&L vs daily limits
- [ ] Positions near stop loss
- [ ] New news affecting positions
- [ ] Unexpected correlations between positions

### Daily Review
- [ ] Analysis of all day's trades
- [ ] Update risk metrics
- [ ] Verify no rules were violated
- [ ] Planning for the next day

---

**Remember**: In trading, it's not how much you earn that matters, but how much you don't lose. A trader who consistently preserves capital will always have another opportunity to profit.

**Next Steps**:
- Read [Performance Metrics](./Performance-Metrics.md) for evaluation metrics
- Implement [Position Sizing Calculator](../scripts/strategy-metrics/position-sizing/)
- Study [Backtesting](../technical-practices/Backtesting.md) to validate risk management
