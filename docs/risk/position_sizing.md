> 🇪🇸 [Leer en Español](position_sizing.es.md) | 🇺🇸 **English**

# Fixed-Risk Position Sizing

## The Art of Not Going Broke

Position sizing is the difference between being profitable and going broke. You can have the best strategy in the world, but if you risk too much on each trade, a losing streak will wipe you out.

> **⚠️ Small Caps Warning**: Extreme volatility requires more conservative position sizing than blue chips. Our rule: maximum 1-2% risk per trade in small caps.

## Philosophy: Risk-First Approach

```python
# ❌ BAD: Thinking about gains first
def bad_position_sizing(account_value, stock_price):
    """I want to buy $10,000 of this stock"""
    return 10000 / stock_price

# ✅ GOOD: Thinking about risk first
def good_position_sizing(account_value, entry_price, stop_price, risk_per_trade=0.02):
    """How much can I lose on this trade?"""
    risk_amount = account_value * risk_per_trade
    risk_per_share = entry_price - stop_price
    return risk_amount / risk_per_share if risk_per_share > 0 else 0
```

## Fixed Risk Position Sizing

### The Base Method: 1-2% Risk
```python
class FixedRiskSizer:
    def __init__(self, account_value, max_risk_per_trade=0.02):
        self.account_value = account_value
        self.max_risk_per_trade = max_risk_per_trade
        
    def calculate_shares(self, entry_price, stop_loss_price):
        """Calculate shares based on fixed risk"""
        # Validations
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        # For long positions
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
        
        # Amount of money I'm willing to lose
        total_risk_amount = self.account_value * self.max_risk_per_trade
        
        # Shares I can buy
        shares = int(total_risk_amount / risk_per_share)
        
        # Verify it doesn't exceed max portfolio %
        max_position_value = self.account_value * 0.20  # 20% max
        max_shares_by_value = int(max_position_value / entry_price)
        
        return min(shares, max_shares_by_value)
    
    def calculate_position_details(self, entry_price, stop_loss_price):
        """Complete position details"""
        shares = self.calculate_shares(entry_price, stop_loss_price)
        
        if shares == 0:
            return None
        
        position_value = shares * entry_price
        risk_amount = shares * abs(entry_price - stop_loss_price)
        
        return {
            'shares': shares,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_percentage': risk_amount / self.account_value,
            'position_percentage': position_value / self.account_value
        }

# Usage example
sizer = FixedRiskSizer(account_value=50000, max_risk_per_trade=0.02)

# Stock at $25, stop loss at $23
position = sizer.calculate_position_details(entry_price=25, stop_loss_price=23)
print(f"Shares to buy: {position['shares']}")
print(f"Risk amount: ${position['risk_amount']:.2f}")
print(f"Risk percentage: {position['risk_percentage']:.2%}")
```

## Position Sizing for Small Caps

### Volatility Adjustments
```python
class SmallCapPositionSizer(FixedRiskSizer):
    def __init__(self, account_value, base_risk=0.02):
        super().__init__(account_value, base_risk)
        self.base_risk = base_risk
        
    def adjust_risk_for_volatility(self, ticker, current_price, lookback_days=20):
        """Adjust risk based on stock volatility"""
        # Get historical data
        historical_data = get_historical_data(ticker, lookback_days)
        
        if historical_data.empty:
            return self.base_risk
        
        # Calculate volatility
        returns = historical_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Adjust risk inversely to volatility
        if volatility > 0.8:  # High vol (>80% annual)
            risk_multiplier = 0.5  # Reduce risk to 50%
        elif volatility > 0.5:  # Medium vol
            risk_multiplier = 0.75
        elif volatility > 0.3:  # Low vol
            risk_multiplier = 1.0
        else:  # Very low vol
            risk_multiplier = 1.25
        
        adjusted_risk = self.base_risk * risk_multiplier
        
        # Cap at maximum 3%
        return min(adjusted_risk, 0.03)
    
    def adjust_risk_for_float(self, float_shares):
        """Adjust risk based on float size"""
        if float_shares < 5_000_000:  # Micro float
            return self.base_risk * 0.5  # 50% of normal risk
        elif float_shares < 20_000_000:  # Low float
            return self.base_risk * 0.75
        else:
            return self.base_risk
    
    def calculate_smart_position(self, ticker, entry_price, stop_loss_price, 
                               float_shares=None, gap_percent=None):
        """Smart position sizing for small caps"""
        
        # Base risk
        risk = self.base_risk
        
        # Adjust for volatility
        risk = self.adjust_risk_for_volatility(ticker, entry_price)
        
        # Adjust for float
        if float_shares:
            float_adjustment = self.adjust_risk_for_float(float_shares)
            risk = min(risk, float_adjustment)
        
        # Adjust for gap size
        if gap_percent and abs(gap_percent) > 20:
            risk *= 0.5  # Reduce risk 50% on large gaps
        
        # Update risk and calculate
        original_risk = self.max_risk_per_trade
        self.max_risk_per_trade = risk
        
        position = self.calculate_position_details(entry_price, stop_loss_price)
        
        # Restore original risk
        self.max_risk_per_trade = original_risk
        
        if position:
            position['adjusted_risk'] = risk
            position['risk_factors'] = {
                'base_risk': self.base_risk,
                'volatility_adjusted': risk != self.base_risk,
                'float_adjusted': float_shares is not None,
                'gap_adjusted': gap_percent is not None and abs(gap_percent) > 20
            }
        
        return position
```

## Risk Scaling Strategies

### 1. Kelly Criterion
```python
def kelly_criterion_sizing(win_rate, avg_win, avg_loss):
    """Kelly Criterion for optimal position sizing"""
    if avg_loss == 0:
        return 0
    
    # Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
    b = avg_win / abs(avg_loss)
    p = win_rate
    q = 1 - win_rate
    
    kelly_fraction = (b * p - q) / b
    
    # Kelly is aggressive, use a fraction
    conservative_kelly = kelly_fraction * 0.25  # 25% of full Kelly
    
    # Cap at maximum 5%
    return min(max(conservative_kelly, 0), 0.05)

def apply_kelly_sizing(historical_trades, current_trade):
    """Apply Kelly to current trade"""
    if len(historical_trades) < 20:  # Need history
        return 0.02  # Default 2%
    
    # Calculate historical statistics
    wins = [t for t in historical_trades if t > 0]
    losses = [t for t in historical_trades if t < 0]
    
    win_rate = len(wins) / len(historical_trades)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    kelly_size = kelly_criterion_sizing(win_rate, avg_win, avg_loss)
    
    return kelly_size
```

### 2. Volatility Adjusted Sizing
```python
def volatility_adjusted_sizing(base_risk, current_volatility, target_volatility=0.15):
    """Adjust position size by volatility"""
    if current_volatility <= 0:
        return base_risk
    
    # Scale inversely to volatility
    vol_adjustment = target_volatility / current_volatility
    adjusted_risk = base_risk * vol_adjustment
    
    # Reasonable limits
    return max(min(adjusted_risk, 0.05), 0.005)  # Between 0.5% and 5%

def calculate_portfolio_volatility_target(positions, target_vol=0.15):
    """Calculate sizing to maintain target portfolio volatility"""
    # Simplified version - you actually need correlations
    individual_vols = [pos['volatility'] for pos in positions]
    individual_weights = [pos['weight'] for pos in positions]
    
    # Portfolio vol (assuming average correlation)
    portfolio_vol = np.sqrt(np.sum([(w * v) ** 2 for w, v in zip(individual_weights, individual_vols)]))
    
    # Scale factor to reach target
    if portfolio_vol > 0:
        scale_factor = target_vol / portfolio_vol
        return min(scale_factor, 2.0)  # Max 2x scaling
    
    return 1.0
```

## Dynamic Position Sizing

### 1. Equity Curve Adjustment
```python
class DynamicRiskManager:
    def __init__(self, base_risk=0.02, lookback_period=20):
        self.base_risk = base_risk
        self.lookback_period = lookback_period
        self.equity_curve = []
        
    def update_equity(self, new_equity_value):
        """Update equity curve"""
        self.equity_curve.append(new_equity_value)
        
        # Keep only the lookback period
        if len(self.equity_curve) > self.lookback_period * 2:
            self.equity_curve = self.equity_curve[-self.lookback_period * 2:]
    
    def calculate_current_risk(self):
        """Calculate current risk based on recent performance"""
        if len(self.equity_curve) < self.lookback_period:
            return self.base_risk
        
        recent_equity = self.equity_curve[-self.lookback_period:]
        
        # Calculate current drawdown
        peak = max(recent_equity)
        current = recent_equity[-1]
        current_drawdown = (peak - current) / peak
        
        # Calculate recent returns
        returns = [(recent_equity[i] - recent_equity[i-1]) / recent_equity[i-1] 
                  for i in range(1, len(recent_equity))]
        
        # Recent win rate
        winning_periods = len([r for r in returns if r > 0])
        recent_win_rate = winning_periods / len(returns)
        
        # Adjust risk based on recent performance
        risk_multiplier = 1.0
        
        # Reduce risk if in drawdown
        if current_drawdown > 0.1:  # 10% drawdown
            risk_multiplier *= 0.5
        elif current_drawdown > 0.05:  # 5% drawdown
            risk_multiplier *= 0.75
        
        # Adjust for recent win rate
        if recent_win_rate < 0.4:  # <40% win rate
            risk_multiplier *= 0.75
        elif recent_win_rate > 0.6:  # >60% win rate
            risk_multiplier *= 1.25
        
        # Cap limits
        risk_multiplier = max(min(risk_multiplier, 2.0), 0.25)
        
        return self.base_risk * risk_multiplier
```

### 2. Market Regime Adjustment
```python
def adjust_risk_for_market_regime(base_risk, vix_level=None, market_trend=None):
    """Adjust risk based on market regime"""
    risk_multiplier = 1.0
    
    # VIX adjustment
    if vix_level:
        if vix_level > 30:  # High fear
            risk_multiplier *= 0.5
        elif vix_level > 20:  # Moderate fear
            risk_multiplier *= 0.75
        elif vix_level < 12:  # Complacency
            risk_multiplier *= 0.8  # Also reduce in complacency
    
    # Market trend adjustment
    if market_trend:
        if market_trend == 'strong_downtrend':
            risk_multiplier *= 0.3
        elif market_trend == 'downtrend':
            risk_multiplier *= 0.6
        elif market_trend == 'sideways':
            risk_multiplier *= 0.8
        elif market_trend == 'uptrend':
            risk_multiplier *= 1.0
        elif market_trend == 'strong_uptrend':
            risk_multiplier *= 1.2
    
    return base_risk * risk_multiplier
```

## Portfolio Level Risk Management

### 1. Correlation Adjustments
```python
def calculate_correlation_adjusted_sizing(positions, new_position, max_correlated_risk=0.1):
    """Adjust sizing considering correlations"""
    if not positions:
        return new_position['base_size']
    
    # Calculate average correlation with existing positions
    correlations = []
    for pos in positions:
        corr = calculate_correlation(pos['ticker'], new_position['ticker'])
        correlations.append(abs(corr))  # Absolute correlation
    
    avg_correlation = np.mean(correlations)
    
    # If high correlation, reduce position size
    if avg_correlation > 0.7:
        correlation_multiplier = 0.5
    elif avg_correlation > 0.5:
        correlation_multiplier = 0.75
    else:
        correlation_multiplier = 1.0
    
    # Calculate total risk of correlated positions
    correlated_risk = sum([pos['risk_amount'] for pos in positions 
                          if calculate_correlation(pos['ticker'], new_position['ticker']) > 0.5])
    
    account_value = sum([pos['account_value'] for pos in positions])
    
    # If there's already too much correlated risk, reduce further
    if correlated_risk / account_value > max_correlated_risk:
        correlation_multiplier *= 0.5
    
    return new_position['base_size'] * correlation_multiplier
```

### 2. Sector Concentration Limits
```python
class SectorRiskManager:
    def __init__(self, max_sector_risk=0.15):
        self.max_sector_risk = max_sector_risk
        self.sector_exposures = {}
        
    def add_position(self, ticker, sector, risk_amount, account_value):
        """Add position and track sector exposure"""
        if sector not in self.sector_exposures:
            self.sector_exposures[sector] = 0
        
        self.sector_exposures[sector] += risk_amount
        
        # Check if exceeding sector limit
        sector_risk_pct = self.sector_exposures[sector] / account_value
        
        if sector_risk_pct > self.max_sector_risk:
            excess = sector_risk_pct - self.max_sector_risk
            recommended_reduction = excess * account_value
            
            return {
                'approved': False,
                'reason': f'Sector {sector} exposure would be {sector_risk_pct:.2%}',
                'recommended_reduction': recommended_reduction
            }
        
        return {'approved': True}
    
    def get_sector_exposures(self, account_value):
        """Get current sector exposures"""
        return {sector: amount / account_value 
                for sector, amount in self.sector_exposures.items()}
```

## Real-Time Position Monitoring

```python
class PositionMonitor:
    def __init__(self):
        self.positions = {}
        self.alerts = []
        
    def add_position(self, ticker, entry_price, shares, stop_loss, max_risk):
        """Add position to monitor"""
        self.positions[ticker] = {
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'max_risk': max_risk,
            'current_risk': 0,
            'entry_time': pd.Timestamp.now()
        }
    
    def update_position(self, ticker, current_price):
        """Update position with current price"""
        if ticker not in self.positions:
            return
        
        pos = self.positions[ticker]
        
        # Calculate current risk
        current_risk = (pos['entry_price'] - current_price) * pos['shares']
        pos['current_risk'] = current_risk
        
        # Check alerts
        risk_pct = current_risk / pos['max_risk']
        
        if risk_pct > 0.8:  # 80% of max risk
            self.alerts.append(f"⚠️ {ticker}: Approaching max risk ({risk_pct:.1%})")
        
        if current_price <= pos['stop_loss']:
            self.alerts.append(f"🚨 {ticker}: Stop loss hit at ${current_price}")
    
    def get_portfolio_risk(self):
        """Calculate total portfolio risk"""
        total_risk = sum([pos['current_risk'] for pos in self.positions.values() if pos['current_risk'] > 0])
        return total_risk
```

## My Personal Setup

```python
# position_sizing_config.py
RISK_CONFIG = {
    'base_risk_per_trade': 0.015,  # 1.5% base risk
    'max_risk_per_trade': 0.03,    # 3% max risk
    'max_portfolio_risk': 0.06,    # 6% total risk
    'max_sector_concentration': 0.20,  # 20% per sector
    'max_single_position': 0.15,   # 15% max position size
    'volatility_target': 0.15,     # 15% portfolio volatility target
    
    # Small cap adjustments
    'micro_float_multiplier': 0.5,  # 50% size for micro floats
    'large_gap_multiplier': 0.5,    # 50% size for >20% gaps
    'high_vol_multiplier': 0.75,    # 75% size for high vol stocks
    
    # Market regime adjustments
    'high_vix_multiplier': 0.5,     # 50% size when VIX >30
    'bear_market_multiplier': 0.6,  # 60% size in bear market
}

def calculate_final_position_size(ticker, entry_price, stop_loss, account_value):
    """My main position sizing function"""
    
    # 1. Base sizing
    sizer = SmallCapPositionSizer(account_value, RISK_CONFIG['base_risk_per_trade'])
    
    # 2. Get market data
    market_data = get_market_context()
    stock_data = get_stock_context(ticker)
    
    # 3. Calculate smart position
    position = sizer.calculate_smart_position(
        ticker=ticker,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        float_shares=stock_data.get('float'),
        gap_percent=stock_data.get('gap_percent')
    )
    
    # 4. Apply market regime adjustments
    market_multiplier = adjust_risk_for_market_regime(
        1.0, 
        vix_level=market_data.get('vix'),
        market_trend=market_data.get('trend')
    )
    
    if position:
        position['shares'] = int(position['shares'] * market_multiplier)
        position['final_adjustments'] = {
            'market_multiplier': market_multiplier,
            'vix_level': market_data.get('vix'),
            'market_trend': market_data.get('trend')
        }
    
    return position
```

## Next Step

With position sizing mastered, let's move on to [Daily Risk Limits](risk_limits.md) to protect capital at the portfolio level.