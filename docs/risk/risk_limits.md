# Límites de Riesgo Diario

## El Circuit Breaker de tu Trading

Los límites de riesgo diario son tu última línea de defensa contra el desastre. Un día malo puede destruir semanas de ganancias si no tienes límites claros y automatizados.

## Por Qué Necesitas Límites

### La Realidad Brutal
- Un trader puede perder todo en un solo día sin límites
- Las emociones se intensifican con las pérdidas
- "Revenge trading" destruye cuentas
- Los mejores traders tienen días terribles

### Ejemplo Real
```python
# Sin límites de riesgo
account_start = 50000
trades = [
    -500,   # Trade 1: -1%
    -750,   # Trade 2: -1.5% (getting emotional)
    -1200,  # Trade 3: -2.5% (revenge trading)
    -2000,  # Trade 4: -4.5% (desperation)
    -3000   # Trade 5: -7% (account bleeding)
]

running_balance = account_start
for trade in trades:
    running_balance += trade
    print(f"After trade: ${running_balance:,} ({trade/account_start:.1%})")

# Result: $42,550 (-14.9% in one day!)
```

## Framework de Límites de Riesgo

### 1. Daily Loss Limit
```python
class DailyRiskManager:
    def __init__(self, account_value, max_daily_loss_pct=0.03):
        self.account_value = account_value
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_loss_amount = account_value * max_daily_loss_pct
        
        # Track daily P&L
        self.daily_trades = []
        self.daily_pnl = 0
        self.trading_halted = False
        
    def add_trade_result(self, pnl):
        """Agregar resultado de trade"""
        self.daily_trades.append({
            'pnl': pnl,
            'timestamp': pd.Timestamp.now(),
            'cumulative_pnl': self.daily_pnl + pnl
        })
        
        self.daily_pnl += pnl
        
        # Check limits
        self.check_daily_limits()
        
    def check_daily_limits(self):
        """Verificar si se excedieron límites"""
        if self.daily_pnl <= -self.max_daily_loss_amount:
            self.trading_halted = True
            self.send_alert(f"🚨 DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f}")
            return False
        
        # Warning at 75% of limit
        warning_threshold = -self.max_daily_loss_amount * 0.75
        if self.daily_pnl <= warning_threshold and not hasattr(self, 'warning_sent'):
            self.send_alert(f"⚠️ 75% of daily loss limit reached: ${self.daily_pnl:.2f}")
            self.warning_sent = True
        
        return True
    
    def can_take_trade(self, potential_loss):
        """¿Puedo tomar este trade?"""
        if self.trading_halted:
            return False, "Trading halted due to daily loss limit"
        
        potential_daily_loss = self.daily_pnl - potential_loss
        
        if potential_daily_loss <= -self.max_daily_loss_amount:
            return False, f"Trade would exceed daily loss limit"
        
        return True, "Trade approved"
    
    def reset_daily_limits(self):
        """Reset para nuevo día de trading"""
        self.daily_trades = []
        self.daily_pnl = 0
        self.trading_halted = False
        if hasattr(self, 'warning_sent'):
            delattr(self, 'warning_sent')
    
    def send_alert(self, message):
        """Enviar alerta (Discord, email, etc.)"""
        print(f"ALERT: {message}")
        # Implementar envío real de alertas
```

### 2. Maximum Positions Limit
```python
class PositionLimitManager:
    def __init__(self, max_simultaneous_positions=5, max_sector_positions=2):
        self.max_simultaneous_positions = max_simultaneous_positions
        self.max_sector_positions = max_sector_positions
        self.current_positions = {}  # {ticker: position_info}
        self.sector_positions = {}   # {sector: [tickers]}
        
    def can_open_position(self, ticker, sector):
        """¿Puedo abrir esta posición?"""
        # Check total positions
        if len(self.current_positions) >= self.max_simultaneous_positions:
            return False, f"Max positions limit ({self.max_simultaneous_positions}) reached"
        
        # Check sector concentration
        if sector in self.sector_positions:
            if len(self.sector_positions[sector]) >= self.max_sector_positions:
                return False, f"Max positions in {sector} sector ({self.max_sector_positions}) reached"
        
        return True, "Position approved"
    
    def open_position(self, ticker, sector, position_info):
        """Abrir nueva posición"""
        can_open, reason = self.can_open_position(ticker, sector)
        
        if not can_open:
            return False, reason
        
        # Add position
        self.current_positions[ticker] = position_info
        
        # Track sector
        if sector not in self.sector_positions:
            self.sector_positions[sector] = []
        self.sector_positions[sector].append(ticker)
        
        return True, f"Position opened: {ticker}"
    
    def close_position(self, ticker):
        """Cerrar posición"""
        if ticker not in self.current_positions:
            return False, "Position not found"
        
        position_info = self.current_positions[ticker]
        sector = position_info.get('sector')
        
        # Remove from tracking
        del self.current_positions[ticker]
        
        if sector and sector in self.sector_positions:
            if ticker in self.sector_positions[sector]:
                self.sector_positions[sector].remove(ticker)
            
            # Clean up empty sectors
            if not self.sector_positions[sector]:
                del self.sector_positions[sector]
        
        return True, f"Position closed: {ticker}"
    
    def get_position_summary(self):
        """Resumen de posiciones actuales"""
        return {
            'total_positions': len(self.current_positions),
            'max_positions': self.max_simultaneous_positions,
            'positions_available': self.max_simultaneous_positions - len(self.current_positions),
            'sector_breakdown': {sector: len(tickers) for sector, tickers in self.sector_positions.items()},
            'current_tickers': list(self.current_positions.keys())
        }
```

### 3. Drawdown-Based Limits
```python
class DrawdownLimitManager:
    def __init__(self, account_value, max_drawdown_pct=0.15):
        self.initial_account_value = account_value
        self.peak_account_value = account_value
        self.current_account_value = account_value
        self.max_drawdown_pct = max_drawdown_pct
        self.max_drawdown_amount = account_value * max_drawdown_pct
        
        self.drawdown_levels = {
            0.05: "conservative",   # 5% - reduce risk
            0.10: "moderate",       # 10% - significant reduction
            0.15: "severe"          # 15% - halt trading
        }
        
    def update_account_value(self, new_value):
        """Update account value y check drawdown"""
        self.current_account_value = new_value
        
        # Update peak si es nuevo high
        if new_value > self.peak_account_value:
            self.peak_account_value = new_value
        
        return self.check_drawdown_limits()
    
    def check_drawdown_limits(self):
        """Check drawdown limits"""
        current_drawdown = (self.peak_account_value - self.current_account_value) / self.peak_account_value
        current_dd_amount = self.peak_account_value - self.current_account_value
        
        status = {
            'current_drawdown_pct': current_drawdown,
            'current_drawdown_amount': current_dd_amount,
            'peak_value': self.peak_account_value,
            'current_value': self.current_account_value,
            'action_required': None,
            'risk_multiplier': 1.0
        }
        
        # Determine action based on drawdown level
        if current_drawdown >= 0.15:  # Severe
            status['action_required'] = "HALT_TRADING"
            status['risk_multiplier'] = 0.0
            status['message'] = f"🚨 SEVERE DRAWDOWN: {current_drawdown:.1%} - Trading halted"
            
        elif current_drawdown >= 0.10:  # Moderate
            status['action_required'] = "REDUCE_RISK_SIGNIFICANTLY"
            status['risk_multiplier'] = 0.25  # 25% of normal risk
            status['message'] = f"⚠️ MODERATE DRAWDOWN: {current_drawdown:.1%} - Risk reduced to 25%"
            
        elif current_drawdown >= 0.05:  # Conservative
            status['action_required'] = "REDUCE_RISK"
            status['risk_multiplier'] = 0.5   # 50% of normal risk
            status['message'] = f"⚠️ CONSERVATIVE DRAWDOWN: {current_drawdown:.1%} - Risk reduced to 50%"
        
        else:
            status['action_required'] = "NORMAL"
            status['message'] = f"✅ Drawdown within limits: {current_drawdown:.1%}"
        
        return status
    
    def get_adjusted_risk(self, base_risk):
        """Get risk adjusted for current drawdown"""
        dd_status = self.check_drawdown_limits()
        return base_risk * dd_status['risk_multiplier']
```

## Time-Based Risk Limits

### 1. Pre-Market Limits
```python
class TimeBasedRiskManager:
    def __init__(self):
        self.time_limits = {
            'premarket': {
                'start': '04:00',
                'end': '09:30',
                'max_risk_per_trade': 0.01,  # 1% max in pre-market
                'max_positions': 2
            },
            'opening': {
                'start': '09:30',
                'end': '10:30',
                'max_risk_per_trade': 0.025,  # 2.5% max in first hour
                'max_positions': 3
            },
            'regular': {
                'start': '10:30',
                'end': '15:30',
                'max_risk_per_trade': 0.02,   # 2% normal hours
                'max_positions': 5
            },
            'closing': {
                'start': '15:30',
                'end': '16:00',
                'max_risk_per_trade': 0.015,  # 1.5% in last 30 min
                'max_positions': 2
            }
        }
    
    def get_current_time_limits(self):
        """Get limits for current time"""
        current_time = pd.Timestamp.now().time()
        
        for period, limits in self.time_limits.items():
            start_time = pd.Timestamp(limits['start']).time()
            end_time = pd.Timestamp(limits['end']).time()
            
            if start_time <= current_time < end_time:
                return period, limits
        
        # After hours - no trading
        return 'after_hours', {
            'max_risk_per_trade': 0,
            'max_positions': 0
        }
    
    def can_trade_now(self):
        """¿Puedo tradear ahora?"""
        period, limits = self.get_current_time_limits()
        
        if period == 'after_hours':
            return False, "Trading not allowed after hours"
        
        return True, f"Trading allowed in {period} period"
```

### 2. Velocity Limits
```python
class VelocityLimitManager:
    def __init__(self, max_trades_per_hour=10, max_trades_per_day=50):
        self.max_trades_per_hour = max_trades_per_hour
        self.max_trades_per_day = max_trades_per_day
        self.trade_timestamps = []
        
    def can_execute_trade(self):
        """¿Puedo ejecutar otro trade?"""
        now = pd.Timestamp.now()
        
        # Clean old trades (keep last 24 hours)
        cutoff = now - pd.Timedelta(hours=24)
        self.trade_timestamps = [ts for ts in self.trade_timestamps if ts > cutoff]
        
        # Check daily limit
        today_trades = [ts for ts in self.trade_timestamps if ts.date() == now.date()]
        if len(today_trades) >= self.max_trades_per_day:
            return False, f"Daily trade limit ({self.max_trades_per_day}) reached"
        
        # Check hourly limit
        hour_ago = now - pd.Timedelta(hours=1)
        recent_trades = [ts for ts in self.trade_timestamps if ts > hour_ago]
        if len(recent_trades) >= self.max_trades_per_hour:
            return False, f"Hourly trade limit ({self.max_trades_per_hour}) reached"
        
        return True, "Trade velocity within limits"
    
    def record_trade(self):
        """Record trade execution"""
        self.trade_timestamps.append(pd.Timestamp.now())
```

## Integrated Risk Management System

```python
class IntegratedRiskManager:
    def __init__(self, account_value, config=None):
        self.account_value = account_value
        
        # Default config
        default_config = {
            'max_daily_loss_pct': 0.03,
            'max_drawdown_pct': 0.15,
            'max_simultaneous_positions': 5,
            'max_sector_positions': 2,
            'max_trades_per_hour': 10,
            'max_trades_per_day': 50
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize sub-managers
        self.daily_risk = DailyRiskManager(account_value, self.config['max_daily_loss_pct'])
        self.position_limits = PositionLimitManager(
            self.config['max_simultaneous_positions'],
            self.config['max_sector_positions']
        )
        self.drawdown_limits = DrawdownLimitManager(account_value, self.config['max_drawdown_pct'])
        self.time_limits = TimeBasedRiskManager()
        self.velocity_limits = VelocityLimitManager(
            self.config['max_trades_per_hour'],
            self.config['max_trades_per_day']
        )
        
    def can_execute_trade(self, ticker, sector, potential_loss):
        """Master check - ¿puedo ejecutar este trade?"""
        checks = {}
        
        # Daily loss check
        can_trade, reason = self.daily_risk.can_take_trade(potential_loss)
        checks['daily_loss'] = {'passed': can_trade, 'reason': reason}
        
        # Position limits check
        can_open, reason = self.position_limits.can_open_position(ticker, sector)
        checks['position_limits'] = {'passed': can_open, 'reason': reason}
        
        # Time-based check
        can_trade_time, reason = self.time_limits.can_trade_now()
        checks['time_limits'] = {'passed': can_trade_time, 'reason': reason}
        
        # Velocity check
        can_trade_velocity, reason = self.velocity_limits.can_execute_trade()
        checks['velocity_limits'] = {'passed': can_trade_velocity, 'reason': reason}
        
        # Drawdown check
        dd_status = self.drawdown_limits.check_drawdown_limits()
        checks['drawdown'] = {
            'passed': dd_status['action_required'] != 'HALT_TRADING',
            'reason': dd_status['message'],
            'risk_multiplier': dd_status['risk_multiplier']
        }
        
        # Overall result
        all_passed = all(check['passed'] for check in checks.values())
        failed_checks = [name for name, check in checks.items() if not check['passed']]
        
        result = {
            'approved': all_passed,
            'checks': checks,
            'failed_checks': failed_checks,
            'risk_multiplier': checks['drawdown']['risk_multiplier']
        }
        
        if not all_passed:
            result['reason'] = f"Failed checks: {', '.join(failed_checks)}"
        
        return result
    
    def execute_trade(self, ticker, sector, pnl, position_info=None):
        """Record trade execution"""
        # Record P&L
        self.daily_risk.add_trade_result(pnl)
        
        # Update positions if opening
        if position_info and pnl == 0:  # Opening trade
            self.position_limits.open_position(ticker, sector, position_info)
        elif pnl != 0:  # Closing trade
            self.position_limits.close_position(ticker)
        
        # Record velocity
        self.velocity_limits.record_trade()
        
        # Update account value (simplified)
        new_account_value = self.account_value + pnl
        self.drawdown_limits.update_account_value(new_account_value)
        self.account_value = new_account_value
    
    def get_risk_dashboard(self):
        """Dashboard completo de riesgo"""
        dd_status = self.drawdown_limits.check_drawdown_limits()
        position_summary = self.position_limits.get_position_summary()
        time_period, time_limits = self.time_limits.get_current_time_limits()
        
        return {
            'account_value': self.account_value,
            'daily_pnl': self.daily_risk.daily_pnl,
            'daily_loss_limit': self.daily_risk.max_daily_loss_amount,
            'trading_halted': self.daily_risk.trading_halted,
            'drawdown_status': dd_status,
            'position_summary': position_summary,
            'current_time_period': time_period,
            'time_limits': time_limits,
            'trades_today': len([ts for ts in self.velocity_limits.trade_timestamps 
                               if ts.date() == pd.Timestamp.now().date()])
        }
```

## Alertas y Notifications

```python
class RiskAlertSystem:
    def __init__(self, discord_webhook=None, email_config=None):
        self.discord_webhook = discord_webhook
        self.email_config = email_config
        self.alert_history = []
        
    def send_risk_alert(self, level, message, data=None):
        """Enviar alerta de riesgo"""
        alert = {
            'timestamp': pd.Timestamp.now(),
            'level': level,  # 'info', 'warning', 'critical'
            'message': message,
            'data': data
        }
        
        self.alert_history.append(alert)
        
        # Enviar según nivel
        if level == 'critical':
            self.send_immediate_alert(message, data)
        elif level == 'warning':
            self.send_warning_alert(message, data)
        else:
            self.log_info_alert(message, data)
    
    def send_immediate_alert(self, message, data):
        """Alerta inmediata (SMS, call, etc.)"""
        print(f"🚨 CRITICAL ALERT: {message}")
        
        # Discord
        if self.discord_webhook:
            self.send_discord_alert(f"🚨 **CRITICAL ALERT** 🚨\n{message}")
        
        # Email
        if self.email_config:
            self.send_email_alert("CRITICAL TRADING ALERT", message, data)
    
    def send_discord_alert(self, message):
        """Enviar a Discord"""
        # Implementar webhook de Discord
        pass
    
    def daily_risk_report(self, risk_manager):
        """Reporte diario de riesgo"""
        dashboard = risk_manager.get_risk_dashboard()
        
        report = f"""
📊 **Daily Risk Report** - {pd.Timestamp.now().strftime('%Y-%m-%d')}

💰 **Account Status**
• Account Value: ${dashboard['account_value']:,.2f}
• Daily P&L: ${dashboard['daily_pnl']:,.2f}
• Drawdown: {dashboard['drawdown_status']['current_drawdown_pct']:.2%}

🔢 **Position Status**
• Active Positions: {dashboard['position_summary']['total_positions']}/{dashboard['position_summary']['max_positions']}
• Trades Today: {dashboard['trades_today']}

⚠️ **Risk Status**
• Trading Halted: {dashboard['trading_halted']}
• Risk Multiplier: {dashboard['drawdown_status']['risk_multiplier']}
• Time Period: {dashboard['current_time_period']}
"""
        
        return report
```

## Mi Setup Personal

```python
# risk_config.py
RISK_LIMITS = {
    # Daily limits
    'max_daily_loss_pct': 0.025,    # 2.5% max daily loss
    'daily_loss_warning_pct': 0.02,  # Warning at 2%
    
    # Drawdown limits
    'max_drawdown_pct': 0.12,        # 12% max drawdown
    'drawdown_warning_pct': 0.08,    # Warning at 8%
    
    # Position limits
    'max_simultaneous_positions': 4,  # Max 4 positions
    'max_sector_positions': 2,        # Max 2 per sector
    'max_single_position_pct': 0.15,  # 15% max single position
    
    # Velocity limits
    'max_trades_per_hour': 8,         # Max 8 trades/hour
    'max_trades_per_day': 30,         # Max 30 trades/day
    
    # Time-based adjustments
    'premarket_risk_multiplier': 0.5,   # 50% risk pre-market
    'closing_risk_multiplier': 0.75,    # 75% risk last 30 min
}

def initialize_risk_manager(account_value):
    """Initialize mi risk manager personal"""
    return IntegratedRiskManager(account_value, RISK_LIMITS)

# Uso diario
risk_manager = initialize_risk_manager(50000)

# Antes de cada trade
trade_check = risk_manager.can_execute_trade('AAPL', 'Technology', 500)
if trade_check['approved']:
    # Execute trade
    pass
else:
    print(f"Trade rejected: {trade_check['reason']}")
```

## Siguiente Paso

Con los límites de riesgo establecidos, vamos a [Stop Loss y Trailing Stops](stops.md) para gestión táctica de riesgo.