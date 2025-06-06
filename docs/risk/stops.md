# Stop Loss y Trailing Stops

## Tu Póliza de Seguro en Cada Trade

Los stops son la diferencia entre una pérdida controlada y un desastre. En small caps, donde los movimientos pueden ser violentos, tener stops automáticos no es opcional - es supervivencia.

## Tipos de Stop Loss

### 1. Fixed Percentage Stop
```python
class FixedPercentageStop:
    def __init__(self, stop_loss_pct=0.03):
        self.stop_loss_pct = stop_loss_pct
        
    def calculate_stop_price(self, entry_price, position_type='long'):
        """Calcular precio de stop loss"""
        if position_type == 'long':
            stop_price = entry_price * (1 - self.stop_loss_pct)
        else:  # short
            stop_price = entry_price * (1 + self.stop_loss_pct)
        
        return round(stop_price, 2)
    
    def is_stopped_out(self, current_price, stop_price, position_type='long'):
        """¿Se activó el stop?"""
        if position_type == 'long':
            return current_price <= stop_price
        else:  # short
            return current_price >= stop_price

# Ejemplo
stop_manager = FixedPercentageStop(stop_loss_pct=0.03)  # 3% stop
entry_price = 25.00
stop_price = stop_manager.calculate_stop_price(entry_price)
print(f"Entry: ${entry_price}, Stop: ${stop_price}")  # Entry: $25.00, Stop: $24.25
```

### 2. ATR-Based Stop
```python
def calculate_atr_stop(df, entry_price, atr_multiplier=1.5, position_type='long'):
    """Stop basado en Average True Range"""
    # Calcular ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(14).mean()
    
    current_atr = df['atr'].iloc[-1]
    
    if position_type == 'long':
        stop_price = entry_price - (current_atr * atr_multiplier)
    else:  # short
        stop_price = entry_price + (current_atr * atr_multiplier)
    
    return round(stop_price, 2)

# Ejemplo de uso
def smart_stop_calculation(ticker, entry_price, position_type='long'):
    """Calcular stop inteligente basado en volatilidad"""
    # Obtener datos históricos
    data = get_historical_data(ticker, period='3mo')
    
    # Método 1: Fixed percentage (3%)
    fixed_stop = entry_price * (0.97 if position_type == 'long' else 1.03)
    
    # Método 2: ATR-based
    atr_stop = calculate_atr_stop(data, entry_price, 1.5, position_type)
    
    # Método 3: Support/Resistance
    if position_type == 'long':
        support_level = find_nearest_support(data, entry_price)
        sr_stop = support_level * 0.99  # 1% below support
    else:
        resistance_level = find_nearest_resistance(data, entry_price)
        sr_stop = resistance_level * 1.01  # 1% above resistance
    
    # Usar el más conservador (closer to entry)
    if position_type == 'long':
        final_stop = max(fixed_stop, atr_stop, sr_stop)  # Highest stop for long
    else:
        final_stop = min(fixed_stop, atr_stop, sr_stop)  # Lowest stop for short
    
    return {
        'final_stop': round(final_stop, 2),
        'fixed_stop': round(fixed_stop, 2),
        'atr_stop': round(atr_stop, 2),
        'sr_stop': round(sr_stop, 2),
        'method_used': 'conservative_composite'
    }
```

### 3. VWAP-Based Stop
```python
class VWAPStopManager:
    def __init__(self, buffer_pct=0.01):
        self.buffer_pct = buffer_pct  # 1% buffer below VWAP
        
    def calculate_vwap_stop(self, df, position_type='long'):
        """Stop basado en VWAP con buffer"""
        # Calcular VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        current_vwap = df['vwap'].iloc[-1]
        
        if position_type == 'long':
            stop_price = current_vwap * (1 - self.buffer_pct)
        else:  # short
            stop_price = current_vwap * (1 + self.buffer_pct)
        
        return round(stop_price, 2)
    
    def is_vwap_intact(self, current_price, current_vwap, position_type='long'):
        """¿Está el precio manteniendo VWAP?"""
        if position_type == 'long':
            return current_price > current_vwap
        else:  # short
            return current_price < current_vwap
```

## Trailing Stops

### 1. Basic Trailing Stop
```python
class TrailingStopManager:
    def __init__(self, initial_stop_pct=0.03, trailing_pct=0.02):
        self.initial_stop_pct = initial_stop_pct
        self.trailing_pct = trailing_pct
        self.positions = {}  # {ticker: position_info}
        
    def initialize_position(self, ticker, entry_price, shares, position_type='long'):
        """Initialize nueva posición con trailing stop"""
        if position_type == 'long':
            initial_stop = entry_price * (1 - self.initial_stop_pct)
        else:  # short
            initial_stop = entry_price * (1 + self.initial_stop_pct)
        
        self.positions[ticker] = {
            'entry_price': entry_price,
            'shares': shares,
            'position_type': position_type,
            'current_stop': initial_stop,
            'highest_price': entry_price,  # For long positions
            'lowest_price': entry_price,   # For short positions
            'unrealized_pnl': 0,
            'max_favorable_excursion': 0
        }
        
        return initial_stop
    
    def update_trailing_stop(self, ticker, current_price):
        """Update trailing stop con nuevo precio"""
        if ticker not in self.positions:
            return None
        
        position = self.positions[ticker]
        
        # Update tracking prices
        if position['position_type'] == 'long':
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
                
                # Calculate new trailing stop
                new_stop = current_price * (1 - self.trailing_pct)
                
                # Only move stop up (never down for longs)
                if new_stop > position['current_stop']:
                    position['current_stop'] = new_stop
        
        else:  # short position
            if current_price < position['lowest_price']:
                position['lowest_price'] = current_price
                
                # Calculate new trailing stop
                new_stop = current_price * (1 + self.trailing_pct)
                
                # Only move stop down (never up for shorts)
                if new_stop < position['current_stop']:
                    position['current_stop'] = new_stop
        
        # Update P&L tracking
        if position['position_type'] == 'long':
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['shares']
            position['max_favorable_excursion'] = max(
                position['max_favorable_excursion'],
                (position['highest_price'] - position['entry_price']) * position['shares']
            )
        else:  # short
            position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['shares']
            position['max_favorable_excursion'] = max(
                position['max_favorable_excursion'],
                (position['entry_price'] - position['lowest_price']) * position['shares']
            )
        
        return position['current_stop']
    
    def is_stopped_out(self, ticker, current_price):
        """Check si se activó el trailing stop"""
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        
        if position['position_type'] == 'long':
            return current_price <= position['current_stop']
        else:  # short
            return current_price >= position['current_stop']
    
    def get_position_status(self, ticker):
        """Get status completo de la posición"""
        if ticker not in self.positions:
            return None
        
        return self.positions[ticker].copy()
```

### 2. Breakeven Stop
```python
class BreakevenStopManager:
    def __init__(self, breakeven_trigger_pct=0.04, breakeven_buffer_pct=0.005):
        self.breakeven_trigger_pct = breakeven_trigger_pct  # 4% gain to trigger
        self.breakeven_buffer_pct = breakeven_buffer_pct    # 0.5% above entry
        self.positions = {}
        
    def should_move_to_breakeven(self, ticker, current_price):
        """¿Debería mover el stop a breakeven?"""
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        
        if position['breakeven_set']:
            return False  # Ya está en breakeven
        
        entry_price = position['entry_price']
        
        if position['position_type'] == 'long':
            gain_pct = (current_price - entry_price) / entry_price
            return gain_pct >= self.breakeven_trigger_pct
        else:  # short
            gain_pct = (entry_price - current_price) / entry_price
            return gain_pct >= self.breakeven_trigger_pct
    
    def set_breakeven_stop(self, ticker):
        """Mover stop a breakeven"""
        if ticker not in self.positions:
            return None
        
        position = self.positions[ticker]
        entry_price = position['entry_price']
        
        if position['position_type'] == 'long':
            breakeven_stop = entry_price * (1 + self.breakeven_buffer_pct)
        else:  # short
            breakeven_stop = entry_price * (1 - self.breakeven_buffer_pct)
        
        position['current_stop'] = breakeven_stop
        position['breakeven_set'] = True
        
        return breakeven_stop
```

### 3. Parabolic SAR Stop
```python
def calculate_parabolic_sar(df, acceleration=0.02, max_acceleration=0.2):
    """Calcular Parabolic SAR para trailing stops"""
    df = df.copy()
    df['sar'] = df['close'].iloc[0]  # Initialize
    df['ep'] = df['high'].iloc[0]   # Extreme point
    df['af'] = acceleration         # Acceleration factor
    df['trend'] = 1                 # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df)):
        prev_sar = df['sar'].iloc[i-1]
        prev_ep = df['ep'].iloc[i-1]
        prev_af = df['af'].iloc[i-1]
        prev_trend = df['trend'].iloc[i-1]
        
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        # Calculate new SAR
        new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
        
        # Check for trend reversal
        if prev_trend == 1:  # Uptrend
            if current_low <= new_sar:
                # Trend reversal to downtrend
                df.loc[df.index[i], 'trend'] = -1
                df.loc[df.index[i], 'sar'] = prev_ep
                df.loc[df.index[i], 'ep'] = current_low
                df.loc[df.index[i], 'af'] = acceleration
            else:
                # Continue uptrend
                df.loc[df.index[i], 'trend'] = 1
                df.loc[df.index[i], 'sar'] = new_sar
                
                # Update extreme point and acceleration
                if current_high > prev_ep:
                    df.loc[df.index[i], 'ep'] = current_high
                    df.loc[df.index[i], 'af'] = min(prev_af + acceleration, max_acceleration)
                else:
                    df.loc[df.index[i], 'ep'] = prev_ep
                    df.loc[df.index[i], 'af'] = prev_af
                    
        else:  # Downtrend
            if current_high >= new_sar:
                # Trend reversal to uptrend
                df.loc[df.index[i], 'trend'] = 1
                df.loc[df.index[i], 'sar'] = prev_ep
                df.loc[df.index[i], 'ep'] = current_high
                df.loc[df.index[i], 'af'] = acceleration
            else:
                # Continue downtrend
                df.loc[df.index[i], 'trend'] = -1
                df.loc[df.index[i], 'sar'] = new_sar
                
                # Update extreme point and acceleration
                if current_low < prev_ep:
                    df.loc[df.index[i], 'ep'] = current_low
                    df.loc[df.index[i], 'af'] = min(prev_af + acceleration, max_acceleration)
                else:
                    df.loc[df.index[i], 'ep'] = prev_ep
                    df.loc[df.index[i], 'af'] = prev_af
    
    return df
```

## Stop Loss Avanzado para Small Caps

### 1. Volatility-Adjusted Stops
```python
class VolatilityAdjustedStop:
    def __init__(self, base_stop_pct=0.03, lookback_days=20):
        self.base_stop_pct = base_stop_pct
        self.lookback_days = lookback_days
        
    def calculate_volatility_multiplier(self, ticker):
        """Calcular multiplicador basado en volatilidad"""
        # Obtener datos históricos
        data = get_historical_data(ticker, self.lookback_days)
        
        # Calcular volatilidad diaria
        returns = data['close'].pct_change().dropna()
        daily_vol = returns.std()
        
        # Clasificar volatilidad
        if daily_vol > 0.08:      # >8% daily vol
            return 1.5            # Wider stops
        elif daily_vol > 0.05:    # 5-8% daily vol
            return 1.25
        elif daily_vol > 0.03:    # 3-5% daily vol
            return 1.0            # Normal stops
        else:                     # <3% daily vol
            return 0.75           # Tighter stops
    
    def calculate_adjusted_stop(self, ticker, entry_price, position_type='long'):
        """Calculate stop ajustado por volatilidad"""
        vol_multiplier = self.calculate_volatility_multiplier(ticker)
        adjusted_stop_pct = self.base_stop_pct * vol_multiplier
        
        if position_type == 'long':
            stop_price = entry_price * (1 - adjusted_stop_pct)
        else:  # short
            stop_price = entry_price * (1 + adjusted_stop_pct)
        
        return {
            'stop_price': round(stop_price, 2),
            'stop_pct': adjusted_stop_pct,
            'vol_multiplier': vol_multiplier,
            'original_stop_pct': self.base_stop_pct
        }
```

### 2. Time-Based Stop Adjustments
```python
class TimeBasedStopManager:
    def __init__(self):
        self.time_adjustments = {
            'premarket': 0.5,     # Tighter stops pre-market
            'opening': 1.5,       # Wider stops first hour
            'regular': 1.0,       # Normal stops
            'closing': 0.75,      # Tighter stops last hour
            'after_hours': 0.5    # Very tight stops after hours
        }
    
    def get_time_period(self):
        """Determinar período actual"""
        current_time = pd.Timestamp.now().time()
        
        if pd.Timestamp('04:00').time() <= current_time < pd.Timestamp('09:30').time():
            return 'premarket'
        elif pd.Timestamp('09:30').time() <= current_time < pd.Timestamp('10:30').time():
            return 'opening'
        elif pd.Timestamp('10:30').time() <= current_time < pd.Timestamp('15:30').time():
            return 'regular'
        elif pd.Timestamp('15:30').time() <= current_time < pd.Timestamp('16:00').time():
            return 'closing'
        else:
            return 'after_hours'
    
    def adjust_stop_for_time(self, base_stop_pct):
        """Ajustar stop según hora del día"""
        time_period = self.get_time_period()
        multiplier = self.time_adjustments[time_period]
        
        return {
            'adjusted_stop_pct': base_stop_pct * multiplier,
            'time_period': time_period,
            'multiplier': multiplier
        }
```

## Stop Loss Automation

### 1. Real-Time Stop Monitoring
```python
class RealTimeStopMonitor:
    def __init__(self):
        self.active_stops = {}  # {ticker: stop_info}
        self.alerts = []
        
    def add_stop_order(self, ticker, stop_price, shares, position_type, order_type='market'):
        """Agregar stop order para monitoring"""
        self.active_stops[ticker] = {
            'stop_price': stop_price,
            'shares': shares,
            'position_type': position_type,
            'order_type': order_type,
            'created_time': pd.Timestamp.now(),
            'triggered': False
        }
    
    def check_stops(self, market_data):
        """Check all active stops contra market data"""
        triggered_stops = []
        
        for ticker, stop_info in self.active_stops.items():
            if stop_info['triggered']:
                continue
                
            if ticker in market_data:
                current_price = market_data[ticker]['price']
                
                # Check if stop triggered
                if self.is_stop_triggered(current_price, stop_info):
                    stop_info['triggered'] = True
                    stop_info['trigger_time'] = pd.Timestamp.now()
                    stop_info['trigger_price'] = current_price
                    
                    triggered_stops.append({
                        'ticker': ticker,
                        'stop_info': stop_info,
                        'current_price': current_price
                    })
                    
                    # Generate alert
                    self.generate_stop_alert(ticker, stop_info, current_price)
        
        return triggered_stops
    
    def is_stop_triggered(self, current_price, stop_info):
        """Check si el stop se activó"""
        stop_price = stop_info['stop_price']
        position_type = stop_info['position_type']
        
        if position_type == 'long':
            return current_price <= stop_price
        else:  # short
            return current_price >= stop_price
    
    def generate_stop_alert(self, ticker, stop_info, trigger_price):
        """Generar alerta de stop activado"""
        alert = f"🚨 STOP TRIGGERED: {ticker} @ ${trigger_price:.2f} (Stop: ${stop_info['stop_price']:.2f})"
        self.alerts.append({
            'timestamp': pd.Timestamp.now(),
            'ticker': ticker,
            'message': alert,
            'trigger_price': trigger_price,
            'stop_price': stop_info['stop_price']
        })
        
        # Send immediate notification
        print(alert)
        # Implement Discord/email notification
```

### 2. Bracket Orders
```python
class BracketOrderManager:
    def __init__(self, broker_api):
        self.broker_api = broker_api
        self.active_brackets = {}
        
    def place_bracket_order(self, ticker, entry_price, shares, stop_loss_price, 
                          take_profit_price, position_type='long'):
        """Place bracket order (entry + stop + target)"""
        
        try:
            # Main order
            if position_type == 'long':
                main_order = self.broker_api.place_buy_order(ticker, shares, entry_price)
            else:
                main_order = self.broker_api.place_sell_order(ticker, shares, entry_price)
            
            # Stop loss order (OCO - One Cancels Other)
            stop_order = self.broker_api.place_stop_order(
                ticker, shares, stop_loss_price, 
                order_type='sell' if position_type == 'long' else 'buy'
            )
            
            # Take profit order
            profit_order = self.broker_api.place_limit_order(
                ticker, shares, take_profit_price,
                order_type='sell' if position_type == 'long' else 'buy'
            )
            
            # Link orders
            bracket_id = f"{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.active_brackets[bracket_id] = {
                'ticker': ticker,
                'main_order_id': main_order['order_id'],
                'stop_order_id': stop_order['order_id'],
                'profit_order_id': profit_order['order_id'],
                'entry_price': entry_price,
                'stop_price': stop_loss_price,
                'target_price': take_profit_price,
                'shares': shares,
                'position_type': position_type,
                'status': 'pending'
            }
            
            return bracket_id
            
        except Exception as e:
            print(f"Error placing bracket order: {e}")
            return None
    
    def update_bracket_status(self, bracket_id):
        """Update status del bracket order"""
        if bracket_id not in self.active_brackets:
            return None
        
        bracket = self.active_brackets[bracket_id]
        
        # Check order status
        main_status = self.broker_api.get_order_status(bracket['main_order_id'])
        
        if main_status == 'filled':
            bracket['status'] = 'active'
            # Now monitor stop and profit orders
        elif main_status == 'cancelled':
            bracket['status'] = 'cancelled'
            # Cancel associated orders
        
        return bracket['status']
```

## Stop Loss Psychology

### 1. Mental Stop vs Hard Stop
```python
class StopLossStrategy:
    def __init__(self, use_hard_stops=True, mental_stop_buffer_pct=0.005):
        self.use_hard_stops = use_hard_stops
        self.mental_stop_buffer_pct = mental_stop_buffer_pct
        
    def set_stop_strategy(self, ticker, entry_price, base_stop_pct):
        """Decidir estrategia de stop"""
        if self.use_hard_stops:
            # Hard stop: orden automática en el broker
            stop_price = entry_price * (1 - base_stop_pct)
            return {
                'type': 'hard_stop',
                'stop_price': stop_price,
                'automatic': True
            }
        else:
            # Mental stop: monitoring manual
            mental_stop = entry_price * (1 - base_stop_pct)
            alert_level = mental_stop * (1 + self.mental_stop_buffer_pct)
            
            return {
                'type': 'mental_stop',
                'stop_price': mental_stop,
                'alert_price': alert_level,
                'automatic': False
            }
    
    def should_use_hard_stop(self, ticker, volatility, float_size):
        """Determinar si usar hard stop o mental"""
        # Usar hard stops para:
        # - Stocks muy volátiles
        # - Low float (pueden gapear)
        # - When can't monitor constantly
        
        if volatility > 0.08:  # >8% daily volatility
            return True
        
        if float_size < 10_000_000:  # Low float
            return True
        
        return self.use_hard_stops  # Default setting
```

## Mi Setup Personal de Stops

```python
# stop_config.py
STOP_CONFIG = {
    # Base stop settings
    'default_stop_pct': 0.025,      # 2.5% default stop
    'max_stop_pct': 0.05,           # 5% max stop (never risk more)
    'min_stop_pct': 0.01,           # 1% min stop (for low vol)
    
    # Trailing stop settings
    'trailing_trigger_pct': 0.04,    # Start trailing at 4% gain
    'trailing_step_pct': 0.02,       # Trail by 2%
    'breakeven_trigger_pct': 0.06,   # Move to breakeven at 6% gain
    
    # Time-based adjustments
    'premarket_multiplier': 0.5,     # Tighter stops pre-market
    'opening_hour_multiplier': 1.5,  # Wider stops first hour
    'closing_multiplier': 0.75,      # Tighter stops last 30 min
    
    # Volatility adjustments
    'high_vol_multiplier': 1.5,      # Wider stops for high vol
    'low_vol_multiplier': 0.8,       # Tighter stops for low vol
    
    # Small cap adjustments
    'micro_float_multiplier': 1.25,  # Wider stops for micro float
    'large_gap_multiplier': 1.5,     # Wider stops on big gaps
}

class MyStopManager:
    def __init__(self, config=STOP_CONFIG):
        self.config = config
        self.trailing_manager = TrailingStopManager()
        self.vol_adjuster = VolatilityAdjustedStop()
        self.time_adjuster = TimeBasedStopManager()
        
    def calculate_optimal_stop(self, ticker, entry_price, stock_info):
        """Mi método principal para calcular stops"""
        
        # 1. Base stop
        base_stop_pct = self.config['default_stop_pct']
        
        # 2. Adjust for volatility
        vol_data = self.vol_adjuster.calculate_adjusted_stop(ticker, entry_price)
        vol_adjusted_pct = vol_data['stop_pct']
        
        # 3. Adjust for time
        time_data = self.time_adjuster.adjust_stop_for_time(vol_adjusted_pct)
        time_adjusted_pct = time_data['adjusted_stop_pct']
        
        # 4. Adjust for stock characteristics
        final_stop_pct = time_adjusted_pct
        
        # Float adjustment
        if stock_info.get('float', float('inf')) < 10_000_000:
            final_stop_pct *= self.config['micro_float_multiplier']
        
        # Gap adjustment
        if abs(stock_info.get('gap_pct', 0)) > 20:
            final_stop_pct *= self.config['large_gap_multiplier']
        
        # Apply limits
        final_stop_pct = max(min(final_stop_pct, self.config['max_stop_pct']), 
                           self.config['min_stop_pct'])
        
        # Calculate final stop price
        stop_price = entry_price * (1 - final_stop_pct)
        
        return {
            'stop_price': round(stop_price, 2),
            'stop_pct': final_stop_pct,
            'adjustments': {
                'base_pct': base_stop_pct,
                'volatility_adjusted': vol_adjusted_pct,
                'time_adjusted': time_adjusted_pct,
                'final_pct': final_stop_pct
            },
            'factors': {
                'volatility_multiplier': vol_data['vol_multiplier'],
                'time_multiplier': time_data['multiplier'],
                'time_period': time_data['time_period']
            }
        }
```

## Siguiente Paso

Con stops implementados, completemos la gestión de riesgo con [Riesgo Asimétrico](asymmetric_risk.md).