# First Green Day & First Red Day Patterns

## Concepto Base

Las estrategias First Green Day y First Red Day se basan en los cambios de momentum direccional después de tendencias sostenidas. Estas reversiones suelen marcar puntos de inflexión importantes en small caps debido a la psicología retail y el flujo institucional.

## First Green Day Strategy

### Fundamentos
Después de una serie de días rojos consecutivos, el primer día verde suele atraer buying interest renovado, especialmente si viene acompañado de volumen.

```python
class FirstGreenDayPattern:
    def __init__(self):
        self.pattern_type = "trend_reversal"
        self.min_red_days = 3
        self.max_red_days = 8  # Más de 8 días puede indicar problemas fundamentales
        
    def identify_pattern(self, price_history):
        """Identificar patrón de First Green Day"""
        
        # Verificar días rojos consecutivos
        consecutive_red = self.count_consecutive_red_days(price_history)
        
        if not (self.min_red_days <= consecutive_red <= self.max_red_days):
            return None
        
        # Verificar que hoy sea green
        today = price_history.iloc[-1]
        if today['close'] <= today['open']:
            return None
        
        # Calcular métricas del patrón
        pattern_metrics = self.calculate_pattern_metrics(price_history, consecutive_red)
        
        return {
            'consecutive_red_days': consecutive_red,
            'pattern_strength': pattern_metrics['strength'],
            'oversold_level': pattern_metrics['oversold'],
            'volume_confirmation': pattern_metrics['volume_confirm'],
            'setup_quality': pattern_metrics['quality_score']
        }
    
    def calculate_pattern_metrics(self, price_history, red_days):
        """Calcular métricas del patrón"""
        
        # Decline magnitude durante red days
        start_price = price_history.iloc[-(red_days + 1)]['close']
        low_price = price_history.tail(red_days)['low'].min()
        total_decline = (start_price - low_price) / start_price
        
        # Volume durante decline vs recovery
        decline_avg_volume = price_history.tail(red_days)['volume'].mean()
        today_volume = price_history.iloc[-1]['volume']
        volume_ratio = today_volume / decline_avg_volume if decline_avg_volume > 0 else 1
        
        # RSI oversold
        rsi = calculate_rsi(price_history['close'], 14)
        current_rsi = rsi.iloc[-1]
        
        # Pattern strength
        strength_score = min(100, (total_decline * 200) + (red_days * 5))
        
        # Quality score
        quality_components = [
            min(30, total_decline * 100),  # Decline severity (max 30)
            min(25, red_days * 3),         # Duration (max 25)
            min(25, (volume_ratio - 1) * 12.5),  # Volume (max 25)
            min(20, max(0, (30 - current_rsi)))  # RSI oversold (max 20)
        ]
        
        quality_score = sum(quality_components)
        
        return {
            'strength': strength_score,
            'oversold': current_rsi < 35,
            'volume_confirm': volume_ratio > 1.5,
            'quality_score': quality_score
        }

def count_consecutive_red_days(price_data):
    """Contar días rojos consecutivos"""
    consecutive = 0
    
    for i in range(len(price_data) - 2, -1, -1):  # Start from yesterday
        day = price_data.iloc[i]
        if day['close'] < day['open']:
            consecutive += 1
        else:
            break
    
    return consecutive
```

### Entry Strategy
```python
class FirstGreenDayEntry:
    def __init__(self, risk_tolerance='medium'):
        self.risk_tolerance = risk_tolerance
        self.entry_strategies = {
            'conservative': self.conservative_entry,
            'aggressive': self.aggressive_entry,
            'scalp': self.scalp_entry
        }
    
    def calculate_entry_plan(self, stock_data, pattern_data):
        """Calcular plan de entrada"""
        
        strategy_func = self.entry_strategies.get(self.risk_tolerance, self.conservative_entry)
        return strategy_func(stock_data, pattern_data)
    
    def conservative_entry(self, stock_data, pattern_data):
        """Entrada conservadora - esperar confirmación"""
        
        current_price = stock_data.price
        today_open = stock_data.open
        yesterday_high = stock_data.previous_day_high
        
        return {
            'primary_entry': {
                'price': max(yesterday_high * 1.01, current_price * 1.02),
                'trigger': 'break_previous_day_high',
                'size_pct': 0.60,
                'confirmation_required': True
            },
            'secondary_entry': {
                'price': current_price * 1.05,
                'trigger': 'sustained_momentum',
                'size_pct': 0.40,
                'time_limit': '11:30'  # Only if momentum continues
            },
            'stop_loss': min(
                today_open * 0.97,  # 3% below today's open
                stock_data.support_level * 0.98
            ),
            'targets': {
                'target_1': current_price * 1.08,    # 8% quick target
                'target_2': current_price * 1.15,    # 15% extended
                'target_3': stock_data.previous_resistance * 1.02
            },
            'risk_reward': 'conservative',
            'max_hold_time': 'same_day'
        }
    
    def aggressive_entry(self, stock_data, pattern_data):
        """Entrada agresiva - temprano en el momentum"""
        
        current_price = stock_data.price
        today_open = stock_data.open
        
        return {
            'immediate_entry': {
                'price': current_price * 1.005,  # Slight premium
                'trigger': 'pattern_confirmation',
                'size_pct': 0.75,
                'market_order': True
            },
            'add_on_strength': {
                'price': current_price * 1.03,
                'trigger': 'momentum_acceleration',
                'size_pct': 0.25,
                'volume_requirement': '2x_average'
            },
            'stop_loss': today_open * 0.95,  # 5% below open
            'targets': {
                'quick_target': current_price * 1.06,
                'momentum_target': current_price * 1.12
            },
            'risk_reward': 'aggressive',
            'time_management': {
                'profit_take_time': '14:00',
                'force_exit_time': '15:45'
            }
        }
```

## First Red Day Strategy (Short Bias)

### Fundamentos
Después de una serie de días verdes consecutivos, el primer día rojo suele indicar que el momentum alcista se está agotando y puede ser el inicio de una corrección.

```python
class FirstRedDayPattern:
    def __init__(self):
        self.pattern_type = "momentum_exhaustion"
        self.min_green_days = 3
        self.max_green_days = 10
        
    def identify_short_setup(self, price_history):
        """Identificar setup para short en First Red Day"""
        
        consecutive_green = self.count_consecutive_green_days(price_history)
        
        if not (self.min_green_days <= consecutive_green <= self.max_green_days):
            return None
        
        # Verificar que hoy sea red day
        today = price_history.iloc[-1]
        if today['close'] >= today['open']:
            return None
        
        # Analizar calidad del setup
        setup_analysis = self.analyze_short_setup_quality(price_history, consecutive_green)
        
        return {
            'consecutive_green_days': consecutive_green,
            'momentum_exhaustion_score': setup_analysis['exhaustion_score'],
            'overbought_level': setup_analysis['overbought'],
            'distribution_signs': setup_analysis['distribution'],
            'short_setup_quality': setup_analysis['quality_score'],
            'entry_recommendation': setup_analysis['recommendation']
        }
    
    def analyze_short_setup_quality(self, price_history, green_days):
        """Analizar calidad para short setup"""
        
        # Extensión del rally
        rally_start = price_history.iloc[-(green_days + 1)]['close']
        rally_high = price_history.tail(green_days)['high'].max()
        rally_magnitude = (rally_high - rally_start) / rally_start
        
        # Volume analysis durante rally
        rally_volume = price_history.tail(green_days)['volume'].mean()
        today_volume = price_history.iloc[-1]['volume']
        volume_ratio = today_volume / rally_volume if rally_volume > 0 else 1
        
        # RSI overbought
        rsi = calculate_rsi(price_history['close'], 14)
        current_rsi = rsi.iloc[-1]
        
        # Distribution signs
        distribution_score = self.detect_distribution_signs(price_history.tail(green_days + 1))
        
        # Exhaustion score
        exhaustion_components = [
            min(40, rally_magnitude * 100),  # Rally magnitude
            min(30, green_days * 3),         # Duration
            min(20, max(0, current_rsi - 70)), # Overbought
            min(10, distribution_score)      # Distribution
        ]
        
        exhaustion_score = sum(exhaustion_components)
        
        # Overall quality
        quality_score = min(100, exhaustion_score + (volume_ratio * 10))
        
        recommendation = 'strong_short' if quality_score > 80 else \
                        'moderate_short' if quality_score > 60 else \
                        'weak_short'
        
        return {
            'exhaustion_score': exhaustion_score,
            'overbought': current_rsi > 75,
            'distribution': distribution_score > 5,
            'quality_score': quality_score,
            'recommendation': recommendation
        }
    
    def detect_distribution_signs(self, recent_data):
        """Detectar signos de distribución"""
        score = 0
        
        # Lower highs en días recientes
        highs = recent_data['high'].values
        if len(highs) >= 3:
            if highs[-1] < highs[-2] < highs[-3]:
                score += 30
        
        # Volume en red days vs green days
        green_days = recent_data[recent_data['close'] > recent_data['open']]
        red_days = recent_data[recent_data['close'] < recent_data['open']]
        
        if len(red_days) > 0 and len(green_days) > 0:
            red_avg_vol = red_days['volume'].mean()
            green_avg_vol = green_days['volume'].mean()
            
            if red_avg_vol > green_avg_vol * 1.2:
                score += 25
        
        # Failed breakouts
        resistance_breaks = self.count_failed_breakouts(recent_data)
        score += min(20, resistance_breaks * 10)
        
        return score

def count_consecutive_green_days(price_data):
    """Contar días verdes consecutivos"""
    consecutive = 0
    
    for i in range(len(price_data) - 2, -1, -1):  # Start from yesterday
        day = price_data.iloc[i]
        if day['close'] > day['open']:
            consecutive += 1
        else:
            break
    
    return consecutive
```

### Short Entry Strategy
```python
class FirstRedDayShortEntry:
    def __init__(self):
        self.short_strategies = {
            'gap_down': self.gap_down_entry,
            'breakdown': self.breakdown_entry,
            'fade_rally': self.fade_rally_entry
        }
    
    def calculate_short_entry_plan(self, stock_data, pattern_data):
        """Calcular plan de entrada para short"""
        
        # Determinar mejor estrategia basada en price action
        if stock_data.gap_percent < -0.03:  # Gap down >3%
            strategy = 'gap_down'
        elif stock_data.price < stock_data.previous_support:
            strategy = 'breakdown'
        else:
            strategy = 'fade_rally'
        
        return self.short_strategies[strategy](stock_data, pattern_data)
    
    def gap_down_entry(self, stock_data, pattern_data):
        """Short en gap down después de rally"""
        
        gap_size = abs(stock_data.gap_percent)
        premarket_high = stock_data.premarket_high
        
        return {
            'primary_short_entry': {
                'price': premarket_high * 0.99,  # Below PM high
                'size_pct': 0.50,
                'trigger': 'failed_gap_fill',
                'confirmation': 'volume_on_weakness'
            },
            'aggressive_short_entry': {
                'price': stock_data.price * 0.98,
                'size_pct': 0.50,
                'trigger': 'continued_weakness',
                'time_window': '10:00-11:00'
            },
            'stop_loss': max(
                premarket_high * 1.03,
                stock_data.previous_day_high * 1.02
            ),
            'targets': {
                'target_1': stock_data.price * 0.92,  # 8% down
                'target_2': stock_data.price * 0.85,  # 15% down
                'gap_fill_target': stock_data.previous_close
            },
            'risk_management': {
                'max_loss_pct': 0.08,
                'time_stop': '15:30',
                'cover_on_squeeze': True
            }
        }
    
    def breakdown_entry(self, stock_data, pattern_data):
        """Short en breakdown de soporte"""
        
        support_level = stock_data.support_level
        
        return {
            'breakdown_short': {
                'price': support_level * 0.995,  # Just below support
                'size_pct': 0.75,
                'trigger': 'confirmed_breakdown',
                'volume_requirement': '1.5x_average'
            },
            'retest_short': {
                'price': support_level * 1.005,  # Failed retest
                'size_pct': 0.25,
                'trigger': 'failed_retest',
                'timeout': '30_minutes'
            },
            'stop_loss': support_level * 1.05,  # 5% above broken support
            'targets': {
                'measured_move': support_level * 0.90,
                'next_support': stock_data.next_support_level
            }
        }
```

## Risk Management Específico

### Position Sizing Dinámico
```python
class PatternPositionSizer:
    def __init__(self, account_size, base_risk=0.015):
        self.account_size = account_size
        self.base_risk = base_risk
        
    def calculate_pattern_position_size(self, stock_data, pattern_data, entry_plan):
        """Calcular tamaño basado en patrón específico"""
        
        # Base risk amount
        base_risk_amount = self.account_size * self.base_risk
        
        # Pattern-specific adjustments
        pattern_adjustments = self.get_pattern_adjustments(pattern_data)
        
        # Market environment adjustments
        market_adjustments = self.get_market_environment_adjustments()
        
        # Final risk calculation
        adjusted_risk = base_risk_amount * pattern_adjustments * market_adjustments
        
        # Calculate position size
        entry_price = entry_plan['price']
        stop_price = entry_plan['stop_loss']
        risk_per_share = abs(entry_price - stop_price)
        
        shares = int(adjusted_risk / risk_per_share) if risk_per_share > 0 else 0
        
        return {
            'shares': shares,
            'dollar_risk': adjusted_risk,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'pattern_adjustment': pattern_adjustments,
            'market_adjustment': market_adjustments
        }
    
    def get_pattern_adjustments(self, pattern_data):
        """Ajustes específicos del patrón"""
        
        base_multiplier = 1.0
        
        # Quality score adjustment
        quality_score = pattern_data.get('setup_quality', 50)
        if quality_score > 80:
            base_multiplier *= 1.3
        elif quality_score > 60:
            base_multiplier *= 1.1
        elif quality_score < 40:
            base_multiplier *= 0.7
        
        # Pattern strength
        if pattern_data.get('pattern_strength', 0) > 75:
            base_multiplier *= 1.2
        
        # Volume confirmation
        if pattern_data.get('volume_confirmation', False):
            base_multiplier *= 1.1
        
        return min(2.0, max(0.5, base_multiplier))
```

### Time-Based Exit Rules
```python
class PatternTimeManager:
    def __init__(self):
        self.time_rules = {
            'first_green_day': self.green_day_time_rules,
            'first_red_day': self.red_day_time_rules
        }
    
    def green_day_time_rules(self, entry_time, current_time, pnl):
        """Reglas de tiempo para First Green Day"""
        
        time_in_trade = (current_time - entry_time).seconds / 60  # minutes
        
        rules = []
        
        # Early momentum check
        if time_in_trade > 30 and pnl < 0.02:  # 30 min, less than 2% profit
            rules.append({
                'action': 'consider_exit',
                'reason': 'lack_of_early_momentum',
                'urgency': 'medium'
            })
        
        # Lunch time warning
        if current_time.hour == 12:
            rules.append({
                'action': 'reduce_position',
                'reason': 'lunch_time_low_volume',
                'urgency': 'low'
            })
        
        # Late day exit
        if current_time.hour >= 15:
            rules.append({
                'action': 'exit_position',
                'reason': 'late_day_exit',
                'urgency': 'high'
            })
        
        return rules
    
    def red_day_time_rules(self, entry_time, current_time, pnl):
        """Reglas de tiempo para First Red Day shorts"""
        
        time_in_trade = (current_time - entry_time).seconds / 60
        
        rules = []
        
        # Quick profit taking on shorts
        if time_in_trade > 60 and pnl > 0.10:  # 1 hour, 10% profit
            rules.append({
                'action': 'take_partial_profit',
                'reason': 'quick_short_profit',
                'urgency': 'medium'
            })
        
        # Cover before close (shorts risky overnight)
        if current_time.hour >= 15 and current_time.minute >= 30:
            rules.append({
                'action': 'cover_short',
                'reason': 'avoid_overnight_short_risk',
                'urgency': 'high'
            })
        
        return rules
```

## Performance Analysis

### Pattern-Specific Metrics
```python
class PatternPerformanceAnalyzer:
    def __init__(self):
        self.pattern_metrics = {}
        
    def analyze_pattern_performance(self, trades_df, pattern_type):
        """Analizar performance específica del patrón"""
        
        pattern_trades = trades_df[trades_df['pattern_type'] == pattern_type]
        
        if len(pattern_trades) == 0:
            return {'error': f'No trades found for pattern {pattern_type}'}
        
        # Basic metrics
        total_trades = len(pattern_trades)
        wins = len(pattern_trades[pattern_trades['pnl'] > 0])
        win_rate = wins / total_trades
        
        # Pattern-specific analysis
        if pattern_type == 'first_green_day':
            specific_analysis = self.analyze_green_day_performance(pattern_trades)
        else:  # first_red_day
            specific_analysis = self.analyze_red_day_performance(pattern_trades)
        
        return {
            'basic_metrics': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_pnl': pattern_trades['pnl'].mean(),
                'total_pnl': pattern_trades['pnl'].sum()
            },
            'pattern_specific': specific_analysis,
            'best_setups': self.identify_best_setups(pattern_trades),
            'worst_setups': self.identify_worst_setups(pattern_trades)
        }
    
    def analyze_green_day_performance(self, green_trades):
        """Análisis específico para First Green Day"""
        
        # Performance by consecutive red days
        red_days_performance = green_trades.groupby('consecutive_red_days')['pnl'].agg(['mean', 'count', 'sum'])
        
        # Performance by setup quality
        quality_bins = pd.cut(green_trades['setup_quality'], bins=[0, 50, 70, 85, 100])
        quality_performance = green_trades.groupby(quality_bins)['pnl'].agg(['mean', 'count'])
        
        # Time of entry analysis
        entry_hour_performance = green_trades.groupby(green_trades['entry_time'].dt.hour)['pnl'].mean()
        
        return {
            'consecutive_red_days_impact': red_days_performance.to_dict(),
            'setup_quality_performance': quality_performance.to_dict(),
            'best_entry_hours': entry_hour_performance.sort_values(ascending=False).head(3).to_dict(),
            'optimal_red_days': red_days_performance['mean'].idxmax()
        }
```

Esta documentación proporciona un framework completo para implementar estrategias First Green Day y First Red Day, con énfasis en la gestión de riesgo y análisis de performance específico para cada patrón.