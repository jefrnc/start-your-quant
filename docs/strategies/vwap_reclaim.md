# VWAP Reclaim Strategy

## Concepto Base

La estrategia VWAP Reclaim se basa en el comportamiento predecible de small caps que han estado traded por debajo del VWAP (Volume Weighted Average Price) y logran reclamarlo con volumen significativo. Este reclaim suele indicar un cambio de momentum institucional.

> **⚠️ DISCLAIMER**: Small caps requieren experiencia avanzada. Esta estrategia involucra timing preciso y gestión de riesgo estricta. Solo para cuentas con $25k+ y conocimiento sólido de market microstructure.

## Fundamentos Teóricos

### ¿Por Qué Funciona?

1. **Algoritmos Institucionales**: Muchos algos usan VWAP como benchmark
2. **Psychological Support**: Traders retail respetan el VWAP como soporte/resistencia  
3. **Volume Confirmation**: El volumen confirma la legitimidad del movimiento
4. **Small Cap Momentum**: En small caps, el momentum tiende a persistir más tiempo

### Anatomía del VWAP Reclaim
```python
class VWAPReclaimPattern:
    def __init__(self):
        self.phases = {
            'accumulation_below': {
                'duration_minutes': (30, 120),
                'price_action': 'Consolidación debajo VWAP',
                'volume_pattern': 'Volumen decreciente',
                'characteristics': 'Compression, low volatility'
            },
            'reclaim_attempt': {
                'duration_minutes': (5, 30),
                'price_action': 'Push through VWAP',
                'volume_pattern': 'Volume spike 2-5x',
                'characteristics': 'Decisive move con volume'
            },
            'confirmation': {
                'duration_minutes': (15, 60),
                'price_action': 'Hold above VWAP',
                'volume_pattern': 'Sustained volume',
                'characteristics': 'Higher lows, buying interest'
            },
            'continuation': {
                'duration_minutes': (60, 240),
                'price_action': 'Trend continuation',
                'volume_pattern': 'Volume on moves up',
                'characteristics': 'VWAP as support, momentum'
            }
        }
```

## Screening Criteria

### Filtros Primarios
```python
def vwap_reclaim_screener(market_data):
    """Screen para VWAP Reclaim opportunities"""
    
    primary_filters = {
        'price_range': (2.00, 50.00),  # Avoid penny stocks
        'avg_volume_20d': 1_000_000,   # Minimum liquidity
        'market_cap': (50_000_000, 2_000_000_000),  # Small to mid cap
        'float_shares': (5_000_000, 100_000_000),
        'time_below_vwap': 30,  # Al menos 30 min bajo VWAP
    }
    
    # Technical filters
    technical_filters = {
        'currently_below_vwap': True,
        'distance_from_vwap': (-0.03, -0.001),  # 0.1% a 3% below
        'volume_last_5min': lambda v: v > market_data['avg_volume_5min'] * 2,
        'price_compression': True,  # ATR decreasing
        'no_major_resistance_above': True
    }
    
    return primary_filters, technical_filters

def calculate_compression_score(price_data, periods=20):
    """Calcular score de compression"""
    atr_current = calculate_atr(price_data, 5)
    atr_baseline = calculate_atr(price_data, periods)
    
    compression_ratio = atr_current / atr_baseline if atr_baseline > 0 else 1
    
    # Score: 0-100, donde 100 = máxima compression
    compression_score = max(0, (1 - compression_ratio) * 100)
    
    return {
        'compression_score': compression_score,
        'current_atr': atr_current,
        'baseline_atr': atr_baseline,
        'is_compressed': compression_score > 60
    }
```

### Setup Quality Score
```python
class VWAPReclaimScorer:
    def __init__(self):
        self.weights = {
            'compression': 0.25,
            'volume_profile': 0.25,
            'price_structure': 0.20,
            'momentum': 0.15,
            'market_context': 0.15
        }
    
    def score_setup(self, stock_data):
        """Score del setup (0-100)"""
        
        # 1. Compression Score (25%)
        compression = calculate_compression_score(stock_data.price_history)
        compression_score = compression['compression_score']
        
        # 2. Volume Profile Score (25%)
        volume_score = self.score_volume_profile(stock_data)
        
        # 3. Price Structure Score (20%)
        structure_score = self.score_price_structure(stock_data)
        
        # 4. Momentum Score (15%)
        momentum_score = self.score_momentum(stock_data)
        
        # 5. Market Context Score (15%)
        context_score = self.score_market_context(stock_data)
        
        # Weighted total
        total_score = (
            compression_score * self.weights['compression'] +
            volume_score * self.weights['volume_profile'] +
            structure_score * self.weights['price_structure'] +
            momentum_score * self.weights['momentum'] +
            context_score * self.weights['market_context']
        )
        
        return {
            'total_score': total_score,
            'components': {
                'compression': compression_score,
                'volume': volume_score,
                'structure': structure_score,
                'momentum': momentum_score,
                'context': context_score
            },
            'grade': self.assign_grade(total_score),
            'tradeable': total_score >= 65
        }
    
    def score_volume_profile(self, stock_data):
        """Score del perfil de volumen"""
        score = 0
        
        # Volume during compression
        if stock_data.volume_during_compression < stock_data.avg_volume * 0.8:
            score += 30  # Low volume durante compression es bueno
        
        # Recent volume spike
        recent_volume_ratio = stock_data.current_volume / stock_data.avg_volume_5min
        if recent_volume_ratio > 3:
            score += 40
        elif recent_volume_ratio > 2:
            score += 25
        
        # Volume trend
        if stock_data.volume_increasing_trend:
            score += 20
        
        # VWAP volume quality
        if stock_data.volume_at_vwap_test > stock_data.avg_volume * 1.5:
            score += 10
        
        return min(score, 100)
    
    def score_price_structure(self, stock_data):
        """Score de estructura de precio"""
        score = 0
        
        # Higher lows pattern
        if stock_data.has_higher_lows:
            score += 25
        
        # Distance from key levels
        distance_from_vwap = abs(stock_data.price - stock_data.vwap) / stock_data.vwap
        if distance_from_vwap < 0.02:  # Very close to VWAP
            score += 20
        elif distance_from_vwap < 0.05:
            score += 10
        
        # Support levels below
        if stock_data.has_clean_support_below:
            score += 15
        
        # Resistance clearing potential
        if stock_data.clear_path_above_vwap:
            score += 25
        
        # Flag/pennant pattern
        if stock_data.has_flag_pattern:
            score += 15
        
        return min(score, 100)
```

## Entry Strategies

### 1. Conservative Entry
```python
class ConservativeVWAPEntry:
    def __init__(self):
        self.entry_type = "conservative"
        self.risk_tolerance = "low"
        
    def calculate_entry_levels(self, stock_data):
        """Calcular niveles de entrada conservadores"""
        
        vwap = stock_data.vwap
        current_price = stock_data.price
        
        return {
            'primary_entry': {
                'price': vwap + 0.01,  # Penny above VWAP
                'size_percentage': 0.60,
                'trigger': 'confirmed_break_above_vwap',
                'timeout_minutes': 15
            },
            'secondary_entry': {
                'price': vwap * 1.005,  # 0.5% above VWAP
                'size_percentage': 0.40,
                'trigger': 'sustained_hold_above_vwap',
                'timeout_minutes': 30
            },
            'stop_loss': max(
                vwap * 0.985,  # 1.5% below VWAP
                stock_data.previous_low * 0.99
            ),
            'targets': {
                'target_1': vwap * 1.08,   # 8% above VWAP
                'target_2': vwap * 1.15,   # 15% above VWAP
                'target_3': stock_data.day_high * 1.02  # Previous high break
            },
            'time_stop': '15:30'  # Exit before close if no progress
        }

    def entry_confirmation_signals(self, real_time_data):
        """Señales de confirmación para entry"""
        signals = []
        
        # Volume confirmation
        if real_time_data.volume_last_5min > real_time_data.avg_volume_5min * 2:
            signals.append('volume_confirmation')
        
        # Price action confirmation
        if real_time_data.price > real_time_data.vwap:
            if real_time_data.consecutive_green_minutes >= 2:
                signals.append('price_momentum')
        
        # Bid/ask spread tightening
        if real_time_data.bid_ask_spread < real_time_data.avg_spread * 0.8:
            signals.append('spread_tightening')
        
        # Level 2 strength
        if real_time_data.bid_size > real_time_data.ask_size * 1.5:
            signals.append('bid_strength')
        
        return {
            'signals': signals,
            'confidence': len(signals) / 4,  # 0-1 scale
            'entry_recommended': len(signals) >= 2
        }
```

### 2. Aggressive Entry
```python
class AggressiveVWAPEntry:
    def __init__(self):
        self.entry_type = "aggressive"
        self.risk_tolerance = "medium"
        
    def calculate_entry_levels(self, stock_data):
        """Entrada agresiva en anticipación al reclaim"""
        
        vwap = stock_data.vwap
        current_price = stock_data.price
        
        return {
            'anticipation_entry': {
                'price': vwap * 0.998,  # Just below VWAP
                'size_percentage': 0.40,
                'trigger': 'approaching_vwap_with_volume',
                'risk_note': 'Higher risk - anticipating move'
            },
            'breakout_entry': {
                'price': vwap * 1.002,  # Slight premium for confirmation
                'size_percentage': 0.60,
                'trigger': 'confirmed_break_with_volume',
                'timeout_minutes': 10
            },
            'stop_loss': current_price * 0.96,  # 4% stop from current
            'targets': {
                'quick_target': vwap * 1.06,   # 6% quick target
                'extended_target': vwap * 1.12  # 12% extended
            },
            'time_management': {
                'max_hold_time': '2 hours',
                'profit_taking_time': '14:00',
                'forced_exit_time': '15:45'
            }
        }
```

## Position Management

### Dynamic Position Sizing
```python
class VWAPReclaimPositionManager:
    def __init__(self, account_size, risk_per_trade=0.02):
        self.account_size = account_size
        self.base_risk = risk_per_trade
        
    def calculate_position_size(self, stock_data, setup_score, entry_price, stop_price):
        """Calcular tamaño basado en múltiples factores"""
        
        # Base risk amount
        base_risk_amount = self.account_size * self.base_risk
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        # Base position size
        base_shares = int(base_risk_amount / risk_per_share)
        
        # Adjustments
        adjustments = self.calculate_size_adjustments(stock_data, setup_score)
        
        # Final position size
        final_shares = int(base_shares * adjustments['total_multiplier'])
        
        return {
            'shares': final_shares,
            'dollar_amount': final_shares * entry_price,
            'risk_amount': final_shares * risk_per_share,
            'risk_percentage': (final_shares * risk_per_share) / self.account_size,
            'adjustments': adjustments,
            'setup_score': setup_score
        }
    
    def calculate_size_adjustments(self, stock_data, setup_score):
        """Ajustes al tamaño base"""
        
        # Setup quality multiplier
        if setup_score >= 85:
            quality_mult = 1.3
        elif setup_score >= 75:
            quality_mult = 1.1
        elif setup_score >= 65:
            quality_mult = 1.0
        else:
            quality_mult = 0.7
        
        # Volatility adjustment
        atr_pct = stock_data.atr_14 / stock_data.price
        if atr_pct > 0.08:  # High volatility
            volatility_mult = 0.7
        elif atr_pct > 0.05:
            volatility_mult = 0.85
        else:
            volatility_mult = 1.0
        
        # Float adjustment
        if stock_data.float_shares < 10_000_000:  # Low float
            float_mult = 1.2
        elif stock_data.float_shares > 50_000_000:  # High float
            float_mult = 0.9
        else:
            float_mult = 1.0
        
        # Market time adjustment
        current_time = pd.Timestamp.now().time()
        if current_time < pd.Timestamp('10:30').time():  # Morning session
            time_mult = 1.1
        elif current_time > pd.Timestamp('15:00').time():  # Late session
            time_mult = 0.8
        else:
            time_mult = 1.0
        
        total_multiplier = quality_mult * volatility_mult * float_mult * time_mult
        
        # Cap adjustments
        total_multiplier = max(0.5, min(2.0, total_multiplier))
        
        return {
            'quality_multiplier': quality_mult,
            'volatility_multiplier': volatility_mult,
            'float_multiplier': float_mult,
            'time_multiplier': time_mult,
            'total_multiplier': total_multiplier
        }
```

### Profit Taking Strategy
```python
class VWAPProfitManager:
    def __init__(self):
        self.profit_levels = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
        self.position_reduction = [0.25, 0.25, 0.30, 0.20]  # How much to sell
        
    def calculate_profit_taking_plan(self, entry_price, position_size):
        """Plan de toma de ganancias escalonado"""
        
        plan = []
        remaining_shares = position_size
        
        for i, (profit_pct, reduction_pct) in enumerate(zip(self.profit_levels, self.position_reduction)):
            target_price = entry_price * (1 + profit_pct)
            shares_to_sell = int(position_size * reduction_pct)
            
            plan.append({
                'level': i + 1,
                'target_price': round(target_price, 2),
                'profit_percentage': profit_pct,
                'shares_to_sell': shares_to_sell,
                'remaining_shares': remaining_shares - shares_to_sell,
                'expected_profit': shares_to_sell * entry_price * profit_pct
            })
            
            remaining_shares -= shares_to_sell
        
        return plan
    
    def dynamic_profit_adjustment(self, current_price, entry_price, time_in_trade, volume_profile):
        """Ajustar profit taking basado en condiciones dinámicas"""
        
        current_profit = (current_price - entry_price) / entry_price
        
        adjustments = {
            'accelerate_taking': False,
            'hold_longer': False,
            'reason': []
        }
        
        # Time-based adjustments
        if time_in_trade > 120:  # 2 hours
            adjustments['accelerate_taking'] = True
            adjustments['reason'].append('long_time_in_trade')
        
        # Volume-based adjustments
        if volume_profile['decreasing'] and current_profit > 0.08:
            adjustments['accelerate_taking'] = True
            adjustments['reason'].append('volume_declining')
        
        # Strong momentum
        if volume_profile['increasing'] and current_profit > 0.05:
            adjustments['hold_longer'] = True
            adjustments['reason'].append('strong_momentum')
        
        # Market close approaching
        market_close_minutes = self.minutes_to_market_close()
        if market_close_minutes < 60:
            adjustments['accelerate_taking'] = True
            adjustments['reason'].append('market_close_approaching')
        
        return adjustments
```

## Risk Management Específico

### Stop Loss Dinámico
```python
class VWAPStopManager:
    def __init__(self):
        self.stop_types = ['fixed', 'trailing', 'time_based', 'technical']
        
    def calculate_stop_levels(self, entry_price, stock_data, strategy_type='conservative'):
        """Calcular niveles de stop multiples"""
        
        vwap = stock_data.vwap
        
        stops = {
            'initial_stop': entry_price * 0.96,  # 4% initial stop
            'vwap_stop': vwap * 0.985,  # 1.5% below VWAP
            'technical_stop': stock_data.support_level * 0.99,
            'time_stop': None,  # Will be calculated based on time
            'recommended_stop': None
        }
        
        # Determine recommended stop
        if strategy_type == 'conservative':
            stops['recommended_stop'] = max(stops['initial_stop'], stops['vwap_stop'])
        else:  # aggressive
            stops['recommended_stop'] = stops['initial_stop']
        
        return stops
    
    def manage_trailing_stop(self, current_price, entry_price, highest_price, current_stop):
        """Gestión de trailing stop"""
        
        current_profit = (current_price - entry_price) / entry_price
        
        # Start trailing after 3% profit
        if current_profit >= 0.03:
            # Trail at 50% of max profit
            max_profit = (highest_price - entry_price) / entry_price
            trailing_profit = max_profit * 0.5
            
            new_stop = entry_price * (1 + trailing_profit)
            
            # Only move stop up
            return max(current_stop, new_stop)
        
        return current_stop
    
    def check_emergency_exit_conditions(self, stock_data):
        """Condiciones para exit de emergencia"""
        
        emergency_conditions = []
        
        # Volume drying up dramatically
        if stock_data.current_volume < stock_data.avg_volume * 0.3:
            emergency_conditions.append('volume_collapse')
        
        # Market selling off
        if stock_data.spy_change < -0.02:  # SPY down 2%
            emergency_conditions.append('market_selloff')
        
        # Failed multiple VWAP tests
        if stock_data.vwap_rejections >= 3:
            emergency_conditions.append('multiple_vwap_failures')
        
        # News/halt risk
        if stock_data.halt_risk_detected:
            emergency_conditions.append('halt_risk')
        
        return {
            'emergency_exit_recommended': len(emergency_conditions) >= 2,
            'conditions': emergency_conditions,
            'urgency': 'high' if len(emergency_conditions) >= 3 else 'medium'
        }
```

## Backtesting Framework

### Historical Analysis
```python
class VWAPReclaimBacktest:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.results = []
        
    def run_backtest(self, stock_universe, entry_strategy='conservative'):
        """Ejecutar backtest de la estrategia"""
        
        for date in pd.date_range(self.start_date, self.end_date):
            daily_opportunities = self.scan_daily_opportunities(date, stock_universe)
            
            for opportunity in daily_opportunities:
                trade_result = self.simulate_trade(opportunity, entry_strategy)
                if trade_result:
                    self.results.append(trade_result)
        
        return self.analyze_results()
    
    def simulate_trade(self, opportunity, strategy_type):
        """Simular un trade individual"""
        
        entry_data = opportunity['entry_data']
        intraday_data = opportunity['intraday_data']
        
        # Calculate entry and exit levels
        if strategy_type == 'conservative':
            entry_manager = ConservativeVWAPEntry()
        else:
            entry_manager = AggressiveVWAPEntry()
        
        levels = entry_manager.calculate_entry_levels(entry_data)
        
        # Simulate execution
        trade_result = self.execute_simulated_trade(
            intraday_data, 
            levels,
            opportunity['symbol'],
            opportunity['date']
        )
        
        return trade_result
    
    def analyze_results(self):
        """Analizar resultados del backtest"""
        
        if not self.results:
            return {'error': 'No trades executed'}
        
        df = pd.DataFrame(self.results)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        # P&L metrics
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(df['cumulative_pnl'])
        sharpe_ratio = self.calculate_sharpe_ratio(df['pnl'])
        
        # Time-based analysis
        avg_hold_time = df['hold_time_minutes'].mean()
        
        return {
            'summary': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_hold_time_minutes': avg_hold_time
            },
            'monthly_performance': self.calculate_monthly_performance(df),
            'setup_score_analysis': self.analyze_by_setup_score(df),
            'time_of_day_analysis': self.analyze_by_time_of_day(df)
        }
```

Esta estrategia VWAP Reclaim ofrece una aproximación sistemática para aprovechar los movimientos predictivos cuando small caps reclaiman su VWAP con volumen confirmatorio.