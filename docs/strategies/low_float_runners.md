# Low Float Runners Strategy

## Concepto Base

La estrategia Low Float Runners se enfoca en stocks con float extremadamente bajo (típicamente <15M shares) que experimentan momentum buying significativo. Estos stocks pueden tener movimientos explosivos debido a la limitada supply de shares disponibles para trading.

## Fundamentos de Low Float

### ¿Por Qué Low Float Causa Explosive Moves?

```python
class LowFloatMechanics:
    def __init__(self):
        self.float_categories = {
            'micro_float': {'range': (1_000_000, 5_000_000), 'volatility': 'extreme'},
            'low_float': {'range': (5_000_000, 15_000_000), 'volatility': 'high'},
            'small_float': {'range': (15_000_000, 30_000_000), 'volatility': 'medium'}
        }
        
    def calculate_supply_demand_dynamics(self, float_shares, daily_volume):
        """Calcular dynamics de supply/demand"""
        
        # Institutional lockup estimate (typically 60-80% of float)
        locked_shares = float_shares * 0.70
        tradeable_float = float_shares - locked_shares
        
        # Daily turnover rate
        turnover_rate = daily_volume / tradeable_float if tradeable_float > 0 else 0
        
        # Scarcity factor
        scarcity_score = min(100, (1 / (tradeable_float / 1_000_000)) * 20)
        
        return {
            'tradeable_float': tradeable_float,
            'daily_turnover_rate': turnover_rate,
            'scarcity_score': scarcity_score,
            'explosive_potential': turnover_rate > 0.5 and scarcity_score > 60,
            'risk_level': 'extreme' if scarcity_score > 80 else 'high'
        }
        
    def estimate_price_impact(self, order_size, tradeable_float, current_price):
        """Estimar impacto en precio de orden grande"""
        
        # Simplified market impact model
        float_ratio = order_size / tradeable_float
        
        # Price impact increases exponentially with order size
        base_impact = float_ratio * 0.1  # 10% impact per 1% of float
        explosive_multiplier = 1 + (float_ratio * 2)  # Exponential component
        
        estimated_impact = base_impact * explosive_multiplier
        
        return {
            'estimated_price_impact_pct': estimated_impact,
            'new_price_estimate': current_price * (1 + estimated_impact),
            'market_impact_category': self.categorize_impact(estimated_impact)
        }
    
    def categorize_impact(self, impact_pct):
        """Categorizar impacto en mercado"""
        if impact_pct > 0.20:
            return 'explosive'
        elif impact_pct > 0.10:
            return 'high'
        elif impact_pct > 0.05:
            return 'medium'
        else:
            return 'low'
```

## Screening System

### Multi-Tier Filtering
```python
class LowFloatScreener:
    def __init__(self):
        self.screening_tiers = {
            'tier_1_universe': self.tier_1_screening,    # 8000 → 200 stocks
            'tier_2_activity': self.tier_2_screening,    # 200 → 50 stocks  
            'tier_3_momentum': self.tier_3_screening,    # 50 → 10 stocks
            'tier_4_execution': self.tier_4_screening    # 10 → 3-5 stocks
        }
    
    def tier_1_screening(self, stock_universe):
        """Filtro inicial - características básicas"""
        
        filtered = []
        
        for stock in stock_universe:
            # Basic float requirements
            if not (1_000_000 <= stock.float_shares <= 15_000_000):
                continue
                
            # Price range for momentum potential
            if not (3.00 <= stock.price <= 50.00):
                continue
                
            # Minimum liquidity
            if stock.avg_volume_20d < 500_000:
                continue
                
            # Market cap range
            if not (10_000_000 <= stock.market_cap <= 500_000_000):
                continue
                
            # Exchange quality
            if stock.exchange not in ['NASDAQ', 'NYSE']:
                continue
                
            # Basic financial health
            if stock.cash_per_share < 0.10:  # Some cash cushion
                continue
                
            filtered.append(stock)
        
        return filtered
    
    def tier_2_screening(self, stocks):
        """Filtro de actividad - anomalías de volumen/precio"""
        
        filtered = []
        
        for stock in stocks:
            activity_metrics = self.calculate_activity_metrics(stock)
            
            # Volume anomaly
            if activity_metrics['volume_ratio'] < 2.0:  # At least 2x normal volume
                continue
                
            # Price movement
            if activity_metrics['price_change_abs'] < 0.15:  # At least 15% move
                continue
                
            # Intraday range
            if activity_metrics['daily_range'] < 0.20:  # At least 20% range
                continue
                
            # Relative strength
            if activity_metrics['relative_strength'] < 60:  # vs SPY
                continue
            
            stock.activity_score = self.calculate_activity_score(activity_metrics)
            
            if stock.activity_score >= 70:
                filtered.append(stock)
        
        return sorted(filtered, key=lambda x: x.activity_score, reverse=True)
    
    def tier_3_screening(self, stocks):
        """Filtro de momentum - calidad del setup"""
        
        filtered = []
        
        for stock in stocks:
            momentum_analysis = self.analyze_momentum_quality(stock)
            
            # Momentum persistence
            if not momentum_analysis['has_persistent_momentum']:
                continue
                
            # Technical setup
            if momentum_analysis['technical_score'] < 65:
                continue
                
            # Catalyst verification
            if not momentum_analysis['has_legitimate_catalyst']:
                continue
                
            # Risk/reward profile
            if momentum_analysis['risk_reward_ratio'] < 2.0:
                continue
            
            stock.momentum_score = momentum_analysis['total_score']
            stock.risk_reward = momentum_analysis['risk_reward_ratio']
            
            filtered.append(stock)
        
        return filtered
    
    def calculate_activity_metrics(self, stock):
        """Calcular métricas de actividad"""
        
        volume_ratio = stock.volume_today / stock.avg_volume_20d
        price_change = (stock.price - stock.prev_close) / stock.prev_close
        daily_range = (stock.high - stock.low) / stock.low
        
        # Relative strength vs market
        spy_change = self.get_spy_performance()
        relative_strength = ((price_change - spy_change) + 1) * 50  # Normalized 0-100
        
        return {
            'volume_ratio': volume_ratio,
            'price_change': price_change,
            'price_change_abs': abs(price_change),
            'daily_range': daily_range,
            'relative_strength': relative_strength
        }
```

### Momentum Quality Analysis
```python
class MomentumQualityAnalyzer:
    def __init__(self):
        self.quality_factors = [
            'volume_profile',
            'price_structure', 
            'catalyst_strength',
            'technical_setup',
            'risk_reward'
        ]
    
    def analyze_momentum_quality(self, stock_data):
        """Análisis completo de calidad del momentum"""
        
        # 1. Volume Profile Analysis
        volume_analysis = self.analyze_volume_profile(stock_data)
        
        # 2. Price Structure
        price_structure = self.analyze_price_structure(stock_data)
        
        # 3. Catalyst Assessment
        catalyst_analysis = self.assess_catalyst_strength(stock_data)
        
        # 4. Technical Setup
        technical_analysis = self.analyze_technical_setup(stock_data)
        
        # 5. Risk/Reward Calculation
        risk_reward = self.calculate_risk_reward(stock_data)
        
        # Composite scoring
        total_score = self.calculate_composite_score({
            'volume': volume_analysis['score'],
            'price_structure': price_structure['score'],
            'catalyst': catalyst_analysis['score'],
            'technical': technical_analysis['score'],
            'risk_reward': min(100, risk_reward * 25)  # Cap at 100
        })
        
        return {
            'has_persistent_momentum': volume_analysis['persistent'] and price_structure['strong'],
            'has_legitimate_catalyst': catalyst_analysis['legitimate'],
            'technical_score': technical_analysis['score'],
            'risk_reward_ratio': risk_reward,
            'total_score': total_score,
            'components': {
                'volume': volume_analysis,
                'price_structure': price_structure,
                'catalyst': catalyst_analysis,
                'technical': technical_analysis
            }
        }
    
    def analyze_volume_profile(self, stock_data):
        """Analizar perfil de volumen"""
        
        score = 0
        
        # Volume surge magnitude
        volume_ratio = stock_data.volume_today / stock_data.avg_volume_20d
        if volume_ratio > 10:
            score += 30
        elif volume_ratio > 5:
            score += 20
        elif volume_ratio > 3:
            score += 10
        
        # Volume trend
        if stock_data.volume_increasing_trend:
            score += 20
        
        # Volume at key levels
        if stock_data.volume_at_breakout > stock_data.avg_volume * 2:
            score += 25
        
        # Sustained volume
        if stock_data.avg_volume_last_hour > stock_data.avg_hourly_volume * 1.5:
            score += 15
        
        # Distribution signs
        if stock_data.has_distribution_volume:
            score -= 20
        
        persistent = score >= 60 and volume_ratio >= 3
        
        return {
            'score': min(100, score),
            'persistent': persistent,
            'volume_ratio': volume_ratio,
            'trend': 'increasing' if stock_data.volume_increasing_trend else 'decreasing'
        }
    
    def analyze_price_structure(self, stock_data):
        """Analizar estructura de precio"""
        
        score = 0
        
        # Higher highs and higher lows
        if stock_data.has_higher_highs:
            score += 25
        if stock_data.has_higher_lows:
            score += 25
        
        # Breakout quality
        if stock_data.clean_breakout:
            score += 20
        
        # Support/resistance clarity
        if stock_data.clear_support_resistance:
            score += 15
        
        # Momentum acceleration
        if stock_data.accelerating_momentum:
            score += 15
        
        strong = score >= 70
        
        return {
            'score': min(100, score),
            'strong': strong,
            'pattern': self.identify_price_pattern(stock_data),
            'breakout_quality': 'clean' if stock_data.clean_breakout else 'messy'
        }
```

## Entry Strategies

### Progressive Entry System
```python
class LowFloatEntryManager:
    def __init__(self, risk_tolerance='medium'):
        self.risk_tolerance = risk_tolerance
        self.entry_approaches = {
            'breakout': self.breakout_entry,
            'pullback': self.pullback_entry,
            'momentum': self.momentum_entry,
            'scalp': self.scalp_entry
        }
    
    def determine_entry_approach(self, stock_data, momentum_analysis):
        """Determinar mejor approach de entrada"""
        
        # Factor analysis
        volume_ratio = stock_data.volume_today / stock_data.avg_volume_20d
        price_momentum = stock_data.price_momentum_score
        volatility = stock_data.atr_14 / stock_data.price
        
        # Decision matrix
        if (volume_ratio > 10 and price_momentum > 80 and 
            stock_data.at_resistance):
            return 'breakout'
        elif (stock_data.pullback_to_support and 
              momentum_analysis['total_score'] > 75):
            return 'pullback'
        elif (volume_ratio > 5 and price_momentum > 70):
            return 'momentum'
        else:
            return 'scalp'
    
    def breakout_entry(self, stock_data, momentum_analysis):
        """Entrada en breakout de resistencia"""
        
        resistance_level = stock_data.resistance_level
        current_price = stock_data.price
        
        return {
            'strategy_type': 'breakout',
            'entry_levels': {
                'aggressive': {
                    'price': resistance_level * 1.005,  # Just above resistance
                    'size_pct': 0.40,
                    'trigger': 'resistance_break',
                    'confirmation': 'volume_surge'
                },
                'conservative': {
                    'price': resistance_level * 1.02,   # Clear break
                    'size_pct': 0.60,
                    'trigger': 'confirmed_breakout',
                    'confirmation': 'sustained_volume'
                }
            },
            'stop_loss': resistance_level * 0.97,  # 3% below resistance
            'targets': {
                'target_1': resistance_level * 1.15,  # 15% above breakout
                'target_2': resistance_level * 1.30,  # 30% extended move
                'measured_move': self.calculate_measured_move(stock_data)
            },
            'risk_management': {
                'max_hold_time': '4_hours',
                'profit_take_schedule': [0.25, 0.35, 0.40],  # % of position
                'profit_levels': [0.08, 0.15, 0.25]           # % profit levels
            }
        }
    
    def pullback_entry(self, stock_data, momentum_analysis):
        """Entrada en pullback a soporte"""
        
        support_level = stock_data.support_level
        vwap = stock_data.vwap
        
        return {
            'strategy_type': 'pullback',
            'entry_levels': {
                'support_test': {
                    'price': support_level * 1.01,
                    'size_pct': 0.50,
                    'trigger': 'support_hold',
                    'confirmation': 'buying_volume'
                },
                'vwap_reclaim': {
                    'price': vwap * 1.005,
                    'size_pct': 0.50,
                    'trigger': 'vwap_reclaim',
                    'confirmation': 'momentum_resumption'
                }
            },
            'stop_loss': support_level * 0.95,  # 5% below support
            'targets': {
                'previous_high': stock_data.recent_high,
                'extension_target': stock_data.recent_high * 1.10
            },
            'time_limits': {
                'entry_window': '2_hours',
                'max_consolidation': '1_hour'
            }
        }
    
    def momentum_entry(self, stock_data, momentum_analysis):
        """Entrada en momentum continuation"""
        
        current_price = stock_data.price
        
        return {
            'strategy_type': 'momentum',
            'entry_approach': 'chase_with_discipline',
            'entry_levels': {
                'immediate': {
                    'price': current_price * 1.01,  # 1% premium
                    'size_pct': 0.30,
                    'trigger': 'market_order',
                    'max_slippage': 0.02
                },
                'strength_add': {
                    'price': current_price * 1.05,  # Add on strength
                    'size_pct': 0.40,
                    'trigger': 'momentum_acceleration',
                    'volume_requirement': '3x_spike'
                },
                'final_add': {
                    'price': current_price * 1.08,  # Final add
                    'size_pct': 0.30,
                    'trigger': 'explosive_move',
                    'max_chase_level': current_price * 1.10
                }
            },
            'stop_loss': current_price * 0.92,  # 8% stop
            'profit_strategy': 'quick_scalp',
            'targets': {
                'quick_target': current_price * 1.12,  # 12% quick
                'extended_target': current_price * 1.25 # 25% if momentum continues
            }
        }
```

## Risk Management Extremo

### Position Sizing para High Volatility
```python
class LowFloatRiskManager:
    def __init__(self, account_size, max_portfolio_exposure=0.15):
        self.account_size = account_size
        self.max_portfolio_exposure = max_portfolio_exposure
        self.low_float_max_exposure = 0.08  # Never more than 8% in one low float
        
    def calculate_low_float_position_size(self, stock_data, entry_plan, momentum_score):
        """Cálculo especializado para low float"""
        
        # Base calculations
        entry_price = entry_plan['entry_price']
        stop_price = entry_plan['stop_loss']
        risk_per_share = abs(entry_price - stop_price)
        
        # Volatility adjustment
        volatility_factor = self.calculate_volatility_factor(stock_data)
        
        # Float size adjustment
        float_factor = self.calculate_float_factor(stock_data.float_shares)
        
        # Momentum quality adjustment
        momentum_factor = self.calculate_momentum_factor(momentum_score)
        
        # Base risk amount
        base_risk = self.account_size * 0.02  # 2% base risk
        
        # Adjusted risk
        adjusted_risk = (base_risk * volatility_factor * 
                        float_factor * momentum_factor)
        
        # Cap at low float maximum
        max_position_value = self.account_size * self.low_float_max_exposure
        max_shares_by_exposure = int(max_position_value / entry_price)
        
        # Calculate shares
        shares_by_risk = int(adjusted_risk / risk_per_share) if risk_per_share > 0 else 0
        final_shares = min(shares_by_risk, max_shares_by_exposure)
        
        return {
            'shares': final_shares,
            'position_value': final_shares * entry_price,
            'risk_amount': final_shares * risk_per_share,
            'portfolio_exposure_pct': (final_shares * entry_price) / self.account_size,
            'adjustments': {
                'volatility_factor': volatility_factor,
                'float_factor': float_factor,
                'momentum_factor': momentum_factor
            },
            'risk_level': self.assess_risk_level(stock_data, final_shares * entry_price)
        }
    
    def calculate_volatility_factor(self, stock_data):
        """Factor de ajuste por volatilidad"""
        
        atr_pct = stock_data.atr_14 / stock_data.price
        
        if atr_pct > 0.20:      # >20% daily ATR
            return 0.4          # Reduce size dramatically
        elif atr_pct > 0.15:    # >15% daily ATR
            return 0.6
        elif atr_pct > 0.10:    # >10% daily ATR
            return 0.8
        else:
            return 1.0
    
    def calculate_float_factor(self, float_shares):
        """Factor de ajuste por float size"""
        
        if float_shares < 3_000_000:      # Micro float
            return 0.5                    # Very risky
        elif float_shares < 8_000_000:    # Low float
            return 0.75
        elif float_shares < 15_000_000:   # Small float
            return 1.0
        else:
            return 1.2                    # Larger float = less risk
```

### Dynamic Stop Management
```python
class LowFloatStopManager:
    def __init__(self):
        self.stop_strategies = {
            'breakout': self.breakout_stops,
            'pullback': self.pullback_stops,
            'momentum': self.momentum_stops
        }
    
    def manage_stops_dynamically(self, position, current_data, strategy_type):
        """Gestión dinámica de stops para low float"""
        
        stop_strategy = self.stop_strategies.get(strategy_type, self.default_stops)
        return stop_strategy(position, current_data)
    
    def breakout_stops(self, position, current_data):
        """Stops específicos para breakout trades"""
        
        entry_price = position['entry_price']
        current_price = current_data['price']
        breakout_level = position['breakout_level']
        
        # Initial stop below breakout level
        initial_stop = breakout_level * 0.97
        
        # Trailing stop logic
        current_profit = (current_price - entry_price) / entry_price
        
        if current_profit > 0.15:  # 15% profit
            # Trail at 50% of max profit
            max_profit = (position['highest_price'] - entry_price) / entry_price
            trailing_profit = max_profit * 0.5
            trailing_stop = entry_price * (1 + trailing_profit)
            
            return {
                'stop_price': max(initial_stop, trailing_stop),
                'stop_type': 'trailing',
                'trigger_reason': 'trailing_profit'
            }
        
        elif current_profit > 0.08:  # 8% profit
            # Move stop to breakeven
            return {
                'stop_price': entry_price * 1.01,  # Slight profit
                'stop_type': 'breakeven_plus',
                'trigger_reason': 'secure_profit'
            }
        
        else:
            return {
                'stop_price': initial_stop,
                'stop_type': 'initial',
                'trigger_reason': 'risk_management'
            }
    
    def momentum_stops(self, position, current_data):
        """Stops para momentum trades - más agresivos"""
        
        entry_price = position['entry_price']
        current_price = current_data['price']
        
        # Tighter stops for momentum chasing
        initial_stop = entry_price * 0.92  # 8% stop
        
        # Quick profit protection
        current_profit = (current_price - entry_price) / entry_price
        
        if current_profit > 0.10:  # 10% profit
            # Lock in 5% profit
            return {
                'stop_price': entry_price * 1.05,
                'stop_type': 'profit_protection',
                'trigger_reason': 'momentum_profit_protection'
            }
        
        elif current_profit > 0.05:  # 5% profit
            # Move to breakeven
            return {
                'stop_price': entry_price * 1.01,
                'stop_type': 'breakeven',
                'trigger_reason': 'quick_momentum_profit'
            }
        
        else:
            return {
                'stop_price': initial_stop,
                'stop_type': 'initial_tight',
                'trigger_reason': 'momentum_risk_control'
            }
    
    def check_emergency_exit_conditions(self, position, current_data):
        """Condiciones de exit de emergencia"""
        
        emergency_conditions = []
        
        # Volume collapse
        if current_data['volume_ratio'] < 0.5:  # Volume fell below 50% of average
            emergency_conditions.append({
                'condition': 'volume_collapse',
                'severity': 'high',
                'action': 'immediate_exit'
            })
        
        # Extreme volatility spike
        current_volatility = current_data['current_atr'] / current_data['price']
        if current_volatility > 0.30:  # 30% intraday ATR
            emergency_conditions.append({
                'condition': 'extreme_volatility',
                'severity': 'high',
                'action': 'reduce_position_50pct'
            })
        
        # Multiple failed attempts at key level
        if current_data.get('failed_attempts_count', 0) >= 3:
            emergency_conditions.append({
                'condition': 'multiple_rejections',
                'severity': 'medium',
                'action': 'consider_exit'
            })
        
        # Time-based exit (low float trades should be quick)
        time_in_trade = current_data['current_time'] - position['entry_time']
        if time_in_trade.seconds > 14400:  # 4 hours
            emergency_conditions.append({
                'condition': 'time_limit_exceeded',
                'severity': 'medium',
                'action': 'exit_before_close'
            })
        
        return emergency_conditions
```

## Profit Taking Strategy

### Staged Profit Taking
```python
class LowFloatProfitManager:
    def __init__(self):
        self.profit_stages = {
            'quick_scalp': [0.05, 0.08, 0.12],      # 5%, 8%, 12%
            'momentum_ride': [0.08, 0.15, 0.25],    # 8%, 15%, 25%
            'breakout_play': [0.10, 0.20, 0.35]     # 10%, 20%, 35%
        }
        
        self.position_reduction = {
            'quick_scalp': [0.50, 0.30, 0.20],      # Reduce position %
            'momentum_ride': [0.30, 0.40, 0.30],
            'breakout_play': [0.25, 0.35, 0.40]
        }
    
    def create_profit_plan(self, entry_price, position_size, strategy_type):
        """Crear plan de toma de ganancias"""
        
        profit_levels = self.profit_stages.get(strategy_type, self.profit_stages['momentum_ride'])
        reduction_schedule = self.position_reduction.get(strategy_type, self.position_reduction['momentum_ride'])
        
        plan = []
        remaining_shares = position_size
        
        for i, (profit_pct, reduction_pct) in enumerate(zip(profit_levels, reduction_schedule)):
            target_price = entry_price * (1 + profit_pct)
            shares_to_sell = int(position_size * reduction_pct)
            
            plan.append({
                'stage': i + 1,
                'target_price': round(target_price, 2),
                'profit_percentage': profit_pct,
                'shares_to_sell': shares_to_sell,
                'remaining_shares': remaining_shares - shares_to_sell,
                'profit_amount': shares_to_sell * entry_price * profit_pct,
                'execution_style': 'limit_order',
                'urgency': 'high' if i == 0 else 'medium'
            })
            
            remaining_shares -= shares_to_sell
        
        return {
            'profit_plan': plan,
            'strategy_type': strategy_type,
            'total_stages': len(plan),
            'final_remaining_shares': remaining_shares
        }
    
    def dynamic_profit_adjustment(self, current_position, market_conditions):
        """Ajustar profit taking basado en condiciones dinámicas"""
        
        adjustments = []
        
        # Market close approaching
        if market_conditions['minutes_to_close'] < 60:
            adjustments.append({
                'action': 'accelerate_exit',
                'reason': 'market_close',
                'adjustment': 'exit_75pct_remaining'
            })
        
        # Volume declining
        if market_conditions['volume_trend'] == 'declining':
            adjustments.append({
                'action': 'take_profits_faster',
                'reason': 'volume_declining',
                'adjustment': 'reduce_profit_targets_10pct'
            })
        
        # Extreme volatility
        if market_conditions['volatility_spike']:
            adjustments.append({
                'action': 'lock_profits',
                'reason': 'volatility_spike',
                'adjustment': 'immediate_partial_exit_50pct'
            })
        
        # Strong momentum continuation
        if (market_conditions['momentum_score'] > 85 and 
            market_conditions['volume_increasing']):
            adjustments.append({
                'action': 'extend_targets',
                'reason': 'strong_momentum',
                'adjustment': 'increase_targets_20pct'
            })
        
        return adjustments
```

La estrategia Low Float Runners requiere extrema disciplina en risk management debido a la naturaleza explosiva y volátil de estos stocks. La clave está en entries precisos, stops ajustados, y profit taking agresivo.