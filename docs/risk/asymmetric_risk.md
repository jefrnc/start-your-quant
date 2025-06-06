# Riesgo Asimétrico

## Apostar Más Cuando las Probabilidades Están a Tu Favor

El riesgo asimétrico es el secreto de los traders exitosos: arriesgar menos en setups mediocres y más en setups de alta probabilidad. No todos los trades son iguales, y tu position sizing debería reflejarlo.

## El Concepto de Riesgo Asimétrico

### ¿Qué es?
```python
# ❌ SIMÉTRICO: Mismo riesgo en todos los trades
def symmetric_risk_sizing(account_value):
    return account_value * 0.02  # Siempre 2%

# ✅ ASIMÉTRICO: Riesgo variable según calidad del setup
def asymmetric_risk_sizing(account_value, setup_quality):
    base_risk = 0.02
    
    if setup_quality == 'A+':
        return account_value * (base_risk * 2.0)    # 4% en mejores setups
    elif setup_quality == 'A':
        return account_value * (base_risk * 1.5)    # 3% en buenos setups
    elif setup_quality == 'B':
        return account_value * (base_risk * 1.0)    # 2% en setups normales
    elif setup_quality == 'C':
        return account_value * (base_risk * 0.5)    # 1% en setups débiles
    else:
        return 0  # No tradear setups malos
```

### Por Qué Funciona
```python
# Ejemplo: 100 trades con diferentes calidades
trade_scenarios = {
    'symmetric': {
        'risk_per_trade': 0.02,
        'a_plus_setups': {'count': 10, 'win_rate': 0.8, 'avg_rr': 3.0},
        'a_setups': {'count': 20, 'win_rate': 0.7, 'avg_rr': 2.5},
        'b_setups': {'count': 40, 'win_rate': 0.6, 'avg_rr': 2.0},
        'c_setups': {'count': 30, 'win_rate': 0.4, 'avg_rr': 1.5}
    },
    'asymmetric': {
        'a_plus_setups': {'count': 10, 'risk': 0.04, 'win_rate': 0.8, 'avg_rr': 3.0},
        'a_setups': {'count': 20, 'risk': 0.03, 'win_rate': 0.7, 'avg_rr': 2.5},
        'b_setups': {'count': 40, 'risk': 0.02, 'win_rate': 0.6, 'avg_rr': 2.0},
        'c_setups': {'count': 0, 'risk': 0.0, 'win_rate': 0.0, 'avg_rr': 0.0}  # Skip C setups
    }
}

def calculate_expected_return(scenario):
    """Calcular retorno esperado de cada escenario"""
    total_return = 0
    
    for setup_type, stats in scenario.items():
        if setup_type == 'symmetric':
            continue
            
        if 'risk' in stats:
            risk = stats['risk']
        else:
            risk = scenario.get('risk_per_trade', 0.02)
        
        # Expected return por trade de este tipo
        win_amount = risk * stats['avg_rr']
        loss_amount = risk
        expected_per_trade = (stats['win_rate'] * win_amount) - ((1 - stats['win_rate']) * loss_amount)
        
        total_return += expected_per_trade * stats['count']
    
    return total_return

symmetric_return = calculate_expected_return(trade_scenarios['symmetric'])
asymmetric_return = calculate_expected_return(trade_scenarios['asymmetric'])

print(f"Symmetric approach: {symmetric_return:.2%} expected return")
print(f"Asymmetric approach: {asymmetric_return:.2%} expected return")
print(f"Improvement: {(asymmetric_return - symmetric_return):.2%}")
```

## Setup Quality Scoring

### 1. Multi-Factor Scoring System
```python
class SetupQualityScorer:
    def __init__(self):
        self.scoring_factors = {
            'technical': {
                'vwap_reclaim': {'weight': 0.2, 'max_score': 10},
                'volume_spike': {'weight': 0.25, 'max_score': 10},
                'price_action': {'weight': 0.15, 'max_score': 10},
                'support_resistance': {'weight': 0.15, 'max_score': 10},
                'trend_alignment': {'weight': 0.25, 'max_score': 10}
            },
            'fundamental': {
                'catalyst_strength': {'weight': 0.4, 'max_score': 10},
                'float_quality': {'weight': 0.3, 'max_score': 10},
                'sector_momentum': {'weight': 0.3, 'max_score': 10}
            },
            'market_context': {
                'market_direction': {'weight': 0.5, 'max_score': 10},
                'volatility_regime': {'weight': 0.3, 'max_score': 10},
                'time_of_day': {'weight': 0.2, 'max_score': 10}
            }
        }
    
    def score_technical_factors(self, data):
        """Score factores técnicos"""
        scores = {}
        
        # VWAP Reclaim
        if 'vwap_reclaim' in data and data['vwap_reclaim']:
            if data['close'] > data['vwap'] * 1.02:  # 2% above VWAP
                scores['vwap_reclaim'] = 10
            elif data['close'] > data['vwap'] * 1.01:  # 1% above VWAP
                scores['vwap_reclaim'] = 7
            else:
                scores['vwap_reclaim'] = 5
        else:
            scores['vwap_reclaim'] = 0
        
        # Volume Spike
        rvol = data.get('rvol', 1)
        if rvol >= 5:
            scores['volume_spike'] = 10
        elif rvol >= 3:
            scores['volume_spike'] = 8
        elif rvol >= 2:
            scores['volume_spike'] = 6
        elif rvol >= 1.5:
            scores['volume_spike'] = 4
        else:
            scores['volume_spike'] = 0
        
        # Price Action
        gap_pct = abs(data.get('gap_pct', 0))
        if 10 <= gap_pct <= 25:  # Sweet spot for gaps
            scores['price_action'] = 10
        elif 5 <= gap_pct < 10:
            scores['price_action'] = 7
        elif gap_pct < 5:
            scores['price_action'] = 5
        else:  # >25% gap is risky
            scores['price_action'] = 3
        
        # Support/Resistance
        distance_from_key_level = data.get('distance_from_key_level', float('inf'))
        if distance_from_key_level < 0.005:  # Within 0.5% of key level
            scores['support_resistance'] = 10
        elif distance_from_key_level < 0.01:  # Within 1%
            scores['support_resistance'] = 7
        else:
            scores['support_resistance'] = 3
        
        # Trend Alignment
        if data.get('trend_alignment', 0) >= 3:  # Multiple timeframes aligned
            scores['trend_alignment'] = 10
        elif data.get('trend_alignment', 0) >= 2:
            scores['trend_alignment'] = 7
        else:
            scores['trend_alignment'] = 3
        
        return scores
    
    def score_fundamental_factors(self, data):
        """Score factores fundamentales"""
        scores = {}
        
        # Catalyst Strength
        catalyst_type = data.get('catalyst_type', 'none')
        if catalyst_type in ['earnings_beat', 'fda_approval', 'major_contract']:
            scores['catalyst_strength'] = 10
        elif catalyst_type in ['analyst_upgrade', 'insider_buying']:
            scores['catalyst_strength'] = 7
        elif catalyst_type in ['technical_breakout']:
            scores['catalyst_strength'] = 5
        else:
            scores['catalyst_strength'] = 0
        
        # Float Quality
        float_shares = data.get('float_shares', float('inf'))
        if float_shares < 5_000_000:
            scores['float_quality'] = 10
        elif float_shares < 15_000_000:
            scores['float_quality'] = 8
        elif float_shares < 30_000_000:
            scores['float_quality'] = 6
        else:
            scores['float_quality'] = 3
        
        # Sector Momentum
        sector_performance = data.get('sector_performance_5d', 0)
        if sector_performance > 0.05:  # Sector up 5%+ in 5 days
            scores['sector_momentum'] = 10
        elif sector_performance > 0.02:
            scores['sector_momentum'] = 7
        elif sector_performance > -0.02:
            scores['sector_momentum'] = 5
        else:
            scores['sector_momentum'] = 2
        
        return scores
    
    def score_market_context(self, data):
        """Score contexto de mercado"""
        scores = {}
        
        # Market Direction
        spy_trend = data.get('spy_trend', 'neutral')
        if spy_trend == 'strong_uptrend':
            scores['market_direction'] = 10
        elif spy_trend == 'uptrend':
            scores['market_direction'] = 8
        elif spy_trend == 'neutral':
            scores['market_direction'] = 5
        elif spy_trend == 'downtrend':
            scores['market_direction'] = 3
        else:  # strong_downtrend
            scores['market_direction'] = 0
        
        # Volatility Regime
        vix_level = data.get('vix', 20)
        if 12 <= vix_level <= 20:  # Sweet spot
            scores['volatility_regime'] = 10
        elif 20 < vix_level <= 25:
            scores['volatility_regime'] = 7
        elif vix_level < 12:  # Too complacent
            scores['volatility_regime'] = 5
        else:  # VIX > 25
            scores['volatility_regime'] = 2
        
        # Time of Day
        current_time = pd.Timestamp.now().time()
        if pd.Timestamp('09:30').time() <= current_time <= pd.Timestamp('11:00').time():
            scores['time_of_day'] = 10  # Best time for momentum
        elif pd.Timestamp('14:00').time() <= current_time <= pd.Timestamp('15:30').time():
            scores['time_of_day'] = 8   # Good afternoon volume
        else:
            scores['time_of_day'] = 5
        
        return scores
    
    def calculate_overall_score(self, data):
        """Calcular score total del setup"""
        technical_scores = self.score_technical_factors(data)
        fundamental_scores = self.score_fundamental_factors(data)
        market_scores = self.score_market_context(data)
        
        # Calculate weighted scores
        technical_weighted = sum(
            score * self.scoring_factors['technical'][factor]['weight'] 
            for factor, score in technical_scores.items()
        )
        
        fundamental_weighted = sum(
            score * self.scoring_factors['fundamental'][factor]['weight'] 
            for factor, score in fundamental_scores.items()
        )
        
        market_weighted = sum(
            score * self.scoring_factors['market_context'][factor]['weight'] 
            for factor, score in market_scores.items()
        )
        
        # Combine scores (equal weight for now)
        total_score = (technical_weighted + fundamental_weighted + market_weighted) / 3
        
        # Convert to letter grade
        if total_score >= 8.5:
            grade = 'A+'
        elif total_score >= 7.5:
            grade = 'A'
        elif total_score >= 6.5:
            grade = 'B+'
        elif total_score >= 5.5:
            grade = 'B'
        elif total_score >= 4.5:
            grade = 'C+'
        elif total_score >= 3.5:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'total_score': total_score,
            'grade': grade,
            'technical_score': technical_weighted,
            'fundamental_score': fundamental_weighted,
            'market_score': market_weighted,
            'detailed_scores': {
                'technical': technical_scores,
                'fundamental': fundamental_scores,
                'market': market_scores
            }
        }
```

### 2. Historical Performance by Setup Quality
```python
class SetupPerformanceTracker:
    def __init__(self):
        self.trade_history = []
        
    def record_trade(self, setup_data, trade_result):
        """Registrar trade con setup quality y resultado"""
        scorer = SetupQualityScorer()
        quality_score = scorer.calculate_overall_score(setup_data)
        
        trade_record = {
            'timestamp': pd.Timestamp.now(),
            'ticker': setup_data['ticker'],
            'setup_grade': quality_score['grade'],
            'setup_score': quality_score['total_score'],
            'entry_price': trade_result['entry_price'],
            'exit_price': trade_result['exit_price'],
            'pnl': trade_result['pnl'],
            'pnl_pct': trade_result['pnl_pct'],
            'hold_time': trade_result['hold_time'],
            'was_winner': trade_result['pnl'] > 0,
            'setup_details': quality_score['detailed_scores']
        }
        
        self.trade_history.append(trade_record)
    
    def analyze_performance_by_grade(self):
        """Analizar performance por grado de setup"""
        if not self.trade_history:
            return {}
        
        df = pd.DataFrame(self.trade_history)
        performance_by_grade = {}
        
        for grade in df['setup_grade'].unique():
            grade_trades = df[df['setup_grade'] == grade]
            
            performance_by_grade[grade] = {
                'total_trades': len(grade_trades),
                'win_rate': grade_trades['was_winner'].mean(),
                'avg_pnl_pct': grade_trades['pnl_pct'].mean(),
                'avg_winner_pct': grade_trades[grade_trades['was_winner']]['pnl_pct'].mean() if grade_trades['was_winner'].any() else 0,
                'avg_loser_pct': grade_trades[~grade_trades['was_winner']]['pnl_pct'].mean() if (~grade_trades['was_winner']).any() else 0,
                'profit_factor': abs(grade_trades[grade_trades['was_winner']]['pnl'].sum() / 
                                   grade_trades[~grade_trades['was_winner']]['pnl'].sum()) if (~grade_trades['was_winner']).any() else float('inf'),
                'avg_hold_time': grade_trades['hold_time'].mean(),
                'expectancy': (
                    grade_trades['win_rate'].iloc[0] * grade_trades[grade_trades['was_winner']]['pnl_pct'].mean() +
                    (1 - grade_trades['win_rate'].iloc[0]) * grade_trades[~grade_trades['was_winner']]['pnl_pct'].mean()
                ) if len(grade_trades) > 0 else 0
            }
        
        return performance_by_grade
    
    def get_optimal_risk_allocation(self):
        """Calcular allocation óptimo basado en historical performance"""
        performance = self.analyze_performance_by_grade()
        
        # Calculate Kelly-inspired allocation
        allocations = {}
        
        for grade, stats in performance.items():
            if stats['total_trades'] < 5:  # Not enough data
                allocations[grade] = 0.01  # Minimal allocation
                continue
            
            win_rate = stats['win_rate']
            avg_win = stats['avg_winner_pct'] / 100 if stats['avg_winner_pct'] else 0
            avg_loss = abs(stats['avg_loser_pct'] / 100) if stats['avg_loser_pct'] else 0.02
            
            if avg_loss == 0:
                avg_loss = 0.02  # Default
            
            # Modified Kelly
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
            
            # Conservative fraction of Kelly
            conservative_allocation = max(0.005, min(0.05, kelly_fraction * 0.25))
            allocations[grade] = conservative_allocation
        
        return allocations
```

## Dynamic Risk Allocation

### 1. Confidence-Based Sizing
```python
class ConfidenceBasedSizer:
    def __init__(self, base_risk=0.02):
        self.base_risk = base_risk
        self.performance_tracker = SetupPerformanceTracker()
        
    def calculate_confidence_multiplier(self, setup_data):
        """Calcular multiplicador de confianza"""
        scorer = SetupQualityScorer()
        quality_score = scorer.calculate_overall_score(setup_data)
        
        # Base multiplier from score
        score_multiplier = quality_score['total_score'] / 5.0  # Normalize to 2.0 max
        
        # Historical performance adjustment
        performance = self.performance_tracker.analyze_performance_by_grade()
        grade = quality_score['grade']
        
        if grade in performance and performance[grade]['total_trades'] >= 10:
            # Use historical expectancy
            expectancy = performance[grade]['expectancy']
            performance_multiplier = max(0.25, min(2.0, 1 + expectancy * 5))
        else:
            performance_multiplier = 1.0
        
        # Market regime adjustment
        market_multiplier = self.get_market_regime_multiplier(setup_data)
        
        # Combined confidence
        final_multiplier = score_multiplier * performance_multiplier * market_multiplier
        
        # Cap between 0.25x and 3.0x
        return max(0.25, min(3.0, final_multiplier))
    
    def get_market_regime_multiplier(self, setup_data):
        """Adjust for market regime"""
        vix = setup_data.get('vix', 20)
        spy_trend = setup_data.get('spy_trend', 'neutral')
        
        # Base market multiplier
        if spy_trend == 'strong_uptrend':
            market_mult = 1.2
        elif spy_trend == 'uptrend':
            market_mult = 1.1
        elif spy_trend == 'neutral':
            market_mult = 1.0
        elif spy_trend == 'downtrend':
            market_mult = 0.8
        else:  # strong_downtrend
            market_mult = 0.5
        
        # VIX adjustment
        if vix > 30:  # High fear
            market_mult *= 0.7
        elif vix > 25:
            market_mult *= 0.85
        elif vix < 12:  # Complacency
            market_mult *= 0.9
        
        return market_mult
    
    def calculate_position_size(self, account_value, setup_data, entry_price, stop_price):
        """Calculate final position size"""
        confidence_multiplier = self.calculate_confidence_multiplier(setup_data)
        adjusted_risk = self.base_risk * confidence_multiplier
        
        # Standard position sizing
        risk_per_share = abs(entry_price - stop_price)
        risk_amount = account_value * adjusted_risk
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        # Position limits
        max_position_value = account_value * 0.25  # 25% max position
        max_shares = int(max_position_value / entry_price)
        
        final_shares = min(shares, max_shares)
        
        return {
            'shares': final_shares,
            'risk_amount': final_shares * risk_per_share,
            'risk_pct': (final_shares * risk_per_share) / account_value,
            'confidence_multiplier': confidence_multiplier,
            'base_risk': self.base_risk,
            'adjusted_risk': adjusted_risk
        }
```

### 2. Streak-Based Adjustments
```python
class StreakAdjustedRisk:
    def __init__(self, base_risk=0.02):
        self.base_risk = base_risk
        self.recent_trades = []
        self.max_lookback = 20
        
    def add_trade_result(self, was_winner, pnl_pct):
        """Add trade result"""
        self.recent_trades.append({
            'timestamp': pd.Timestamp.now(),
            'was_winner': was_winner,
            'pnl_pct': pnl_pct
        })
        
        # Keep only recent trades
        if len(self.recent_trades) > self.max_lookback:
            self.recent_trades = self.recent_trades[-self.max_lookback:]
    
    def calculate_streak_multiplier(self):
        """Calculate risk multiplier based on recent performance"""
        if len(self.recent_trades) < 5:
            return 1.0
        
        recent_10 = self.recent_trades[-10:] if len(self.recent_trades) >= 10 else self.recent_trades
        
        # Calculate recent win rate
        recent_win_rate = sum(1 for trade in recent_10 if trade['was_winner']) / len(recent_10)
        
        # Calculate recent avg return
        recent_avg_return = sum(trade['pnl_pct'] for trade in recent_10) / len(recent_10)
        
        # Current streak
        current_streak = 0
        streak_type = None
        
        for trade in reversed(self.recent_trades):
            if streak_type is None:
                streak_type = 'win' if trade['was_winner'] else 'loss'
                current_streak = 1
            elif (streak_type == 'win' and trade['was_winner']) or (streak_type == 'loss' and not trade['was_winner']):
                current_streak += 1
            else:
                break
        
        # Adjustment logic
        multiplier = 1.0
        
        # Winning streak: gradually increase confidence
        if streak_type == 'win' and current_streak >= 3:
            multiplier = min(1.5, 1.0 + (current_streak - 2) * 0.1)
        
        # Losing streak: reduce risk
        elif streak_type == 'loss' and current_streak >= 2:
            multiplier = max(0.25, 1.0 - (current_streak - 1) * 0.15)
        
        # Poor recent performance: reduce risk
        if recent_win_rate < 0.3:  # <30% win rate
            multiplier *= 0.5
        elif recent_avg_return < -0.02:  # Losing money on average
            multiplier *= 0.75
        
        # Great recent performance: slight increase
        elif recent_win_rate > 0.7 and recent_avg_return > 0.03:
            multiplier *= 1.25
        
        return max(0.25, min(2.0, multiplier))
    
    def get_adjusted_risk(self):
        """Get current risk adjustment"""
        streak_multiplier = self.calculate_streak_multiplier()
        return self.base_risk * streak_multiplier
```

## Implementation Framework

### 1. Integrated Asymmetric Risk Manager
```python
class AsymmetricRiskManager:
    def __init__(self, account_value, base_risk=0.02):
        self.account_value = account_value
        self.base_risk = base_risk
        
        # Components
        self.scorer = SetupQualityScorer()
        self.performance_tracker = SetupPerformanceTracker()
        self.confidence_sizer = ConfidenceBasedSizer(base_risk)
        self.streak_adjuster = StreakAdjustedRisk(base_risk)
        
    def calculate_optimal_position_size(self, setup_data, entry_price, stop_price):
        """Master function para calcular position size óptimo"""
        
        # 1. Score the setup
        quality_assessment = self.scorer.calculate_overall_score(setup_data)
        
        # 2. Confidence-based sizing
        confidence_sizing = self.confidence_sizer.calculate_position_size(
            self.account_value, setup_data, entry_price, stop_price
        )
        
        # 3. Streak adjustment
        streak_multiplier = self.streak_adjuster.calculate_streak_multiplier()
        
        # 4. Combine adjustments
        final_shares = int(confidence_sizing['shares'] * streak_multiplier)
        
        # 5. Final validation
        final_risk_amount = final_shares * abs(entry_price - stop_price)
        final_risk_pct = final_risk_amount / self.account_value
        
        # Safety caps
        max_risk_pct = 0.06  # Never risk more than 6%
        if final_risk_pct > max_risk_pct:
            final_shares = int((self.account_value * max_risk_pct) / abs(entry_price - stop_price))
            final_risk_amount = final_shares * abs(entry_price - stop_price)
            final_risk_pct = final_risk_amount / self.account_value
        
        return {
            'final_shares': final_shares,
            'final_risk_amount': final_risk_amount,
            'final_risk_pct': final_risk_pct,
            'setup_quality': quality_assessment,
            'confidence_multiplier': confidence_sizing['confidence_multiplier'],
            'streak_multiplier': streak_multiplier,
            'base_risk': self.base_risk,
            'decision_breakdown': {
                'setup_grade': quality_assessment['grade'],
                'setup_score': quality_assessment['total_score'],
                'base_shares': confidence_sizing['shares'],
                'streak_adjusted_shares': final_shares
            }
        }
    
    def should_take_trade(self, setup_data, min_grade='C'):
        """¿Debería tomar este trade?"""
        quality_assessment = self.scorer.calculate_overall_score(setup_data)
        grade = quality_assessment['grade']
        
        # Grade hierarchy
        grade_values = {'A+': 10, 'A': 9, 'B+': 8, 'B': 7, 'C+': 6, 'C': 5, 'D': 4}
        min_value = grade_values.get(min_grade, 5)
        current_value = grade_values.get(grade, 0)
        
        should_trade = current_value >= min_value
        
        return {
            'should_trade': should_trade,
            'grade': grade,
            'score': quality_assessment['total_score'],
            'reason': f"Grade {grade} {'meets' if should_trade else 'below'} minimum {min_grade}"
        }
```

### 2. Monitoring and Reporting
```python
def generate_asymmetric_risk_report(risk_manager):
    """Generate comprehensive risk report"""
    
    # Historical performance by grade
    performance = risk_manager.performance_tracker.analyze_performance_by_grade()
    
    # Current allocations
    optimal_allocations = risk_manager.performance_tracker.get_optimal_risk_allocation()
    
    # Recent streak performance
    streak_multiplier = risk_manager.streak_adjuster.calculate_streak_multiplier()
    
    report = f"""
📊 **Asymmetric Risk Management Report**
{'='*50}

🎯 **Setup Performance by Grade**
"""
    
    for grade in ['A+', 'A', 'B+', 'B', 'C+', 'C']:
        if grade in performance:
            stats = performance[grade]
            report += f"""
{grade} Grade Setups:
  • Total Trades: {stats['total_trades']}
  • Win Rate: {stats['win_rate']:.1%}
  • Avg Return: {stats['avg_pnl_pct']:.2%}
  • Profit Factor: {stats['profit_factor']:.2f}
  • Expectancy: {stats['expectancy']:.2%}
"""
    
    report += f"""

💰 **Current Risk Allocations**
"""
    for grade, allocation in optimal_allocations.items():
        report += f"  • {grade} Grade: {allocation:.1%} risk per trade\n"
    
    report += f"""

📈 **Current Adjustments**
  • Streak Multiplier: {streak_multiplier:.2f}x
  • Base Risk: {risk_manager.base_risk:.1%}
  • Effective Risk Range: {risk_manager.base_risk * streak_multiplier * 0.25:.1%} - {risk_manager.base_risk * streak_multiplier * 3.0:.1%}
"""
    
    return report
```

## Mi Setup Personal

```python
# asymmetric_config.py
ASYMMETRIC_CONFIG = {
    'base_risk': 0.015,  # 1.5% base risk
    'grade_multipliers': {
        'A+': 2.5,  # 3.75% risk for A+ setups
        'A': 2.0,   # 3.0% risk for A setups
        'B+': 1.5,  # 2.25% risk for B+ setups
        'B': 1.0,   # 1.5% risk for B setups
        'C+': 0.5,  # 0.75% risk for C+ setups
        'C': 0.25,  # 0.375% risk for C setups
        'D': 0.0    # No trading D setups
    },
    'min_tradeable_grade': 'B',  # Don't trade below B grade
    'max_single_risk': 0.05,     # Never risk more than 5%
    'streak_adjustment': True,    # Enable streak-based adjustments
    'market_regime_adjustment': True  # Enable market-based adjustments
}

# Usage
def my_position_sizing_workflow(ticker, setup_data, entry_price, stop_price):
    """Mi workflow completo de position sizing"""
    
    # Initialize manager
    risk_manager = AsymmetricRiskManager(
        account_value=50000, 
        base_risk=ASYMMETRIC_CONFIG['base_risk']
    )
    
    # Should I trade this?
    trade_decision = risk_manager.should_take_trade(
        setup_data, 
        min_grade=ASYMMETRIC_CONFIG['min_tradeable_grade']
    )
    
    if not trade_decision['should_trade']:
        return {
            'approved': False,
            'reason': trade_decision['reason'],
            'grade': trade_decision['grade']
        }
    
    # Calculate position size
    position_info = risk_manager.calculate_optimal_position_size(
        setup_data, entry_price, stop_price
    )
    
    # Final approval
    if position_info['final_risk_pct'] > ASYMMETRIC_CONFIG['max_single_risk']:
        return {
            'approved': False,
            'reason': f"Risk too high: {position_info['final_risk_pct']:.2%}",
            'max_allowed': ASYMMETRIC_CONFIG['max_single_risk']
        }
    
    return {
        'approved': True,
        'shares': position_info['final_shares'],
        'risk_amount': position_info['final_risk_amount'],
        'risk_pct': position_info['final_risk_pct'],
        'setup_grade': position_info['setup_quality']['grade'],
        'confidence_multiplier': position_info['confidence_multiplier'],
        'decision_detail': position_info['decision_breakdown']
    }
```

Completada la sección de gestión de riesgo. Continuemos con las estrategias aplicadas a small caps.