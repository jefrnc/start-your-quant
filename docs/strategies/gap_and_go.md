# Gap & Go Strategy

## La Estrategia Rey de Small Caps

Gap & Go es la estrategia más conocida y rentable para small caps. Cuando un stock gappea arriba con volumen y continúa, puedes capturar movimientos explosivos. La clave está en el screening y timing.

> **⚠️ DISCLAIMER**: Small caps son extremadamente volátiles. Esta estrategia requiere experiencia, capital suficiente y gestión de riesgo estricta. Los ejemplos son educativos, no consejos financieros.

## Filosofía del Gap & Go

### ¿Por Qué Funciona?
- **FOMO Retail**: Gaps grandes atraen compradores retail
- **Momentum Cascade**: Un gap con volumen crea más compradores
- **Short Squeeze**: Shorts atrapados cubren posiciones
- **Algorithmic Following**: Bots siguen momentum con volumen

### Tipos de Gaps
```python
def classify_gap_type(gap_pct, volume_ratio, float_shares, catalyst):
    """Clasificar tipo de gap para strategy selection"""
    
    gap_types = {
        'explosive': {
            'gap_range': (25, 100),
            'volume_min': 10,
            'float_max': 10_000_000,
            'catalysts': ['fda_approval', 'buyout_rumor', 'major_contract']
        },
        'momentum': {
            'gap_range': (10, 25),
            'volume_min': 5,
            'float_max': 30_000_000,
            'catalysts': ['earnings_beat', 'analyst_upgrade', 'breakthrough']
        },
        'continuation': {
            'gap_range': (5, 15),
            'volume_min': 3,
            'float_max': 50_000_000,
            'catalysts': ['follow_through', 'sector_momentum']
        },
        'weak': {
            'gap_range': (2, 8),
            'volume_min': 1.5,
            'float_max': 100_000_000,
            'catalysts': ['technical', 'sympathy']
        }
    }
    
    for gap_type, criteria in gap_types.items():
        if (criteria['gap_range'][0] <= gap_pct <= criteria['gap_range'][1] and
            volume_ratio >= criteria['volume_min'] and
            float_shares <= criteria['float_max']):
            
            return {
                'type': gap_type,
                'strength': 'high' if catalyst in criteria['catalysts'] else 'medium',
                'expected_continuation': get_continuation_probability(gap_type, catalyst)
            }
    
    return {'type': 'no_play', 'strength': 'none', 'expected_continuation': 0}

def get_continuation_probability(gap_type, catalyst):
    """Historical continuation probabilities"""
    probabilities = {
        'explosive': {'fda_approval': 0.85, 'buyout_rumor': 0.75, 'major_contract': 0.70},
        'momentum': {'earnings_beat': 0.65, 'analyst_upgrade': 0.55, 'breakthrough': 0.60},
        'continuation': {'follow_through': 0.45, 'sector_momentum': 0.40},
        'weak': {'technical': 0.25, 'sympathy': 0.20}
    }
    
    return probabilities.get(gap_type, {}).get(catalyst, 0.30)
```

## Pre-Market Scanner

### 1. Core Screening Logic
```python
class GapAndGoScanner:
    def __init__(self):
        self.min_gap_pct = 10
        self.min_volume = 500_000
        self.max_float = 50_000_000
        self.min_price = 1.00
        self.max_price = 50.00
        
    def scan_premarket_gaps(self, scan_time='08:00'):
        """Scan pre-market para gap candidates"""
        candidates = []
        
        # Get pre-market movers
        movers = self.get_premarket_movers()
        
        for ticker in movers:
            try:
                data = self.get_stock_data(ticker)
                
                # Basic filters
                if not self.passes_basic_filters(data):
                    continue
                
                # Calculate metrics
                metrics = self.calculate_gap_metrics(data)
                
                # Score the setup
                score = self.score_gap_setup(data, metrics)
                
                if score['total_score'] >= 7:  # Minimum score
                    candidates.append({
                        'ticker': ticker,
                        'gap_pct': metrics['gap_pct'],
                        'premarket_volume': metrics['pm_volume'],
                        'rvol': metrics['rvol'],
                        'float': data['float_shares'],
                        'price': data['current_price'],
                        'catalyst': data.get('catalyst', 'unknown'),
                        'score': score['total_score'],
                        'rank': self.calculate_rank(metrics, score)
                    })
                    
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                continue
        
        # Sort by rank
        return sorted(candidates, key=lambda x: x['rank'], reverse=True)
    
    def passes_basic_filters(self, data):
        """Basic filtering criteria"""
        return (
            self.min_price <= data['current_price'] <= self.max_price and
            data['gap_pct'] >= self.min_gap_pct and
            data['premarket_volume'] >= self.min_volume and
            data['float_shares'] <= self.max_float and
            data['avg_volume_20d'] >= 100_000  # Minimum liquidity
        )
    
    def calculate_gap_metrics(self, data):
        """Calculate key gap metrics"""
        prev_close = data['prev_close']
        current_price = data['current_price']
        pm_volume = data['premarket_volume']
        avg_volume = data['avg_volume_20d']
        
        return {
            'gap_pct': (current_price - prev_close) / prev_close * 100,
            'gap_dollars': current_price - prev_close,
            'pm_volume': pm_volume,
            'rvol': pm_volume / (avg_volume * 0.1),  # Assuming 10% of volume pre-market
            'price_vs_high': current_price / data['prev_day_high'],
            'float_turnover': pm_volume / data['float_shares'] * 100
        }
    
    def score_gap_setup(self, data, metrics):
        """Score the gap setup quality"""
        scores = {}
        
        # Gap size score
        gap_pct = metrics['gap_pct']
        if gap_pct >= 30:
            scores['gap_size'] = 10
        elif gap_pct >= 20:
            scores['gap_size'] = 8
        elif gap_pct >= 15:
            scores['gap_size'] = 6
        else:
            scores['gap_size'] = 4
        
        # Volume score
        rvol = metrics['rvol']
        if rvol >= 20:
            scores['volume'] = 10
        elif rvol >= 10:
            scores['volume'] = 8
        elif rvol >= 5:
            scores['volume'] = 6
        else:
            scores['volume'] = 3
        
        # Float score
        float_shares = data['float_shares']
        if float_shares < 5_000_000:
            scores['float'] = 10
        elif float_shares < 15_000_000:
            scores['float'] = 8
        elif float_shares < 30_000_000:
            scores['float'] = 6
        else:
            scores['float'] = 3
        
        # Catalyst score
        catalyst = data.get('catalyst', 'none')
        catalyst_scores = {
            'fda_approval': 10, 'buyout_rumor': 9, 'major_contract': 8,
            'earnings_beat': 7, 'analyst_upgrade': 6, 'breakthrough': 6,
            'follow_through': 4, 'technical': 3, 'sympathy': 2, 'none': 1
        }
        scores['catalyst'] = catalyst_scores.get(catalyst, 1)
        
        # Price level score
        price = data['current_price']
        if 5 <= price <= 20:  # Sweet spot
            scores['price_level'] = 10
        elif 2 <= price < 5 or 20 < price <= 30:
            scores['price_level'] = 7
        else:
            scores['price_level'] = 4
        
        # Calculate weighted total
        weights = {
            'gap_size': 0.25,
            'volume': 0.30,
            'float': 0.20,
            'catalyst': 0.15,
            'price_level': 0.10
        }
        
        total_score = sum(scores[factor] * weights[factor] for factor in scores)
        
        return {
            'total_score': total_score,
            'individual_scores': scores,
            'weights': weights
        }
```

### 2. Real-Time Monitoring
```python
class GapMonitor:
    def __init__(self, watchlist):
        self.watchlist = watchlist
        self.alerts = []
        self.position_tracker = {}
        
    def monitor_gaps(self):
        """Monitor gap stocks during market hours"""
        for ticker in self.watchlist:
            current_data = self.get_real_time_data(ticker)
            
            # Entry signals
            if self.check_entry_signal(ticker, current_data):
                self.send_entry_alert(ticker, current_data)
            
            # Position management
            if ticker in self.position_tracker:
                self.manage_position(ticker, current_data)
    
    def check_entry_signal(self, ticker, data):
        """Check for gap continuation entry signals"""
        # Must be above pre-market high
        if data['price'] <= data['premarket_high']:
            return False
        
        # Volume confirmation
        if data['volume'] < data['avg_volume'] * 2:
            return False
        
        # Time filter (usually first 2 hours work best)
        current_time = pd.Timestamp.now().time()
        if current_time > pd.Timestamp('11:30').time():
            return False
        
        # Momentum check
        if data['price'] < data['vwap']:
            return False
        
        return True
    
    def manage_position(self, ticker, data):
        """Manage existing gap position"""
        position = self.position_tracker[ticker]
        
        # Trailing stop management
        if data['price'] > position['highest_price']:
            position['highest_price'] = data['price']
            
            # Update trailing stop
            new_stop = data['price'] * 0.85  # 15% trailing stop
            if new_stop > position['stop_price']:
                position['stop_price'] = new_stop
                self.send_stop_update_alert(ticker, new_stop)
        
        # Exit signals
        if self.check_exit_signal(ticker, data, position):
            self.send_exit_alert(ticker, data)
    
    def check_exit_signal(self, ticker, data, position):
        """Check for exit signals"""
        # Stop loss hit
        if data['price'] <= position['stop_price']:
            return True
        
        # VWAP loss
        if data['price'] < data['vwap'] and data['volume'] > data['avg_volume']:
            return True
        
        # End of day exit
        current_time = pd.Timestamp.now().time()
        if current_time >= pd.Timestamp('15:45').time():
            return True
        
        return False
```

## Entry Strategies

### 1. Breakout Entry
```python
def breakout_entry_strategy(data, position_type='aggressive'):
    """Entry on breakout of pre-market high"""
    
    entry_signals = []
    
    # Aggressive entry: break of pre-market high
    if position_type == 'aggressive':
        if (data['price'] > data['premarket_high'] and
            data['volume'] > data['avg_volume'] * 3):
            
            entry_signals.append({
                'type': 'aggressive_breakout',
                'entry_price': data['premarket_high'] + 0.01,
                'stop_loss': data['vwap'] * 0.97,
                'target': data['premarket_high'] * 1.20,
                'confidence': 0.75
            })
    
    # Conservative entry: pullback to VWAP then breakout
    elif position_type == 'conservative':
        if (data['price'] > data['premarket_high'] and
            data['previous_candle_low'] <= data['vwap'] and
            data['price'] > data['vwap']):
            
            entry_signals.append({
                'type': 'vwap_reclaim_breakout',
                'entry_price': data['vwap'] + 0.01,
                'stop_loss': data['vwap'] * 0.98,
                'target': data['premarket_high'] * 1.15,
                'confidence': 0.85
            })
    
    return entry_signals

def layer_entry_strategy(data, total_size):
    """Layered entry approach"""
    entries = []
    
    # Entry 1: 30% on initial breakout
    entries.append({
        'percentage': 0.30,
        'trigger': data['premarket_high'],
        'stop': data['vwap'] * 0.97
    })
    
    # Entry 2: 40% on continuation above resistance
    resistance_level = data['premarket_high'] * 1.05
    entries.append({
        'percentage': 0.40,
        'trigger': resistance_level,
        'stop': data['premarket_high'] * 0.98
    })
    
    # Entry 3: 30% on strong momentum
    momentum_level = data['premarket_high'] * 1.15
    entries.append({
        'percentage': 0.30,
        'trigger': momentum_level,
        'stop': resistance_level * 0.95
    })
    
    return entries
```

### 2. VWAP Strategy Integration
```python
class GapVWAPStrategy:
    def __init__(self):
        self.positions = {}
        self.alerts = []
        
    def analyze_gap_vwap_setup(self, ticker, data):
        """Analyze gap stock usando VWAP"""
        
        analysis = {
            'ticker': ticker,
            'gap_pct': data['gap_pct'],
            'current_price': data['price'],
            'vwap': data['vwap'],
            'signals': []
        }
        
        # Signal 1: Holding above VWAP after gap
        if (data['price'] > data['vwap'] and 
            data['volume'] > data['avg_volume'] * 2):
            
            analysis['signals'].append({
                'type': 'vwap_hold',
                'strength': 'medium',
                'entry': data['price'],
                'stop': data['vwap'] * 0.99,
                'target': data['price'] * 1.10
            })
        
        # Signal 2: VWAP reclaim after pullback
        if (data['price'] > data['vwap'] and
            data['low_of_day'] < data['vwap'] and
            data['volume'] > data['avg_volume'] * 1.5):
            
            analysis['signals'].append({
                'type': 'vwap_reclaim',
                'strength': 'high',
                'entry': data['vwap'] + 0.01,
                'stop': data['low_of_day'] * 0.99,
                'target': data['premarket_high'] * 1.05
            })
        
        # Signal 3: VWAP rejection (short setup)
        if (data['price'] < data['vwap'] and
            data['high_of_day'] > data['vwap'] and
            data['volume'] > data['avg_volume'] * 2):
            
            analysis['signals'].append({
                'type': 'vwap_rejection',
                'strength': 'medium',
                'entry': data['vwap'] - 0.01,
                'stop': data['vwap'] * 1.02,
                'target': data['price'] * 0.90,
                'direction': 'short'
            })
        
        return analysis
```

## Risk Management para Gap & Go

### 1. Position Sizing Específico
```python
def calculate_gap_position_size(account_value, gap_data, risk_tolerance=0.015):
    """Position sizing específico para gaps"""
    
    base_risk = account_value * risk_tolerance
    
    # Adjust risk based on gap characteristics
    risk_multipliers = {
        'gap_size': 1.0,
        'volume': 1.0,
        'float': 1.0,
        'time': 1.0
    }
    
    # Gap size adjustment
    gap_pct = gap_data['gap_pct']
    if gap_pct > 50:
        risk_multipliers['gap_size'] = 0.5  # Reduce risk on huge gaps
    elif gap_pct > 30:
        risk_multipliers['gap_size'] = 0.75
    elif gap_pct < 10:
        risk_multipliers['gap_size'] = 1.25  # Increase risk on smaller gaps
    
    # Volume adjustment
    rvol = gap_data['rvol']
    if rvol < 2:
        risk_multipliers['volume'] = 0.5  # Low volume = lower confidence
    elif rvol > 10:
        risk_multipliers['volume'] = 1.25
    
    # Float adjustment
    float_shares = gap_data['float_shares']
    if float_shares < 5_000_000:
        risk_multipliers['float'] = 0.75  # Micro float = more risk
    elif float_shares > 30_000_000:
        risk_multipliers['float'] = 1.25  # Larger float = less risk
    
    # Time adjustment
    current_time = pd.Timestamp.now().time()
    if current_time > pd.Timestamp('11:00').time():
        risk_multipliers['time'] = 0.75  # Later in day = less reliable
    
    # Final risk calculation
    final_multiplier = 1.0
    for factor, multiplier in risk_multipliers.items():
        final_multiplier *= multiplier
    
    adjusted_risk = base_risk * final_multiplier
    
    # Calculate shares
    entry_price = gap_data['entry_price']
    stop_price = gap_data['stop_price']
    risk_per_share = entry_price - stop_price
    
    if risk_per_share <= 0:
        return 0
    
    shares = int(adjusted_risk / risk_per_share)
    
    # Position limits
    max_position_value = account_value * 0.10  # 10% max en small caps volátiles
    max_shares = int(max_position_value / entry_price)
    
    return min(shares, max_shares)
```

### 2. Stop Loss Strategies
```python
class GapStopManager:
    def __init__(self):
        self.stop_strategies = ['vwap', 'premarket_low', 'trailing', 'time']
        
    def calculate_stop_levels(self, data, entry_price, strategy='adaptive'):
        """Calculate multiple stop levels"""
        
        stops = {}
        
        # VWAP stop
        stops['vwap'] = data['vwap'] * 0.99
        
        # Pre-market low stop
        stops['premarket_low'] = data['premarket_low'] * 0.98
        
        # Percentage stop
        stops['percentage'] = entry_price * 0.95  # 5% stop
        
        # ATR stop
        if 'atr' in data:
            stops['atr'] = entry_price - (data['atr'] * 1.5)
        
        # Time-based stop
        stops['time_based'] = self.get_time_based_stop(data, entry_price)
        
        if strategy == 'adaptive':
            # Choose best stop based on context
            chosen_stop = self.choose_optimal_stop(stops, data)
        else:
            chosen_stop = stops.get(strategy, stops['vwap'])
        
        return {
            'recommended_stop': chosen_stop,
            'all_stops': stops,
            'stop_strategy': strategy
        }
    
    def choose_optimal_stop(self, stops, data):
        """Choose optimal stop based on gap characteristics"""
        
        # For small gaps, use tighter stops
        if data['gap_pct'] < 15:
            return max(stops['vwap'], stops['percentage'])
        
        # For large gaps, use wider stops
        elif data['gap_pct'] > 30:
            return min(stops['premarket_low'], stops['atr'])
        
        # For medium gaps, use VWAP
        else:
            return stops['vwap']
    
    def get_time_based_stop(self, data, entry_price):
        """Stop based on time of day"""
        current_time = pd.Timestamp.now().time()
        
        if current_time < pd.Timestamp('10:30').time():
            # Early morning - use wider stops
            return entry_price * 0.93
        elif current_time < pd.Timestamp('14:00').time():
            # Mid-day - normal stops
            return entry_price * 0.95
        else:
            # Late day - tighter stops
            return entry_price * 0.97
```

## Exit Strategies

### 1. Profit Taking Levels
```python
def calculate_profit_targets(entry_price, gap_data):
    """Calculate multiple profit targets"""
    
    targets = {}
    
    # Technical targets
    targets['r1'] = entry_price * 1.05  # 5% quick profit
    targets['r2'] = entry_price * 1.10  # 10% target
    targets['r3'] = entry_price * 1.20  # Extension target
    
    # Gap-specific targets
    gap_pct = gap_data['gap_pct']
    
    # Target based on gap size
    if gap_pct > 30:
        targets['gap_extension'] = entry_price * 1.50  # Large gaps can run far
    elif gap_pct > 15:
        targets['gap_extension'] = entry_price * 1.25
    else:
        targets['gap_extension'] = entry_price * 1.15
    
    # Resistance levels
    if 'resistance_levels' in gap_data:
        targets['resistance'] = gap_data['resistance_levels'][0]
    
    # Previous day high
    if 'prev_day_high' in gap_data:
        if gap_data['prev_day_high'] > entry_price:
            targets['prev_high'] = gap_data['prev_day_high']
    
    # Float-based targets
    float_shares = gap_data['float_shares']
    if float_shares < 10_000_000:
        # Low float - can run further
        targets['float_adjusted'] = entry_price * 1.30
    else:
        targets['float_adjusted'] = entry_price * 1.15
    
    # Sort targets by price
    sorted_targets = sorted(
        [(name, price) for name, price in targets.items() if price > entry_price],
        key=lambda x: x[1]
    )
    
    return {
        'targets': targets,
        'sorted_targets': sorted_targets,
        'recommended_sequence': sorted_targets[:3]  # First 3 targets
    }

def scaling_exit_strategy(position_size, targets):
    """Scaling exit strategy"""
    
    exit_plan = []
    remaining_size = position_size
    
    # Target 1: Take 25% at first resistance
    t1_size = int(position_size * 0.25)
    exit_plan.append({
        'target_price': targets['sorted_targets'][0][1],
        'shares_to_sell': t1_size,
        'remaining': remaining_size - t1_size,
        'reason': 'quick_profit'
    })
    remaining_size -= t1_size
    
    # Target 2: Take 50% at second target
    t2_size = int(remaining_size * 0.50)
    exit_plan.append({
        'target_price': targets['sorted_targets'][1][1] if len(targets['sorted_targets']) > 1 else targets['sorted_targets'][0][1] * 1.1,
        'shares_to_sell': t2_size,
        'remaining': remaining_size - t2_size,
        'reason': 'main_target'
    })
    remaining_size -= t2_size
    
    # Target 3: Let remaining ride with trailing stop
    exit_plan.append({
        'target_price': 'trailing_stop',
        'shares_to_sell': remaining_size,
        'remaining': 0,
        'reason': 'let_winners_run'
    })
    
    return exit_plan
```

## Backtesting Gap & Go

### 1. Historical Performance Analysis
```python
def backtest_gap_and_go(historical_data, start_date, end_date):
    """Backtest gap and go strategy"""
    
    results = {
        'trades': [],
        'daily_pnl': [],
        'metrics': {}
    }
    
    scanner = GapAndGoScanner()
    
    # Iterate through each trading day
    for date in pd.date_range(start_date, end_date, freq='B'):  # Business days
        
        # Get gap candidates for this day
        daily_candidates = scanner.scan_historical_gaps(date)
        
        for candidate in daily_candidates:
            # Simulate trade
            trade_result = simulate_gap_trade(candidate, historical_data[date])
            
            if trade_result:
                results['trades'].append(trade_result)
                results['daily_pnl'].append(trade_result['pnl'])
    
    # Calculate metrics
    results['metrics'] = calculate_gap_strategy_metrics(results['trades'])
    
    return results

def simulate_gap_trade(candidate, intraday_data):
    """Simulate a single gap trade"""
    
    ticker = candidate['ticker']
    
    if ticker not in intraday_data:
        return None
    
    data = intraday_data[ticker]
    
    # Entry logic
    entry_price = None
    entry_time = None
    
    # Look for breakout of pre-market high
    for timestamp, bar in data.iterrows():
        if (bar['high'] > candidate['premarket_high'] and
            bar['volume'] > candidate['avg_volume'] * 2):
            
            entry_price = candidate['premarket_high'] + 0.01
            entry_time = timestamp
            break
    
    if not entry_price:
        return None  # No entry signal
    
    # Exit logic
    stop_price = candidate['vwap'] * 0.97
    target_price = entry_price * 1.15
    
    exit_price = None
    exit_time = None
    exit_reason = None
    
    # Look for exit
    for timestamp, bar in data[data.index > entry_time].iterrows():
        
        # Stop loss
        if bar['low'] <= stop_price:
            exit_price = stop_price
            exit_time = timestamp
            exit_reason = 'stop_loss'
            break
        
        # Target hit
        if bar['high'] >= target_price:
            exit_price = target_price
            exit_time = timestamp
            exit_reason = 'target_hit'
            break
        
        # End of day exit
        if timestamp.time() >= pd.Timestamp('15:45').time():
            exit_price = bar['close']
            exit_time = timestamp
            exit_reason = 'eod_exit'
            break
    
    if not exit_price:
        exit_price = data.iloc[-1]['close']
        exit_time = data.index[-1]
        exit_reason = 'eod_exit'
    
    # Calculate results
    pnl_pct = (exit_price - entry_price) / entry_price
    hold_time = (exit_time - entry_time).total_seconds() / 60  # minutes
    
    return {
        'ticker': ticker,
        'date': entry_time.date(),
        'entry_time': entry_time,
        'exit_time': exit_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'hold_time': hold_time,
        'exit_reason': exit_reason,
        'gap_pct': candidate['gap_pct'],
        'rvol': candidate['rvol'],
        'float': candidate['float']
    }
```

## Mi Setup Personal

```python
# gap_and_go_config.py
GAP_CONFIG = {
    'screening': {
        'min_gap_pct': 12,
        'min_premarket_volume': 750_000,
        'max_float': 40_000_000,
        'min_price': 2.00,
        'max_price': 40.00,
        'min_rvol': 3.0
    },
    'entry': {
        'strategy': 'layered',  # 'aggressive', 'conservative', 'layered'
        'entry_1_pct': 0.30,    # 30% on breakout
        'entry_2_pct': 0.50,    # 50% on continuation
        'entry_3_pct': 0.20,    # 20% on momentum
        'max_entry_time': '11:00'
    },
    'risk_management': {
        'base_risk_pct': 0.015,        # 1.5% para principiantes, 2.5% para avanzados
        'max_position_pct': 0.10,      # 10% max en small caps volátiles
        'stop_strategy': 'adaptive',
        'time_stop': '15:45',
        'beginner_mode': True          # Usar parámetros conservadores
    },
    'profit_targets': {
        'quick_profit': 0.08,   # 8% quick scalp
        'main_target': 0.15,    # 15% main target
        'extension': 0.25,      # 25% moon shot
        'scaling': True
    }
}

def my_gap_and_go_workflow():
    """Mi workflow completo de Gap & Go"""
    
    # 1. Pre-market scan (7:30 AM)
    scanner = GapAndGoScanner()
    candidates = scanner.scan_premarket_gaps()
    
    # 2. Filter top candidates
    top_candidates = [c for c in candidates if c['score'] >= 8][:5]
    
    # 3. Set alerts for market open
    for candidate in top_candidates:
        setup_entry_alerts(candidate)
    
    # 4. Monitor and execute during market hours
    return execute_gap_strategy(top_candidates)
```

## Siguiente Paso

Continuemos con [VWAP Reclaim](vwap_reclaim.md), otra estrategia fundamental para small caps.