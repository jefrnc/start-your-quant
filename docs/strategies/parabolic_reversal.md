# Parabolic Reversal Strategy

## Concepto Base

La estrategia Parabolic Reversal se enfoca en identificar y capitalizar reversiones en stocks que han experimentado movimientos parabólicos extremos. Estos setups ofrecen algunas de las mejores oportunidades risk/reward cuando se ejecutan correctamente.

## Anatomía de un Movimiento Parabólico

### Características de Movimientos Parabólicos
```python
class ParabolicMovementAnalyzer:
    def __init__(self):
        self.parabolic_criteria = {
            'magnitude': 100,        # >100% move from base
            'timeframe': 21,         # Within 21 days
            'acceleration': 1.5,     # Accelerating daily gains
            'volume_profile': 'increasing_then_decreasing'
        }
        
    def identify_parabolic_move(self, price_data):
        """Identificar si un stock está en movimiento parabólico"""
        
        # Calculate move from base
        lookback_period = 30
        base_price = price_data.tail(lookback_period)['low'].min()
        current_price = price_data.iloc[-1]['close']
        total_move = (current_price - base_price) / base_price
        
        # Calculate time to move
        base_date_idx = price_data.tail(lookback_period)['low'].idxmin()
        base_date_position = price_data.index.get_loc(base_date_idx)
        days_to_move = len(price_data) - base_date_position
        
        # Acceleration analysis
        acceleration_score = self.calculate_acceleration(price_data.tail(10))
        
        # Volume profile analysis
        volume_profile = self.analyze_volume_profile(price_data.tail(15))
        
        # Determine if parabolic
        is_parabolic = (
            total_move > 1.0 and          # >100% move
            days_to_move <= 21 and        # Within 21 days
            acceleration_score > 60       # Strong acceleration
        )
        
        return {
            'is_parabolic': is_parabolic,
            'total_move_pct': total_move,
            'days_to_move': days_to_move,
            'acceleration_score': acceleration_score,
            'volume_profile': volume_profile,
            'exhaustion_signals': self.detect_exhaustion_signals(price_data),
            'reversal_probability': self.calculate_reversal_probability(
                total_move, days_to_move, acceleration_score, volume_profile
            )
        }
    
    def calculate_acceleration(self, recent_data):
        """Calcular score de aceleración del momentum"""
        
        daily_returns = recent_data['close'].pct_change().fillna(0)
        
        # Check for accelerating pattern
        early_period = daily_returns.head(5).mean()
        late_period = daily_returns.tail(5).mean()
        
        # Acceleration ratio
        acceleration_ratio = late_period / early_period if early_period != 0 else 1
        
        # Consistency of gains
        positive_days = (daily_returns > 0).sum()
        consistency_score = (positive_days / len(daily_returns)) * 100
        
        # Combined acceleration score
        acceleration_score = min(100, (acceleration_ratio * 30) + consistency_score)
        
        return acceleration_score
    
    def analyze_volume_profile(self, volume_data):
        """Analizar perfil de volumen durante movimiento parabólico"""
        
        early_volume = volume_data.head(7)['volume'].mean()
        middle_volume = volume_data.iloc[4:11]['volume'].mean()
        late_volume = volume_data.tail(7)['volume'].mean()
        
        # Classic parabolic pattern: increasing then decreasing volume
        volume_pattern = {
            'early_avg': early_volume,
            'middle_avg': middle_volume,
            'late_avg': late_volume,
            'peak_volume': volume_data['volume'].max(),
            'current_vs_peak': volume_data.iloc[-1]['volume'] / volume_data['volume'].max()
        }
        
        # Determine pattern type
        if middle_volume > early_volume and late_volume < middle_volume:
            pattern_type = 'classic_parabolic'  # Good for reversal
        elif late_volume > middle_volume > early_volume:
            pattern_type = 'accelerating'       # Still building
        else:
            pattern_type = 'irregular'          # Mixed signals
        
        volume_pattern['pattern_type'] = pattern_type
        
        return volume_pattern
    
    def detect_exhaustion_signals(self, price_data):
        """Detectar señales de agotamiento"""
        
        signals = {}
        recent_data = price_data.tail(5)
        
        # 1. Shooting star / doji patterns
        signals['bearish_candlestick'] = self.detect_bearish_patterns(recent_data)
        
        # 2. Lower highs despite volume
        signals['lower_highs'] = self.detect_lower_highs(recent_data)
        
        # 3. Gap ups with immediate selling
        signals['gap_fade'] = self.detect_gap_fade_pattern(recent_data)
        
        # 4. Volume divergence
        signals['volume_divergence'] = self.detect_volume_divergence(price_data.tail(10))
        
        # 5. Extended time at highs
        signals['time_at_highs'] = self.calculate_time_at_highs(recent_data)
        
        # 6. Failed breakout attempts
        signals['failed_breakouts'] = self.count_failed_breakouts(recent_data)
        
        # Composite exhaustion score
        exhaustion_score = sum([
            signals['bearish_candlestick'] * 20,
            signals['lower_highs'] * 25,
            signals['gap_fade'] * 15,
            signals['volume_divergence'] * 20,
            signals['time_at_highs'] * 10,
            signals['failed_breakouts'] * 10
        ])
        
        signals['exhaustion_score'] = min(100, exhaustion_score)
        signals['reversal_imminent'] = exhaustion_score > 70
        
        return signals
```

## Entry Strategies para Reversals

### Long Setup (Bounce from Oversold)
```python
class ParabolicBounceStrategy:
    def __init__(self):
        self.strategy_type = "oversold_bounce"
        self.min_decline = 0.30  # 30% decline from high
        
    def identify_bounce_setup(self, stock_data, parabolic_data):
        """Identificar setup para bounce después de crash parabólico"""
        
        # Verify parabolic crash
        recent_high = stock_data.tail(10)['high'].max()
        current_price = stock_data.iloc[-1]['close']
        decline_from_high = (recent_high - current_price) / recent_high
        
        if decline_from_high < self.min_decline:
            return None
        
        # Check for bounce signals
        bounce_signals = self.detect_bounce_signals(stock_data)
        
        if bounce_signals['signal_count'] >= 3:
            return self.calculate_bounce_entry_levels(stock_data, bounce_signals)
        
        return None
    
    def detect_bounce_signals(self, stock_data):
        """Detectar señales de bounce potencial"""
        
        signals = {}
        recent_data = stock_data.tail(5)
        
        # 1. Oversold RSI
        rsi = calculate_rsi(stock_data['close'], 14)
        signals['oversold_rsi'] = rsi.iloc[-1] < 25
        
        # 2. Volume spike on decline
        current_volume = stock_data.iloc[-1]['volume']
        avg_volume = stock_data['volume'].rolling(20).mean().iloc[-1]
        signals['volume_spike'] = current_volume > avg_volume * 2
        
        # 3. Hammer/doji formation
        signals['reversal_candle'] = self.detect_reversal_candlestick(recent_data.iloc[-1])
        
        # 4. Support level test
        signals['support_test'] = self.check_support_level_test(stock_data)
        
        # 5. Positive divergence
        signals['bullish_divergence'] = self.detect_bullish_divergence(stock_data.tail(10))
        
        # 6. Gap down with recovery
        signals['gap_recovery'] = self.detect_gap_recovery(recent_data)
        
        signal_count = sum([signals[key] for key in signals if isinstance(signals[key], bool)])
        signals['signal_count'] = signal_count
        
        return signals
    
    def calculate_bounce_entry_levels(self, stock_data, signals):
        """Calcular niveles de entrada para bounce"""
        
        current_price = stock_data.iloc[-1]['close']
        recent_low = stock_data.tail(5)['low'].min()
        support_level = self.identify_support_level(stock_data)
        
        return {
            'strategy': 'parabolic_bounce',
            'entry_levels': {
                'aggressive': {
                    'price': current_price * 1.02,  # 2% above current
                    'size_pct': 0.40,
                    'trigger': 'reversal_confirmation',
                    'confirmation': 'green_candle_with_volume'
                },
                'conservative': {
                    'price': support_level * 1.05,  # 5% above support
                    'size_pct': 0.60,
                    'trigger': 'support_hold_confirmation',
                    'confirmation': 'multiple_green_candles'
                }
            },
            'stop_loss': recent_low * 0.95,  # 5% below recent low
            'targets': {
                'target_1': current_price * 1.20,   # 20% bounce
                'target_2': current_price * 1.40,   # 40% bounce  
                'fibonacci_618': self.calculate_fib_retracement(stock_data, 0.618)
            },
            'time_limits': {
                'entry_window': '2_hours',
                'max_hold': '3_days'
            },
            'risk_reward': 2.5,  # Target 2.5:1 R/R minimum
            'confidence': self.calculate_setup_confidence(signals)
        }
```

### Short Setup (Parabolic Top)
```python
class ParabolicTopStrategy:
    def __init__(self):
        self.strategy_type = "parabolic_top_short"
        self.min_exhaustion_score = 70
        
    def identify_top_setup(self, stock_data, parabolic_data):
        """Identificar setup para short en top parabólico"""
        
        exhaustion_signals = parabolic_data['exhaustion_signals']
        
        if exhaustion_signals['exhaustion_score'] < self.min_exhaustion_score:
            return None
        
        # Additional confirmation signals
        top_confirmation = self.analyze_top_formation(stock_data)
        
        if top_confirmation['confidence'] >= 0.7:
            return self.calculate_short_entry_levels(stock_data, exhaustion_signals)
        
        return None
    
    def analyze_top_formation(self, stock_data):
        """Analizar formación de top"""
        
        recent_data = stock_data.tail(10)
        
        analysis = {}
        
        # 1. Multiple tests of resistance
        analysis['resistance_tests'] = self.count_resistance_tests(recent_data)
        
        # 2. Decreasing volume on up moves
        analysis['volume_exhaustion'] = self.detect_volume_exhaustion(recent_data)
        
        # 3. Bearish divergence
        analysis['bearish_divergence'] = self.detect_bearish_divergence(recent_data)
        
        # 4. Failed breakout pattern
        analysis['failed_breakout'] = self.detect_failed_breakout(recent_data)
        
        # 5. Time spent at highs
        analysis['distribution_time'] = self.calculate_distribution_time(recent_data)
        
        # Calculate confidence
        confidence_factors = [
            analysis['resistance_tests'] >= 2,
            analysis['volume_exhaustion'],
            analysis['bearish_divergence'],
            analysis['failed_breakout'],
            analysis['distribution_time'] > 3  # 3+ days at highs
        ]
        
        confidence = sum(confidence_factors) / len(confidence_factors)
        analysis['confidence'] = confidence
        
        return analysis
    
    def calculate_short_entry_levels(self, stock_data, exhaustion_signals):
        """Calcular niveles de entrada para short"""
        
        current_price = stock_data.iloc[-1]['close']
        recent_high = stock_data.tail(5)['high'].max()
        resistance_level = self.identify_resistance_level(stock_data)
        
        return {
            'strategy': 'parabolic_top_short',
            'entry_levels': {
                'aggressive': {
                    'price': current_price * 0.98,  # 2% below current
                    'size_pct': 0.50,
                    'trigger': 'first_weakness',
                    'confirmation': 'volume_on_decline'
                },
                'conservative': {
                    'price': resistance_level * 0.95,  # 5% below resistance
                    'size_pct': 0.50,
                    'trigger': 'confirmed_breakdown',
                    'confirmation': 'sustained_selling'
                }
            },
            'stop_loss': recent_high * 1.05,  # 5% above recent high
            'targets': {
                'target_1': current_price * 0.80,   # 20% decline
                'target_2': current_price * 0.65,   # 35% decline
                'fibonacci_618': self.calculate_fib_retracement(stock_data, 0.382)
            },
            'risk_management': {
                'max_hold_time': '5_days',
                'profit_taking_schedule': [0.30, 0.40, 0.30],  # 30%, 40%, 30%
                'stop_tightening': 'after_10pct_profit'
            }
        }
```

## Pattern Recognition System

### Machine Learning para Pattern Detection
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class ParabolicPatternRecognizer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, price_data):
        """Crear features para ML pattern recognition"""
        
        features = {}
        
        # Price-based features
        features['total_move_pct'] = self.calculate_total_move(price_data)
        features['days_to_peak'] = self.calculate_days_to_peak(price_data)
        features['volatility_increasing'] = self.is_volatility_increasing(price_data)
        features['price_acceleration'] = self.calculate_price_acceleration(price_data)
        
        # Volume features
        features['volume_pattern_score'] = self.score_volume_pattern(price_data)
        features['volume_divergence'] = self.detect_volume_divergence(price_data)
        features['avg_volume_ratio'] = self.calculate_avg_volume_ratio(price_data)
        
        # Technical features
        features['rsi_overbought'] = self.calculate_rsi_extreme(price_data)
        features['bb_position'] = self.calculate_bollinger_position(price_data)
        features['distance_from_ma'] = self.calculate_ma_distance(price_data)
        
        # Pattern features
        features['exhaustion_candles'] = self.count_exhaustion_candles(price_data)
        features['failed_breakouts'] = self.count_failed_breakouts(price_data)
        features['gap_behavior'] = self.analyze_gap_behavior(price_data)
        
        # Time features
        features['time_at_highs'] = self.calculate_time_at_highs(price_data)
        features['momentum_duration'] = self.calculate_momentum_duration(price_data)
        
        return features
    
    def train_model(self, historical_data, labels):
        """Entrenar modelo de reconocimiento de patrones"""
        
        # Create feature matrix
        X = []
        y = []
        
        for i, data in enumerate(historical_data):
            features = self.create_features(data)
            feature_vector = [features[key] for key in sorted(features.keys())]
            X.append(feature_vector)
            y.append(labels[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Feature importance
        feature_names = sorted(self.create_features(historical_data[0]).keys())
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'training_accuracy': self.model.score(X_scaled, y),
            'feature_importance': importance_df,
            'model_trained': True
        }
    
    def predict_reversal_probability(self, current_data):
        """Predecir probabilidad de reversal"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = self.create_features(current_data)
        feature_vector = np.array([[features[key] for key in sorted(features.keys())]])
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Get prediction probability
        reversal_probability = self.model.predict_proba(feature_vector_scaled)[0][1]
        
        # Get feature contributions
        feature_names = sorted(features.keys())
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        return {
            'reversal_probability': reversal_probability,
            'confidence_level': 'high' if reversal_probability > 0.8 else 'medium' if reversal_probability > 0.6 else 'low',
            'key_features': self.get_top_contributing_features(features, feature_importance),
            'model_features': features
        }
```

## Risk Management Específico

### Gestión de Volatilidad Extrema
```python
class ParabolicRiskManager:
    def __init__(self, account_size):
        self.account_size = account_size
        self.max_parabolic_exposure = 0.10  # Max 10% in parabolic plays
        
    def calculate_parabolic_position_size(self, stock_data, strategy_type, confidence_score):
        """Cálculo de position size para trades parabólicos"""
        
        # Base risk is lower for parabolic trades due to volatility
        base_risk_pct = 0.015  # 1.5% base risk
        
        # Volatility adjustment
        volatility_factor = self.calculate_volatility_adjustment(stock_data)
        
        # Confidence adjustment
        confidence_factor = max(0.5, confidence_score)
        
        # Strategy-specific adjustment
        strategy_factors = {
            'parabolic_bounce': 1.2,    # Slightly more size (better R/R)
            'parabolic_top_short': 0.8  # Less size (unlimited upside risk)
        }
        
        strategy_factor = strategy_factors.get(strategy_type, 1.0)
        
        # Final risk calculation
        adjusted_risk_pct = (base_risk_pct * volatility_factor * 
                           confidence_factor * strategy_factor)
        
        # Cap at maximum parabolic exposure
        max_position_value = self.account_size * self.max_parabolic_exposure
        
        return {
            'risk_percentage': adjusted_risk_pct,
            'max_position_value': max_position_value,
            'volatility_adjustment': volatility_factor,
            'confidence_adjustment': confidence_factor,
            'strategy_adjustment': strategy_factor,
            'recommended_risk': min(adjusted_risk_pct, self.max_parabolic_exposure)
        }
    
    def manage_parabolic_stops(self, position, current_data, strategy_type):
        """Gestión especializada de stops para trades parabólicos"""
        
        if strategy_type == 'parabolic_bounce':
            return self.manage_bounce_stops(position, current_data)
        else:  # parabolic_top_short
            return self.manage_short_stops(position, current_data)
    
    def manage_bounce_stops(self, position, current_data):
        """Stops para bounce trades"""
        
        entry_price = position['entry_price']
        current_price = current_data['price']
        
        # Initial stop below recent low
        initial_stop = position['recent_low'] * 0.95
        
        current_profit = (current_price - entry_price) / entry_price
        
        # Aggressive profit protection due to volatility
        if current_profit > 0.15:  # 15% profit
            # Lock in 8% profit
            return entry_price * 1.08
        elif current_profit > 0.08:  # 8% profit
            # Move to breakeven + 2%
            return entry_price * 1.02
        else:
            return initial_stop
    
    def manage_short_stops(self, position, current_data):
        """Stops para short trades en parabolic tops"""
        
        entry_price = position['entry_price']
        current_price = current_data['price']
        
        # Initial stop above recent high
        initial_stop = position['recent_high'] * 1.05
        
        current_profit = (entry_price - current_price) / entry_price
        
        # Quick profit protection for shorts
        if current_profit > 0.12:  # 12% profit
            # Lock in 6% profit
            return entry_price * 0.94
        elif current_profit > 0.06:  # 6% profit
            # Move to breakeven - 2%
            return entry_price * 0.98
        else:
            return initial_stop
```

## Backtesting Framework

### Parabolic-Specific Backtesting
```python
class ParabolicBacktester:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.parabolic_recognizer = ParabolicPatternRecognizer()
        
    def run_parabolic_backtest(self, stock_universe):
        """Ejecutar backtest específico para patrones parabólicos"""
        
        results = {
            'bounce_trades': [],
            'short_trades': [],
            'pattern_detection_accuracy': {},
            'performance_metrics': {}
        }
        
        for symbol in stock_universe:
            symbol_results = self.backtest_symbol(symbol)
            
            results['bounce_trades'].extend(symbol_results['bounce_trades'])
            results['short_trades'].extend(symbol_results['short_trades'])
        
        # Analyze results
        results['performance_metrics'] = self.analyze_parabolic_performance(results)
        
        return results
    
    def backtest_symbol(self, symbol):
        """Backtest para un símbolo específico"""
        
        # Get historical data
        historical_data = self.get_historical_data(symbol, self.start_date, self.end_date)
        
        symbol_results = {
            'bounce_trades': [],
            'short_trades': []
        }
        
        # Sliding window analysis
        window_size = 30
        
        for i in range(window_size, len(historical_data)):
            window_data = historical_data.iloc[i-window_size:i]
            
            # Check for parabolic patterns
            parabolic_analysis = ParabolicMovementAnalyzer().identify_parabolic_move(window_data)
            
            if parabolic_analysis['is_parabolic']:
                # Check for reversal setups
                bounce_setup = ParabolicBounceStrategy().identify_bounce_setup(
                    window_data, parabolic_analysis
                )
                
                top_setup = ParabolicTopStrategy().identify_top_setup(
                    window_data, parabolic_analysis
                )
                
                if bounce_setup:
                    trade_result = self.simulate_bounce_trade(
                        symbol, window_data, bounce_setup, historical_data.iloc[i:]
                    )
                    if trade_result:
                        symbol_results['bounce_trades'].append(trade_result)
                
                if top_setup:
                    trade_result = self.simulate_short_trade(
                        symbol, window_data, top_setup, historical_data.iloc[i:]
                    )
                    if trade_result:
                        symbol_results['short_trades'].append(trade_result)
        
        return symbol_results
    
    def analyze_parabolic_performance(self, results):
        """Analizar performance específica de trades parabólicos"""
        
        bounce_df = pd.DataFrame(results['bounce_trades'])
        short_df = pd.DataFrame(results['short_trades'])
        
        analysis = {}
        
        if len(bounce_df) > 0:
            analysis['bounce_performance'] = {
                'total_trades': len(bounce_df),
                'win_rate': (bounce_df['pnl'] > 0).mean(),
                'avg_pnl': bounce_df['pnl'].mean(),
                'avg_hold_time': bounce_df['hold_time_hours'].mean(),
                'best_trade': bounce_df['pnl'].max(),
                'worst_trade': bounce_df['pnl'].min()
            }
        
        if len(short_df) > 0:
            analysis['short_performance'] = {
                'total_trades': len(short_df),
                'win_rate': (short_df['pnl'] > 0).mean(),
                'avg_pnl': short_df['pnl'].mean(),
                'avg_hold_time': short_df['hold_time_hours'].mean(),
                'best_trade': short_df['pnl'].max(),
                'worst_trade': short_df['pnl'].min()
            }
        
        # Combined analysis
        all_trades = pd.concat([bounce_df, short_df], ignore_index=True)
        
        if len(all_trades) > 0:
            analysis['combined_performance'] = {
                'total_trades': len(all_trades),
                'overall_win_rate': (all_trades['pnl'] > 0).mean(),
                'total_pnl': all_trades['pnl'].sum(),
                'sharpe_ratio': all_trades['pnl'].mean() / all_trades['pnl'].std(),
                'max_drawdown': self.calculate_max_drawdown(all_trades['cumulative_pnl'])
            }
        
        return analysis
```

La estrategia Parabolic Reversal requiere timing extremadamente preciso y gestión de riesgo agresiva, pero puede ofrecer algunas de las mejores oportunidades risk/reward en small cap trading cuando se ejecuta correctamente.