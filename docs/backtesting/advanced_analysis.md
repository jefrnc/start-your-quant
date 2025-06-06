# Análisis Cuantitativo Avanzado

## Detección de Pump & Dump con Machine Learning

### Características del Modelo
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class PumpDumpDetector:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42
        )
        self.feature_columns = []
        self.is_trained = False
        
    def create_features(self, df):
        """Crear features para detectar pump & dump"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['price_change_1d'] = df['close'].pct_change()
        features['price_change_3d'] = df['close'].pct_change(3)
        features['price_change_5d'] = df['close'].pct_change(5)
        features['price_volatility_5d'] = df['close'].pct_change().rolling(5).std()
        
        # Volume-based features
        features['volume_ratio_1d'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_spike'] = (features['volume_ratio_1d'] > 5).astype(int)
        features['volume_trend_3d'] = df['volume'].rolling(3).mean() / df['volume'].rolling(10).mean()
        
        # Technical indicators
        features['rsi_14'] = self.calculate_rsi(df['close'], 14)
        features['distance_from_sma20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()
        features['bb_position'] = self.calculate_bollinger_position(df['close'])
        
        # Pattern-based features
        features['consecutive_green_days'] = self.count_consecutive_green(df)
        features['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1) > 0.10).astype(int)
        features['failed_breakout'] = self.detect_failed_breakout(df)
        
        # Momentum features
        features['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        features['acceleration'] = features['price_change_1d'] - features['price_change_1d'].shift(1)
        
        # Market cap and float features (if available)
        if 'market_cap' in df.columns:
            features['market_cap_category'] = pd.cut(
                df['market_cap'], 
                bins=[0, 50e6, 200e6, 1e9, np.inf], 
                labels=[0, 1, 2, 3]
            ).astype(float)
        
        return features.fillna(0)
    
    def calculate_rsi(self, prices, periods=14):
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_position(self, prices, periods=20):
        """Posición dentro de Bollinger Bands"""
        sma = prices.rolling(periods).mean()
        std = prices.rolling(periods).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return bb_position.clip(0, 1)
    
    def count_consecutive_green(self, df):
        """Contar días verdes consecutivos"""
        green_days = (df['close'] > df['open']).astype(int)
        consecutive = green_days * (green_days.groupby((green_days != green_days.shift()).cumsum()).cumcount() + 1)
        return consecutive
    
    def detect_failed_breakout(self, df, lookback=5):
        """Detectar failed breakouts"""
        rolling_high = df['high'].rolling(lookback).max()
        breakout = df['high'] > rolling_high.shift(1)
        
        # Failed breakout: rompe high pero no puede mantenerlo
        failed = breakout & (df['close'] < df['high'] * 0.95)
        return failed.astype(int)
    
    def create_labels(self, df, forward_days=5, dump_threshold=-0.30):
        """Crear labels para training (1 = dump incoming, 0 = normal)"""
        # Look forward para ver si hay dump
        future_returns = df['close'].shift(-forward_days) / df['close'] - 1
        
        # Label = 1 si hay caída >30% en los próximos 5 días
        labels = (future_returns < dump_threshold).astype(int)
        
        return labels[:-forward_days]  # Remove last days without future data
    
    def train_model(self, historical_data, symbols):
        """Entrenar modelo con datos históricos"""
        all_features = []
        all_labels = []
        
        for symbol in symbols:
            if symbol in historical_data:
                df = historical_data[symbol]
                
                # Create features and labels
                features = self.create_features(df)
                labels = self.create_labels(df)
                
                # Align features and labels
                min_length = min(len(features), len(labels))
                features = features.iloc[-min_length:]
                labels = labels.iloc[-min_length:]
                
                all_features.append(features)
                all_labels.extend(labels)
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = np.array(all_labels)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        return {
            'train_accuracy': self.model.score(X_train, y_train),
            'test_accuracy': self.model.score(X_test, y_test),
            'feature_importance': importance_df
        }
    
    def predict_dump_probability(self, current_data):
        """Predecir probabilidad de dump"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = self.create_features(current_data)
        latest_features = features.iloc[-1:][self.feature_columns]
        
        # Predict probability
        dump_probability = self.model.predict_proba(latest_features)[0][1]
        
        return {
            'dump_probability': dump_probability,
            'risk_level': 'HIGH' if dump_probability > 0.7 else 'MEDIUM' if dump_probability > 0.4 else 'LOW',
            'features_used': latest_features.to_dict('records')[0]
        }
```

## Análisis de Correlaciones Dinámicas

### Correlaciones Rolling
```python
class DynamicCorrelationAnalysis:
    def __init__(self, window=30):
        self.window = window
        self.correlation_history = {}
        
    def calculate_rolling_correlations(self, returns_data, benchmark='SPY'):
        """Calcular correlaciones rolling con benchmark"""
        correlations = {}
        
        for symbol in returns_data.columns:
            if symbol != benchmark and benchmark in returns_data.columns:
                rolling_corr = returns_data[symbol].rolling(
                    window=self.window
                ).corr(returns_data[benchmark])
                
                correlations[symbol] = {
                    'current_correlation': rolling_corr.iloc[-1],
                    'avg_correlation': rolling_corr.mean(),
                    'correlation_trend': self.calculate_trend(rolling_corr),
                    'correlation_stability': rolling_corr.std(),
                    'rolling_series': rolling_corr
                }
        
        return correlations
    
    def calculate_trend(self, series, periods=10):
        """Calcular tendencia de correlación"""
        if len(series) < periods:
            return 0
        
        recent = series.tail(periods).mean()
        previous = series.tail(periods * 2).head(periods).mean()
        
        return (recent - previous) / abs(previous) if previous != 0 else 0
    
    def identify_correlation_breakdowns(self, correlations, threshold=0.3):
        """Identificar breakdowns de correlación (oportunidades de short)"""
        breakdowns = []
        
        for symbol, corr_data in correlations.items():
            # Breakdown = correlación históricamente alta pero actualmente baja
            if (corr_data['avg_correlation'] > 0.5 and 
                corr_data['current_correlation'] < threshold):
                
                severity = abs(corr_data['avg_correlation'] - corr_data['current_correlation'])
                
                breakdowns.append({
                    'symbol': symbol,
                    'current_corr': corr_data['current_correlation'],
                    'avg_corr': corr_data['avg_correlation'],
                    'severity': severity,
                    'trend': corr_data['correlation_trend'],
                    'opportunity_type': 'mean_reversion_short' if severity > 0.4 else 'momentum_short'
                })
        
        return sorted(breakdowns, key=lambda x: x['severity'], reverse=True)
    
    def sector_correlation_heatmap(self, returns_data, sector_mapping):
        """Crear heatmap de correlaciones por sector"""
        sector_correlations = {}
        
        # Group by sector
        sectors = {}
        for symbol, sector in sector_mapping.items():
            if sector not in sectors:
                sectors[sector] = []
            if symbol in returns_data.columns:
                sectors[sector].append(symbol)
        
        # Calculate sector average returns
        sector_returns = {}
        for sector, symbols in sectors.items():
            if len(symbols) > 0:
                sector_returns[sector] = returns_data[symbols].mean(axis=1)
        
        # Calculate correlation matrix
        sector_df = pd.DataFrame(sector_returns)
        correlation_matrix = sector_df.corr()
        
        return correlation_matrix
```

## Statistical Arbitrage Framework

### Pairs Trading Avanzado
```python
class StatisticalArbitrageFramework:
    def __init__(self):
        self.pairs = {}
        self.cointegration_results = {}
        
    def find_cointegrated_pairs(self, price_data, min_correlation=0.7):
        """Encontrar pares cointegrados"""
        from statsmodels.tsa.stattools import coint
        
        symbols = list(price_data.columns)
        cointegrated_pairs = []
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Check basic correlation first
                correlation = price_data[symbol1].corr(price_data[symbol2])
                
                if correlation > min_correlation:
                    # Test cointegration
                    series1 = price_data[symbol1].dropna()
                    series2 = price_data[symbol2].dropna()
                    
                    # Align series
                    aligned_data = pd.concat([series1, series2], axis=1).dropna()
                    
                    if len(aligned_data) > 50:  # Minimum observations
                        try:
                            coint_stat, p_value, critical_values = coint(
                                aligned_data.iloc[:, 0], 
                                aligned_data.iloc[:, 1]
                            )
                            
                            if p_value < 0.05:  # Cointegrated at 5% level
                                cointegrated_pairs.append({
                                    'pair': (symbol1, symbol2),
                                    'correlation': correlation,
                                    'coint_stat': coint_stat,
                                    'p_value': p_value,
                                    'critical_value_5pct': critical_values[1]
                                })
                                
                        except Exception as e:
                            continue
        
        return sorted(cointegrated_pairs, key=lambda x: x['p_value'])
    
    def calculate_spread_metrics(self, price_data, pair):
        """Calcular métricas del spread"""
        symbol1, symbol2 = pair
        
        # Calculate spread using linear regression
        from sklearn.linear_model import LinearRegression
        
        aligned_data = pd.concat([
            price_data[symbol1], 
            price_data[symbol2]
        ], axis=1).dropna()
        
        X = aligned_data.iloc[:, 1].values.reshape(-1, 1)  # symbol2
        y = aligned_data.iloc[:, 0].values  # symbol1
        
        # Fit regression
        reg = LinearRegression().fit(X, y)
        hedge_ratio = reg.coef_[0]
        
        # Calculate spread
        spread = aligned_data.iloc[:, 0] - hedge_ratio * aligned_data.iloc[:, 1]
        
        # Spread statistics
        spread_stats = {
            'hedge_ratio': hedge_ratio,
            'spread_mean': spread.mean(),
            'spread_std': spread.std(),
            'current_spread': spread.iloc[-1],
            'z_score': (spread.iloc[-1] - spread.mean()) / spread.std(),
            'spread_series': spread
        }
        
        return spread_stats
    
    def generate_pairs_signals(self, spread_stats, entry_threshold=2.0, exit_threshold=0.5):
        """Generar señales de pairs trading"""
        z_score = spread_stats['z_score']
        
        if abs(z_score) > entry_threshold:
            if z_score > entry_threshold:
                # Spread too high: short symbol1, long symbol2
                signal = {
                    'action': 'enter',
                    'symbol1_position': 'short',
                    'symbol2_position': 'long',
                    'hedge_ratio': spread_stats['hedge_ratio'],
                    'confidence': min(abs(z_score) / entry_threshold, 2.0) / 2.0,
                    'expected_return': abs(z_score) * spread_stats['spread_std']
                }
            else:
                # Spread too low: long symbol1, short symbol2
                signal = {
                    'action': 'enter',
                    'symbol1_position': 'long',
                    'symbol2_position': 'short',
                    'hedge_ratio': spread_stats['hedge_ratio'],
                    'confidence': min(abs(z_score) / entry_threshold, 2.0) / 2.0,
                    'expected_return': abs(z_score) * spread_stats['spread_std']
                }
        
        elif abs(z_score) < exit_threshold:
            signal = {
                'action': 'exit',
                'reason': 'spread_normalized'
            }
        
        else:
            signal = {
                'action': 'hold',
                'z_score': z_score
            }
        
        return signal
```

## Regime Detection

### Market Regime Classification
```python
class MarketRegimeDetector:
    def __init__(self):
        self.regimes = {}
        self.current_regime = None
        
    def detect_volatility_regime(self, returns, lookback=20):
        """Detectar régimen de volatilidad"""
        rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)  # Annualized
        
        vol_percentiles = {
            'low': rolling_vol.quantile(0.25),
            'medium': rolling_vol.quantile(0.75),
            'high': rolling_vol.quantile(0.95)
        }
        
        current_vol = rolling_vol.iloc[-1]
        
        if current_vol < vol_percentiles['low']:
            regime = 'low_volatility'
        elif current_vol < vol_percentiles['medium']:
            regime = 'normal_volatility'
        elif current_vol < vol_percentiles['high']:
            regime = 'high_volatility'
        else:
            regime = 'extreme_volatility'
        
        return {
            'regime': regime,
            'current_volatility': current_vol,
            'percentiles': vol_percentiles,
            'regime_persistence': self.calculate_regime_persistence(rolling_vol, regime)
        }
    
    def detect_trend_regime(self, prices, short_window=20, long_window=50):
        """Detectar régimen de tendencia"""
        short_ma = prices.rolling(short_window).mean()
        long_ma = prices.rolling(long_window).mean()
        
        # Trend direction
        trend_direction = np.where(short_ma > long_ma, 1, -1)
        
        # Trend strength
        price_position = (prices - long_ma) / long_ma
        trend_strength = abs(price_position.iloc[-1])
        
        # Classify regime
        current_trend = trend_direction[-1]
        
        if trend_strength > 0.1:  # Strong trend
            regime = 'strong_uptrend' if current_trend == 1 else 'strong_downtrend'
        elif trend_strength > 0.05:  # Moderate trend
            regime = 'uptrend' if current_trend == 1 else 'downtrend'
        else:
            regime = 'sideways'
        
        return {
            'regime': regime,
            'trend_strength': trend_strength,
            'price_position': price_position.iloc[-1],
            'ma_distance': (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
        }
    
    def adaptive_strategy_parameters(self, volatility_regime, trend_regime):
        """Adaptar parámetros de estrategia según régimen"""
        adaptations = {
            'position_sizing': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'holding_period_adjustment': 1.0,
            'strategy_selection': []
        }
        
        # Volatility adaptations
        if volatility_regime['regime'] == 'low_volatility':
            adaptations['position_sizing'] = 1.5  # Increase size in low vol
            adaptations['stop_loss_multiplier'] = 0.8  # Tighter stops
            adaptations['strategy_selection'].append('mean_reversion')
            
        elif volatility_regime['regime'] == 'high_volatility':
            adaptations['position_sizing'] = 0.5  # Reduce size in high vol
            adaptations['stop_loss_multiplier'] = 1.5  # Wider stops
            adaptations['strategy_selection'].append('momentum')
            
        elif volatility_regime['regime'] == 'extreme_volatility':
            adaptations['position_sizing'] = 0.25  # Very small positions
            adaptations['stop_loss_multiplier'] = 2.0  # Much wider stops
            adaptations['holding_period_adjustment'] = 0.5  # Shorter holds
        
        # Trend adaptations
        if trend_regime['regime'] in ['strong_uptrend', 'strong_downtrend']:
            adaptations['strategy_selection'].append('trend_following')
            adaptations['take_profit_multiplier'] = 1.5  # Let winners run
            
        elif trend_regime['regime'] == 'sideways':
            adaptations['strategy_selection'].append('range_trading')
            adaptations['take_profit_multiplier'] = 0.8  # Take profits quickly
        
        return adaptations
```

## Performance Attribution

### Factor-Based Analysis
```python
class PerformanceAttribution:
    def __init__(self):
        self.factor_exposures = {}
        self.attribution_results = {}
        
    def calculate_factor_exposures(self, returns, factor_returns):
        """Calcular exposiciones a factores"""
        from sklearn.linear_model import LinearRegression
        
        # Align data
        aligned_data = pd.concat([returns, factor_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:  # Minimum observations
            return None
        
        y = aligned_data.iloc[:, 0].values  # Strategy returns
        X = aligned_data.iloc[:, 1:].values  # Factor returns
        
        # Regression
        reg = LinearRegression().fit(X, y)
        
        exposures = dict(zip(factor_returns.columns, reg.coef_))
        alpha = reg.intercept_
        r_squared = reg.score(X, y)
        
        return {
            'factor_exposures': exposures,
            'alpha': alpha,
            'r_squared': r_squared,
            'residual_volatility': np.std(y - reg.predict(X))
        }
    
    def decompose_returns(self, strategy_returns, factor_returns, factor_exposures):
        """Descomponer returns en factores + alpha"""
        decomposition = pd.DataFrame(index=strategy_returns.index)
        
        # Factor contributions
        for factor, exposure in factor_exposures['factor_exposures'].items():
            if factor in factor_returns.columns:
                decomposition[f'{factor}_contribution'] = (
                    factor_returns[factor] * exposure
                )
        
        # Total factor return
        decomposition['total_factor_return'] = decomposition.sum(axis=1)
        
        # Alpha
        decomposition['alpha'] = factor_exposures['alpha']
        
        # Residual (unexplained)
        decomposition['residual'] = (
            strategy_returns - 
            decomposition['total_factor_return'] - 
            decomposition['alpha']
        )
        
        return decomposition
```

Este framework de análisis cuantitativo avanzado permite identificar oportunidades de short selling con mayor precisión y gestionar el riesgo de manera más sofisticada.