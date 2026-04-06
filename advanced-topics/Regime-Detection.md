> 🇪🇸 [Leer en Español](Regime-Detection.es.md) | 🇺🇸 **English**

# Market Regime Detection

## Why Do Market Regimes Matter?

Markets do not behave consistently. They alternate between **distinct regimes** where price dynamics, volatility, and correlations change dramatically. A strategy that works excellently in a trending regime can be disastrous in a mean-reverting regime.

For small caps, this is especially critical because:
- **Volatility clustering**: Periods of high/low volatility tend to cluster together
- **Correlation shifts**: Small caps decouple/couple with the broader market
- **Liquidity regimes**: Liquidity availability varies dramatically
- **Risk appetite cycles**: Institutional flows into/out of small caps

### Relevant Regime Types

```python
REGIME_TYPES = {
    'volatility_regimes': ['low_vol', 'medium_vol', 'high_vol', 'crisis'],
    'trend_regimes': ['strong_bull', 'weak_bull', 'sideways', 'weak_bear', 'strong_bear'],
    'liquidity_regimes': ['abundant', 'normal', 'constrained', 'crisis'],
    'risk_appetite': ['risk_on', 'risk_neutral', 'risk_off', 'panic'],
    'small_cap_specific': ['rotation_into', 'rotation_out_of', 'overlooked', 'crowded']
}
```

## Regime Detection Framework

### 1. Hidden Markov Models (HMM) - Primary Approach

HMMs are ideal for regime detection because:
- They capture unobservable market states
- They allow probabilistic transitions between regimes
- They automatically adapt to structural changes
- They provide confidence levels for each regime

```python
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """
    Regime detector using Hidden Markov Models

    Identifies regimes based on:
    - Returns patterns
    - Volatility clustering
    - Volume characteristics
    - Cross-asset correlations
    """

    def __init__(self, n_regimes: int = 3):
        """
        Args:
            n_regimes: Number of regimes to detect (typically 2-4)
        """
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_labels = {}

    def prepare_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares features for regime detection

        Args:
            price_data: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=price_data.index)

        # 1. Returns features
        features['returns'] = price_data['close'].pct_change()
        features['returns_abs'] = features['returns'].abs()
        features['returns_squared'] = features['returns'] ** 2

        # 2. Volatility features
        features['realized_vol_5'] = features['returns'].rolling(5).std()
        features['realized_vol_20'] = features['returns'].rolling(20).std()
        features['vol_ratio'] = features['realized_vol_5'] / features['realized_vol_20']

        # 3. Volume features
        features['volume_norm'] = price_data['volume'] / price_data['volume'].rolling(20).mean()
        features['volume_volatility'] = (
            price_data['volume'].rolling(10).std() /
            price_data['volume'].rolling(10).mean()
        )

        # 4. Trend features
        features['ma_5'] = price_data['close'].rolling(5).mean()
        features['ma_20'] = price_data['close'].rolling(20).mean()
        features['trend_strength'] = (features['ma_5'] - features['ma_20']) / features['ma_20']

        # 5. Price action features
        features['high_low_ratio'] = (price_data['high'] - price_data['low']) / price_data['close']
        features['close_position'] = (
            (price_data['close'] - price_data['low']) /
            (price_data['high'] - price_data['low'])
        )

        # 6. Lag features to capture autocorrelations
        features['returns_lag1'] = features['returns'].shift(1)
        features['vol_lag1'] = features['realized_vol_5'].shift(1)

        return features.dropna()

    def fit(self, features: pd.DataFrame) -> 'MarketRegimeDetector':
        """
        Fit the HMM model to the features
        """
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)

        # Fit HMM
        self.model.fit(features_scaled)

        # Predict regimes for labeling
        regimes = self.model.predict(features_scaled)

        # Label regimes based on characteristics
        self.regime_labels = self._label_regimes(features, regimes)

        self.is_fitted = True
        return self

    def predict_regime(self, features: pd.DataFrame) -> Dict:
        """
        Predicts the current regime and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Scale features
        features_scaled = self.scaler.transform(features.iloc[-1:])

        # Get regime probabilities
        regime_probs = self.model.predict_proba(features_scaled)[0]

        # Most likely regime
        most_likely_regime = np.argmax(regime_probs)

        return {
            'regime': most_likely_regime,
            'regime_label': self.regime_labels.get(most_likely_regime, f'Regime_{most_likely_regime}'),
            'probabilities': {
                f'regime_{i}': prob for i, prob in enumerate(regime_probs)
            },
            'confidence': regime_probs[most_likely_regime]
        }

    def _label_regimes(self, features: pd.DataFrame, regimes: np.array) -> Dict:
        """
        Label regimes based on their statistical characteristics
        """
        regime_stats = {}

        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            regime_data = features[regime_mask]

            if len(regime_data) > 0:
                regime_stats[regime] = {
                    'avg_return': regime_data['returns'].mean(),
                    'volatility': regime_data['returns'].std(),
                    'volume_level': regime_data['volume_norm'].mean(),
                    'trend_strength': regime_data['trend_strength'].mean()
                }

        # Label based on characteristics
        labels = {}

        # Sort regimes by volatility
        sorted_by_vol = sorted(regime_stats.items(),
                              key=lambda x: x[1]['volatility'])

        if self.n_regimes == 3:
            labels[sorted_by_vol[0][0]] = 'Low_Volatility'
            labels[sorted_by_vol[1][0]] = 'Medium_Volatility'
            labels[sorted_by_vol[2][0]] = 'High_Volatility'

        elif self.n_regimes == 4:
            labels[sorted_by_vol[0][0]] = 'Calm'
            labels[sorted_by_vol[1][0]] = 'Normal'
            labels[sorted_by_vol[2][0]] = 'Stressed'
            labels[sorted_by_vol[3][0]] = 'Crisis'

        return labels

    def get_regime_history(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Gets historical regimes for the entire dataset
        """
        features_scaled = self.scaler.transform(features)
        regimes = self.model.predict(features_scaled)
        regime_probs = self.model.predict_proba(features_scaled)

        result = pd.DataFrame(index=features.index)
        result['regime'] = regimes
        result['regime_label'] = [self.regime_labels.get(r, f'Regime_{r}') for r in regimes]

        # Add probabilities
        for i in range(self.n_regimes):
            result[f'prob_regime_{i}'] = regime_probs[:, i]

        result['confidence'] = regime_probs.max(axis=1)

        return result


# Usage example
def example_regime_detection():
    """
    Complete regime detection example for small caps
    """

    # 1. Load data (example with synthetic data)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')

    # Simulate regime-changing data
    np.random.seed(42)
    n_days = len(dates)

    # Create synthetic regime data
    regimes_true = np.concatenate([
        np.ones(n_days//3) * 0,      # Low vol regime
        np.ones(n_days//3) * 1,      # Medium vol regime
        np.ones(n_days - 2*(n_days//3)) * 2  # High vol regime
    ])

    # Generate price data with regime-dependent characteristics
    returns = []
    vol_base = [0.01, 0.02, 0.04]  # Volatility per regime

    for i, regime in enumerate(regimes_true):
        if i == 0:
            ret = np.random.normal(0.001, vol_base[int(regime)])
        else:
            ret = np.random.normal(0.001, vol_base[int(regime)])
        returns.append(ret)

    returns = np.array(returns)
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV data
    price_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_days)
    }, index=dates)

    # Add OHLC
    price_data['open'] = price_data['close'].shift(1) * (1 + np.random.normal(0, 0.005, n_days))
    price_data['high'] = price_data[['open', 'close']].max(axis=1) * (1 + np.random.exponential(0.01, n_days))
    price_data['low'] = price_data[['open', 'close']].min(axis=1) * (1 - np.random.exponential(0.01, n_days))

    price_data = price_data.dropna()

    # 2. Initialize regime detector
    detector = MarketRegimeDetector(n_regimes=3)

    # 3. Prepare features
    features = detector.prepare_features(price_data)

    # 4. Fit model
    detector.fit(features)

    # 5. Get regime history
    regime_history = detector.get_regime_history(features)

    # 6. Current regime prediction
    current_regime = detector.predict_regime(features)

    print("Current Market Regime:")
    print(f"Regime: {current_regime['regime_label']}")
    print(f"Confidence: {current_regime['confidence']:.2%}")
    print("\nRegime Probabilities:")
    for regime, prob in current_regime['probabilities'].items():
        print(f"{regime}: {prob:.2%}")

    return detector, regime_history, price_data

if __name__ == "__main__":
    detector, history, data = example_regime_detection()
```

### 2. Volatility Regime Detection

```python
class VolatilityRegimeDetector:
    """
    Volatility-specific regime detector
    Uses GARCH models and threshold detection
    """

    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.thresholds = {}

    def detect_vol_regime(self, returns: pd.Series) -> Dict:
        """
        Detects the current volatility regime
        """
        # Calculate realized volatility
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

        # Historical volatility distribution
        hist_vol = returns.rolling(20).std() * np.sqrt(252)
        hist_vol = hist_vol.dropna()

        # Define thresholds based on percentiles
        if len(hist_vol) >= self.lookback_window:
            self.thresholds = {
                'low': hist_vol.quantile(0.25),
                'medium_low': hist_vol.quantile(0.50),
                'medium_high': hist_vol.quantile(0.75),
                'high': hist_vol.quantile(0.90)
            }

        # Classify current regime
        if current_vol <= self.thresholds.get('low', 0.1):
            regime = 'Ultra_Low_Vol'
            risk_adjustment = 1.5  # Increase position size
        elif current_vol <= self.thresholds.get('medium_low', 0.15):
            regime = 'Low_Vol'
            risk_adjustment = 1.2
        elif current_vol <= self.thresholds.get('medium_high', 0.25):
            regime = 'Normal_Vol'
            risk_adjustment = 1.0
        elif current_vol <= self.thresholds.get('high', 0.35):
            regime = 'High_Vol'
            risk_adjustment = 0.7
        else:
            regime = 'Crisis_Vol'
            risk_adjustment = 0.3  # Dramatically reduce exposure

        return {
            'regime': regime,
            'current_vol': current_vol,
            'vol_percentile': (hist_vol <= current_vol).mean(),
            'risk_adjustment_factor': risk_adjustment,
            'thresholds': self.thresholds
        }
```

### 3. Correlation Regime Detection

```python
class CorrelationRegimeDetector:
    """
    Detects shifts in correlation structures
    Critical for small caps that alternate between correlation with the market
    """

    def __init__(self, benchmark_symbols: List[str] = ['SPY', 'IWM']):
        self.benchmark_symbols = benchmark_symbols
        self.correlation_history = {}

    def detect_correlation_regime(self,
                                 target_returns: pd.Series,
                                 benchmark_returns: pd.DataFrame,
                                 window: int = 60) -> Dict:
        """
        Detects the correlation regime using rolling correlations
        """

        correlation_results = {}

        for benchmark in self.benchmark_symbols:
            if benchmark in benchmark_returns.columns:
                # Rolling correlation
                rolling_corr = target_returns.rolling(window).corr(
                    benchmark_returns[benchmark]
                ).dropna()

                current_corr = rolling_corr.iloc[-1]
                hist_corr = rolling_corr.iloc[:-1]

                # Correlation percentile
                corr_percentile = (hist_corr <= current_corr).mean()

                # Correlation stability (std of recent correlations)
                recent_corr_std = rolling_corr.tail(20).std()

                correlation_results[benchmark] = {
                    'current_correlation': current_corr,
                    'correlation_percentile': corr_percentile,
                    'correlation_stability': recent_corr_std,
                    'avg_correlation': hist_corr.mean()
                }

        # Overall correlation regime
        avg_corr = np.mean([r['current_correlation'] for r in correlation_results.values()])

        if avg_corr > 0.7:
            regime = 'High_Correlation'  # Risk-off, everything moves together
            strategy_implications = 'Reduce diversification benefit, focus on market timing'
        elif avg_corr > 0.3:
            regime = 'Medium_Correlation'  # Normal market
            strategy_implications = 'Standard stock selection approaches work'
        elif avg_corr > 0:
            regime = 'Low_Correlation'  # Stock picking environment
            strategy_implications = 'Strong stock selection opportunities'
        else:
            regime = 'Negative_Correlation'  # Unusual regime
            strategy_implications = 'Investigate for structural breaks'

        return {
            'regime': regime,
            'avg_correlation': avg_corr,
            'individual_correlations': correlation_results,
            'strategy_implications': strategy_implications
        }
```

## Integration with Trading Strategies

### Adaptive Strategy Framework

```python
class RegimeAdaptiveStrategy:
    """
    Strategy that adapts parameters based on regime detection
    """

    def __init__(self, base_strategy, regime_detector):
        self.base_strategy = base_strategy
        self.regime_detector = regime_detector
        self.regime_configs = self._define_regime_configs()

    def _define_regime_configs(self) -> Dict:
        """
        Define strategy parameters for each regime
        """
        return {
            'Low_Volatility': {
                'position_size_multiplier': 1.5,
                'stop_loss_multiplier': 0.8,
                'profit_target_multiplier': 1.2,
                'entry_threshold': 0.6  # Lower threshold for entries
            },
            'Medium_Volatility': {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'profit_target_multiplier': 1.0,
                'entry_threshold': 0.7  # Standard threshold
            },
            'High_Volatility': {
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.3,
                'profit_target_multiplier': 0.8,
                'entry_threshold': 0.8  # Higher threshold, more selective
            }
        }

    def generate_signal(self, market_data: Dict) -> Optional[Dict]:
        """
        Generate signal adapted to the current regime
        """
        # Detect current regime
        features = self.regime_detector.prepare_features(
            market_data['price_history']
        )
        current_regime = self.regime_detector.predict_regime(features)

        # Get regime-specific configuration
        regime_config = self.regime_configs.get(
            current_regime['regime_label'],
            self.regime_configs['Medium_Volatility']  # Default
        )

        # Adjust base strategy parameters
        adjusted_config = self._adjust_strategy_config(
            self.base_strategy.config,
            regime_config,
            current_regime['confidence']
        )

        # Generate signal with adjusted parameters
        self.base_strategy.update_config(adjusted_config)
        signal = self.base_strategy.generate_signal(market_data)

        # Add regime information to signal
        if signal:
            signal['regime_info'] = {
                'regime': current_regime['regime_label'],
                'confidence': current_regime['confidence'],
                'adjustments_applied': regime_config
            }

        return signal

    def _adjust_strategy_config(self,
                               base_config: Dict,
                               regime_config: Dict,
                               confidence: float) -> Dict:
        """
        Adjust strategy config based on regime, weighted by confidence
        """
        adjusted_config = base_config.copy()

        # Apply adjustments weighted by confidence
        for param, multiplier in regime_config.items():
            if param in base_config:
                adjustment = (multiplier - 1.0) * confidence + 1.0
                adjusted_config[param] *= adjustment

        return adjusted_config
```

### Regime-Based Risk Management

```python
class RegimeAwareRiskManager:
    """
    Risk management that adapts limits based on regimes
    """

    def __init__(self):
        self.base_limits = {
            'max_position_size': 1000,
            'max_daily_risk': 500,
            'max_correlation_exposure': 0.6
        }

    def get_adjusted_limits(self, regime_info: Dict) -> Dict:
        """
        Adjust risk limits based on the current regime
        """
        regime = regime_info['regime']
        confidence = regime_info['confidence']

        # Base adjustments per regime
        regime_adjustments = {
            'Low_Volatility': {
                'max_position_size': 1.3,
                'max_daily_risk': 1.2,
                'max_correlation_exposure': 0.8
            },
            'Medium_Volatility': {
                'max_position_size': 1.0,
                'max_daily_risk': 1.0,
                'max_correlation_exposure': 0.6
            },
            'High_Volatility': {
                'max_position_size': 0.6,
                'max_daily_risk': 0.7,
                'max_correlation_exposure': 0.4
            },
            'Crisis': {
                'max_position_size': 0.3,
                'max_daily_risk': 0.4,
                'max_correlation_exposure': 0.2
            }
        }

        adjustments = regime_adjustments.get(regime, regime_adjustments['Medium_Volatility'])

        # Apply adjustments weighted by confidence
        adjusted_limits = {}
        for limit, base_value in self.base_limits.items():
            if limit in adjustments:
                multiplier = adjustments[limit]
                # Weight adjustment by confidence
                effective_multiplier = (multiplier - 1.0) * confidence + 1.0
                adjusted_limits[limit] = base_value * effective_multiplier
            else:
                adjusted_limits[limit] = base_value

        return adjusted_limits
```

## Practical Implementation Guide

### 1. Data Requirements
```python
# Minimum data needed for robust regime detection
DATA_REQUIREMENTS = {
    'price_data': {
        'frequency': 'daily',  # Can work with daily, better with intraday
        'history': '2+ years',  # Minimum for regime detection
        'fields': ['open', 'high', 'low', 'close', 'volume']
    },
    'market_data': {
        'benchmarks': ['SPY', 'IWM', 'QQQ'],  # For correlation analysis
        'volatility_indices': ['VIX', 'RVX'],  # Volatility regime context
        'sentiment': ['AAII', 'Put/Call Ratio']  # Risk appetite indicators
    }
}
```

### 2. Implementation Workflow
```python
# Daily regime detection workflow
def daily_regime_update():
    """
    Daily process to update regime detection
    """

    # 1. Fetch latest market data
    latest_data = fetch_market_data()

    # 2. Update regime models
    regime_detector.update_with_new_data(latest_data)

    # 3. Detect current regime
    current_regime = regime_detector.predict_regime(latest_data)

    # 4. Adjust strategy parameters if regime changed
    if regime_has_changed(current_regime):
        update_strategy_parameters(current_regime)

        # 5. Notify about regime change
        send_regime_change_alert(current_regime)

    # 6. Log regime information
    log_regime_data(current_regime)

    return current_regime
```

### 3. Performance Monitoring
```python
def monitor_regime_detection_performance():
    """
    Monitor the effectiveness of regime detection
    """

    # Track regime transition accuracy
    regime_transitions = detect_regime_transitions()

    # Measure strategy performance by regime
    performance_by_regime = calculate_performance_by_regime()

    # Evaluate regime detection leading indicators
    regime_prediction_accuracy = evaluate_prediction_accuracy()

    return {
        'transition_accuracy': regime_transitions,
        'performance_by_regime': performance_by_regime,
        'prediction_accuracy': regime_prediction_accuracy
    }
```

## Integration with the Quant Playbook

### Next Steps:
1. **Implement Basic Regime Detection**: Start with volatility regimes
2. **Adapt Existing Strategies**: Modify Gap & Go and VWAP templates
3. **Create Regime Dashboard**: Monitor current regime in real-time
4. **Backtest Adaptive Strategies**: Compare regime-aware vs static approaches

### Related Concepts:
- **[Strategy Development](../technical-practices/Strategy-Development.md)**: Integrate regime awareness
- **[Risk Management](../core-concepts/Risk-Management.md)**: Regime-based risk adjustments
- **[Portfolio Optimization](./Portfolio-Optimization.md)**: Multi-strategy allocation by regime

---

**Remember**: Regime detection is not about predicting the future - it's about **adapting quickly** to changing market conditions. The goal is to recognize shifts early and adjust strategy parameters accordingly.
