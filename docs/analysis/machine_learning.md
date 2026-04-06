> 🇪🇸 [Leer en Español](machine_learning.es.md) | 🇺🇸 **English**

# Machine Learning Applied to Quantitative Trading

## Introduction

Machine Learning offers powerful tools for detecting non-linear patterns in financial markets, identifying market regimes, and creating predictive models. This documentation covers practical implementations specifically validated for quantitative trading.

## Unsupervised Models

### Hidden Markov Models (HMM) for Regime Detection

HMMs are especially useful for identifying hidden market states (bullish/bearish, high/low volatility) that are not directly observable but influence price behavior.

#### Core Concepts

**What Are Markov States?**
- They represent discrete conditions a system can occupy
- Only the current state matters for predicting the next step
- Hidden states influence the observations (prices) we see

**Applications in Trading:**
- Market regime detection (bullish/bearish)
- Identifying high/low volatility periods
- Changes in market structure
- Entry/exit signals based on state transitions

#### Basic Implementation with HMM

```python
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class MarketRegimeDetector:
    """
    Market regime detector using Hidden Markov Models
    """
    
    def __init__(self, n_components=2, covariance_type="full", random_state=42):
        """
        Parameters
        ----------
        n_components : int
            Number of hidden states (typically 2-4 for markets)
        covariance_type : str
            Covariance matrix type ('full', 'diag', 'tied', 'spherical')
        random_state : int
            Seed for reproducibility
        """
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, df):
        """
        Prepare features for the HMM model
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV columns
            
        Returns
        -------
        np.array
            Array of normalized features
        """
        features = pd.DataFrame(index=df.index)
        
        # Log returns
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Range diario normalizado
        features['daily_range'] = (df['High'] / df['Low']) - 1
        
        # Realized volatility (5-day window)
        features['realized_vol'] = features['log_returns'].rolling(5).std()
        
        # Relative volume
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # RSI as momentum proxy
        features['rsi'] = self.calculate_rsi(df['Close'], period=14)
        
        # Remove NaN and normalize
        features_clean = features.dropna()
        features_scaled = self.scaler.fit_transform(features_clean)
        
        return features_scaled, features_clean.index
    
    def calculate_rsi(self, series, period=14):
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def fit(self, df):
        """
        Train the HMM model
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data with OHLCV columns
        """
        features, self.feature_index = self.prepare_features(df)
        
        # Train the model
        self.model.fit(features)
        self.is_fitted = True
        
        # Predict states
        self.hidden_states = self.model.predict(features)
        
        # Save data for analysis
        self.features = features
        self.original_data = df.loc[self.feature_index]
        
        return self
    
    def predict_current_regime(self, df):
        """
        Predict the current market regime
        
        Returns
        -------
        dict
            Current regime information
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Run fit() first.")
        
        features, _ = self.prepare_features(df)
        
        # Predict current state
        current_state = self.model.predict(features[-1:].reshape(1, -1))[0]
        
        # Calculate probabilities
        log_prob, state_sequence = self.model.decode(features[-10:], algorithm="viterbi")
        state_probs = np.exp(self.model.predict_proba(features[-1:].reshape(1, -1)))[0]
        
        return {
            'current_state': current_state,
            'state_probabilities': state_probs,
            'confidence': np.max(state_probs),
            'recent_sequence': state_sequence
        }
    
    def analyze_regimes(self):
        """
        Analyze the characteristics of each regime
        """
        if not self.is_fitted:
            raise ValueError("Model not trained.")
        
        regime_analysis = {}
        
        for state in range(self.n_components):
            mask = self.hidden_states == state
            state_data = self.original_data[mask]
            
            if len(state_data) > 0:
                avg_return = state_data['Close'].pct_change().mean()
                volatility = state_data['Close'].pct_change().std()
                avg_volume = state_data['Volume'].mean()
                duration = len(state_data)
                
                regime_analysis[f'State_{state}'] = {
                    'average_return': avg_return,
                    'volatility': volatility,
                    'average_volume': avg_volume,
                    'duration_days': duration,
                    'percentage_time': duration / len(self.hidden_states),
                    'regime_type': 'Bullish' if avg_return > 0 else 'Bearish'
                }
        
        return regime_analysis
    
    def plot_regimes(self, title="Market Regimes Detection"):
        """
        Visualize the detected regimes
        """
        if not self.is_fitted:
            raise ValueError("Model not trained.")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Price with regimes
        colors = ['green', 'red', 'blue', 'orange'][:self.n_components]
        
        for state in range(self.n_components):
            mask = self.hidden_states == state
            state_data = self.original_data[mask]
            
            ax1.scatter(state_data.index, state_data['Close'], 
                       c=colors[state], label=f'Regime {state}', alpha=0.6, s=10)
        
        ax1.plot(self.original_data.index, self.original_data['Close'], 
                'k-', alpha=0.3, linewidth=0.5)
        ax1.set_title(f'{title} - Price Action')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: State sequence
        ax2.plot(self.feature_index, self.hidden_states, 'k-', linewidth=2)
        ax2.fill_between(self.feature_index, 0, self.hidden_states, alpha=0.3)
        ax2.set_title('Hidden States Sequence')
        ax2.set_ylabel('State')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def hmm_trading_strategy(df, detector, confidence_threshold=0.7):
    """
    HMM-based trading strategy
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical data
    detector : MarketRegimeDetector
        Trained detector
    confidence_threshold : float
        Confidence threshold for generating signals
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['signal'] = 0
    signals['regime'] = np.nan
    signals['confidence'] = np.nan
    
    # Rolling window for predictions
    window_size = 252  # 1 year of data
    
    for i in range(window_size, len(df)):
        # Training data
        train_data = df.iloc[i-window_size:i]
        
        # Train detector
        temp_detector = MarketRegimeDetector(n_components=2)
        temp_detector.fit(train_data)
        
        # Predict current regime
        current_data = df.iloc[i-50:i+1]  # Last 50 days for context
        regime_info = temp_detector.predict_current_regime(current_data)
        
        current_idx = df.index[i]
        signals.loc[current_idx, 'regime'] = regime_info['current_state']
        signals.loc[current_idx, 'confidence'] = regime_info['confidence']
        
        # Generate signals only with high confidence
        if regime_info['confidence'] > confidence_threshold:
            # Analyze regime characteristics
            regime_analysis = temp_detector.analyze_regimes()
            current_regime = f"State_{regime_info['current_state']}"
            
            if current_regime in regime_analysis:
                regime_return = regime_analysis[current_regime]['average_return']
                
                # Signal based on regime type
                if regime_return > 0.001:  # Bullish regime
                    signals.loc[current_idx, 'signal'] = 1
                elif regime_return < -0.001:  # Bearish regime
                    signals.loc[current_idx, 'signal'] = -1
    
    return signals

# Usage example completo
def hmm_example_analysis():
    """
    Complete HMM analysis example for trading
    """
    # Get data
    ticker = "SPY"
    df = yf.download(ticker, start="2020-01-01", end="2024-01-01", interval="1d")
    
    print(f"=== HMM ANALYSIS: {ticker} ===\n")
    
    # Create and train detector
    detector = MarketRegimeDetector(n_components=2, random_state=42)
    detector.fit(df)
    
    # Analyze regimes
    regime_analysis = detector.analyze_regimes()
    
    print("📊 REGIME ANALYSIS:")
    for regime, stats in regime_analysis.items():
        print(f"\n{regime} ({stats['regime_type']}):")
        print(f"   Average Return: {stats['average_return']:.4f}")
        print(f"   Volatility: {stats['volatility']:.4f}")
        print(f"   Duration: {stats['duration_days']} days")
        print(f"   % of Time: {stats['percentage_time']:.1%}")
    
    # Predict current regime
    current_regime = detector.predict_current_regime(df)
    print(f"\n🎯 CURRENT REGIME:")
    print(f"   State: {current_regime['current_state']}")
    print(f"   Confidence: {current_regime['confidence']:.1%}")
    print(f"   Probabilities: {current_regime['state_probabilities']}")
    
    # Generate strategy
    strategy_signals = hmm_trading_strategy(df, detector)
    
    # Strategy statistics
    total_signals = strategy_signals['signal'].abs().sum()
    long_signals = (strategy_signals['signal'] == 1).sum()
    short_signals = (strategy_signals['signal'] == -1).sum()
    
    print(f"\n📈 STRATEGY STATISTICS:")
    print(f"   Total Signals: {total_signals}")
    print(f"   Long Signals: {long_signals}")
    print(f"   Short Signals: {short_signals}")
    
    # Visualize
    detector.plot_regimes(f"HMM Regime Detection - {ticker}")
    
    return detector, strategy_signals

if __name__ == "__main__":
    hmm_example_analysis()
```

#### Advanced Strategy: Multi-State HMM

```python
class AdvancedMarketRegimeDetector:
    """
    Advanced detector with multiple states for complex markets
    """
    
    def __init__(self, n_components=4):
        """
        4 States típicos:
        0: Bull Market
        1: Bear Market  
        2: High Volatility (crisis)
        3: Low Volatility (consolidation)
        """
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            random_state=42
        )
        
    def prepare_advanced_features(self, df):
        """
        Advanced features for multi-state detection
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns across multiple timeframes
        features['returns_1d'] = df['Close'].pct_change()
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_20d'] = df['Close'].pct_change(20)
        
        # Volatilityes realizadas
        features['vol_5d'] = features['returns_1d'].rolling(5).std()
        features['vol_20d'] = features['returns_1d'].rolling(20).std()
        features['vol_60d'] = features['returns_1d'].rolling(60).std()
        
        # Momentum indicators
        features['rsi'] = self.calculate_rsi(df['Close'])
        features['macd'] = self.calculate_macd(df['Close'])
        
        # Volume patterns
        features['volume_trend'] = df['Volume'].rolling(20).mean() / df['Volume'].rolling(60).mean()
        features['volume_spike'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # VIX proxy (volatility of volatility)
        features['vol_of_vol'] = features['vol_20d'].rolling(10).std()
        
        return features.dropna()
    
    def calculate_macd(self, series, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        return macd_line
    
    def fit_advanced(self, df):
        """Train advanced model"""
        features = self.prepare_advanced_features(df)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled)
        self.hidden_states = self.model.predict(features_scaled)
        
        # Interpret states
        self.regime_interpretation = self.interpret_regimes(df, features)
        
        return self
    
    def interpret_regimes(self, df, features):
        """
        Automatically interpret what each state represents
        """
        interpretation = {}
        
        for state in range(self.n_components):
            mask = self.hidden_states == state
            state_features = features[mask]
            
            if len(state_features) > 0:
                avg_return = state_features['returns_1d'].mean()
                avg_vol = state_features['vol_20d'].mean()
                avg_rsi = state_features['rsi'].mean()
                
                # Classify state based on features
                if avg_return > 0.001 and avg_vol < state_features['vol_20d'].quantile(0.5):
                    regime_type = "Bull Market"
                elif avg_return < -0.001 and avg_vol < state_features['vol_20d'].quantile(0.5):
                    regime_type = "Bear Market"
                elif avg_vol > state_features['vol_20d'].quantile(0.75):
                    regime_type = "High Volatility/Crisis"
                else:
                    regime_type = "Consolidation/Low Volatility"
                
                interpretation[state] = {
                    'type': regime_type,
                    'avg_return': avg_return,
                    'avg_volatility': avg_vol,
                    'avg_rsi': avg_rsi,
                    'frequency': np.mean(mask)
                }
        
        return interpretation

def small_cap_hmm_strategy(df, lookback_days=252):
    """
    Small cap-specific HMM strategy
    """
    # Small cap-specific parameters
    detector = MarketRegimeDetector(n_components=3)  # 3 states: bullish, bearish, volatile
    
    # Small cap-specific features
    features = pd.DataFrame(index=df.index)
    
    # Gap detection
    features['gap_pct'] = (df['Open'] / df['Close'].shift(1)) - 1
    
    # Intraday range
    features['intraday_range'] = (df['High'] - df['Low']) / df['Open']
    
    # Volume spikes (crucial for small caps)
    features['volume_spike'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Price momentum
    features['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
    
    # Relative strength vs market
    spy_data = yf.download("SPY", start=df.index[0], end=df.index[-1])
    features['relative_strength'] = (df['Close'].pct_change() - 
                                   spy_data['Close'].pct_change().reindex(df.index))
    
    # Fit model with small cap specific features
    features_clean = features.dropna()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean)
    
    detector.model.fit(features_scaled)
    hidden_states = detector.model.predict(features_scaled)
    
    return {
        'states': hidden_states,
        'features': features_clean,
        'index': features_clean.index,
        'detector': detector
    }
```

## Supervised Models for Trading

### Price Prediction with XGBoost

Supervised models can predict future price movements based on historical features.

```python
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class TradingPredictor:
    """
    Price and direction predictor using XGBoost
    """
    
    def __init__(self, prediction_type='price', target_days=1):
        """
        Parameters
        ----------
        prediction_type : str
            'price' for regression, 'direction' for classification
        target_days : int
            Days ahead to predict
        """
        self.prediction_type = prediction_type
        self.target_days = target_days
        
        if prediction_type == 'price':
            self.model = XGBRegressor(
                max_depth=6,
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = XGBClassifier(
                max_depth=6,
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
    
    def create_features(self, df):
        """
        Create features for machine learning
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['sma_5'] = df['Close'].rolling(5).mean() / df['Close']
        features['sma_10'] = df['Close'].rolling(10).mean() / df['Close']
        features['sma_20'] = df['Close'].rolling(20).mean() / df['Close']
        
        # Volatility features
        features['volatility_5'] = df['Close'].pct_change().rolling(5).std()
        features['volatility_20'] = df['Close'].pct_change().rolling(20).std()
        
        # Momentum features
        features['roc_5'] = df['Close'].pct_change(5)
        features['roc_10'] = df['Close'].pct_change(10)
        features['roc_20'] = df['Close'].pct_change(20)
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(df['Close'])
        features['bb_position'] = self.calculate_bb_position(df['Close'])
        
        # Volume features
        features['volume_sma'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['price_volume'] = df['Close'] * df['Volume']
        
        # OHLC features
        features['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        features['open_close_pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = df['Close'].pct_change().shift(lag)
            features[f'volume_lag_{lag}'] = df['Volume'].pct_change().shift(lag)
        
        return features.dropna()
    
    def calculate_rsi(self, series, period=14):
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bb_position(self, series, period=20, std_mult=2):
        """Calculate position within Bollinger Bands"""
        sma = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = sma + (std * std_mult)
        lower = sma - (std * std_mult)
        return (series - lower) / (upper - lower)
    
    def create_targets(self, df):
        """
        Create target variables
        """
        if self.prediction_type == 'price':
            # Predict future price
            target = df['Close'].shift(-self.target_days)
        else:
            # Predict direction (classification)
            future_return = df['Close'].pct_change(self.target_days).shift(-self.target_days)
            target = (future_return > 0).astype(int)  # 1 if up, 0 if down
        
        return target
    
    def fit(self, df, test_size=0.2):
        """
        Train the model
        """
        # Create features and targets
        features = self.create_features(df)
        targets = self.create_targets(df)
        
        # Align data
        aligned_data = pd.concat([features, targets], axis=1).dropna()
        X = aligned_data.iloc[:, :-1]  # All columns except the last one
        y = aligned_data.iloc[:, -1]   # Última columna (target)
        
        # Temporal split (important for time series)
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Metrics
        if self.prediction_type == 'price':
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            self.metrics = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_rmse': np.sqrt(train_mse),
                'test_rmse': np.sqrt(test_mse)
            }
        else:
            train_accuracy = (train_pred.round() == y_train).mean()
            test_accuracy = (test_pred.round() == y_test).mean()
            
            self.metrics = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            }
        
        # Save data for analysis
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.train_pred, self.test_pred = train_pred, test_pred
        self.feature_names = X.columns.tolist()
        
        return self
    
    def predict_next(self, df, periods=1):
        """
        Predict next periods
        """
        features = self.create_features(df)
        latest_features = features.iloc[-periods:].values
        
        if latest_features.shape[0] == 0:
            raise ValueError("Not enough data to generate features")
        
        predictions = self.model.predict(latest_features)
        
        return predictions
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_predictions(self, title="Predictions vs Reality"):
        """
        Visualize predicciones vs realidad
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set
        ax1.scatter(self.y_train, self.train_pred, alpha=0.6)
        ax1.plot([self.y_train.min(), self.y_train.max()], 
                [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Training Set')
        ax1.grid(True, alpha=0.3)
        
        # Test set
        ax2.scatter(self.y_test, self.test_pred, alpha=0.6, color='orange')
        ax2.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Test Set')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        return fig

# ML-based trading strategy
def ml_trading_strategy(df, prediction_threshold=0.6):
    """
    Trading strategy using ML predictions
    """
    # Train direction predictor
    direction_predictor = TradingPredictor(prediction_type='direction', target_days=1)
    direction_predictor.fit(df)
    
    # Train price predictor
    price_predictor = TradingPredictor(prediction_type='price', target_days=1)
    price_predictor.fit(df)
    
    # Generate signals
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['signal'] = 0
    signals['confidence'] = 0
    
    # Rolling window for predictions
    window_size = 252
    
    for i in range(window_size, len(df) - 1):
        train_data = df.iloc[i-window_size:i]
        
        try:
            # Train models with data up to this point
            temp_direction = TradingPredictor(prediction_type='direction')
            temp_direction.fit(train_data, test_size=0.3)
            
            # Predict direction
            direction_pred = temp_direction.predict_next(train_data)[0]
            
            # Only generate signal if confidence is high
            if temp_direction.metrics['test_accuracy'] > prediction_threshold:
                current_idx = df.index[i]
                
                if direction_pred > 0.5:  # Bullish prediction
                    signals.loc[current_idx, 'signal'] = 1
                else:  # Bearish prediction
                    signals.loc[current_idx, 'signal'] = -1
                
                signals.loc[current_idx, 'confidence'] = temp_direction.metrics['test_accuracy']
        
        except Exception as e:
            continue
    
    return signals

# Usage example completo
def ml_example_analysis():
    """
    Complete ML analysis example for trading
    """
    # Get data
    ticker = "AAPL"
    df = yf.download(ticker, start="2020-01-01", end="2024-01-01", interval="1d")
    
    print(f"=== ML ANALYSIS: {ticker} ===\n")
    
    # Direction predictor
    print("🎯 DIRECTION PREDICTOR:")
    direction_model = TradingPredictor(prediction_type='direction', target_days=1)
    direction_model.fit(df)
    
    print(f"   Training Accuracy: {direction_model.metrics['train_accuracy']:.1%}")
    print(f"   Test Accuracy: {direction_model.metrics['test_accuracy']:.1%}")
    
    # Price predictor
    print(f"\n📈 PRICE PREDICTOR:")
    price_model = TradingPredictor(prediction_type='price', target_days=1)
    price_model.fit(df)
    
    print(f"   Training RMSE: ${price_model.metrics['train_rmse']:.2f}")
    print(f"   Test RMSE: ${price_model.metrics['test_rmse']:.2f}")
    
    # Importancia de características
    print(f"\n🔍 TOP FEATURES:")
    importance = direction_model.get_feature_importance(5)
    for _, row in importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Current predictions
    latest_direction = direction_model.predict_next(df, 1)[0]
    latest_price = price_model.predict_next(df, 1)[0]
    current_price = df['Close'].iloc[-1]
    
    print(f"\n🔮 PREDICTIONS:")
    print(f"   Next Day Direction: {'Up' if latest_direction > 0.5 else 'Down'}")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Predicted Price: ${latest_price:.2f}")
    print(f"   Expected Change: {(latest_price/current_price - 1):.1%}")
    
    # Generate strategy
    strategy_signals = ml_trading_strategy(df)
    
    # Strategy statistics
    total_signals = strategy_signals['signal'].abs().sum()
    avg_confidence = strategy_signals[strategy_signals['confidence'] > 0]['confidence'].mean()
    
    print(f"\n📊 ML STRATEGY:")
    print(f"   Total Signals: {total_signals}")
    print(f"   Confidence Promedio: {avg_confidence:.1%}")
    
    # Visualize
    direction_model.plot_predictions(f"Direction Prediction - {ticker}")
    price_model.plot_predictions(f"Price Prediction - {ticker}")
    
    return direction_model, price_model, strategy_signals

if __name__ == "__main__":
    ml_example_analysis()
```

## Best Practices for ML in Trading

### 1. Temporal Validation
```python
def time_series_cross_validation(df, model_class, n_splits=5):
    """
    Time series-specific cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(df):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        model = model_class()
        model.fit(train_data)
        
        # Evaluate on test data
        test_score = model.evaluate(test_data)
        scores.append(test_score)
    
    return np.array(scores)
```

### 2. Advanced Feature Engineering
```python
def create_advanced_features(df, market_data=None):
    """
    Create advanced features for ML
    """
    features = df.copy()
    
    # Market regime features
    if market_data is not None:
        features['beta'] = calculate_rolling_beta(df['Close'], market_data['Close'])
        features['relative_strength'] = df['Close'].pct_change() - market_data['Close'].pct_change()
    
    # Technical pattern features
    features['doji'] = detect_doji_patterns(df)
    features['hammer'] = detect_hammer_patterns(df)
    features['engulfing'] = detect_engulfing_patterns(df)
    
    # Volatility clustering
    features['vol_regime'] = detect_volatility_regime(df['Close'])
    
    # Seasonal features
    features['day_of_week'] = df.index.dayofweek
    features['month'] = df.index.month
    features['quarter'] = df.index.quarter
    
    return features
```

### 3. Overfitting Management
```python
class OverfittingDetector:
    """
    Overfitting detector for trading models
    """
    
    def __init__(self):
        self.warnings = []
    
    def check_overfitting(self, train_score, test_score, threshold=0.1):
        """
        Detect overfitting by comparing scores
        """
        if abs(train_score - test_score) > threshold:
            self.warnings.append("High difference between train/test scores")
        
        if train_score > 0.95:  # Too perfect
            self.warnings.append("Training score suspiciously high")
        
        return len(self.warnings) == 0
    
    def suggest_fixes(self):
        """
        Suggest fixes for overfitting
        """
        suggestions = [
            "Reduce model complexity (max_depth, n_estimators)",
            "Add regularization (L1/L2)",
            "Increase training data",
            "Use feature selection",
            "Implement early stopping"
        ]
        return suggestions
```

## Specific Applications for Small Caps

### 1. Gap Prediction
```python
def gap_prediction_model(df, gap_threshold=0.02):
    """
    Model specifically for predicting gaps in small caps
    """
    features = pd.DataFrame(index=df.index)
    
    # Previous day features
    features['prev_close_vol'] = df['Volume'].shift(1)
    features['prev_range'] = ((df['High'] - df['Low']) / df['Close']).shift(1)
    features['prev_return'] = df['Close'].pct_change().shift(1)
    
    # After-hours indicators
    features['ah_volume'] = df['Volume'].rolling(5).mean()  # Proxy
    features['news_sentiment'] = 0  # Placeholder for news sentiment
    
    # Create gap target
    gap_pct = (df['Open'] / df['Close'].shift(1)) - 1
    target = (abs(gap_pct) > gap_threshold).astype(int)
    
    return features, target
```

### 2. Volatility Prediction
```python
def volatility_prediction_model(df, horizon=5):
    """
    Predict future volatility for small caps
    """
    # GARCH-like features
    returns = df['Close'].pct_change()
    
    features = pd.DataFrame(index=df.index)
    features['returns_lag1'] = returns.shift(1)
    features['returns_lag2'] = returns.shift(2)
    features['vol_lag1'] = returns.rolling(5).std().shift(1)
    features['vol_lag2'] = returns.rolling(10).std().shift(1)
    
    # Target: future volatility
    target = returns.rolling(horizon).std().shift(-horizon)
    
    return features.dropna(), target.dropna()
```

## Evaluation Metrics for Trading

```python
def evaluate_trading_model(predictions, actual_returns, transaction_cost=0.001):
    """
    Evaluate modelo desde perspectiva de trading
    """
    # Convert predictions to trading signals
    signals = np.where(predictions > 0.5, 1, -1)
    
    # Calculate strategy returns
    strategy_returns = signals * actual_returns - abs(np.diff(signals, prepend=signals[0])) * transaction_cost
    
    # Trading-specific metrics
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    max_drawdown = calculate_max_drawdown(strategy_returns.cumsum())
    hit_rate = (strategy_returns > 0).mean()
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate,
        'total_return': strategy_returns.sum(),
        'volatility': strategy_returns.std() * np.sqrt(252)
    }

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()
```

## Next Step

With Machine Learning mastered, let's continue with [Sentiment Analysis](sentiment_analysis.md) to incorporate alternative data into our strategies.