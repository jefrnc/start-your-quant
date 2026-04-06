> 🇪🇸 [Leer en Español](Dynamic-Position-Sizing.es.md) | 🇺🇸 **English**

# Dynamic Position Sizing with Volatility Clustering

## Why Dynamic Position Sizing?

Traditional position sizing (fixed per trade) ignores a fundamental reality of financial markets: **volatility is not constant**. In small caps, this is especially critical because:

- **Volatility clustering**: Periods of high volatility tend to cluster together
- **Regime shifts**: Market conditions change dramatically
- **Liquidity cycles**: Available liquidity varies over time
- **News-driven spikes**: Small caps react violently to news
- **Float dynamics**: Changes in float affect price volatility

### The Problem with Fixed Position Sizing

```python
# Fixed position sizing - PROBLEMATIC
FIXED_RISK_PER_TRADE = 10.0  # $10 per trade always

# Problems:
# 1. In low volatility: Leaving money on the table
# 2. In high volatility: Taking excessive risk
# 3. During crisis: No adjustment for market stress
# 4. Earnings season: Same size despite elevated risk
```

## Dynamic Position Sizing Framework

### 1. Volatility-Based Position Sizing

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class VolatilityRegime(Enum):
    """Volatility regimes for position sizing"""
    ULTRA_LOW = "ultra_low"      # < 10th percentile
    LOW = "low"                  # 10th-30th percentile
    NORMAL = "normal"            # 30th-70th percentile
    HIGH = "high"                # 70th-90th percentile
    EXTREME = "extreme"          # > 90th percentile


@dataclass
class PositionSizingConfig:
    """Configuration for dynamic position sizing"""

    # Base parameters
    base_risk_per_trade: float = 10.0      # Base risk amount ($)
    max_position_value: float = 100.0      # Maximum position value ($)
    min_position_value: float = 20.0       # Minimum position value ($)

    # Volatility adjustments
    volatility_lookback_days: int = 60     # Days for volatility calculation
    volatility_adjustment_factor: float = 2.0  # Max adjustment multiplier

    # Regime-based multipliers
    regime_multipliers: Dict[VolatilityRegime, float] = None

    # Correlation adjustments
    max_correlation_exposure: float = 0.6  # Max exposure to correlated positions
    correlation_lookback_days: int = 30    # Days for correlation calculation

    # Market stress indicators
    vix_threshold_high: float = 25.0       # VIX level for stress adjustment
    vix_threshold_extreme: float = 35.0    # VIX level for extreme adjustment

    def __post_init__(self):
        if self.regime_multipliers is None:
            self.regime_multipliers = {
                VolatilityRegime.ULTRA_LOW: 1.8,   # Increase size in low vol
                VolatilityRegime.LOW: 1.3,
                VolatilityRegime.NORMAL: 1.0,      # Baseline
                VolatilityRegime.HIGH: 0.7,        # Reduce size in high vol
                VolatilityRegime.EXTREME: 0.4      # Dramatically reduce in extreme vol
            }


class DynamicPositionSizer:
    """
    Advanced position sizing that adapts to:
    - Volatility regimes
    - Market correlation
    - Portfolio heat
    - Risk parity principles
    """

    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.volatility_history = {}
        self.correlation_matrix = None
        self.current_positions = {}
        self.portfolio_heat = 0.0

    def calculate_position_size(self,
                               symbol: str,
                               entry_price: float,
                               stop_loss_price: float,
                               market_data: Dict,
                               current_positions: Dict = None) -> Dict:
        """
        Calculate optimal position size based on multiple factors

        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            market_data: Current market data including volatility info
            current_positions: Current portfolio positions

        Returns:
            Dict with position sizing recommendation
        """

        if current_positions:
            self.current_positions = current_positions

        # 1. Base position size (traditional risk-based)
        base_size = self._calculate_base_position_size(
            entry_price, stop_loss_price
        )

        # 2. Volatility adjustment
        volatility_multiplier = self._calculate_volatility_adjustment(
            symbol, market_data
        )

        # 3. Market regime adjustment
        regime_multiplier = self._calculate_regime_adjustment(market_data)

        # 4. Correlation adjustment
        correlation_multiplier = self._calculate_correlation_adjustment(
            symbol, market_data
        )

        # 5. Portfolio heat adjustment
        heat_multiplier = self._calculate_portfolio_heat_adjustment()

        # 6. Liquidity adjustment (small caps specific)
        liquidity_multiplier = self._calculate_liquidity_adjustment(
            symbol, market_data
        )

        # Combine all adjustments
        total_multiplier = (
            volatility_multiplier *
            regime_multiplier *
            correlation_multiplier *
            heat_multiplier *
            liquidity_multiplier
        )

        # Apply adjustments
        adjusted_size = base_size * total_multiplier

        # Apply position limits
        final_position_value = entry_price * adjusted_size
        final_position_value = max(
            min(final_position_value, self.config.max_position_value),
            self.config.min_position_value
        )

        final_shares = int(final_position_value / entry_price)

        # Calculate actual risk
        actual_risk = (entry_price - stop_loss_price) * final_shares

        return {
            'symbol': symbol,
            'recommended_shares': final_shares,
            'position_value': final_position_value,
            'actual_risk': actual_risk,
            'risk_percentage': actual_risk / final_position_value * 100,
            'adjustments': {
                'base_size': base_size,
                'volatility_multiplier': volatility_multiplier,
                'regime_multiplier': regime_multiplier,
                'correlation_multiplier': correlation_multiplier,
                'heat_multiplier': heat_multiplier,
                'liquidity_multiplier': liquidity_multiplier,
                'total_multiplier': total_multiplier
            },
            'volatility_regime': self._determine_volatility_regime(symbol, market_data),
            'confidence': self._calculate_sizing_confidence(market_data)
        }

    def _calculate_base_position_size(self,
                                     entry_price: float,
                                     stop_loss_price: float) -> int:
        """Calculate base position size using fixed risk"""

        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            return 0

        base_shares = int(self.config.base_risk_per_trade / risk_per_share)
        return max(base_shares, 1)

    def _calculate_volatility_adjustment(self,
                                       symbol: str,
                                       market_data: Dict) -> float:
        """
        Adjust position size based on realized volatility

        Lower volatility = larger positions
        Higher volatility = smaller positions
        """

        # Get realized volatility
        returns = market_data.get('returns_series', pd.Series())

        if len(returns) < 20:
            return 1.0  # Default if insufficient data

        # Calculate rolling realized volatility
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        current_vol = realized_vol.iloc[-1] if len(realized_vol) > 0 else 0.2

        # Historical volatility for percentile calculation
        if len(realized_vol) >= self.config.volatility_lookback_days:
            hist_vol = realized_vol.tail(self.config.volatility_lookback_days)
            vol_percentile = (hist_vol <= current_vol).mean()
        else:
            vol_percentile = 0.5  # Default to median

        # Volatility adjustment curve
        if vol_percentile <= 0.1:  # Ultra low volatility
            multiplier = 1.8
        elif vol_percentile <= 0.3:  # Low volatility
            multiplier = 1.3
        elif vol_percentile <= 0.7:  # Normal volatility
            multiplier = 1.0
        elif vol_percentile <= 0.9:  # High volatility
            multiplier = 0.7
        else:  # Extreme volatility
            multiplier = 0.4

        # Store for regime detection
        self.volatility_history[symbol] = {
            'current_vol': current_vol,
            'vol_percentile': vol_percentile,
            'multiplier': multiplier
        }

        return multiplier

    def _calculate_regime_adjustment(self, market_data: Dict) -> float:
        """
        Adjust based on broader market regime

        Uses VIX, market correlation, and other regime indicators
        """

        vix_level = market_data.get('vix', 20.0)
        market_stress_indicator = market_data.get('market_stress', 0.0)

        # VIX-based adjustment
        if vix_level >= self.config.vix_threshold_extreme:
            vix_multiplier = 0.3  # Extreme caution
        elif vix_level >= self.config.vix_threshold_high:
            vix_multiplier = 0.6  # High caution
        elif vix_level <= 15:
            vix_multiplier = 1.2  # Low fear = slightly larger positions
        else:
            vix_multiplier = 1.0  # Normal

        # Market stress adjustment
        stress_multiplier = max(0.3, 1.0 - market_stress_indicator)

        return vix_multiplier * stress_multiplier

    def _calculate_correlation_adjustment(self,
                                        symbol: str,
                                        market_data: Dict) -> float:
        """
        Adjust based on correlation with existing positions

        High correlation = reduce position size to avoid concentration
        """

        if not self.current_positions:
            return 1.0

        # Get correlation data
        symbol_returns = market_data.get('returns_series', pd.Series())

        if len(symbol_returns) < 30:
            return 1.0  # Insufficient data

        # Calculate correlation with existing positions
        correlations = []

        for existing_symbol, position_info in self.current_positions.items():
            if existing_symbol == symbol:
                continue

            existing_returns = market_data.get(f'{existing_symbol}_returns', pd.Series())

            if len(existing_returns) >= 30:
                # Align series and calculate correlation
                aligned_data = pd.DataFrame({
                    'target': symbol_returns,
                    'existing': existing_returns
                }).dropna()

                if len(aligned_data) >= 20:
                    correlation = aligned_data['target'].corr(aligned_data['existing'])
                    position_weight = position_info.get('weight', 0.1)

                    # Weight correlation by position size
                    correlations.append(abs(correlation) * position_weight)

        if not correlations:
            return 1.0

        # Calculate exposure-weighted average correlation
        avg_correlation = np.mean(correlations)

        # Adjustment based on correlation
        if avg_correlation >= 0.7:
            correlation_multiplier = 0.5  # Highly correlated - reduce significantly
        elif avg_correlation >= 0.5:
            correlation_multiplier = 0.7  # Moderately correlated - reduce
        elif avg_correlation >= 0.3:
            correlation_multiplier = 0.9  # Slightly correlated - slight reduction
        else:
            correlation_multiplier = 1.1  # Low correlation - slight increase

        return correlation_multiplier

    def _calculate_portfolio_heat_adjustment(self) -> float:
        """
        Adjust based on current portfolio heat (total risk exposure)

        High portfolio heat = reduce new position sizes
        """

        if not self.current_positions:
            return 1.0

        # Calculate total portfolio heat
        total_risk = 0
        total_value = 0

        for symbol, position in self.current_positions.items():
            position_risk = position.get('current_risk', 0)
            position_value = position.get('current_value', 0)

            total_risk += position_risk
            total_value += position_value

        if total_value <= 0:
            return 1.0

        heat_ratio = total_risk / total_value

        # Adjustment based on portfolio heat
        if heat_ratio >= 0.15:  # Very high heat (15%+ portfolio at risk)
            heat_multiplier = 0.4
        elif heat_ratio >= 0.10:  # High heat (10%+ portfolio at risk)
            heat_multiplier = 0.6
        elif heat_ratio >= 0.05:  # Moderate heat (5%+ portfolio at risk)
            heat_multiplier = 0.8
        else:  # Low heat
            heat_multiplier = 1.1

        self.portfolio_heat = heat_ratio
        return heat_multiplier

    def _calculate_liquidity_adjustment(self,
                                      symbol: str,
                                      market_data: Dict) -> float:
        """
        Adjust for small cap-specific liquidity constraints

        Lower liquidity = smaller positions to avoid market impact
        """

        avg_volume = market_data.get('avg_volume_20d', 100000)
        current_volume = market_data.get('current_volume', avg_volume)
        bid_ask_spread = market_data.get('bid_ask_spread_pct', 0.01)

        # Volume-based adjustment
        if avg_volume >= 1000000:  # High volume
            volume_multiplier = 1.0
        elif avg_volume >= 500000:  # Medium volume
            volume_multiplier = 0.9
        elif avg_volume >= 100000:  # Low volume
            volume_multiplier = 0.7
        else:  # Very low volume
            volume_multiplier = 0.5

        # Spread-based adjustment
        if bid_ask_spread <= 0.005:  # Tight spread (0.5%)
            spread_multiplier = 1.0
        elif bid_ask_spread <= 0.01:  # Normal spread (1%)
            spread_multiplier = 0.9
        elif bid_ask_spread <= 0.02:  # Wide spread (2%)
            spread_multiplier = 0.7
        else:  # Very wide spread
            spread_multiplier = 0.5

        return volume_multiplier * spread_multiplier

    def _determine_volatility_regime(self,
                                   symbol: str,
                                   market_data: Dict) -> VolatilityRegime:
        """Determine current volatility regime"""

        if symbol not in self.volatility_history:
            return VolatilityRegime.NORMAL

        vol_percentile = self.volatility_history[symbol]['vol_percentile']

        if vol_percentile <= 0.1:
            return VolatilityRegime.ULTRA_LOW
        elif vol_percentile <= 0.3:
            return VolatilityRegime.LOW
        elif vol_percentile <= 0.7:
            return VolatilityRegime.NORMAL
        elif vol_percentile <= 0.9:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def _calculate_sizing_confidence(self, market_data: Dict) -> float:
        """
        Calculate confidence in position sizing recommendation

        Higher confidence when:
        - More historical data available
        - Stable volatility regime
        - Clear market conditions
        """

        data_quality = market_data.get('data_quality_score', 0.5)
        regime_stability = market_data.get('regime_stability', 0.5)
        market_clarity = 1.0 - market_data.get('market_stress', 0.0)

        confidence = (data_quality + regime_stability + market_clarity) / 3
        return max(0.1, min(confidence, 1.0))


# Kelly Criterion Implementation for Advanced Users
class KellyCriterionSizer:
    """
    Kelly Criterion position sizing with modifications for practical trading

    Kelly Criterion: f* = (bp - q) / b
    Where:
    - f* = fraction of capital to wager
    - b = odds of winning (profit/loss ratio)
    - p = probability of winning
    - q = probability of losing (1-p)
    """

    def __init__(self, max_kelly_fraction: float = 0.25):
        """
        Args:
            max_kelly_fraction: Maximum Kelly fraction to prevent over-leveraging
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.historical_trades = []

    def calculate_kelly_size(self,
                           win_probability: float,
                           avg_win: float,
                           avg_loss: float,
                           account_value: float) -> Dict:
        """
        Calculate position size using Kelly Criterion

        Args:
            win_probability: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            account_value: Current account value

        Returns:
            Dict with Kelly sizing recommendation
        """

        if avg_loss <= 0 or win_probability <= 0 or win_probability >= 1:
            return {'kelly_fraction': 0, 'position_value': 0, 'warning': 'Invalid parameters'}

        # Kelly fraction calculation
        win_loss_ratio = avg_win / avg_loss
        lose_probability = 1 - win_probability

        kelly_fraction = (
            (win_loss_ratio * win_probability - lose_probability) /
            win_loss_ratio
        )

        # Apply safety constraints
        # 1. Cap at maximum Kelly fraction
        capped_kelly = min(kelly_fraction, self.max_kelly_fraction)

        # 2. Never bet more than makes sense
        capped_kelly = max(capped_kelly, 0)

        # 3. Fractional Kelly (more conservative)
        conservative_kelly = capped_kelly * 0.5  # Half Kelly for safety

        position_value = account_value * conservative_kelly

        return {
            'theoretical_kelly_fraction': kelly_fraction,
            'capped_kelly_fraction': capped_kelly,
            'conservative_kelly_fraction': conservative_kelly,
            'position_value': position_value,
            'win_probability': win_probability,
            'win_loss_ratio': win_loss_ratio,
            'recommendation': self._get_kelly_recommendation(kelly_fraction)
        }

    def _get_kelly_recommendation(self, kelly_fraction: float) -> str:
        """Provide recommendation based on Kelly fraction"""

        if kelly_fraction <= 0:
            return "AVOID - Negative expectancy"
        elif kelly_fraction <= 0.02:
            return "MINIMAL - Very small edge"
        elif kelly_fraction <= 0.05:
            return "SMALL - Small edge, conservative sizing"
        elif kelly_fraction <= 0.15:
            return "MODERATE - Good edge, moderate sizing"
        elif kelly_fraction <= 0.25:
            return "LARGE - Strong edge, aggressive sizing"
        else:
            return "EXTREME - Very strong edge, but cap position size"

    def update_trade_history(self, trade_result: Dict):
        """Update historical trade data for Kelly calculation"""
        self.historical_trades.append(trade_result)

        # Keep only recent trades (e.g., last 100)
        if len(self.historical_trades) > 100:
            self.historical_trades = self.historical_trades[-100:]

    def get_current_kelly_stats(self) -> Dict:
        """Get current Kelly statistics based on trade history"""

        if len(self.historical_trades) < 10:
            return {'error': 'Insufficient trade history (minimum 10 trades)'}

        pnl_values = [trade['pnl'] for trade in self.historical_trades]
        wins = [pnl for pnl in pnl_values if pnl > 0]
        losses = [abs(pnl) for pnl in pnl_values if pnl < 0]

        if not wins or not losses:
            return {'error': 'Need both winning and losing trades'}

        win_probability = len(wins) / len(pnl_values)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        return {
            'win_probability': win_probability,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.historical_trades),
            'profit_factor': sum(wins) / sum(losses) if losses else float('inf')
        }


# Integrated Position Sizing System
class IntegratedPositionSizingSystem:
    """
    Integrated system that combines multiple position sizing approaches
    """

    def __init__(self, config: PositionSizingConfig):
        self.dynamic_sizer = DynamicPositionSizer(config)
        self.kelly_sizer = KellyCriterionSizer()
        self.config = config

    def get_optimal_position_size(self,
                                 symbol: str,
                                 entry_price: float,
                                 stop_loss_price: float,
                                 market_data: Dict,
                                 current_positions: Dict = None,
                                 account_value: float = 10000) -> Dict:
        """
        Get optimal position size using multiple approaches
        """

        # 1. Dynamic volatility-based sizing
        dynamic_result = self.dynamic_sizer.calculate_position_size(
            symbol, entry_price, stop_loss_price, market_data, current_positions
        )

        # 2. Kelly criterion sizing (if sufficient trade history)
        kelly_stats = self.kelly_sizer.get_current_kelly_stats()
        kelly_result = None

        if 'error' not in kelly_stats:
            kelly_result = self.kelly_sizer.calculate_kelly_size(
                kelly_stats['win_probability'],
                kelly_stats['avg_win'],
                kelly_stats['avg_loss'],
                account_value
            )

        # 3. Combine approaches
        if kelly_result:
            # Use Kelly as a check on dynamic sizing
            kelly_position_value = kelly_result['position_value']
            dynamic_position_value = dynamic_result['position_value']

            # If Kelly suggests much smaller position, use the smaller one
            if kelly_position_value < dynamic_position_value * 0.5:
                recommended_value = kelly_position_value
                sizing_method = 'kelly_conservative'
            elif kelly_position_value > dynamic_position_value * 2:
                recommended_value = dynamic_position_value
                sizing_method = 'dynamic_capped'
            else:
                # Use average of both methods
                recommended_value = (kelly_position_value + dynamic_position_value) / 2
                sizing_method = 'hybrid'

            recommended_shares = int(recommended_value / entry_price)
        else:
            recommended_shares = dynamic_result['recommended_shares']
            recommended_value = dynamic_result['position_value']
            sizing_method = 'dynamic_only'

        actual_risk = (entry_price - stop_loss_price) * recommended_shares

        return {
            'symbol': symbol,
            'recommended_shares': recommended_shares,
            'position_value': recommended_value,
            'actual_risk': actual_risk,
            'sizing_method': sizing_method,
            'dynamic_result': dynamic_result,
            'kelly_result': kelly_result,
            'confidence': dynamic_result['confidence'],
            'warnings': self._generate_warnings(dynamic_result, kelly_result)
        }

    def _generate_warnings(self,
                         dynamic_result: Dict,
                         kelly_result: Optional[Dict]) -> List[str]:
        """Generate warnings based on sizing analysis"""

        warnings = []

        # High volatility warning
        if dynamic_result['volatility_regime'] == VolatilityRegime.EXTREME:
            warnings.append("⚠️ EXTREME VOLATILITY - Consider skipping or using minimal size")

        # High correlation warning
        correlation_mult = dynamic_result['adjustments']['correlation_multiplier']
        if correlation_mult <= 0.6:
            warnings.append("⚠️ HIGH CORRELATION - Position highly correlated with existing holdings")

        # Portfolio heat warning
        heat_mult = dynamic_result['adjustments']['heat_multiplier']
        if heat_mult <= 0.5:
            warnings.append("⚠️ HIGH PORTFOLIO HEAT - Consider reducing overall exposure")

        # Kelly criterion warnings
        if kelly_result:
            kelly_fraction = kelly_result['theoretical_kelly_fraction']
            if kelly_fraction <= 0:
                warnings.append("🛑 KELLY NEGATIVE - Strategy has negative expectancy")
            elif kelly_fraction >= 0.5:
                warnings.append("⚠️ KELLY EXTREME - Very high Kelly fraction, use caution")

        # Low confidence warning
        if dynamic_result['confidence'] < 0.3:
            warnings.append("⚠️ LOW CONFIDENCE - Insufficient data for reliable sizing")

        return warnings


# Example usage
def example_dynamic_position_sizing():
    """
    Complete dynamic position sizing example
    """

    # Configure position sizing
    config = PositionSizingConfig(
        base_risk_per_trade=15.0,
        max_position_value=150.0,
        volatility_lookback_days=60
    )

    # Initialize integrated system
    sizing_system = IntegratedPositionSizingSystem(config)

    # Mock market data
    market_data = {
        'returns_series': pd.Series(np.random.normal(0.001, 0.02, 100)),  # 100 days of returns
        'vix': 18.5,
        'market_stress': 0.2,
        'avg_volume_20d': 250000,
        'current_volume': 300000,
        'bid_ask_spread_pct': 0.008,
        'data_quality_score': 0.8,
        'regime_stability': 0.7
    }

    # Current positions
    current_positions = {
        'STOCK1': {'current_value': 500, 'current_risk': 25, 'weight': 0.3},
        'STOCK2': {'current_value': 300, 'current_risk': 15, 'weight': 0.2}
    }

    # Calculate position size
    symbol = "NEWSTOCK"
    entry_price = 5.50
    stop_loss_price = 5.20

    result = sizing_system.get_optimal_position_size(
        symbol=symbol,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        market_data=market_data,
        current_positions=current_positions,
        account_value=2000
    )

    print(f"Dynamic Position Sizing for ${symbol}")
    print(f"Entry: ${entry_price}, Stop: ${stop_loss_price}")
    print(f"\nRecommendation:")
    print(f"Shares: {result['recommended_shares']}")
    print(f"Position Value: ${result['position_value']:.2f}")
    print(f"Actual Risk: ${result['actual_risk']:.2f}")
    print(f"Sizing Method: {result['sizing_method']}")
    print(f"Confidence: {result['confidence']:.2%}")

    if result['warnings']:
        print(f"\nWarnings:")
        for warning in result['warnings']:
            print(f"  {warning}")

    # Show detailed adjustments
    adjustments = result['dynamic_result']['adjustments']
    print(f"\nDynamic Adjustments:")
    print(f"Volatility: {adjustments['volatility_multiplier']:.2f}x")
    print(f"Regime: {adjustments['regime_multiplier']:.2f}x")
    print(f"Correlation: {adjustments['correlation_multiplier']:.2f}x")
    print(f"Portfolio Heat: {adjustments['heat_multiplier']:.2f}x")
    print(f"Liquidity: {adjustments['liquidity_multiplier']:.2f}x")
    print(f"Total: {adjustments['total_multiplier']:.2f}x")

if __name__ == "__main__":
    example_dynamic_position_sizing()
```

## Integration with Trading Strategies

### 1. **Adaptive Strategy Implementation**

```python
class VolatilityAdaptiveStrategy:
    """
    Strategy that adapts position sizing based on volatility clustering
    """

    def __init__(self, base_strategy, position_sizer):
        self.base_strategy = base_strategy
        self.position_sizer = position_sizer

    def execute_trade(self, signal, market_data):
        """Execute trade with dynamic position sizing"""

        if not signal:
            return None

        # Get optimal position size
        sizing_result = self.position_sizer.get_optimal_position_size(
            symbol=signal['symbol'],
            entry_price=signal['price'],
            stop_loss_price=signal['stop_loss'],
            market_data=market_data,
            current_positions=self.get_current_positions()
        )

        # Check warnings
        if any('EXTREME' in warning for warning in sizing_result['warnings']):
            # Skip trade in extreme conditions
            return None

        # Execute with calculated size
        signal['quantity'] = sizing_result['recommended_shares']
        signal['sizing_info'] = sizing_result

        return self.base_strategy.execute_trade(signal)
```

### 2. **Risk Parity Portfolio Construction**

```python
class RiskParityPositionSizer:
    """
    Position sizing based on risk parity principles
    Equal risk contribution from each position
    """

    def calculate_risk_parity_sizes(self,
                                   symbols: List[str],
                                   prices: Dict[str, float],
                                   volatilities: Dict[str, float],
                                   target_portfolio_risk: float = 0.15) -> Dict:
        """
        Calculate position sizes for equal risk contribution
        """

        # Target risk per position
        risk_per_position = target_portfolio_risk / len(symbols)

        position_sizes = {}

        for symbol in symbols:
            price = prices[symbol]
            volatility = volatilities[symbol]

            # Position size for equal risk contribution
            # Risk = Position_Value * Volatility
            # Position_Value = Risk / Volatility
            position_value = risk_per_position / volatility
            shares = int(position_value / price)

            position_sizes[symbol] = {
                'shares': shares,
                'position_value': shares * price,
                'expected_risk': (shares * price) * volatility,
                'volatility': volatility
            }

        return position_sizes
```

## Advanced Applications

### 1. **Machine Learning-Enhanced Sizing**

```python
from sklearn.ensemble import RandomForestRegressor
import joblib

class MLPositionSizer:
    """
    ML-enhanced position sizing that learns optimal sizing decisions
    """

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_scaler = None

    def prepare_features(self, market_data: Dict, historical_data: pd.DataFrame) -> np.array:
        """
        Prepare features for ML model

        Features include:
        - Current volatility vs historical
        - Market regime indicators
        - Volume characteristics
        - Correlation measures
        - Previous trade outcomes
        """

        features = []

        # Volatility features
        current_vol = market_data.get('realized_volatility', 0.2)
        hist_vol_mean = historical_data['volatility'].mean()
        hist_vol_std = historical_data['volatility'].std()
        vol_zscore = (current_vol - hist_vol_mean) / hist_vol_std if hist_vol_std > 0 else 0

        features.extend([current_vol, hist_vol_mean, vol_zscore])

        # Market features
        features.extend([
            market_data.get('vix', 20),
            market_data.get('market_stress', 0),
            market_data.get('volume_ratio', 1.0)
        ])

        # Portfolio features
        features.extend([
            market_data.get('portfolio_correlation', 0.5),
            market_data.get('portfolio_heat', 0.05)
        ])

        return np.array(features).reshape(1, -1)

    def train_model(self, historical_trades: pd.DataFrame):
        """
        Train ML model on historical trade outcomes

        Target variable: Optimal position size multiplier
        """

        # Prepare training data
        X = []
        y = []

        for _, trade in historical_trades.iterrows():
            features = self._extract_trade_features(trade)
            optimal_multiplier = self._calculate_optimal_multiplier(trade)

            X.append(features)
            y.append(optimal_multiplier)

        X = np.array(X)
        y = np.array(y)

        # Train model
        self.model.fit(X, y)
        self.is_trained = True

        return self

    def predict_optimal_multiplier(self, market_data: Dict, historical_data: pd.DataFrame) -> float:
        """
        Predict optimal position size multiplier
        """

        if not self.is_trained:
            return 1.0  # Default multiplier

        features = self.prepare_features(market_data, historical_data)
        multiplier = self.model.predict(features)[0]

        # Constrain to reasonable range
        return np.clip(multiplier, 0.1, 3.0)
```

## Performance Monitoring and Optimization

### 1. **Position Sizing Performance Tracker**

```python
class PositionSizingTracker:
    """
    Track performance of position sizing decisions
    """

    def __init__(self):
        self.sizing_decisions = []
        self.performance_metrics = {}

    def log_sizing_decision(self, decision: Dict):
        """Log position sizing decision for analysis"""
        self.sizing_decisions.append({
            'timestamp': datetime.now(),
            **decision
        })

    def analyze_sizing_performance(self) -> Dict:
        """Analyze effectiveness of position sizing decisions"""

        if len(self.sizing_decisions) < 10:
            return {'error': 'Insufficient data'}

        df = pd.DataFrame(self.sizing_decisions)

        # Analyze by volatility regime
        regime_performance = df.groupby('volatility_regime').agg({
            'actual_return': ['mean', 'std', 'count'],
            'sharpe_ratio': 'mean'
        })

        # Analyze sizing method effectiveness
        method_performance = df.groupby('sizing_method').agg({
            'actual_return': ['mean', 'std'],
            'max_drawdown': 'mean'
        })

        return {
            'regime_performance': regime_performance,
            'method_performance': method_performance,
            'total_decisions': len(df),
            'avg_confidence': df['confidence'].mean(),
            'recommendations': self._generate_sizing_recommendations(df)
        }

    def _generate_sizing_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on performance analysis"""

        recommendations = []

        # Check for consistent over/under-sizing
        if df['position_too_large'].mean() > 0.3:
            recommendations.append("Consider reducing position sizes - over-sizing detected")

        if df['volatility_regime'].value_counts().get('EXTREME', 0) > 0:
            extreme_performance = df[df['volatility_regime'] == 'EXTREME']['actual_return'].mean()
            if extreme_performance < -0.02:
                recommendations.append("Avoid trading in extreme volatility regimes")

        return recommendations
```

---

**Integration Points**:
- **[Risk Management](../core-concepts/Risk-Management.md)**: Enhanced risk controls
- **[Strategy Development](../technical-practices/Strategy-Development.md)**: Adaptive strategies
- **[Regime Detection](./Regime-Detection.md)**: Regime-aware sizing
- **[Portfolio Optimization](./Portfolio-Optimization.md)**: Multi-strategy coordination

This dynamic position sizing framework enables **automatic adaptation** of position sizes based on real market conditions, delivering better risk-adjusted returns and drawdown control.
