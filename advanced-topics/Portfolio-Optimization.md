> 🇪🇸 [Leer en Español](Portfolio-Optimization.es.md) | 🇺🇸 **English**

# Multi-Strategy Portfolio Management

## Why Multi-Strategy Portfolio Management?

Running multiple strategies simultaneously is not simply "making more trades" - it requires **sophisticated orchestration** to maximize portfolio-level returns while controlling aggregate risk. For small caps, this is especially critical because:

- **Strategy correlation varies** with market regimes
- **Capacity constraints** limit individual strategy scaling
- **Risk concentration** can emerge unexpectedly
- **Suboptimal capital allocation** destroys alpha
- **Interaction effects** between strategies can be positive or negative

### Problems with Naive Approaches

```python
# NAIVE approach - PROBLEMATIC
strategies = ['gap_go', 'vwap_reclaim', 'mean_reversion']

# Common problems:
# 1. Equal allocation without considering performance
# 2. No correlation monitoring
# 3. Risk budgeting ignored
# 4. No regime adaptation
# 5. Capacity limits ignored
```

## Portfolio Management Framework

### 1. Multi-Strategy Architecture

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import cvxpy as cp
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StrategyStatus(Enum):
    """Strategy states in the portfolio"""
    ACTIVE = "active"
    PAUSED = "paused"
    DEGRADED = "degraded"
    DISABLED = "disabled"


@dataclass
class StrategyMetrics:
    """Individual strategy performance metrics"""
    name: str
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    avg_holding_period: float = 0.0
    capacity_used: float = 0.0
    correlation_with_market: float = 0.0
    recent_performance: float = 0.0
    confidence: float = 0.5
    status: StrategyStatus = StrategyStatus.ACTIVE


@dataclass
class PortfolioConfig:
    """Multi-strategy portfolio configuration"""

    # Capital allocation
    total_capital: float = 100000.0
    max_strategy_allocation: float = 0.4      # Max 40% per strategy
    min_strategy_allocation: float = 0.05     # Min 5% per strategy
    reserve_cash_pct: float = 0.1             # 10% cash reserve

    # Risk parameters
    max_portfolio_volatility: float = 0.20    # 20% annual volatility
    max_correlation_limit: float = 0.7        # Max correlation between strategies
    max_drawdown_limit: float = 0.15          # 15% max portfolio drawdown

    # Rebalancing
    rebalance_frequency: str = "weekly"       # daily, weekly, monthly
    min_rebalance_threshold: float = 0.05     # 5% allocation drift trigger
    lookback_periods: int = 60                # Days for performance calculation

    # Strategy management
    min_trades_for_inclusion: int = 20        # Min trades before inclusion
    underperformance_threshold: float = -0.1  # -10% relative performance threshold
    confidence_threshold: float = 0.3         # Min confidence for inclusion


class MultiStrategyPortfolioManager:
    """
    Portfolio manager that optimizes allocation across multiple strategies

    Key responsibilities:
    1. Strategy allocation optimization
    2. Risk budgeting and monitoring
    3. Correlation management
    4. Performance attribution
    5. Dynamic rebalancing
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.strategies = {}
        self.current_allocations = {}
        self.performance_history = pd.DataFrame()
        self.correlation_matrix = None
        self.last_rebalance = None

    def register_strategy(self, strategy_name: str, strategy_instance):
        """Register a new strategy in the portfolio"""

        self.strategies[strategy_name] = {
            'instance': strategy_instance,
            'metrics': StrategyMetrics(name=strategy_name),
            'trade_history': [],
            'allocation': 0.0,
            'capital_allocated': 0.0
        }

        print(f"Strategy '{strategy_name}' registered successfully")

    def calculate_optimal_allocation(self) -> Dict[str, float]:
        """
        Calculates optimal allocation using modern portfolio theory with constraints
        """

        # Get strategy performance data
        strategy_data = self._prepare_strategy_data()

        if len(strategy_data) < 2:
            # Single strategy or insufficient data
            return self._equal_weight_allocation()

        # Extract returns and calculate statistics
        returns_matrix = np.array([data['returns'] for data in strategy_data.values()])
        strategy_names = list(strategy_data.keys())

        # Calculate expected returns and covariance matrix
        expected_returns = np.array([np.mean(returns) for returns in returns_matrix])
        cov_matrix = np.cov(returns_matrix)

        # Optimization using CVXPY
        try:
            optimal_weights = self._solve_portfolio_optimization(
                expected_returns, cov_matrix, strategy_names
            )

            return dict(zip(strategy_names, optimal_weights))

        except Exception as e:
            print(f"Optimization failed: {e}. Using fallback allocation.")
            return self._risk_parity_allocation(strategy_data)

    def _prepare_strategy_data(self) -> Dict:
        """Prepare strategy data for optimization"""

        strategy_data = {}

        for name, strategy_info in self.strategies.items():
            metrics = strategy_info['metrics']

            # Only include strategies that meet minimum criteria
            if (metrics.status == StrategyStatus.ACTIVE and
                len(strategy_info['trade_history']) >= self.config.min_trades_for_inclusion and
                metrics.confidence >= self.config.confidence_threshold):

                # Extract returns from trade history
                trades = strategy_info['trade_history']
                returns = [trade['return_pct'] for trade in trades[-self.config.lookback_periods:]]

                if len(returns) >= 10:  # Minimum data requirement
                    strategy_data[name] = {
                        'returns': returns,
                        'sharpe': metrics.sharpe_ratio,
                        'max_dd': metrics.max_drawdown,
                        'capacity': 1.0 - metrics.capacity_used,
                        'confidence': metrics.confidence
                    }

        return strategy_data

    def _solve_portfolio_optimization(self,
                                    expected_returns: np.array,
                                    cov_matrix: np.array,
                                    strategy_names: List[str]) -> np.array:
        """
        Solve portfolio optimization using mean-variance optimization with constraints
        """

        n = len(expected_returns)
        weights = cp.Variable(n)

        # Risk-adjusted returns (Sharpe optimization)
        portfolio_return = weights.T @ expected_returns
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        portfolio_volatility = cp.sqrt(portfolio_variance)

        # Objective: maximize risk-adjusted return
        objective = cp.Maximize(portfolio_return / portfolio_volatility)

        # Constraints
        constraints = [
            # Weights sum to 1 (allowing for cash reserve)
            cp.sum(weights) == (1 - self.config.reserve_cash_pct),

            # Individual strategy limits
            weights >= self.config.min_strategy_allocation,
            weights <= self.config.max_strategy_allocation,

            # Portfolio volatility constraint
            portfolio_volatility <= self.config.max_portfolio_volatility,
        ]

        # Add correlation constraints if needed
        for i, strategy_i in enumerate(strategy_names):
            for j, strategy_j in enumerate(strategy_names):
                if i < j:  # Avoid double counting
                    correlation = cov_matrix[i, j] / (np.sqrt(cov_matrix[i, i]) * np.sqrt(cov_matrix[j, j]))
                    if abs(correlation) > self.config.max_correlation_limit:
                        # Limit allocation to highly correlated strategies
                        constraints.append(weights[i] + weights[j] <= self.config.max_strategy_allocation)

        # Solve optimization
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError(f"Optimization failed with status: {problem.status}")

        return weights.value

    def _equal_weight_allocation(self) -> Dict[str, float]:
        """Fallback: equal weight allocation"""

        active_strategies = [
            name for name, info in self.strategies.items()
            if info['metrics'].status == StrategyStatus.ACTIVE
        ]

        if not active_strategies:
            return {}

        weight_per_strategy = (1 - self.config.reserve_cash_pct) / len(active_strategies)

        return {name: weight_per_strategy for name in active_strategies}

    def _risk_parity_allocation(self, strategy_data: Dict) -> Dict[str, float]:
        """Risk parity allocation based on volatility"""

        if not strategy_data:
            return self._equal_weight_allocation()

        # Calculate inverse volatility weights
        volatilities = {}
        for name, data in strategy_data.items():
            returns = np.array(data['returns'])
            volatilities[name] = np.std(returns) if len(returns) > 1 else 0.1

        # Inverse volatility weighting
        inv_vol_weights = {name: 1/vol for name, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol_weights.values())

        # Normalize to sum to (1 - cash_reserve)
        allocation_budget = 1 - self.config.reserve_cash_pct

        return {
            name: (weight / total_inv_vol) * allocation_budget
            for name, weight in inv_vol_weights.items()
        }

    def rebalance_portfolio(self, market_data: Dict = None) -> Dict:
        """
        Rebalance portfolio based on optimal allocation
        """

        # Check if rebalancing is needed
        if not self._should_rebalance():
            return {'action': 'no_rebalance', 'reason': 'Within rebalance threshold'}

        # Update strategy metrics
        self._update_all_strategy_metrics()

        # Calculate new optimal allocation
        new_allocation = self.calculate_optimal_allocation()

        if not new_allocation:
            return {'action': 'no_rebalance', 'reason': 'No valid strategies'}

        # Calculate rebalancing actions
        rebalancing_actions = self._calculate_rebalancing_actions(new_allocation)

        # Execute rebalancing
        execution_results = self._execute_rebalancing(rebalancing_actions)

        # Update allocations
        self.current_allocations = new_allocation
        self.last_rebalance = datetime.now()

        return {
            'action': 'rebalanced',
            'new_allocation': new_allocation,
            'rebalancing_actions': rebalancing_actions,
            'execution_results': execution_results,
            'total_strategies': len(new_allocation)
        }

    def _should_rebalance(self) -> bool:
        """Determine if portfolio rebalancing is needed"""

        if self.last_rebalance is None:
            return True  # First rebalancing

        # Time-based rebalancing
        time_since_rebalance = datetime.now() - self.last_rebalance

        if self.config.rebalance_frequency == "daily" and time_since_rebalance.days >= 1:
            return True
        elif self.config.rebalance_frequency == "weekly" and time_since_rebalance.days >= 7:
            return True
        elif self.config.rebalance_frequency == "monthly" and time_since_rebalance.days >= 30:
            return True

        # Drift-based rebalancing
        current_optimal = self.calculate_optimal_allocation()

        for strategy, optimal_weight in current_optimal.items():
            current_weight = self.current_allocations.get(strategy, 0)
            drift = abs(optimal_weight - current_weight)

            if drift > self.config.min_rebalance_threshold:
                return True

        return False

    def _calculate_rebalancing_actions(self, new_allocation: Dict[str, float]) -> List[Dict]:
        """Calculate specific actions needed for rebalancing"""

        actions = []
        current_total_capital = self._calculate_current_portfolio_value()

        for strategy_name, target_weight in new_allocation.items():
            current_weight = self.current_allocations.get(strategy_name, 0)
            current_capital = current_weight * current_total_capital
            target_capital = target_weight * current_total_capital

            capital_change = target_capital - current_capital

            if abs(capital_change) > 100:  # Minimum change threshold ($100)
                action = {
                    'strategy': strategy_name,
                    'action': 'increase' if capital_change > 0 else 'decrease',
                    'capital_change': capital_change,
                    'current_allocation': current_weight,
                    'target_allocation': target_weight
                }
                actions.append(action)

        return actions

    def _execute_rebalancing(self, actions: List[Dict]) -> List[Dict]:
        """Execute rebalancing actions"""

        execution_results = []

        for action in actions:
            strategy_name = action['strategy']
            strategy_info = self.strategies.get(strategy_name)

            if not strategy_info:
                continue

            try:
                if action['action'] == 'increase':
                    # Allocate more capital to strategy
                    result = self._allocate_capital_to_strategy(
                        strategy_name, action['capital_change']
                    )
                else:
                    # Reduce capital from strategy
                    result = self._reduce_capital_from_strategy(
                        strategy_name, abs(action['capital_change'])
                    )

                execution_results.append({
                    'strategy': strategy_name,
                    'action': action['action'],
                    'success': result['success'],
                    'amount': action['capital_change'],
                    'details': result
                })

            except Exception as e:
                execution_results.append({
                    'strategy': strategy_name,
                    'action': action['action'],
                    'success': False,
                    'error': str(e)
                })

        return execution_results

    def add_strategy_trade(self, strategy_name: str, trade_result: Dict):
        """Add trade result for strategy tracking"""

        if strategy_name not in self.strategies:
            print(f"Warning: Strategy '{strategy_name}' not registered")
            return

        # Add timestamp if not present
        if 'timestamp' not in trade_result:
            trade_result['timestamp'] = datetime.now()

        # Calculate return percentage
        if 'return_pct' not in trade_result and 'pnl' in trade_result and 'position_value' in trade_result:
            trade_result['return_pct'] = trade_result['pnl'] / trade_result['position_value']

        self.strategies[strategy_name]['trade_history'].append(trade_result)

        # Update strategy metrics
        self._update_strategy_metrics(strategy_name)

    def _update_strategy_metrics(self, strategy_name: str):
        """Update metrics for a specific strategy"""

        strategy_info = self.strategies[strategy_name]
        trades = strategy_info['trade_history']

        if len(trades) < 10:
            return  # Insufficient data

        # Extract recent trades for calculation
        recent_trades = trades[-self.config.lookback_periods:]
        returns = [trade.get('return_pct', 0) for trade in recent_trades]
        pnl_values = [trade.get('pnl', 0) for trade in recent_trades]

        # Calculate metrics
        metrics = strategy_info['metrics']

        # Sharpe ratio
        if len(returns) > 1:
            returns_array = np.array(returns)
            metrics.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0

        # Win rate
        winning_trades = [pnl for pnl in pnl_values if pnl > 0]
        metrics.win_rate = len(winning_trades) / len(pnl_values)

        # Profit factor
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum([pnl for pnl in pnl_values if pnl < 0]))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        equity_curve = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        metrics.max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        # Recent performance
        metrics.recent_performance = np.mean(returns[-10:]) if len(returns) >= 10 else 0

        # Update confidence based on trade count and consistency
        metrics.confidence = min(
            len(trades) / 100,  # More trades = higher confidence
            1.0 - abs(metrics.max_drawdown)  # Lower drawdown = higher confidence
        )

    def _update_all_strategy_metrics(self):
        """Update metrics for all strategies"""
        for strategy_name in self.strategies:
            self._update_strategy_metrics(strategy_name)

    def get_portfolio_performance(self) -> Dict:
        """Calculate overall portfolio performance"""

        # Aggregate all trades across strategies
        all_trades = []
        strategy_contributions = {}

        for strategy_name, strategy_info in self.strategies.items():
            trades = strategy_info['trade_history']
            allocation = self.current_allocations.get(strategy_name, 0)

            strategy_pnl = sum([trade.get('pnl', 0) for trade in trades])
            strategy_contributions[strategy_name] = {
                'pnl': strategy_pnl,
                'allocation': allocation,
                'trade_count': len(trades),
                'contribution_pct': 0  # Will calculate below
            }

            # Weight trades by allocation for portfolio calculation
            for trade in trades:
                weighted_trade = trade.copy()
                weighted_trade['weighted_pnl'] = trade.get('pnl', 0) * allocation
                all_trades.append(weighted_trade)

        if not all_trades:
            return {'error': 'No trades available'}

        # Calculate portfolio-level metrics
        total_pnl = sum([trade['weighted_pnl'] for trade in all_trades])
        portfolio_returns = [trade['weighted_pnl'] / self.config.total_capital for trade in all_trades]

        # Calculate contribution percentages
        for strategy_name, contrib in strategy_contributions.items():
            contrib['contribution_pct'] = (contrib['pnl'] * contrib['allocation'] / total_pnl * 100) if total_pnl != 0 else 0

        # Portfolio Sharpe ratio
        portfolio_sharpe = (
            np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
            if np.std(portfolio_returns) > 0 else 0
        )

        # Portfolio max drawdown
        portfolio_equity = np.cumsum([trade['weighted_pnl'] for trade in all_trades])
        running_max = np.maximum.accumulate(portfolio_equity)
        drawdowns = (portfolio_equity - running_max) / running_max
        portfolio_max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0

        return {
            'total_pnl': total_pnl,
            'portfolio_sharpe': portfolio_sharpe,
            'portfolio_max_drawdown': portfolio_max_dd,
            'total_trades': len(all_trades),
            'strategy_contributions': strategy_contributions,
            'current_allocations': self.current_allocations,
            'diversification_ratio': self._calculate_diversification_ratio()
        }

    def _calculate_diversification_ratio(self) -> float:
        """
        Calculate diversification ratio = weighted avg vol / portfolio vol
        Higher ratio = better diversification
        """

        if len(self.strategies) < 2:
            return 1.0

        strategy_vols = []
        weights = []

        for strategy_name, allocation in self.current_allocations.items():
            if allocation > 0 and strategy_name in self.strategies:
                trades = self.strategies[strategy_name]['trade_history']
                if len(trades) >= 10:
                    returns = [trade.get('return_pct', 0) for trade in trades[-30:]]
                    vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.2

                    strategy_vols.append(vol)
                    weights.append(allocation)

        if not strategy_vols:
            return 1.0

        # Weighted average volatility
        weights = np.array(weights) / sum(weights)  # Normalize weights
        weighted_avg_vol = np.sum(np.array(strategy_vols) * weights)

        # Portfolio volatility (need correlation matrix for exact calculation)
        # Simplified: assume moderate correlation
        portfolio_vol = weighted_avg_vol * 0.8  # Assuming some diversification benefit

        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

    def _allocate_capital_to_strategy(self, strategy_name: str, capital_amount: float) -> Dict:
        """Allocate capital to specific strategy"""

        strategy_info = self.strategies[strategy_name]
        current_capital = strategy_info['capital_allocated']
        new_capital = current_capital + capital_amount

        strategy_info['capital_allocated'] = new_capital

        return {
            'success': True,
            'previous_capital': current_capital,
            'new_capital': new_capital,
            'change': capital_amount
        }

    def _reduce_capital_from_strategy(self, strategy_name: str, capital_amount: float) -> Dict:
        """Reduce capital from specific strategy"""

        strategy_info = self.strategies[strategy_name]
        current_capital = strategy_info['capital_allocated']
        new_capital = max(0, current_capital - capital_amount)

        strategy_info['capital_allocated'] = new_capital

        return {
            'success': True,
            'previous_capital': current_capital,
            'new_capital': new_capital,
            'change': -capital_amount
        }

    def _calculate_current_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        return sum([info['capital_allocated'] for info in self.strategies.values()])

    def generate_portfolio_report(self) -> str:
        """Generate comprehensive portfolio report"""

        performance = self.get_portfolio_performance()

        if 'error' in performance:
            return "Insufficient data for portfolio report"

        report = f"""
MULTI-STRATEGY PORTFOLIO REPORT
{'='*50}

PORTFOLIO OVERVIEW
Total P&L: ${performance['total_pnl']:,.2f}
Portfolio Sharpe: {performance['portfolio_sharpe']:.2f}
Max Drawdown: {performance['portfolio_max_drawdown']:.1%}
Total Trades: {performance['total_trades']}
Diversification Ratio: {performance['diversification_ratio']:.2f}

STRATEGY ALLOCATIONS
"""

        for strategy, allocation in performance['current_allocations'].items():
            contrib = performance['strategy_contributions'].get(strategy, {})
            report += f"""
{strategy}:
  Allocation: {allocation:.1%}
  P&L: ${contrib.get('pnl', 0):,.2f}
  Contribution: {contrib.get('contribution_pct', 0):.1f}%
  Trades: {contrib.get('trade_count', 0)}
"""

        report += f"""
STRATEGY PERFORMANCE
"""

        for strategy_name, strategy_info in self.strategies.items():
            metrics = strategy_info['metrics']
            report += f"""
{strategy_name}:
  Status: {metrics.status.value}
  Sharpe: {metrics.sharpe_ratio:.2f}
  Win Rate: {metrics.win_rate:.1%}
  Profit Factor: {metrics.profit_factor:.2f}
  Max DD: {metrics.max_drawdown:.1%}
  Confidence: {metrics.confidence:.2f}
"""

        return report


# Portfolio Correlation Monitor
class PortfolioCorrelationMonitor:
    """
    Monitor correlations between strategies in real time
    """

    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
        self.strategy_returns = {}

    def update_strategy_returns(self, strategy_name: str, returns: List[float]):
        """Update returns data for strategy"""

        self.strategy_returns[strategy_name] = returns[-self.lookback_window:]

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between strategies"""

        if len(self.strategy_returns) < 2:
            return pd.DataFrame()

        # Align returns data
        aligned_data = {}
        min_length = min([len(returns) for returns in self.strategy_returns.values()])

        for strategy, returns in self.strategy_returns.items():
            aligned_data[strategy] = returns[-min_length:]

        df = pd.DataFrame(aligned_data)
        return df.corr()

    def detect_correlation_changes(self, threshold: float = 0.3) -> List[Dict]:
        """Detect significant changes in strategy correlations"""

        current_corr = self.calculate_correlation_matrix()

        if current_corr.empty:
            return []

        alerts = []

        for i in range(len(current_corr.columns)):
            for j in range(i + 1, len(current_corr.columns)):
                strategy1 = current_corr.columns[i]
                strategy2 = current_corr.columns[j]
                correlation = current_corr.iloc[i, j]

                if abs(correlation) > 0.7:  # High correlation threshold
                    alerts.append({
                        'type': 'HIGH_CORRELATION',
                        'strategy1': strategy1,
                        'strategy2': strategy2,
                        'correlation': correlation,
                        'recommendation': 'Consider reducing allocation to one strategy'
                    })

        return alerts


# Example usage and testing
def example_multi_strategy_portfolio():
    """
    Complete multi-strategy portfolio management example
    """

    # Configure portfolio
    config = PortfolioConfig(
        total_capital=50000,
        max_strategy_allocation=0.5,
        rebalance_frequency="weekly"
    )

    # Initialize portfolio manager
    portfolio = MultiStrategyPortfolioManager(config)

    # Register strategies (mock implementations)
    portfolio.register_strategy("gap_and_go", "GapAndGoStrategy()")
    portfolio.register_strategy("vwap_reclaim", "VWAPReclaimStrategy()")
    portfolio.register_strategy("mean_reversion", "MeanReversionStrategy()")

    # Simulate trade results
    strategies = ["gap_and_go", "vwap_reclaim", "mean_reversion"]

    for day in range(30):  # 30 days of trading
        for strategy in strategies:
            # Simulate trades (random for example)
            if np.random.random() > 0.7:  # 30% chance of trade per day
                trade_result = {
                    'pnl': np.random.normal(5, 15),  # Random P&L
                    'position_value': 1000,
                    'return_pct': np.random.normal(0.01, 0.03),
                    'timestamp': datetime.now() - timedelta(days=30-day)
                }
                portfolio.add_strategy_trade(strategy, trade_result)

    # Calculate optimal allocation
    optimal_allocation = portfolio.calculate_optimal_allocation()
    print("Optimal Strategy Allocation:")
    for strategy, allocation in optimal_allocation.items():
        print(f"  {strategy}: {allocation:.1%}")

    # Rebalance portfolio
    rebalance_result = portfolio.rebalance_portfolio()
    print(f"\nRebalancing: {rebalance_result['action']}")

    # Get performance report
    performance_report = portfolio.generate_portfolio_report()
    print(performance_report)

    return portfolio

if __name__ == "__main__":
    example_portfolio = example_multi_strategy_portfolio()
```

## Advanced Portfolio Optimization Techniques

### 1. **Black-Litterman Model for Strategy Views**

```python
class BlackLittermanPortfolio:
    """
    Black-Litterman model adaptation for strategy allocation
    Incorporates specific views on strategy performance
    """

    def __init__(self, returns_data: pd.DataFrame, tau: float = 0.05):
        self.returns_data = returns_data
        self.tau = tau  # Confidence in equilibrium returns

    def calculate_bl_allocation(self, views: Dict, view_confidence: np.array) -> np.array:
        """
        Calculate allocation using Black-Litterman with strategy views

        Args:
            views: Dict with strategy views (expected outperformance)
            view_confidence: Array with confidence levels for each view
        """

        # Equilibrium returns (reverse optimization)
        sigma = self.returns_data.cov().values
        market_weights = np.ones(len(self.returns_data.columns)) / len(self.returns_data.columns)
        pi = sigma @ market_weights  # Implied equilibrium returns

        # Views matrix
        P = np.eye(len(pi))  # Each view corresponds to one strategy
        Q = np.array(list(views.values()))  # View returns
        omega = np.diag(1 / view_confidence)  # View uncertainty

        # Black-Litterman formula
        M1 = np.linalg.inv(self.tau * sigma)
        M2 = P.T @ np.linalg.inv(omega) @ P
        M3 = np.linalg.inv(self.tau * sigma) @ pi
        M4 = P.T @ np.linalg.inv(omega) @ Q

        # New expected returns
        mu_bl = np.linalg.inv(M1 + M2) @ (M3 + M4)

        # New covariance matrix
        sigma_bl = np.linalg.inv(M1 + M2)

        return mu_bl, sigma_bl
```

### 2. **Risk Parity with Regime Adaptation**

```python
class RegimeAdaptiveRiskParity:
    """
    Risk parity allocation that adapts with market regimes
    """

    def __init__(self):
        self.regime_detector = None  # Would integrate with regime detection

    def calculate_risk_parity_weights(self,
                                    returns_data: pd.DataFrame,
                                    current_regime: str) -> np.array:
        """
        Calculate risk parity weights adjusted for regime
        """

        # Base risk parity calculation
        cov_matrix = returns_data.cov().values
        inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
        base_weights = inv_vol / np.sum(inv_vol)

        # Regime-specific adjustments
        regime_adjustments = {
            'low_volatility': np.array([1.2, 1.0, 0.8]),    # Favor momentum strategies
            'high_volatility': np.array([0.7, 0.8, 1.3]),   # Favor mean reversion
            'trending': np.array([1.3, 1.1, 0.6]),          # Favor directional strategies
            'sideways': np.array([0.8, 1.2, 1.0])           # Favor range-bound strategies
        }

        adjustment = regime_adjustments.get(current_regime, np.ones(len(base_weights)))

        # Apply adjustments
        adjusted_weights = base_weights * adjustment
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

        return adjusted_weights
```

### 3. **Dynamic Portfolio Rebalancing**

```python
class DynamicRebalancer:
    """
    Dynamic rebalancing that considers transaction costs and market impact
    """

    def __init__(self, transaction_cost_pct: float = 0.001):
        self.transaction_cost_pct = transaction_cost_pct

    def calculate_optimal_rebalancing(self,
                                    current_weights: np.array,
                                    target_weights: np.array,
                                    portfolio_value: float) -> Dict:
        """
        Calculate optimal rebalancing considering costs
        """

        weight_diffs = target_weights - current_weights
        rebalancing_costs = np.abs(weight_diffs) * portfolio_value * self.transaction_cost_pct

        # Only rebalance if benefit > cost
        rebalancing_benefit = self._estimate_rebalancing_benefit(weight_diffs, portfolio_value)

        net_benefit = rebalancing_benefit - np.sum(rebalancing_costs)

        if net_benefit > 0:
            return {
                'should_rebalance': True,
                'weight_changes': weight_diffs,
                'estimated_costs': np.sum(rebalancing_costs),
                'estimated_benefit': rebalancing_benefit,
                'net_benefit': net_benefit
            }
        else:
            return {
                'should_rebalance': False,
                'reason': 'Costs exceed benefits',
                'net_benefit': net_benefit
            }

    def _estimate_rebalancing_benefit(self, weight_diffs: np.array, portfolio_value: float) -> float:
        """
        Estimate benefit from rebalancing (simplified model)
        """
        # In practice, this would use more sophisticated models
        # For now, assume benefit proportional to deviation from optimal
        deviation = np.sum(np.abs(weight_diffs))
        return deviation * portfolio_value * 0.01  # 1% benefit per 1% deviation
```

## Integration with Trading System

### **Real-Time Portfolio Monitor**

```python
class RealTimePortfolioMonitor:
    """
    Real-time portfolio monitor with automatic alerts
    """

    def __init__(self, portfolio_manager: MultiStrategyPortfolioManager):
        self.portfolio_manager = portfolio_manager
        self.alert_thresholds = {
            'max_drawdown': -0.10,      # -10% portfolio drawdown
            'correlation_spike': 0.8,    # 80% strategy correlation
            'concentration_risk': 0.6,   # 60% in single strategy
            'underperformance': -0.05    # -5% vs benchmark
        }

    def run_real_time_monitoring(self):
        """Run continuous portfolio monitoring"""

        while True:
            try:
                # Update portfolio metrics
                performance = self.portfolio_manager.get_portfolio_performance()

                # Check for alerts
                alerts = self._check_portfolio_alerts(performance)

                if alerts:
                    self._send_alerts(alerts)

                # Auto-rebalance if needed
                if self._should_auto_rebalance(performance):
                    self.portfolio_manager.rebalance_portfolio()

                # Sleep before next check
                time.sleep(60)  # Check every minute

            except Exception as e:
                print(f"Portfolio monitoring error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _check_portfolio_alerts(self, performance: Dict) -> List[Dict]:
        """Check for portfolio-level alerts"""

        alerts = []

        # Drawdown alert
        if performance['portfolio_max_drawdown'] < self.alert_thresholds['max_drawdown']:
            alerts.append({
                'type': 'MAX_DRAWDOWN_BREACH',
                'severity': 'HIGH',
                'value': performance['portfolio_max_drawdown'],
                'threshold': self.alert_thresholds['max_drawdown']
            })

        # Concentration risk
        max_allocation = max(performance['current_allocations'].values()) if performance['current_allocations'] else 0
        if max_allocation > self.alert_thresholds['concentration_risk']:
            alerts.append({
                'type': 'CONCENTRATION_RISK',
                'severity': 'MEDIUM',
                'value': max_allocation,
                'threshold': self.alert_thresholds['concentration_risk']
            })

        return alerts

    def _send_alerts(self, alerts: List[Dict]):
        """Send alerts via configured channels"""
        for alert in alerts:
            print(f"🚨 PORTFOLIO ALERT: {alert['type']} - {alert['severity']}")
            # In practice: send to Telegram, Discord, email, etc.
```

---

**Integration Points**:
- **[Dynamic Position Sizing](./Dynamic-Position-Sizing.md)**: Portfolio-level sizing
- **[Regime Detection](./Regime-Detection.md)**: Regime-adaptive allocation
- **[Risk Management](../core-concepts/Risk-Management.md)**: Portfolio risk controls
- **[Performance Metrics](../core-concepts/Performance-Metrics.md)**: Portfolio attribution

This framework enables **sophisticated orchestration** of multiple strategies, optimizing allocation, controlling risk concentration, and maximizing diversification benefits - critical for scaling systematic trading operations.
