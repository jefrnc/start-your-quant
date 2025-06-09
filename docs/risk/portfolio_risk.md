# Portfolio Risk Management: Gestión Integral del Riesgo

## Introducción: Del Riesgo Individual al Riesgo de Portfolio

Mientras que la gestión de riesgo individual se enfoca en trades específicos, la gestión de riesgo de portfolio considera las interacciones, correlaciones y efectos sistémicos across todas las posiciones. Este documento presenta un framework completo para optimizar el riesgo a nivel portfolio.

## Architecture de Portfolio Risk Management

### Core Risk Engine

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import scipy.optimize as optimize
from scipy import stats
import warnings

@dataclass
class Position:
    """Estructura de posición individual"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_date: datetime
    strategy: str
    sector: str
    market_cap: str
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.entry_price)
    
    @property
    def weight(self) -> float:
        # Will be calculated by portfolio manager
        return 0.0

@dataclass
class RiskMetrics:
    """Métricas de riesgo del portfolio"""
    total_var: float
    component_var: Dict[str, float]
    marginal_var: Dict[str, float]
    portfolio_beta: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    correlation_risk: float
    concentration_risk: float

class PortfolioRiskManager:
    """Manager central de riesgo de portfolio"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.risk_limits = self._default_risk_limits()
        self.correlation_matrix = None
        self.sector_exposures = {}
        self.strategy_exposures = {}
        
        # Historical data for risk calculations
        self.price_history = {}
        self.returns_history = {}
        
        # Risk monitoring
        self.risk_alerts = []
        self.risk_metrics_history = []
        
    def _default_risk_limits(self) -> Dict[str, float]:
        """Límites de riesgo por defecto"""
        return {
            'max_portfolio_var': 0.02,          # 2% diario VaR máximo
            'max_position_weight': 0.10,         # 10% máximo por posición
            'max_sector_weight': 0.25,           # 25% máximo por sector
            'max_strategy_weight': 0.40,         # 40% máximo por estrategia
            'max_correlation': 0.80,             # 80% correlación máxima
            'max_concentration_ratio': 0.30,     # 30% concentración máxima
            'max_leverage': 1.0,                 # Sin leverage por defecto
            'max_drawdown': 0.15,                # 15% drawdown máximo
            'min_liquidity_ratio': 0.05          # 5% del volumen diario mínimo
        }
    
    def add_position(self, position: Position) -> bool:
        """Agrega nueva posición al portfolio"""
        # 1. Pre-trade risk check
        risk_check = self._pre_trade_risk_check(position)
        if not risk_check['approved']:
            print(f"❌ Position rejected: {risk_check['reason']}")
            return False
        
        # 2. Add position
        self.positions[position.symbol] = position
        
        # 3. Update exposures
        self._update_exposures()
        
        # 4. Post-trade risk assessment
        self._post_trade_risk_assessment()
        
        print(f"✅ Position added: {position.symbol}")
        return True
    
    def remove_position(self, symbol: str) -> bool:
        """Remueve posición del portfolio"""
        if symbol in self.positions:
            del self.positions[symbol]
            self._update_exposures()
            self._post_trade_risk_assessment()
            return True
        return False
    
    def update_prices(self, price_updates: Dict[str, float]):
        """Actualiza precios de las posiciones"""
        for symbol, price in price_updates.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
        
        # Update risk metrics
        self._calculate_real_time_risk_metrics()
    
    def _pre_trade_risk_check(self, new_position: Position) -> Dict[str, Any]:
        """Verificación de riesgo pre-trade"""
        checks = {
            'position_size': self._check_position_size_limit(new_position),
            'sector_concentration': self._check_sector_limit(new_position),
            'strategy_concentration': self._check_strategy_limit(new_position),
            'correlation_risk': self._check_correlation_risk(new_position),
            'liquidity_risk': self._check_liquidity_risk(new_position),
            'var_impact': self._check_var_impact(new_position)
        }
        
        failed_checks = [name for name, passed in checks.items() if not passed]
        
        return {
            'approved': len(failed_checks) == 0,
            'checks': checks,
            'failed_checks': failed_checks,
            'reason': f"Failed checks: {failed_checks}" if failed_checks else "All checks passed"
        }
    
    def _check_position_size_limit(self, position: Position) -> bool:
        """Verifica límite de tamaño de posición"""
        portfolio_value = self._calculate_portfolio_value()
        position_weight = position.market_value / portfolio_value
        
        return position_weight <= self.risk_limits['max_position_weight']
    
    def _check_sector_limit(self, new_position: Position) -> bool:
        """Verifica límite de concentración sectorial"""
        portfolio_value = self._calculate_portfolio_value()
        
        # Calculate current sector exposure
        current_sector_exposure = self.sector_exposures.get(new_position.sector, 0)
        new_sector_exposure = (current_sector_exposure + new_position.market_value) / portfolio_value
        
        return new_sector_exposure <= self.risk_limits['max_sector_weight']
    
    def _check_correlation_risk(self, new_position: Position) -> bool:
        """Verifica riesgo de correlación"""
        if not self.positions:  # First position
            return True
        
        # Get correlation with existing positions
        correlations = self._calculate_position_correlations(new_position.symbol)
        
        # Check if any correlation exceeds limit
        max_correlation = max(abs(corr) for corr in correlations.values()) if correlations else 0
        
        return max_correlation <= self.risk_limits['max_correlation']
    
    def calculate_portfolio_var(self, confidence_level: float = 0.05, 
                               lookback_days: int = 252) -> Dict[str, float]:
        """Calcula Value at Risk del portfolio"""
        if not self.positions:
            return {'portfolio_var': 0, 'component_vars': {}}
        
        # Get returns matrix
        returns_matrix = self._get_returns_matrix(lookback_days)
        
        if returns_matrix.empty:
            return {'portfolio_var': 0, 'component_vars': {}}
        
        # Portfolio weights
        weights = self._get_portfolio_weights()
        
        # Portfolio returns
        portfolio_returns = (returns_matrix * weights).sum(axis=1)
        
        # Calculate VaR
        portfolio_var = np.percentile(portfolio_returns, confidence_level * 100)
        
        # Component VaR
        component_vars = self._calculate_component_var(returns_matrix, weights, confidence_level)
        
        # Marginal VaR
        marginal_vars = self._calculate_marginal_var(returns_matrix, weights, confidence_level)
        
        return {
            'portfolio_var': abs(portfolio_var),
            'component_vars': component_vars,
            'marginal_vars': marginal_vars,
            'confidence_level': confidence_level
        }
    
    def _calculate_component_var(self, returns_matrix: pd.DataFrame, 
                               weights: np.ndarray, confidence_level: float) -> Dict[str, float]:
        """Calcula VaR componente para cada posición"""
        component_vars = {}
        
        # Covariance matrix
        cov_matrix = returns_matrix.cov().values
        
        # Portfolio variance
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Component contributions
        marginal_contributions = np.dot(cov_matrix, weights) / portfolio_vol
        component_contributions = weights * marginal_contributions
        
        # Scale by VaR multiplier
        var_multiplier = stats.norm.ppf(confidence_level)
        
        for i, symbol in enumerate(returns_matrix.columns):
            component_vars[symbol] = abs(component_contributions[i] * var_multiplier)
        
        return component_vars
    
    def optimize_portfolio_risk(self, target_return: Optional[float] = None) -> Dict[str, Any]:
        """Optimiza weights del portfolio para minimizar riesgo"""
        if not self.positions:
            return {'status': 'error', 'message': 'No positions to optimize'}
        
        # Get expected returns and covariance matrix
        returns_data = self._get_returns_matrix(252)  # 1 year
        
        if returns_data.empty:
            return {'status': 'error', 'message': 'Insufficient historical data'}
        
        expected_returns = returns_data.mean().values
        cov_matrix = returns_data.cov().values
        
        n_assets = len(expected_returns)
        
        # Current weights
        current_weights = self._get_portfolio_weights()
        
        # Optimization objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Add return constraint if specified
        if target_return:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(expected_returns, x) - target_return
            })
        
        # Bounds: individual position limits
        bounds = []
        for i in range(n_assets):
            bounds.append((0, self.risk_limits['max_position_weight']))
        
        # Optimize
        result = optimize.minimize(
            objective,
            current_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            optimal_risk = np.sqrt(result.fun)
            
            # Calculate rebalancing trades needed
            rebalancing_trades = self._calculate_rebalancing_trades(optimal_weights)
            
            return {
                'status': 'success',
                'optimal_weights': dict(zip(returns_data.columns, optimal_weights)),
                'optimal_risk': optimal_risk,
                'current_risk': np.sqrt(objective(current_weights)),
                'risk_reduction': np.sqrt(objective(current_weights)) - optimal_risk,
                'rebalancing_trades': rebalancing_trades
            }
        else:
            return {
                'status': 'error',
                'message': 'Optimization failed',
                'details': result.message
            }
```

### Advanced Risk Models

```python
class AdvancedRiskModels:
    """Modelos avanzados de riesgo"""
    
    def __init__(self):
        self.factor_models = {}
        self.regime_models = {}
        
    def fama_french_risk_model(self, returns_data: pd.DataFrame, 
                              market_returns: pd.Series,
                              smb_factor: pd.Series,
                              hml_factor: pd.Series) -> Dict[str, Any]:
        """Modelo de riesgo Fama-French 3-factor"""
        
        risk_decomposition = {}
        
        for symbol in returns_data.columns:
            asset_returns = returns_data[symbol].dropna()
            
            # Align data
            common_index = asset_returns.index.intersection(market_returns.index)
            common_index = common_index.intersection(smb_factor.index)
            common_index = common_index.intersection(hml_factor.index)
            
            if len(common_index) < 50:  # Minimum data requirement
                continue
            
            y = asset_returns[common_index] - 0.02/252  # Excess returns (assume 2% risk-free)
            X = pd.DataFrame({
                'market': market_returns[common_index] - 0.02/252,
                'smb': smb_factor[common_index],
                'hml': hml_factor[common_index]
            })
            
            # Regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # Risk decomposition
            factor_exposures = {
                'beta_market': model.coef_[0],
                'beta_size': model.coef_[1],
                'beta_value': model.coef_[2],
                'alpha': model.intercept_
            }
            
            # Calculate factor contributions to risk
            predictions = model.predict(X)
            residuals = y - predictions
            
            total_variance = np.var(y)
            explained_variance = np.var(predictions)
            idiosyncratic_variance = np.var(residuals)
            
            risk_decomposition[symbol] = {
                'factor_exposures': factor_exposures,
                'total_risk': np.sqrt(total_variance),
                'systematic_risk': np.sqrt(explained_variance),
                'idiosyncratic_risk': np.sqrt(idiosyncratic_variance),
                'r_squared': explained_variance / total_variance
            }
        
        return risk_decomposition
    
    def regime_switching_risk_model(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Modelo de riesgo con cambios de régimen"""
        from sklearn.mixture import GaussianMixture
        
        # Identify market regimes using portfolio returns
        portfolio_returns = returns_data.mean(axis=1)
        
        # Features for regime identification
        features = pd.DataFrame({
            'returns': portfolio_returns,
            'volatility': portfolio_returns.rolling(20).std(),
            'skewness': portfolio_returns.rolling(60).skew(),
            'kurtosis': portfolio_returns.rolling(60).kurt()
        }).dropna()
        
        # Fit Gaussian Mixture Model
        n_regimes = 3  # Bull, Bear, Sideways
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regime_labels = gmm.fit_predict(features)
        
        # Calculate risk metrics by regime
        regime_analysis = {}
        
        for regime in range(n_regimes):
            regime_mask = regime_labels == regime
            regime_returns = returns_data[features.index[regime_mask]]
            
            if len(regime_returns) > 20:  # Minimum observations
                regime_analysis[f'regime_{regime}'] = {
                    'mean_return': regime_returns.mean().mean(),
                    'volatility': regime_returns.std().mean(),
                    'correlation_avg': regime_returns.corr().values[np.triu_indices_from(regime_returns.corr().values, k=1)].mean(),
                    'max_drawdown': self._calculate_regime_max_drawdown(regime_returns),
                    'observations': len(regime_returns),
                    'probability': np.sum(regime_mask) / len(regime_labels)
                }
        
        return {
            'regimes': regime_analysis,
            'current_regime': regime_labels[-1] if len(regime_labels) > 0 else 0,
            'regime_probabilities': gmm.predict_proba(features.iloc[-1:].values)[0] if len(features) > 0 else [0.33, 0.33, 0.34]
        }

class RiskAttributionEngine:
    """Engine de atribución de riesgo"""
    
    def __init__(self):
        self.attribution_methods = {
            'component_var': self._component_var_attribution,
            'marginal_var': self._marginal_var_attribution,
            'factor_based': self._factor_based_attribution,
            'scenario_based': self._scenario_based_attribution
        }
    
    def attribute_portfolio_risk(self, portfolio_manager: PortfolioRiskManager,
                                method: str = 'component_var') -> Dict[str, Any]:
        """Atribuye riesgo del portfolio a diferentes fuentes"""
        
        if method not in self.attribution_methods:
            raise ValueError(f"Unknown attribution method: {method}")
        
        return self.attribution_methods[method](portfolio_manager)
    
    def _component_var_attribution(self, portfolio_manager: PortfolioRiskManager) -> Dict[str, Any]:
        """Atribución basada en VaR componente"""
        var_results = portfolio_manager.calculate_portfolio_var()
        
        attribution = {
            'total_portfolio_var': var_results['portfolio_var'],
            'component_contributions': var_results['component_vars'],
            'percentage_contributions': {}
        }
        
        total_var = var_results['portfolio_var']
        for symbol, component_var in var_results['component_vars'].items():
            attribution['percentage_contributions'][symbol] = (component_var / total_var) * 100 if total_var > 0 else 0
        
        return attribution
    
    def _factor_based_attribution(self, portfolio_manager: PortfolioRiskManager) -> Dict[str, Any]:
        """Atribución basada en factores de riesgo"""
        
        # Get positions data
        positions_data = {}
        for symbol, position in portfolio_manager.positions.items():
            positions_data[symbol] = {
                'weight': position.market_value / portfolio_manager._calculate_portfolio_value(),
                'sector': position.sector,
                'strategy': position.strategy,
                'market_cap': position.market_cap
            }
        
        # Calculate risk attribution by factors
        factor_attribution = {
            'sector_risk': self._calculate_sector_risk_attribution(positions_data),
            'strategy_risk': self._calculate_strategy_risk_attribution(positions_data),
            'size_risk': self._calculate_size_risk_attribution(positions_data),
            'concentration_risk': self._calculate_concentration_risk(positions_data)
        }
        
        return factor_attribution
    
    def _calculate_sector_risk_attribution(self, positions_data: Dict) -> Dict[str, float]:
        """Calcula atribución de riesgo por sector"""
        sector_weights = {}
        
        for symbol, data in positions_data.items():
            sector = data['sector']
            weight = data['weight']
            
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += weight
        
        # Risk attribution based on concentration
        sector_risk = {}
        for sector, weight in sector_weights.items():
            # Higher concentration = higher risk contribution
            sector_risk[sector] = weight ** 2  # Quadratic penalty for concentration
        
        return sector_risk

class StressTestingFramework:
    """Framework de stress testing"""
    
    def __init__(self):
        self.stress_scenarios = {
            'market_crash': {'market_return': -0.20, 'volatility_multiplier': 3.0},
            'sector_rotation': {'sector_specific_shock': -0.15},
            'liquidity_crisis': {'liquidity_discount': 0.10, 'correlation_increase': 0.30},
            'interest_rate_shock': {'rate_change': 0.02, 'duration_impact': True},
            'black_swan': {'extreme_event': True, 'correlation_breakdown': True}
        }
    
    def run_stress_tests(self, portfolio_manager: PortfolioRiskManager) -> Dict[str, Any]:
        """Ejecuta suite completo de stress tests"""
        
        stress_results = {}
        
        for scenario_name, scenario_params in self.stress_scenarios.items():
            try:
                scenario_result = self._run_scenario(portfolio_manager, scenario_params)
                stress_results[scenario_name] = scenario_result
            except Exception as e:
                stress_results[scenario_name] = {'error': str(e)}
        
        # Aggregate results
        worst_case = self._find_worst_case_scenario(stress_results)
        resilience_score = self._calculate_resilience_score(stress_results)
        
        return {
            'scenario_results': stress_results,
            'worst_case_scenario': worst_case,
            'portfolio_resilience_score': resilience_score,
            'recommendations': self._generate_stress_test_recommendations(stress_results)
        }
    
    def _run_scenario(self, portfolio_manager: PortfolioRiskManager, 
                     scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta escenario específico de stress"""
        
        # Calculate impact on each position
        position_impacts = {}
        total_portfolio_impact = 0
        
        for symbol, position in portfolio_manager.positions.items():
            position_impact = self._calculate_position_impact(position, scenario_params)
            position_impacts[symbol] = position_impact
            total_portfolio_impact += position_impact['pnl_impact']
        
        # Calculate portfolio-level metrics under stress
        stressed_var = self._calculate_stressed_var(portfolio_manager, scenario_params)
        
        return {
            'total_pnl_impact': total_portfolio_impact,
            'percentage_impact': total_portfolio_impact / portfolio_manager._calculate_portfolio_value(),
            'stressed_var': stressed_var,
            'position_impacts': position_impacts,
            'survival_probability': self._calculate_survival_probability(total_portfolio_impact, portfolio_manager)
        }
    
    def _calculate_position_impact(self, position: Position, 
                                 scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula impacto en posición individual"""
        
        # Base impact from market crash
        market_impact = scenario_params.get('market_return', 0)
        
        # Sector-specific impact
        sector_impact = 0
        if 'sector_specific_shock' in scenario_params:
            # Apply sector shock (simplified - would need sector mapping)
            sector_impact = scenario_params['sector_specific_shock']
        
        # Liquidity impact
        liquidity_impact = 0
        if 'liquidity_discount' in scenario_params:
            liquidity_impact = -scenario_params['liquidity_discount']
        
        # Total impact
        total_return_impact = market_impact + sector_impact + liquidity_impact
        pnl_impact = position.market_value * total_return_impact
        
        return {
            'return_impact': total_return_impact,
            'pnl_impact': pnl_impact,
            'market_component': market_impact,
            'sector_component': sector_impact,
            'liquidity_component': liquidity_impact
        }
```

### Real-Time Risk Monitoring

```python
class RealTimeRiskMonitor:
    """Monitor de riesgo en tiempo real"""
    
    def __init__(self, portfolio_manager: PortfolioRiskManager):
        self.portfolio_manager = portfolio_manager
        self.monitoring_active = False
        self.alert_thresholds = {
            'var_breach': 0.80,      # 80% of VaR limit
            'concentration_warning': 0.80,  # 80% of concentration limit
            'correlation_spike': 0.90,      # 90% of correlation limit
            'drawdown_warning': 0.75        # 75% of max drawdown
        }
        
    async def start_monitoring(self):
        """Inicia monitoreo en tiempo real"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Calculate current risk metrics
                current_metrics = self._calculate_current_metrics()
                
                # Check for threshold breaches
                alerts = self._check_risk_thresholds(current_metrics)
                
                # Process alerts
                if alerts:
                    await self._process_alerts(alerts)
                
                # Update risk dashboard
                self._update_risk_dashboard(current_metrics)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Risk monitoring error: {e}")
                await asyncio.sleep(5)
    
    def _calculate_current_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de riesgo actuales"""
        
        # VaR calculation
        var_metrics = self.portfolio_manager.calculate_portfolio_var()
        
        # Concentration metrics
        concentration_metrics = self._calculate_concentration_metrics()
        
        # Correlation metrics
        correlation_metrics = self._calculate_correlation_metrics()
        
        # Drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics()
        
        return {
            'timestamp': datetime.now(),
            'var_metrics': var_metrics,
            'concentration_metrics': concentration_metrics,
            'correlation_metrics': correlation_metrics,
            'drawdown_metrics': drawdown_metrics
        }
    
    def _check_risk_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verifica umbrales de riesgo"""
        alerts = []
        
        # VaR threshold check
        current_var = metrics['var_metrics']['portfolio_var']
        var_limit = self.portfolio_manager.risk_limits['max_portfolio_var']
        
        if current_var > var_limit * self.alert_thresholds['var_breach']:
            alerts.append({
                'type': 'var_breach',
                'severity': 'HIGH' if current_var > var_limit else 'MEDIUM',
                'current_value': current_var,
                'threshold': var_limit,
                'message': f"Portfolio VaR ({current_var:.3f}) approaching limit ({var_limit:.3f})"
            })
        
        # Add other threshold checks...
        
        return alerts
    
    async def _process_alerts(self, alerts: List[Dict[str, Any]]):
        """Procesa alertas de riesgo"""
        for alert in alerts:
            # Log alert
            print(f"🚨 RISK ALERT: {alert['message']}")
            
            # Take automated action if needed
            if alert['severity'] == 'HIGH':
                await self._take_emergency_action(alert)
            
            # Send notifications
            await self._send_risk_notification(alert)
    
    async def _take_emergency_action(self, alert: Dict[str, Any]):
        """Toma acción automática de emergencia"""
        if alert['type'] == 'var_breach':
            # Reduce position sizes automatically
            reduction_factor = 0.9  # Reduce positions by 10%
            
            for symbol, position in self.portfolio_manager.positions.items():
                new_quantity = position.quantity * reduction_factor
                position.quantity = new_quantity
            
            print("⚠️ Emergency position reduction executed")

class RiskReportingEngine:
    """Engine de reportes de riesgo"""
    
    def __init__(self):
        self.report_templates = {
            'daily_risk_report': self._generate_daily_risk_report,
            'weekly_risk_summary': self._generate_weekly_risk_summary,
            'monthly_risk_analysis': self._generate_monthly_risk_analysis,
            'stress_test_report': self._generate_stress_test_report
        }
    
    def generate_risk_report(self, portfolio_manager: PortfolioRiskManager,
                           report_type: str = 'daily_risk_report') -> str:
        """Genera reporte de riesgo"""
        
        if report_type not in self.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")
        
        return self.report_templates[report_type](portfolio_manager)
    
    def _generate_daily_risk_report(self, portfolio_manager: PortfolioRiskManager) -> str:
        """Genera reporte diario de riesgo"""
        
        # Calculate metrics
        var_metrics = portfolio_manager.calculate_portfolio_var()
        attribution = RiskAttributionEngine().attribute_portfolio_risk(portfolio_manager)
        
        report = f"""
# Daily Risk Report - {datetime.now().strftime('%Y-%m-%d')}

## Portfolio Overview
- Total Portfolio Value: ${portfolio_manager._calculate_portfolio_value():,.2f}
- Number of Positions: {len(portfolio_manager.positions)}
- Portfolio VaR (95%): ${var_metrics['portfolio_var']*portfolio_manager._calculate_portfolio_value():,.2f}

## Risk Attribution
### Top Risk Contributors:
"""
        
        # Sort positions by risk contribution
        sorted_contributions = sorted(
            var_metrics['component_vars'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for symbol, contribution in sorted_contributions[:5]:
            percentage = (contribution / var_metrics['portfolio_var']) * 100
            report += f"- {symbol}: ${contribution*portfolio_manager._calculate_portfolio_value():,.2f} ({percentage:.1f}%)\n"
        
        report += f"""

## Risk Limits Status:
- VaR Utilization: {(var_metrics['portfolio_var'] / portfolio_manager.risk_limits['max_portfolio_var']) * 100:.1f}%
- Max Position Weight: {max(pos.market_value / portfolio_manager._calculate_portfolio_value() for pos in portfolio_manager.positions.values()) * 100:.1f}%

## Recommendations:
{self._generate_risk_recommendations(portfolio_manager)}
"""
        
        return report
```

### Usage Example

```python
# Ejemplo de uso completo del sistema de riesgo de portfolio
async def main():
    """Ejemplo completo de gestión de riesgo de portfolio"""
    
    # 1. Initialize portfolio risk manager
    portfolio_manager = PortfolioRiskManager(initial_capital=1000000)
    
    # 2. Add positions
    positions = [
        Position("AAPL", 100, 150.0, 155.0, datetime.now(), "momentum", "technology", "large_cap"),
        Position("TSLA", 50, 200.0, 195.0, datetime.now(), "momentum", "automotive", "large_cap"),
        Position("NVDA", 75, 300.0, 310.0, datetime.now(), "growth", "technology", "large_cap")
    ]
    
    for position in positions:
        portfolio_manager.add_position(position)
    
    # 3. Calculate risk metrics
    var_metrics = portfolio_manager.calculate_portfolio_var()
    print(f"Portfolio VaR: ${var_metrics['portfolio_var']*1000000:.2f}")
    
    # 4. Risk attribution
    attribution_engine = RiskAttributionEngine()
    attribution = attribution_engine.attribute_portfolio_risk(portfolio_manager)
    print("Risk Attribution:", attribution['percentage_contributions'])
    
    # 5. Stress testing
    stress_framework = StressTestingFramework()
    stress_results = stress_framework.run_stress_tests(portfolio_manager)
    print("Worst Case Scenario:", stress_results['worst_case_scenario'])
    
    # 6. Risk optimization
    optimization_result = portfolio_manager.optimize_portfolio_risk()
    if optimization_result['status'] == 'success':
        print(f"Risk can be reduced by {optimization_result['risk_reduction']:.4f}")
    
    # 7. Start real-time monitoring
    risk_monitor = RealTimeRiskMonitor(portfolio_manager)
    await risk_monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
```

---

*La gestión de riesgo de portfolio es fundamental para el éxito a largo plazo en trading algorítmico. Un framework robusto que considere correlaciones, concentraciones y efectos sistémicos permite optimizar el balance riesgo-rendimiento mientras protege el capital.*