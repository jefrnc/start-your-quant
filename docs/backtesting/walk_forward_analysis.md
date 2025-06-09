# Walk-Forward Analysis: Validación Dinámica de Estrategias

## Introducción: Más Allá del Backtesting Estático

El walk-forward analysis es una metodología avanzada de validación que simula cómo una estrategia se habría adaptado a condiciones de mercado cambiantes en tiempo real. A diferencia del backtesting tradicional, que optimiza parámetros en todo el período histórico, el walk-forward utiliza ventanas móviles para reoptimizar periódicamente, proporcionando una evaluación más realista del rendimiento futuro.

## ¿Por Qué Es Crítico el Walk-Forward?

### Problemas del Backtesting Tradicional

**Look-Ahead Bias:**
```python
# INCORRECTO: Usar datos futuros para optimización
def bad_backtest(data):
    # Optimiza parámetros usando TODOS los datos históricos
    best_params = optimize_parameters(data['2020':'2023'])
    
    # Evalúa rendimiento en el mismo período
    results = backtest_strategy(data['2020':'2023'], best_params)
    return results  # Resultados inflados por look-ahead bias

# CORRECTO: Walk-forward approach
def good_walkforward(data):
    results = []
    for window_start in date_range:
        # Solo usa datos hasta la fecha actual
        training_data = data[window_start:current_date]
        
        # Optimiza parámetros solo con datos pasados
        params = optimize_parameters(training_data)
        
        # Evalúa en período futuro (out-of-sample)
        future_data = data[current_date:next_date]
        period_result = backtest_strategy(future_data, params)
        results.append(period_result)
    
    return combine_results(results)
```

**Régimen de Mercado Estático:**
- Parámetros optimizados para condiciones específicas
- No adaptación a cambios estructurales del mercado
- Degradación de performance en nuevos regímenes

## Implementación de Walk-Forward Analysis

### Architecture Base

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

class WalkForwardAnalyzer:
    def __init__(self, 
                 strategy_function,
                 optimization_function,
                 parameter_space: Dict[str, List],
                 in_sample_period: int = 252,  # 1 año
                 out_sample_period: int = 63,  # 3 meses
                 reoptimization_frequency: int = 21):  # Mensual
        
        self.strategy_function = strategy_function
        self.optimization_function = optimization_function
        self.parameter_space = parameter_space
        self.in_sample_period = in_sample_period
        self.out_sample_period = out_sample_period
        self.reopt_frequency = reoptimization_frequency
        
        # Results storage
        self.walk_forward_results = []
        self.parameter_evolution = []
        self.performance_metrics = {}
    
    def run_walk_forward(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta análisis walk-forward completo
        """
        print("Iniciando Walk-Forward Analysis...")
        
        # Configurar ventanas de análisis
        analysis_windows = self._create_analysis_windows(data)
        
        all_trades = []
        all_returns = []
        
        for i, window in enumerate(analysis_windows):
            print(f"Procesando ventana {i+1}/{len(analysis_windows)}")
            
            # Extraer datos de entrenamiento y testing
            training_data = data[window['train_start']:window['train_end']]
            testing_data = data[window['test_start']:window['test_end']]
            
            if len(training_data) < self.in_sample_period or len(testing_data) == 0:
                continue
            
            # Optimizar parámetros en datos de entrenamiento
            optimal_params = self._optimize_parameters(training_data)
            
            # Ejecutar estrategia en datos de testing
            window_results = self._execute_strategy(testing_data, optimal_params)
            
            # Almacenar resultados
            window_info = {
                'window_id': i,
                'train_period': f"{window['train_start']} to {window['train_end']}",
                'test_period': f"{window['test_start']} to {window['test_end']}",
                'optimal_params': optimal_params,
                'trades': window_results['trades'],
                'returns': window_results['returns'],
                'metrics': window_results['metrics']
            }
            
            self.walk_forward_results.append(window_info)
            self.parameter_evolution.append({
                'date': window['test_start'],
                'parameters': optimal_params
            })
            
            all_trades.extend(window_results['trades'])
            all_returns.extend(window_results['returns'])
        
        # Calcular métricas agregadas
        self.performance_metrics = self._calculate_aggregate_metrics(all_trades, all_returns)
        
        return {
            'aggregate_metrics': self.performance_metrics,
            'window_results': self.walk_forward_results,
            'parameter_evolution': self.parameter_evolution
        }
    
    def _create_analysis_windows(self, data: pd.DataFrame) -> List[Dict]:
        """
        Crea ventanas de análisis para walk-forward
        """
        windows = []
        dates = data.index
        
        start_idx = self.in_sample_period
        
        while start_idx + self.out_sample_period < len(dates):
            train_start_idx = max(0, start_idx - self.in_sample_period)
            train_end_idx = start_idx
            test_start_idx = start_idx
            test_end_idx = min(len(dates), start_idx + self.out_sample_period)
            
            window = {
                'train_start': dates[train_start_idx],
                'train_end': dates[train_end_idx - 1],
                'test_start': dates[test_start_idx],
                'test_end': dates[test_end_idx - 1]
            }
            
            windows.append(window)
            start_idx += self.reopt_frequency
        
        return windows
    
    def _optimize_parameters(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimiza parámetros usando solo datos de entrenamiento
        """
        return self.optimization_function(training_data, self.parameter_space)
    
    def _execute_strategy(self, testing_data: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """
        Ejecuta estrategia con parámetros optimizados en datos de testing
        """
        trades, returns = self.strategy_function(testing_data, params)
        
        metrics = self._calculate_window_metrics(trades, returns)
        
        return {
            'trades': trades,
            'returns': returns,
            'metrics': metrics
        }
```

### Parameter Optimization Engine

```python
class ParameterOptimizer:
    def __init__(self, optimization_metric='sharpe_ratio'):
        self.optimization_metric = optimization_metric
        self.optimization_history = []
    
    def grid_search_optimization(self, data: pd.DataFrame, 
                                strategy_function, 
                                parameter_space: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimización por grid search
        """
        from itertools import product
        
        # Generar todas las combinaciones de parámetros
        param_combinations = list(product(*parameter_space.values()))
        param_names = list(parameter_space.keys())
        
        best_score = float('-inf')
        best_params = None
        optimization_results = []
        
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            
            try:
                # Ejecutar estrategia con estos parámetros
                trades, returns = strategy_function(data, params)
                
                # Calcular métricas
                metrics = self._calculate_optimization_metrics(trades, returns)
                score = metrics[self.optimization_metric]
                
                optimization_results.append({
                    'parameters': params,
                    'score': score,
                    'metrics': metrics
                })
                
                # Actualizar mejor resultado
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"Error con parámetros {params}: {e}")
                continue
        
        self.optimization_history.append({
            'date': data.index[-1],
            'best_params': best_params,
            'best_score': best_score,
            'all_results': optimization_results
        })
        
        return best_params
    
    def genetic_algorithm_optimization(self, data: pd.DataFrame,
                                     strategy_function,
                                     parameter_space: Dict[str, List],
                                     population_size: int = 50,
                                     generations: int = 100) -> Dict[str, Any]:
        """
        Optimización usando algoritmo genético
        """
        import random
        
        # Inicializar población
        population = self._initialize_population(parameter_space, population_size)
        
        for generation in range(generations):
            # Evaluar fitness de cada individuo
            fitness_scores = []
            for individual in population:
                try:
                    trades, returns = strategy_function(data, individual)
                    metrics = self._calculate_optimization_metrics(trades, returns)
                    fitness = metrics[self.optimization_metric]
                    fitness_scores.append(fitness)
                except:
                    fitness_scores.append(float('-inf'))
            
            # Selección, crossover y mutación
            population = self._evolve_population(population, fitness_scores, parameter_space)
        
        # Retornar mejor individuo
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]
    
    def bayesian_optimization(self, data: pd.DataFrame,
                            strategy_function,
                            parameter_space: Dict[str, List],
                            n_iterations: int = 50) -> Dict[str, Any]:
        """
        Optimización bayesiana para búsqueda eficiente de parámetros
        """
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        
        # Convertir parameter_space a formato skopt
        dimensions = []
        param_names = []
        
        for param_name, param_range in parameter_space.items():
            param_names.append(param_name)
            if isinstance(param_range[0], int):
                dimensions.append(Integer(min(param_range), max(param_range)))
            else:
                dimensions.append(Real(min(param_range), max(param_range)))
        
        def objective(params_list):
            params = dict(zip(param_names, params_list))
            try:
                trades, returns = strategy_function(data, params)
                metrics = self._calculate_optimization_metrics(trades, returns)
                # Negativo porque skopt minimiza
                return -metrics[self.optimization_metric]
            except:
                return 1000  # Penalty for invalid parameters
        
        # Ejecutar optimización bayesiana
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_iterations,
            random_state=42
        )
        
        # Convertir resultado a dict
        best_params = dict(zip(param_names, result.x))
        return best_params
```

### Stability Analysis

```python
class WalkForwardStabilityAnalyzer:
    def __init__(self):
        self.stability_metrics = {}
    
    def analyze_parameter_stability(self, parameter_evolution: List[Dict]) -> Dict[str, Any]:
        """
        Analiza estabilidad de parámetros a lo largo del tiempo
        """
        if not parameter_evolution:
            return {}
        
        # Extraer series temporales de cada parámetro
        param_series = {}
        dates = []
        
        for evolution_point in parameter_evolution:
            dates.append(evolution_point['date'])
            params = evolution_point['parameters']
            
            for param_name, param_value in params.items():
                if param_name not in param_series:
                    param_series[param_name] = []
                param_series[param_name].append(param_value)
        
        # Calcular métricas de estabilidad
        stability_analysis = {}
        
        for param_name, values in param_series.items():
            values_array = np.array(values)
            
            stability_analysis[param_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'coefficient_of_variation': float(np.std(values_array) / np.mean(values_array)) if np.mean(values_array) != 0 else float('inf'),
                'trend': self._calculate_trend(values_array),
                'regime_changes': self._detect_regime_changes(values_array),
                'stability_score': self._calculate_stability_score(values_array)
            }
        
        return stability_analysis
    
    def analyze_performance_consistency(self, walk_forward_results: List[Dict]) -> Dict[str, Any]:
        """
        Analiza consistencia de performance a través de las ventanas
        """
        window_metrics = []
        
        for window in walk_forward_results:
            metrics = window['metrics']
            window_metrics.append(metrics)
        
        # Agregar métricas por ventana
        metric_names = window_metrics[0].keys() if window_metrics else []
        consistency_analysis = {}
        
        for metric_name in metric_names:
            metric_values = [w[metric_name] for w in window_metrics if metric_name in w]
            
            if metric_values:
                metric_array = np.array(metric_values)
                
                consistency_analysis[metric_name] = {
                    'mean': float(np.mean(metric_array)),
                    'std': float(np.std(metric_array)),
                    'min': float(np.min(metric_array)),
                    'max': float(np.max(metric_array)),
                    'positive_periods': int(np.sum(metric_array > 0)),
                    'negative_periods': int(np.sum(metric_array < 0)),
                    'consistency_ratio': float(np.sum(metric_array > 0) / len(metric_array)),
                    'worst_period': float(np.min(metric_array)),
                    'best_period': float(np.max(metric_array))
                }
        
        return consistency_analysis
    
    def calculate_degradation_analysis(self, walk_forward_results: List[Dict]) -> Dict[str, Any]:
        """
        Analiza degradación de performance a lo largo del tiempo
        """
        dates = []
        returns = []
        sharpe_ratios = []
        
        for window in walk_forward_results:
            dates.append(pd.to_datetime(window['test_period'].split(' to ')[0]))
            returns.append(window['metrics'].get('total_return', 0))
            sharpe_ratios.append(window['metrics'].get('sharpe_ratio', 0))
        
        if not dates:
            return {}
        
        # Crear series temporales
        performance_df = pd.DataFrame({
            'date': dates,
            'returns': returns,
            'sharpe_ratio': sharpe_ratios
        }).set_index('date').sort_index()
        
        # Calcular tendencias
        degradation_analysis = {
            'returns_trend': self._calculate_trend(performance_df['returns'].values),
            'sharpe_trend': self._calculate_trend(performance_df['sharpe_ratio'].values),
            'returns_correlation_with_time': self._calculate_time_correlation(performance_df['returns']),
            'sharpe_correlation_with_time': self._calculate_time_correlation(performance_df['sharpe_ratio']),
            'performance_half_life': self._calculate_performance_half_life(performance_df),
            'regime_detection': self._detect_performance_regimes(performance_df)
        }
        
        return degradation_analysis
```

### Practical Example: Gap and Go Walk-Forward

```python
class GapAndGoWalkForward:
    def __init__(self):
        self.parameter_space = {
            'min_gap_percent': [0.01, 0.02, 0.03, 0.04, 0.05],
            'volume_multiplier': [1.5, 2.0, 2.5, 3.0],
            'vwap_threshold': [0.5, 1.0, 1.5, 2.0],
            'stop_loss_atr': [1.0, 1.5, 2.0, 2.5, 3.0],
            'take_profit_ratio': [1.5, 2.0, 2.5, 3.0]
        }
    
    def gap_and_go_strategy(self, data: pd.DataFrame, params: Dict) -> Tuple[List, List]:
        """
        Implementación de estrategia Gap and Go
        """
        trades = []
        returns = []
        
        # Calcular indicadores
        data['gap'] = data['open'] / data['close'].shift(1) - 1
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        data['atr'] = self._calculate_atr(data)
        
        position = None
        
        for i in range(20, len(data)):  # Start after enough data for indicators
            current = data.iloc[i]
            
            # Entry conditions
            if position is None:
                gap_condition = current['gap'] >= params['min_gap_percent']
                volume_condition = current['volume_ratio'] >= params['volume_multiplier']
                vwap_condition = current['close'] > current['vwap'] + params['vwap_threshold']
                
                if gap_condition and volume_condition and vwap_condition:
                    # Enter position
                    entry_price = current['close']
                    stop_loss = entry_price - (current['atr'] * params['stop_loss_atr'])
                    take_profit = entry_price + ((entry_price - stop_loss) * params['take_profit_ratio'])
                    
                    position = {
                        'entry_price': entry_price,
                        'entry_date': current.name,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
            
            # Exit conditions
            elif position is not None:
                exit_price = None
                exit_reason = None
                
                # Check stop loss
                if current['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                
                # Check take profit
                elif current['high'] >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'take_profit'
                
                # Time-based exit (end of day)
                elif current.name.time() >= pd.Timestamp('15:30').time():
                    exit_price = current['close']
                    exit_reason = 'time_exit'
                
                if exit_price:
                    # Calculate return
                    trade_return = (exit_price - position['entry_price']) / position['entry_price']
                    
                    trade_info = {
                        'entry_date': position['entry_date'],
                        'exit_date': current.name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'return': trade_return,
                        'exit_reason': exit_reason,
                        'parameters': params.copy()
                    }
                    
                    trades.append(trade_info)
                    returns.append(trade_return)
                    position = None
        
        return trades, returns
    
    def run_gap_and_go_walkforward(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta walk-forward analysis para Gap and Go
        """
        # Initialize walk-forward analyzer
        wfa = WalkForwardAnalyzer(
            strategy_function=self.gap_and_go_strategy,
            optimization_function=self._optimize_gap_and_go,
            parameter_space=self.parameter_space,
            in_sample_period=252,  # 1 year training
            out_sample_period=63,  # 3 months testing
            reoptimization_frequency=21  # Monthly reoptimization
        )
        
        # Run analysis
        results = wfa.run_walk_forward(data)
        
        # Additional analysis
        stability_analyzer = WalkForwardStabilityAnalyzer()
        
        stability_analysis = stability_analyzer.analyze_parameter_stability(
            results['parameter_evolution']
        )
        
        consistency_analysis = stability_analyzer.analyze_performance_consistency(
            results['window_results']
        )
        
        degradation_analysis = stability_analyzer.calculate_degradation_analysis(
            results['window_results']
        )
        
        return {
            'walk_forward_results': results,
            'stability_analysis': stability_analysis,
            'consistency_analysis': consistency_analysis,
            'degradation_analysis': degradation_analysis,
            'summary_report': self._generate_summary_report(results, stability_analysis)
        }
```

### Visualization and Reporting

```python
class WalkForwardVisualizer:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_parameter_evolution(self, parameter_evolution: List[Dict]):
        """
        Visualiza evolución de parámetros a lo largo del tiempo
        """
        import matplotlib.pyplot as plt
        
        if not parameter_evolution:
            return
        
        # Extract data
        dates = [pe['date'] for pe in parameter_evolution]
        param_names = list(parameter_evolution[0]['parameters'].keys())
        
        fig, axes = plt.subplots(len(param_names), 1, figsize=(12, 3*len(param_names)))
        if len(param_names) == 1:
            axes = [axes]
        
        for i, param_name in enumerate(param_names):
            values = [pe['parameters'][param_name] for pe in parameter_evolution]
            
            axes[i].plot(dates, values, marker='o', linewidth=2, color=self.colors[i % len(self.colors)])
            axes[i].set_title(f'Evolution of {param_name}')
            axes[i].set_ylabel(param_name)
            axes[i].grid(True, alpha=0.3)
            
            # Add trend line
            x_numeric = np.arange(len(dates))
            z = np.polyfit(x_numeric, values, 1)
            p = np.poly1d(z)
            axes[i].plot(dates, p(x_numeric), "--", alpha=0.8, color='red')
        
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_performance(self, walk_forward_results: List[Dict]):
        """
        Visualiza performance rolling de las ventanas
        """
        import matplotlib.pyplot as plt
        
        dates = []
        returns = []
        sharpe_ratios = []
        win_rates = []
        
        for window in walk_forward_results:
            window_start = pd.to_datetime(window['test_period'].split(' to ')[0])
            dates.append(window_start)
            
            metrics = window['metrics']
            returns.append(metrics.get('total_return', 0))
            sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            win_rates.append(metrics.get('win_rate', 0))
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Returns
        axes[0].plot(dates, returns, marker='o', linewidth=2, color=self.colors[0])
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].set_title('Rolling Period Returns')
        axes[0].set_ylabel('Return (%)')
        axes[0].grid(True, alpha=0.3)
        
        # Sharpe Ratio
        axes[1].plot(dates, sharpe_ratios, marker='s', linewidth=2, color=self.colors[1])
        axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Good Sharpe (>1.0)')
        axes[1].set_title('Rolling Sharpe Ratio')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Win Rate
        axes[2].plot(dates, win_rates, marker='^', linewidth=2, color=self.colors[2])
        axes[2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% Win Rate')
        axes[2].set_title('Rolling Win Rate')
        axes[2].set_ylabel('Win Rate')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_walk_forward_report(self, analysis_results: Dict) -> str:
        """
        Genera reporte comprehensive de walk-forward analysis
        """
        report = """
# Walk-Forward Analysis Report

## Executive Summary
This report presents the results of a comprehensive walk-forward analysis,
providing insights into strategy robustness, parameter stability, and 
performance consistency across different market conditions.

## Key Findings

### Parameter Stability
"""
        
        stability = analysis_results.get('stability_analysis', {})
        for param_name, stability_metrics in stability.items():
            report += f"""
**{param_name}:**
- Mean Value: {stability_metrics['mean']:.4f}
- Standard Deviation: {stability_metrics['std']:.4f}
- Coefficient of Variation: {stability_metrics['coefficient_of_variation']:.4f}
- Stability Score: {stability_metrics['stability_score']:.4f}
"""

        consistency = analysis_results.get('consistency_analysis', {})
        report += f"""
### Performance Consistency

**Return Metrics:**
- Mean Return: {consistency.get('total_return', {}).get('mean', 0):.2%}
- Return Volatility: {consistency.get('total_return', {}).get('std', 0):.2%}
- Positive Periods: {consistency.get('total_return', {}).get('positive_periods', 0)}
- Consistency Ratio: {consistency.get('total_return', {}).get('consistency_ratio', 0):.2%}

**Sharpe Ratio:**
- Mean Sharpe: {consistency.get('sharpe_ratio', {}).get('mean', 0):.2f}
- Sharpe Volatility: {consistency.get('sharpe_ratio', {}).get('std', 0):.2f}
"""

        degradation = analysis_results.get('degradation_analysis', {})
        report += f"""
### Performance Degradation Analysis

- Returns Trend: {degradation.get('returns_trend', 'N/A')}
- Sharpe Trend: {degradation.get('sharpe_trend', 'N/A')}
- Time Correlation (Returns): {degradation.get('returns_correlation_with_time', 0):.3f}
- Time Correlation (Sharpe): {degradation.get('sharpe_correlation_with_time', 0):.3f}

## Recommendations

Based on the walk-forward analysis:
"""

        # Generate recommendations based on results
        recommendations = self._generate_recommendations(analysis_results)
        for rec in recommendations:
            report += f"- {rec}\n"

        return report
```

## Best Practices y Consideraciones

### 1. Configuración de Ventanas

```python
def configure_walk_forward_windows(market_type: str, strategy_frequency: str) -> Dict:
    """
    Configuración recomendada de ventanas según tipo de mercado y estrategia
    """
    configurations = {
        'day_trading_stocks': {
            'in_sample_period': 126,  # 6 meses
            'out_sample_period': 21,  # 1 mes
            'reopt_frequency': 7,     # Semanal
            'min_trades_required': 30
        },
        'swing_trading_stocks': {
            'in_sample_period': 252,  # 1 año
            'out_sample_period': 63,  # 3 meses
            'reopt_frequency': 21,    # Mensual
            'min_trades_required': 50
        },
        'crypto_trading': {
            'in_sample_period': 90,   # 3 meses (mercado 24/7)
            'out_sample_period': 30,  # 1 mes
            'reopt_frequency': 7,     # Semanal
            'min_trades_required': 20
        },
        'forex_scalping': {
            'in_sample_period': 30,   # 1 mes
            'out_sample_period': 7,   # 1 semana
            'reopt_frequency': 2,     # Cada 2 días
            'min_trades_required': 100
        }
    }
    
    key = f"{strategy_frequency}_{market_type}"
    return configurations.get(key, configurations['swing_trading_stocks'])
```

### 2. Validation Criteria

```python
def validate_walk_forward_results(results: Dict) -> Dict[str, bool]:
    """
    Valida resultados de walk-forward según criterios profesionales
    """
    validation_criteria = {
        'sufficient_windows': len(results['window_results']) >= 10,
        'parameter_stability': all(
            metrics.get('stability_score', 0) > 0.6 
            for metrics in results['stability_analysis'].values()
        ),
        'consistent_performance': (
            results['consistency_analysis'].get('total_return', {}).get('consistency_ratio', 0) > 0.6
        ),
        'no_severe_degradation': (
            abs(results['degradation_analysis'].get('returns_correlation_with_time', 0)) < 0.5
        ),
        'positive_expectancy': (
            results['walk_forward_results']['aggregate_metrics'].get('total_return', 0) > 0
        )
    }
    
    overall_validation = all(validation_criteria.values())
    
    return {
        **validation_criteria,
        'overall_valid': overall_validation,
        'validation_score': sum(validation_criteria.values()) / len(validation_criteria)
    }
```

---

*El walk-forward analysis es la metodología gold standard para validación robusta de estrategias de trading. Proporciona una evaluación realista de cómo se comportaría una estrategia en condiciones de mercado reales, incluyendo la adaptación continua de parámetros y la degradación natural del rendimiento a lo largo del tiempo.*