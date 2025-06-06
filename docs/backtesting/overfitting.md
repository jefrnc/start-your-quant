# Cómo Evitar Overfitting

## La Trampa Más Peligrosa del Backtesting

Overfitting es cuando tu estrategia funciona perfectamente en datos históricos pero falla miserablemente en vivo. Es como memorizar las respuestas de un examen específico sin entender el tema.

## ¿Qué es Overfitting?

### Definición Simple
Tu estrategia se ajusta tanto a los datos históricos que captura ruido en lugar de señales reales. Funciona en el pasado pero no se generaliza al futuro.

### Ejemplo Visual
```python
# ❌ OVERFITTED: 15 parámetros para explicar 100 trades
def overfitted_strategy(data):
    return (
        (data['sma5'] > data['sma10']) &
        (data['sma10'] > data['sma15']) &
        (data['rsi'] > 52.3) &  # Muy específico
        (data['rsi'] < 67.8) &  # Muy específico
        (data['volume'] > data['volume_sma'] * 1.847) &  # Decimal específico
        (data['hour'] == 10) &  # Solo 10 AM
        (data['minute'] >= 15) &  # Entre 10:15-10:30
        (data['minute'] <= 30) &
        (data['vwap_distance'] > 0.0023) &  # Muy específico
        (data['day_of_week'] != 2) &  # No Tuesday
        # ... 5 condiciones más específicas
    )

# ✅ GENERALIZABLE: 3 parámetros simples
def robust_strategy(data):
    return (
        (data['close'] > data['vwap']) &
        (data['rvol'] > 2) &
        (data['rsi'] > 50)
    )
```

## Señales de Overfitting

### 1. Métricas Demasiado Buenas
```python
def detect_overfitting_signals(backtest_results):
    """Detectar señales de overfitting"""
    red_flags = []
    
    # Sharpe ratio irreal
    if backtest_results['sharpe_ratio'] > 3:
        red_flags.append("Sharpe ratio demasiado alto (>3)")
    
    # Win rate irreal
    if backtest_results['win_rate'] > 0.8:
        red_flags.append("Win rate demasiado alto (>80%)")
    
    # Drawdown demasiado bajo
    if backtest_results['max_drawdown'] < 0.03:
        red_flags.append("Max drawdown demasiado bajo (<3%)")
    
    # Profit factor irreal
    if backtest_results['profit_factor'] > 4:
        red_flags.append("Profit factor demasiado alto (>4)")
    
    # Pocos trades
    if backtest_results['total_trades'] < 50:
        red_flags.append("Muy pocos trades para ser estadísticamente significativo")
    
    return red_flags

# Ejemplo de uso
suspicious_results = {
    'sharpe_ratio': 4.2,
    'win_rate': 0.87,
    'max_drawdown': 0.018,
    'profit_factor': 5.8,
    'total_trades': 23
}

flags = detect_overfitting_signals(suspicious_results)
print("🚨 Red flags detectadas:")
for flag in flags:
    print(f"  - {flag}")
```

### 2. Performance Inconsistente
```python
def test_temporal_stability(strategy, data, periods=4):
    """Test de estabilidad temporal"""
    results = []
    period_length = len(data) // periods
    
    for i in range(periods):
        start_idx = i * period_length
        end_idx = (i + 1) * period_length if i < periods - 1 else len(data)
        
        period_data = data.iloc[start_idx:end_idx]
        period_result = backtest_strategy(strategy, period_data)
        results.append(period_result['total_return'])
    
    # Calcular consistencia
    consistency = {
        'results_by_period': results,
        'mean_return': np.mean(results),
        'std_return': np.std(results),
        'min_return': min(results),
        'max_return': max(results),
        'coefficient_of_variation': np.std(results) / np.mean(results) if np.mean(results) != 0 else float('inf')
    }
    
    # Red flag si la variación es muy alta
    if consistency['coefficient_of_variation'] > 1:
        consistency['warning'] = "Alta variabilidad entre períodos - posible overfitting"
    
    return consistency
```

## Técnicas Anti-Overfitting

### 1. Cross-Validation Temporal
```python
def walk_forward_validation(strategy, data, train_periods=252, test_periods=63):
    """Walk-forward analysis para validar robustez"""
    results = []
    
    for start in range(0, len(data) - train_periods - test_periods, test_periods):
        # Training period
        train_start = start
        train_end = start + train_periods
        train_data = data.iloc[train_start:train_end]
        
        # Test period
        test_start = train_end
        test_end = train_start + test_periods
        test_data = data.iloc[test_start:test_end]
        
        # Optimize strategy on training data
        optimized_params = optimize_strategy_parameters(strategy, train_data)
        
        # Test on out-of-sample data
        test_result = backtest_strategy(strategy, test_data, optimized_params)
        
        results.append({
            'train_period': (train_start, train_end),
            'test_period': (test_start, test_end),
            'train_return': backtest_strategy(strategy, train_data, optimized_params)['total_return'],
            'test_return': test_result['total_return'],
            'params': optimized_params
        })
    
    # Analizar degradación
    train_returns = [r['train_return'] for r in results]
    test_returns = [r['test_return'] for r in results]
    
    degradation = np.mean(train_returns) - np.mean(test_returns)
    
    return {
        'results': results,
        'avg_train_return': np.mean(train_returns),
        'avg_test_return': np.mean(test_returns),
        'degradation': degradation,
        'degradation_pct': degradation / np.mean(train_returns) if np.mean(train_returns) != 0 else 0
    }
```

### 2. Parameter Stability Test
```python
def parameter_stability_test(strategy, data, param_ranges, num_tests=100):
    """Test de estabilidad de parámetros"""
    results = []
    
    for _ in range(num_tests):
        # Generar parámetros aleatorios dentro de rangos
        random_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int):
                random_params[param] = np.random.randint(min_val, max_val + 1)
            else:
                random_params[param] = np.random.uniform(min_val, max_val)
        
        # Test strategy con estos parámetros
        result = backtest_strategy(strategy, data, random_params)
        results.append({
            'params': random_params,
            'return': result['total_return'],
            'sharpe': result['sharpe_ratio'],
            'max_dd': result['max_drawdown']
        })
    
    # Analizar distribución de resultados
    returns = [r['return'] for r in results]
    
    stability_analysis = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'profitable_percentage': len([r for r in returns if r > 0]) / len(returns),
        'robust_percentage': len([r for r in returns if r > 0.1]) / len(returns),  # >10% return
        'parameter_sensitivity': np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else float('inf')
    }
    
    return stability_analysis, results
```

### 3. Bootstrap Analysis
```python
def bootstrap_analysis(strategy, returns, num_bootstrap=1000):
    """Bootstrap para estimar distribución de métricas"""
    bootstrap_results = []
    
    for _ in range(num_bootstrap):
        # Sample returns con replacement
        bootstrap_sample = np.random.choice(returns, len(returns), replace=True)
        
        # Calcular métricas para esta muestra
        sharpe = calculate_sharpe_ratio(bootstrap_sample)
        max_dd = calculate_max_drawdown(np.cumprod(1 + bootstrap_sample))
        
        bootstrap_results.append({
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_return': np.prod(1 + bootstrap_sample) - 1
        })
    
    # Confidence intervals
    sharpe_values = [r['sharpe'] for r in bootstrap_results]
    return_values = [r['total_return'] for r in bootstrap_results]
    
    confidence_intervals = {
        'sharpe_95_ci': (np.percentile(sharpe_values, 2.5), np.percentile(sharpe_values, 97.5)),
        'return_95_ci': (np.percentile(return_values, 2.5), np.percentile(return_values, 97.5)),
        'sharpe_median': np.median(sharpe_values),
        'return_median': np.median(return_values)
    }
    
    return confidence_intervals
```

## Validation Framework

### 1. In-Sample vs Out-of-Sample
```python
class ValidationFramework:
    def __init__(self, data, split_ratio=0.7):
        self.data = data
        self.split_point = int(len(data) * split_ratio)
        
        self.in_sample = data.iloc[:self.split_point]
        self.out_sample = data.iloc[self.split_point:]
        
    def develop_strategy(self, base_strategy):
        """Desarrollar estrategia en in-sample data"""
        # Optimize parameters
        best_params = self.optimize_parameters(base_strategy, self.in_sample)
        
        # Test en in-sample
        in_sample_results = backtest_strategy(base_strategy, self.in_sample, best_params)
        
        return best_params, in_sample_results
    
    def validate_strategy(self, strategy, params):
        """Una sola validación en out-of-sample"""
        out_sample_results = backtest_strategy(strategy, self.out_sample, params)
        return out_sample_results
    
    def full_validation(self, strategy):
        """Proceso completo de desarrollo y validación"""
        # Desarrollo
        best_params, in_sample_results = self.develop_strategy(strategy)
        
        # Validación (solo una vez!)
        out_sample_results = self.validate_strategy(strategy, best_params)
        
        # Comparar performance
        degradation = in_sample_results['total_return'] - out_sample_results['total_return']
        degradation_pct = degradation / in_sample_results['total_return']
        
        # Veredicto
        if degradation_pct < 0.3:  # Menos del 30% de degradación
            verdict = "PASSED - Strategy is robust"
        elif degradation_pct < 0.5:
            verdict = "WARNING - Moderate degradation"
        else:
            verdict = "FAILED - Significant overfitting detected"
        
        return {
            'in_sample': in_sample_results,
            'out_sample': out_sample_results,
            'degradation': degradation,
            'degradation_pct': degradation_pct,
            'verdict': verdict,
            'best_params': best_params
        }
```

### 2. Monte Carlo Validation
```python
def monte_carlo_validation(strategy, returns, num_simulations=1000):
    """Monte Carlo para test de robustez"""
    simulation_results = []
    
    for _ in range(num_simulations):
        # Shuffle returns (mantener distribución pero cambiar orden)
        shuffled_returns = np.random.permutation(returns)
        
        # Crear equity curve simulada
        equity_curve = np.cumprod(1 + shuffled_returns)
        
        # Calcular métricas
        total_return = equity_curve[-1] - 1
        max_dd = calculate_max_drawdown(equity_curve)
        sharpe = calculate_sharpe_ratio(shuffled_returns)
        
        simulation_results.append({
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe
        })
    
    # Comparar con resultado original
    original_sharpe = calculate_sharpe_ratio(returns)
    simulated_sharpes = [r['sharpe_ratio'] for r in simulation_results]
    
    # Percentile del resultado original
    percentile = (np.sum(np.array(simulated_sharpes) < original_sharpe) / len(simulated_sharpes)) * 100
    
    analysis = {
        'original_sharpe': original_sharpe,
        'simulated_mean_sharpe': np.mean(simulated_sharpes),
        'simulated_std_sharpe': np.std(simulated_sharpes),
        'percentile_rank': percentile,
        'is_significant': percentile > 95,  # Top 5%
        'simulation_results': simulation_results
    }
    
    return analysis
```

## Best Practices Anti-Overfitting

### 1. Principio de Parsimonia (Occam's Razor)
```python
def strategy_complexity_score(strategy_conditions):
    """Scoring de complejidad de estrategia"""
    complexity_factors = {
        'num_conditions': len(strategy_conditions),
        'num_parameters': count_unique_parameters(strategy_conditions),
        'decimal_precision': check_decimal_precision(strategy_conditions),
        'time_specificity': check_time_specificity(strategy_conditions)
    }
    
    # Score: menor es mejor
    complexity_score = (
        complexity_factors['num_conditions'] * 2 +
        complexity_factors['num_parameters'] * 3 +
        complexity_factors['decimal_precision'] * 5 +
        complexity_factors['time_specificity'] * 4
    )
    
    if complexity_score < 10:
        assessment = "Simple y robusto"
    elif complexity_score < 20:
        assessment = "Moderadamente complejo"
    else:
        assessment = "Demasiado complejo - riesgo de overfitting"
    
    return {
        'score': complexity_score,
        'assessment': assessment,
        'factors': complexity_factors
    }
```

### 2. Economic Intuition Check
```python
def economic_intuition_check(strategy_logic):
    """Verificar si la estrategia tiene sentido económico"""
    intuition_questions = [
        "¿Por qué debería funcionar esta estrategia?",
        "¿Qué ineficiencia del mercado explota?",
        "¿Por qué otros traders no la están usando?",
        "¿Funcionaría en diferentes condiciones de mercado?",
        "¿Es escalable con más capital?"
    ]
    
    # Esta función requiere input humano, pero el framework ayuda
    return {
        'questions': intuition_questions,
        'reminder': "Si no puedes explicar por qué funciona, probablemente sea overfitting"
    }
```

### 3. Regime Testing
```python
def test_across_market_regimes(strategy, data):
    """Test en diferentes regímenes de mercado"""
    
    # Identificar regímenes (simplificado)
    market_returns = data['SPY'].pct_change() if 'SPY' in data else data.iloc[:, 0].pct_change()
    volatility = market_returns.rolling(20).std()
    
    # Clasificar períodos
    regimes = {}
    
    # Bull/Bear markets
    rolling_returns = market_returns.rolling(60).sum()
    regimes['bull'] = data[rolling_returns > 0.1]  # >10% en 60 días
    regimes['bear'] = data[rolling_returns < -0.1]  # <-10% en 60 días
    
    # High/Low volatility
    vol_median = volatility.median()
    regimes['low_vol'] = data[volatility < vol_median]
    regimes['high_vol'] = data[volatility >= vol_median]
    
    # Test strategy en cada régimen
    regime_results = {}
    
    for regime_name, regime_data in regimes.items():
        if len(regime_data) > 100:  # Sufficient data
            result = backtest_strategy(strategy, regime_data)
            regime_results[regime_name] = result
    
    # Análizar consistencia
    returns_by_regime = [r['total_return'] for r in regime_results.values()]
    sharpe_by_regime = [r['sharpe_ratio'] for r in regime_results.values()]
    
    consistency_analysis = {
        'regime_results': regime_results,
        'return_consistency': np.std(returns_by_regime) / np.mean(returns_by_regime) if np.mean(returns_by_regime) != 0 else float('inf'),
        'works_in_all_regimes': all(r > 0 for r in returns_by_regime),
        'consistent_sharpe': all(s > 0.5 for s in sharpe_by_regime)
    }
    
    return consistency_analysis
```

## Mi Checklist Anti-Overfitting

```python
def overfitting_checklist(strategy, backtest_results, validation_results):
    """Checklist completo anti-overfitting"""
    
    checklist = {
        'metrics_realistic': True,
        'sufficient_trades': True,
        'out_sample_validation': True,
        'economic_intuition': True,
        'parameter_stability': True,
        'regime_robustness': True,
        'complexity_reasonable': True
    }
    
    issues = []
    
    # 1. Métricas realistas
    if (backtest_results['sharpe_ratio'] > 3 or 
        backtest_results['win_rate'] > 0.8 or 
        backtest_results['max_drawdown'] < 0.03):
        checklist['metrics_realistic'] = False
        issues.append("Métricas irrealísticamente buenas")
    
    # 2. Trades suficientes
    if backtest_results['total_trades'] < 100:
        checklist['sufficient_trades'] = False
        issues.append("Insuficientes trades para significancia estadística")
    
    # 3. Validación out-of-sample
    if validation_results['degradation_pct'] > 0.5:
        checklist['out_sample_validation'] = False
        issues.append("Degradación significativa en out-of-sample")
    
    # 4. Complejidad
    complexity = strategy_complexity_score(strategy.conditions)
    if complexity['score'] > 20:
        checklist['complexity_reasonable'] = False
        issues.append("Estrategia demasiado compleja")
    
    # Score final
    passed_checks = sum(checklist.values())
    total_checks = len(checklist)
    
    if passed_checks == total_checks:
        verdict = "✅ STRATEGY APPROVED - Low overfitting risk"
    elif passed_checks >= total_checks * 0.8:
        verdict = "⚠️ PROCEED WITH CAUTION - Some concerns"
    else:
        verdict = "❌ HIGH OVERFITTING RISK - Needs rework"
    
    return {
        'checklist': checklist,
        'issues': issues,
        'score': f"{passed_checks}/{total_checks}",
        'verdict': verdict
    }
```

## Próximos Pasos Después del Backtest

```python
def post_backtest_roadmap(validation_results):
    """Roadmap después de validar estrategia"""
    
    if validation_results['verdict'] == "PASSED":
        return {
            'next_step': 'Paper Trading',
            'duration': '2-3 meses',
            'success_criteria': 'Correlación >70% con backtest',
            'position_size': 'Empezar con 25% del tamaño planeado'
        }
    elif "WARNING" in validation_results['verdict']:
        return {
            'next_step': 'Refinement',
            'actions': [
                'Simplificar estrategia',
                'Ampliar período de test',
                'Test en más regímenes de mercado'
            ]
        }
    else:
        return {
            'next_step': 'Back to Drawing Board',
            'actions': [
                'Revisar lógica fundamental',
                'Buscar nueva edge',
                'Empezar desde cero'
            ]
        }
```

## Siguiente Paso

Con el backtesting dominado, vamos a [Gestión de Riesgo](../risk/position_sizing.md) para proteger tu capital.