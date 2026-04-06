> 🇪🇸 [Leer en Español](overfitting.es.md) | 🇺🇸 **English**

# How to Avoid Overfitting

## The Most Dangerous Trap in Backtesting

Overfitting is when your strategy works perfectly on historical data but fails miserably in live trading. It's like memorizing the answers to a specific exam without understanding the subject.

## What Is Overfitting?

### Simple Definition
Your strategy fits so closely to historical data that it captures noise instead of real signals. It works in the past but doesn't generalize to the future.

### Visual Example
```python
# ❌ OVERFITTED: 15 parameters to explain 100 trades
def overfitted_strategy(data):
    return (
        (data['sma5'] > data['sma10']) &
        (data['sma10'] > data['sma15']) &
        (data['rsi'] > 52.3) &  # Too specific
        (data['rsi'] < 67.8) &  # Too specific
        (data['volume'] > data['volume_sma'] * 1.847) &  # Specific decimal
        (data['hour'] == 10) &  # Only 10 AM
        (data['minute'] >= 15) &  # Between 10:15-10:30
        (data['minute'] <= 30) &
        (data['vwap_distance'] > 0.0023) &  # Too specific
        (data['day_of_week'] != 2) &  # Not Tuesday
        # ... 5 more specific conditions
    )

# ✅ GENERALIZABLE: 3 simple parameters
def robust_strategy(data):
    return (
        (data['close'] > data['vwap']) &
        (data['rvol'] > 2) &
        (data['rsi'] > 50)
    )
```

## Signs of Overfitting

### 1. Metrics That Are Too Good
```python
def detect_overfitting_signals(backtest_results):
    """Detect overfitting signals"""
    red_flags = []
    
    # Unrealistic Sharpe ratio
    if backtest_results['sharpe_ratio'] > 3:
        red_flags.append("Sharpe ratio too high (>3)")
    
    # Unrealistic win rate
    if backtest_results['win_rate'] > 0.8:
        red_flags.append("Win rate too high (>80%)")
    
    # Drawdown too low
    if backtest_results['max_drawdown'] < 0.03:
        red_flags.append("Max drawdown too low (<3%)")
    
    # Unrealistic profit factor
    if backtest_results['profit_factor'] > 4:
        red_flags.append("Profit factor too high (>4)")
    
    # Too few trades
    if backtest_results['total_trades'] < 50:
        red_flags.append("Too few trades to be statistically significant")
    
    return red_flags

# Usage example
suspicious_results = {
    'sharpe_ratio': 4.2,
    'win_rate': 0.87,
    'max_drawdown': 0.018,
    'profit_factor': 5.8,
    'total_trades': 23
}

flags = detect_overfitting_signals(suspicious_results)
print("Red flags detected:")
for flag in flags:
    print(f"  - {flag}")
```

### 2. Inconsistent Performance
```python
def test_temporal_stability(strategy, data, periods=4):
    """Temporal stability test"""
    results = []
    period_length = len(data) // periods
    
    for i in range(periods):
        start_idx = i * period_length
        end_idx = (i + 1) * period_length if i < periods - 1 else len(data)
        
        period_data = data.iloc[start_idx:end_idx]
        period_result = backtest_strategy(strategy, period_data)
        results.append(period_result['total_return'])
    
    # Calculate consistency
    consistency = {
        'results_by_period': results,
        'mean_return': np.mean(results),
        'std_return': np.std(results),
        'min_return': min(results),
        'max_return': max(results),
        'coefficient_of_variation': np.std(results) / np.mean(results) if np.mean(results) != 0 else float('inf')
    }
    
    # Red flag if variation is very high
    if consistency['coefficient_of_variation'] > 1:
        consistency['warning'] = "High variability between periods - possible overfitting"
    
    return consistency
```

## Anti-Overfitting Techniques

### 1. Temporal Cross-Validation
```python
def walk_forward_validation(strategy, data, train_periods=252, test_periods=63):
    """Walk-forward analysis to validate robustness"""
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
    
    # Analyze degradation
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
    """Parameter stability test"""
    results = []
    
    for _ in range(num_tests):
        # Generate random parameters within ranges
        random_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int):
                random_params[param] = np.random.randint(min_val, max_val + 1)
            else:
                random_params[param] = np.random.uniform(min_val, max_val)
        
        # Test strategy with these parameters
        result = backtest_strategy(strategy, data, random_params)
        results.append({
            'params': random_params,
            'return': result['total_return'],
            'sharpe': result['sharpe_ratio'],
            'max_dd': result['max_drawdown']
        })
    
    # Analyze distribution of results
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
    """Bootstrap to estimate metric distributions"""
    bootstrap_results = []
    
    for _ in range(num_bootstrap):
        # Sample returns with replacement
        bootstrap_sample = np.random.choice(returns, len(returns), replace=True)
        
        # Calculate metrics for this sample
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
        """Develop strategy on in-sample data"""
        # Optimize parameters
        best_params = self.optimize_parameters(base_strategy, self.in_sample)
        
        # Test on in-sample
        in_sample_results = backtest_strategy(base_strategy, self.in_sample, best_params)
        
        return best_params, in_sample_results
    
    def validate_strategy(self, strategy, params):
        """Single validation on out-of-sample"""
        out_sample_results = backtest_strategy(strategy, self.out_sample, params)
        return out_sample_results
    
    def full_validation(self, strategy):
        """Complete development and validation process"""
        # Development
        best_params, in_sample_results = self.develop_strategy(strategy)
        
        # Validation (only once!)
        out_sample_results = self.validate_strategy(strategy, best_params)
        
        # Compare performance
        degradation = in_sample_results['total_return'] - out_sample_results['total_return']
        degradation_pct = degradation / in_sample_results['total_return']
        
        # Verdict
        if degradation_pct < 0.3:  # Less than 30% degradation
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
    """Monte Carlo robustness test"""
    simulation_results = []
    
    for _ in range(num_simulations):
        # Shuffle returns (maintain distribution but change order)
        shuffled_returns = np.random.permutation(returns)
        
        # Create simulated equity curve
        equity_curve = np.cumprod(1 + shuffled_returns)
        
        # Calculate metrics
        total_return = equity_curve[-1] - 1
        max_dd = calculate_max_drawdown(equity_curve)
        sharpe = calculate_sharpe_ratio(shuffled_returns)
        
        simulation_results.append({
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe
        })
    
    # Compare with original result
    original_sharpe = calculate_sharpe_ratio(returns)
    simulated_sharpes = [r['sharpe_ratio'] for r in simulation_results]
    
    # Percentile of original result
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

## Best Practices Against Overfitting

### 1. Principle of Parsimony (Occam's Razor)
```python
def strategy_complexity_score(strategy_conditions):
    """Strategy complexity scoring"""
    complexity_factors = {
        'num_conditions': len(strategy_conditions),
        'num_parameters': count_unique_parameters(strategy_conditions),
        'decimal_precision': check_decimal_precision(strategy_conditions),
        'time_specificity': check_time_specificity(strategy_conditions)
    }
    
    # Score: lower is better
    complexity_score = (
        complexity_factors['num_conditions'] * 2 +
        complexity_factors['num_parameters'] * 3 +
        complexity_factors['decimal_precision'] * 5 +
        complexity_factors['time_specificity'] * 4
    )
    
    if complexity_score < 10:
        assessment = "Simple and robust"
    elif complexity_score < 20:
        assessment = "Moderately complex"
    else:
        assessment = "Too complex - overfitting risk"
    
    return {
        'score': complexity_score,
        'assessment': assessment,
        'factors': complexity_factors
    }
```

### 2. Economic Intuition Check
```python
def economic_intuition_check(strategy_logic):
    """Verify if the strategy makes economic sense"""
    intuition_questions = [
        "Why should this strategy work?",
        "What market inefficiency does it exploit?",
        "Why aren't other traders already using it?",
        "Would it work under different market conditions?",
        "Is it scalable with more capital?"
    ]
    
    # This function requires human input, but the framework helps
    return {
        'questions': intuition_questions,
        'reminder': "If you can't explain why it works, it's probably overfitting"
    }
```

### 3. Regime Testing
```python
def test_across_market_regimes(strategy, data):
    """Test across different market regimes"""
    
    # Identify regimes (simplified)
    market_returns = data['SPY'].pct_change() if 'SPY' in data else data.iloc[:, 0].pct_change()
    volatility = market_returns.rolling(20).std()
    
    # Classify periods
    regimes = {}
    
    # Bull/Bear markets
    rolling_returns = market_returns.rolling(60).sum()
    regimes['bull'] = data[rolling_returns > 0.1]  # >10% in 60 days
    regimes['bear'] = data[rolling_returns < -0.1]  # <-10% in 60 days
    
    # High/Low volatility
    vol_median = volatility.median()
    regimes['low_vol'] = data[volatility < vol_median]
    regimes['high_vol'] = data[volatility >= vol_median]
    
    # Test strategy in each regime
    regime_results = {}
    
    for regime_name, regime_data in regimes.items():
        if len(regime_data) > 100:  # Sufficient data
            result = backtest_strategy(strategy, regime_data)
            regime_results[regime_name] = result
    
    # Analyze consistency
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

## My Anti-Overfitting Checklist

```python
def overfitting_checklist(strategy, backtest_results, validation_results):
    """Complete anti-overfitting checklist"""
    
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
    
    # 1. Realistic metrics
    if (backtest_results['sharpe_ratio'] > 3 or 
        backtest_results['win_rate'] > 0.8 or 
        backtest_results['max_drawdown'] < 0.03):
        checklist['metrics_realistic'] = False
        issues.append("Unrealistically good metrics")
    
    # 2. Sufficient trades
    if backtest_results['total_trades'] < 100:
        checklist['sufficient_trades'] = False
        issues.append("Insufficient trades for statistical significance")
    
    # 3. Out-of-sample validation
    if validation_results['degradation_pct'] > 0.5:
        checklist['out_sample_validation'] = False
        issues.append("Significant degradation in out-of-sample")
    
    # 4. Complexity
    complexity = strategy_complexity_score(strategy.conditions)
    if complexity['score'] > 20:
        checklist['complexity_reasonable'] = False
        issues.append("Strategy is too complex")
    
    # Final score
    passed_checks = sum(checklist.values())
    total_checks = len(checklist)
    
    if passed_checks == total_checks:
        verdict = "STRATEGY APPROVED - Low overfitting risk"
    elif passed_checks >= total_checks * 0.8:
        verdict = "PROCEED WITH CAUTION - Some concerns"
    else:
        verdict = "HIGH OVERFITTING RISK - Needs rework"
    
    return {
        'checklist': checklist,
        'issues': issues,
        'score': f"{passed_checks}/{total_checks}",
        'verdict': verdict
    }
```

## Next Steps After the Backtest

```python
def post_backtest_roadmap(validation_results):
    """Roadmap after validating strategy"""
    
    if validation_results['verdict'] == "PASSED":
        return {
            'next_step': 'Paper Trading',
            'duration': '2-3 months',
            'success_criteria': 'Correlation >70% with backtest',
            'position_size': 'Start with 25% of planned size'
        }
    elif "WARNING" in validation_results['verdict']:
        return {
            'next_step': 'Refinement',
            'actions': [
                'Simplify strategy',
                'Extend test period',
                'Test in more market regimes'
            ]
        }
    else:
        return {
            'next_step': 'Back to Drawing Board',
            'actions': [
                'Review fundamental logic',
                'Look for a new edge',
                'Start from scratch'
            ]
        }
```

## Next Step

With backtesting mastered, let's move on to [Risk Management](../risk/position_sizing.md) to protect your capital.
