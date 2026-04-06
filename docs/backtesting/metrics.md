> 🇪🇸 [Leer en Español](metrics.es.md) | 🇺🇸 **English**

# Key Backtesting Metrics

## The Numbers That Really Matter

You can have a backtest that shows +500% return, but if you don't understand the right metrics, you're seeing mirages. These are the metrics I use to evaluate whether a strategy is real or fantasy.

## Profitability Metrics

### 1. CAGR (Compound Annual Growth Rate)

**Definition**: Measures the annualized growth rate of an investment over a specific time period.

**Mathematical Formula**:
```
CAGR = (Final_Value / Initial_Value)^(1/n) - 1
```
Where n = number of years (total_days / 252)

```python
def CAGR(datos: pd.DataFrame, calculo_optimizado: bool = True, columna: str = "Close") -> float:
    """
    Compound Annual Growth Rate - Optimized reference implementation
    
    Parameters
    ----------
    datos : pd.DataFrame
        Historical data of a financial asset
    calculo_optimizado : bool, default True
        Whether to use direct method (True) or returns-based method (False)
    columna : str, default "Close"
        Column to use for calculation
    
    Returns
    -------
    float
        Annualized growth rate
    """
    # Calculate years
    n = np.ceil(datos.shape[0] / 252)
    
    if calculo_optimizado:
        # Direct method (more efficient)
        valor_inicial = datos[columna].iloc[0]
        valor_final = datos[columna].iloc[-1]
        return (valor_final / valor_inicial) ** (1 / n) - 1
    else:
        # Method using daily returns
        retornos_diarios = datos[columna].pct_change()
        retornos_acumulados = (1 + retornos_diarios).cumprod()
        return retornos_acumulados.iloc[-1] ** (1 / n) - 1

def calculate_returns(equity_curve):
    """Calculate different types of return"""
    start_value = equity_curve[0]
    end_value = equity_curve[-1]
    num_years = len(equity_curve) / 252  # Assuming daily data
    
    # Total return
    total_return = (end_value - start_value) / start_value
    
    # CAGR using reference formula
    cagr = (end_value / start_value) ** (1/num_years) - 1
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'absolute_profit': end_value - start_value,
        'years': num_years
    }
```

### 2. Benchmark Comparison
```python
def compare_to_benchmark(strategy_returns, benchmark_returns):
    """Compare against benchmark (SPY)"""
    strategy_cumret = (1 + strategy_returns).cumprod()
    benchmark_cumret = (1 + benchmark_returns).cumprod()
    
    # Alpha (excess return)
    alpha = strategy_cumret.iloc[-1] - benchmark_cumret.iloc[-1]
    
    # Beta (correlation with market)
    correlation = strategy_returns.corr(benchmark_returns)
    beta = strategy_returns.cov(benchmark_returns) / benchmark_returns.var()
    
    return {
        'alpha': alpha,
        'beta': beta,
        'correlation': correlation,
        'outperformed': alpha > 0
    }
```

## Risk Metrics

### 1. Sharpe Ratio - The King of Metrics

**Definition**: Measures the return an investment offers per unit of risk taken.

**Mathematical Formula**:
```
Sharpe = (Asset_Return - Risk_Free_Rate) / Annualized_Standard_Deviation
```

**Interpretation**:
- **> 0**: Return exceeds the risk-free rate
- **< 0**: Better to invest in risk-free assets

```python
def coef_sharpe(datos: pd.DataFrame, tasa_lr: float = 0.03, columna: str = "Close") -> float:
    """
    Sharpe Coefficient - Exact reference implementation
    
    Parameters
    ----------
    datos : pd.DataFrame
        Historical data of a financial asset
    tasa_lr : float, default 0.03
        Risk-free rate (3% by default)
    columna : str, default "Close"
        Column to use for calculation
    
    Returns
    -------
    float
        Sharpe Coefficient
    """
    # Calculate annualized asset return
    retorno_activo = (datos[columna].iloc[-1] / datos[columna].iloc[0]) ** (1 / np.ceil(datos.shape[0] / 252)) - 1
    
    # Annualized standard deviation
    desviacion_estandar_anualizada = datos[columna].pct_change().std() * np.sqrt(252)
    
    return (retorno_activo - tasa_lr) / desviacion_estandar_anualizada

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Sharpe Ratio: Return per unit of risk - Modern version"""
    excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    sharpe = excess_returns / volatility if volatility > 0 else 0
    
    # Enhanced interpretation
    if sharpe > 2:
        quality = "Excellent"
        interpretation = "Exceptional strategy - check for overfitting"
    elif sharpe > 1:
        quality = "Good"
        interpretation = "Solid strategy with good risk-return tradeoff"
    elif sharpe > 0.5:
        quality = "Acceptable"
        interpretation = "Viable strategy but improvable"
    elif sharpe > 0:
        quality = "Poor"
        interpretation = "Barely beats the risk-free rate"
    else:
        quality = "Negative"
        interpretation = "Losing money - better to invest in bonds"
    
    return {
        'sharpe_ratio': sharpe,
        'quality': quality,
        'interpretation': interpretation,
        'excess_return': excess_returns,
        'volatility': volatility
    }
```

### 2. Maximum Drawdown - Your Worst Nightmare

**Definition**: Measures the worst loss suffered by an investment from a historical peak, allowing you to evaluate downside risk.

**Mathematical Formula**:
```
Drawdown = (Highest_Cumulative_Return - Current_Return) / Highest_Cumulative_Return
Max_Drawdown = MAX(Drawdown_Series)
```

```python
def max_dd(datos: pd.DataFrame, columna: str = "Close") -> float:
    """
    Maximum Drawdown - Reference implementation
    
    Parameters
    ----------
    datos : pd.DataFrame
        Historical data of a financial instrument
    columna : str, default "Close"
        Column to use for calculation
    
    Returns
    -------
    float
        Maximum drawdown (as decimal, e.g.: 0.15 = 15%)
    """
    # Calculate daily returns
    rendimientos_diarios = datos[columna].pct_change()
    
    # Cumulative returns
    rendimientos_acumulados = (1 + rendimientos_diarios).cumprod()
    
    # Highest cumulative return up to each point
    mayor_rendimiento_acumulado = rendimientos_acumulados.cummax()
    
    # Difference between the maximum and current value
    diferencia = mayor_rendimiento_acumulado - rendimientos_acumulados
    
    # Convert to percentage
    diferencia_porcentaje = diferencia / mayor_rendimiento_acumulado
    
    # Maximum drawdown
    retroceso_maximo = diferencia_porcentaje.max()
    
    return retroceso_maximo

def calculate_drawdown(equity_curve):
    """Drawdown: Your worst loss from the peak - Extended version"""
    equity_series = pd.Series(equity_curve)
    
    # Running maximum
    peak = equity_series.cummax()
    
    # Drawdown at each point
    drawdown = (equity_series - peak) / peak
    
    # Maximum drawdown
    max_drawdown = drawdown.min()
    
    # Drawdown duration
    drawdown_duration = []
    in_drawdown = False
    start_dd = None
    
    for i, dd in enumerate(drawdown):
        if dd < 0 and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            start_dd = i
        elif dd == 0 and in_drawdown:
            # End of drawdown
            in_drawdown = False
            drawdown_duration.append(i - start_dd)
    
    max_dd_duration = max(drawdown_duration) if drawdown_duration else 0
    
    # Drawdown interpretation
    dd_abs = abs(max_drawdown)
    if dd_abs < 0.05:
        risk_level = "Very Low (Too good to be true?)"
    elif dd_abs < 0.10:
        risk_level = "Low"
    elif dd_abs < 0.20:
        risk_level = "Moderate"
    elif dd_abs < 0.30:
        risk_level = "High"
    else:
        risk_level = "Very High"
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'max_drawdown_duration': max_dd_duration,
        'drawdown_series': drawdown,
        'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0,
        'risk_level': risk_level,
        'peak_values': peak
    }
```

### 3. Calmar Ratio
```python
def calculate_calmar_ratio(returns, equity_curve):
    """Calmar: CAGR / Max Drawdown"""
    cagr = calculate_returns(equity_curve)['cagr']
    max_dd = abs(calculate_drawdown(equity_curve)['max_drawdown'])
    
    calmar = cagr / max_dd if max_dd > 0 else 0
    
    return {
        'calmar_ratio': calmar,
        'interpretation': 'Excellent' if calmar > 1 else 'Good' if calmar > 0.5 else 'Poor'
    }
```

## Trading Metrics

### 1. Win Rate and Profit Factor
```python
def calculate_trade_metrics(trades_df):
    """Trade-specific metrics"""
    if trades_df.empty:
        return {'error': 'No trades to analyze'}
    
    # Win rate
    winning_trades = (trades_df['pnl'] > 0).sum()
    total_trades = len(trades_df)
    win_rate = winning_trades / total_trades
    
    # Average win/loss
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] < 0]['pnl']
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    # Profit factor
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Largest win/loss
    largest_win = wins.max() if len(wins) > 0 else 0
    largest_loss = losses.min() if len(losses) > 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }
```

### 2. Consecutive Wins/Losses
```python
def analyze_streaks(trades_df):
    """Analyze winning and losing streaks"""
    if trades_df.empty:
        return {}
    
    # Create wins/losses series
    wins_losses = (trades_df['pnl'] > 0).astype(int)
    
    # Calculate streaks
    streaks = []
    current_streak = 1
    current_type = wins_losses.iloc[0]
    
    for i in range(1, len(wins_losses)):
        if wins_losses.iloc[i] == current_type:
            current_streak += 1
        else:
            streaks.append({
                'type': 'win' if current_type else 'loss',
                'length': current_streak
            })
            current_streak = 1
            current_type = wins_losses.iloc[i]
    
    # Last streak
    streaks.append({
        'type': 'win' if current_type else 'loss',
        'length': current_streak
    })
    
    # Statistics
    win_streaks = [s['length'] for s in streaks if s['type'] == 'win']
    loss_streaks = [s['length'] for s in streaks if s['type'] == 'loss']
    
    return {
        'max_consecutive_wins': max(win_streaks) if win_streaks else 0,
        'max_consecutive_losses': max(loss_streaks) if loss_streaks else 0,
        'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
        'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
        'all_streaks': streaks
    }
```

## Consistency Metrics

### 1. Monthly Returns Analysis
```python
def analyze_monthly_returns(equity_curve, timestamps):
    """Analyze monthly returns"""
    equity_df = pd.DataFrame({
        'timestamp': timestamps,
        'equity': equity_curve
    })
    equity_df.set_index('timestamp', inplace=True)
    
    # Monthly returns
    monthly_equity = equity_df.resample('M').last()
    monthly_returns = monthly_equity['equity'].pct_change().dropna()
    
    # Metrics
    positive_months = (monthly_returns > 0).sum()
    total_months = len(monthly_returns)
    monthly_win_rate = positive_months / total_months
    
    # Best and worst month
    best_month = monthly_returns.max()
    worst_month = monthly_returns.min()
    
    # Consistency (std of monthly returns)
    consistency = monthly_returns.std()
    
    return {
        'monthly_win_rate': monthly_win_rate,
        'best_month': best_month,
        'worst_month': worst_month,
        'avg_monthly_return': monthly_returns.mean(),
        'monthly_consistency': consistency,
        'total_months': total_months,
        'positive_months': positive_months,
        'monthly_returns': monthly_returns
    }
```

### 2. Rolling Performance
```python
def rolling_performance(returns, window=252):
    """Performance in rolling windows"""
    rolling_sharpe = []
    rolling_returns = []
    
    for i in range(window, len(returns)):
        period_returns = returns[i-window:i]
        
        # Rolling Sharpe
        sharpe = calculate_sharpe_ratio(period_returns)['sharpe_ratio']
        rolling_sharpe.append(sharpe)
        
        # Rolling annual return
        annual_return = period_returns.mean() * 252
        rolling_returns.append(annual_return)
    
    return {
        'rolling_sharpe': rolling_sharpe,
        'rolling_returns': rolling_returns,
        'sharpe_stability': np.std(rolling_sharpe),
        'return_stability': np.std(rolling_returns)
    }
```

## Advanced Metrics

### 1. Value at Risk (VaR)
```python
def calculate_var(returns, confidence_level=0.05):
    """Value at Risk: Maximum expected loss"""
    # Historical VaR
    var_historical = np.percentile(returns, confidence_level * 100)
    
    # Parametric VaR (assuming normal distribution)
    mean_return = returns.mean()
    std_return = returns.std()
    var_parametric = mean_return - (1.96 * std_return)  # 95% confidence
    
    # Expected Shortfall (CVaR)
    shortfall_returns = returns[returns <= var_historical]
    expected_shortfall = shortfall_returns.mean() if len(shortfall_returns) > 0 else 0
    
    return {
        'var_historical': var_historical,
        'var_parametric': var_parametric,
        'expected_shortfall': expected_shortfall,
        'confidence_level': confidence_level
    }
```

### 2. Sortino Ratio - Penalizing Only Losses

**Definition**: Measures risk-adjusted return considering only downside volatility, making it more sensitive to losses than the Sharpe Ratio.

**Mathematical Formula**:
```
Sortino = (Asset_Return - Risk_Free_Rate) / Negative_Standard_Deviation
```

**Advantage over Sharpe**: Only penalizes downside volatility (unwanted losses), not gains.

```python
def coef_Sortino(datos: pd.DataFrame, tasa_lr: float = 0.03, columna: str = "Close") -> float:
    """
    Sortino Coefficient - Exact reference implementation
    
    Parameters
    ----------
    datos : pd.DataFrame
        Historical data of a financial asset
    tasa_lr : float, default 0.03
        Risk-free rate (3% by default)
    columna : str, default "Close"
        Column to use for calculation
    
    Returns
    -------
    float
        Sortino Coefficient
    """
    # Calculate annualized asset return
    rendimiento_activo = (datos[columna].iloc[-1] / datos[columna].iloc[0]) ** (1 / np.ceil(datos.shape[0] / 252)) - 1
    
    # Daily returns
    rendimientos_diarios = datos[columna].pct_change()
    
    # Only negative returns
    rendimientos_diarios_negativos = rendimientos_diarios[rendimientos_diarios < 0]
    
    # Annualized standard deviation of negative returns
    desviacion_estandar_negativos = rendimientos_diarios_negativos.std() * np.sqrt(252)
    
    return (rendimiento_activo - tasa_lr) / desviacion_estandar_negativos

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Sortino: Like Sharpe but only penalizes downside - Modern version"""
    excess_returns = returns.mean() * 252 - risk_free_rate
    
    # Downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252)
    
    sortino = excess_returns / downside_deviation if downside_deviation > 0 else 0
    
    # Interpretation
    if sortino > 2:
        quality = "Excellent"
        interpretation = "Very good loss risk management"
    elif sortino > 1:
        quality = "Good"
        interpretation = "Good downside risk management"
    elif sortino > 0.5:
        quality = "Acceptable"
        interpretation = "Moderate loss risk management"
    elif sortino > 0:
        quality = "Poor"
        interpretation = "Positive return but poor loss control"
    else:
        quality = "Negative"
        interpretation = "Negative return with significant losses"
    
    return {
        'sortino_ratio': sortino,
        'quality': quality,
        'interpretation': interpretation,
        'downside_deviation': downside_deviation,
        'excess_return': excess_returns,
        'negative_periods': len(negative_returns)
    }
```

## Benchmarking Framework

```python
class PerformanceAnalyzer:
    """Complete framework for performance analysis"""
    
    def __init__(self, equity_curve, returns, trades_df=None, benchmark_returns=None):
        self.equity_curve = equity_curve
        self.returns = returns
        self.trades_df = trades_df
        self.benchmark_returns = benchmark_returns
        
    def full_analysis(self):
        """Complete analysis"""
        analysis = {}
        
        # Basic metrics
        analysis['returns'] = calculate_returns(self.equity_curve)
        analysis['sharpe'] = calculate_sharpe_ratio(self.returns)
        analysis['drawdown'] = calculate_drawdown(self.equity_curve)
        analysis['calmar'] = calculate_calmar_ratio(self.returns, self.equity_curve)
        
        # Trading metrics
        if self.trades_df is not None:
            analysis['trades'] = calculate_trade_metrics(self.trades_df)
            analysis['streaks'] = analyze_streaks(self.trades_df)
        
        # Advanced metrics
        analysis['var'] = calculate_var(self.returns)
        analysis['sortino'] = calculate_sortino_ratio(self.returns)
        
        # Benchmark comparison
        if self.benchmark_returns is not None:
            analysis['vs_benchmark'] = compare_to_benchmark(
                self.returns, self.benchmark_returns
            )
        
        # Consistency
        timestamps = pd.date_range(start='2023-01-01', periods=len(self.equity_curve), freq='D')
        analysis['monthly'] = analyze_monthly_returns(self.equity_curve, timestamps)
        
        return analysis
    
    def generate_report(self):
        """Generate readable report"""
        analysis = self.full_analysis()
        
        report = f"""
BACKTEST PERFORMANCE REPORT
{'='*50}

PROFITABILITY
Total Return: {analysis['returns']['total_return']:.2%}
CAGR: {analysis['returns']['cagr']:.2%}
Profit: ${analysis['returns']['absolute_profit']:,.2f}

RISK
Sharpe Ratio: {analysis['sharpe']['sharpe_ratio']:.2f} ({analysis['sharpe']['quality']})
Max Drawdown: {analysis['drawdown']['max_drawdown']:.2%}
Calmar Ratio: {analysis['calmar']['calmar_ratio']:.2f}
Volatility: {analysis['sharpe']['volatility']:.2%}

TRADING METRICS
"""
        
        if 'trades' in analysis:
            trades = analysis['trades']
            report += f"""Total Trades: {trades['total_trades']}
Win Rate: {trades['win_rate']:.2%}
Profit Factor: {trades['profit_factor']:.2f}
Expectancy: ${trades['expectancy']:.2f}
Avg Win: ${trades['avg_win']:.2f}
Avg Loss: ${trades['avg_loss']:.2f}

STREAKS
Max Consecutive Wins: {analysis['streaks']['max_consecutive_wins']}
Max Consecutive Losses: {analysis['streaks']['max_consecutive_losses']}
"""
        
        report += f"""
ADVANCED METRICS
Sortino Ratio: {analysis['sortino']['sortino_ratio']:.2f}
VaR (95%): {analysis['var']['var_historical']:.2%}
Expected Shortfall: {analysis['var']['expected_shortfall']:.2%}

CONSISTENCY
Monthly Win Rate: {analysis['monthly']['monthly_win_rate']:.2%}
Best Month: {analysis['monthly']['best_month']:.2%}
Worst Month: {analysis['monthly']['worst_month']:.2%}
"""
        
        return report
```

## Red Flags in Metrics

```python
def identify_red_flags(analysis):
    """Identify warning signals in metrics"""
    red_flags = []
    
    # Returns too good to be true
    if analysis['returns']['cagr'] > 1.0:  # >100% CAGR
        red_flags.append("CAGR too high - possible overfitting")
    
    # Win rate too high
    if 'trades' in analysis and analysis['trades']['win_rate'] > 0.8:
        red_flags.append("Win rate too high - check for look-ahead bias")
    
    # Drawdown too low
    if abs(analysis['drawdown']['max_drawdown']) < 0.05:
        red_flags.append("Drawdown too low - not realistic")
    
    # Too few trades
    if 'trades' in analysis and analysis['trades']['total_trades'] < 30:
        red_flags.append("Too few trades - lacks statistical significance")
    
    # Profit factor too high
    if 'trades' in analysis and analysis['trades']['profit_factor'] > 3:
        red_flags.append("Profit factor too high - possible curve fitting")
    
    # Sharpe too high
    if analysis['sharpe']['sharpe_ratio'] > 3:
        red_flags.append("Sharpe ratio too high - check data quality")
    
    return red_flags

def validate_backtest(analysis):
    """Complete backtest validation"""
    red_flags = identify_red_flags(analysis)
    
    # Overall score
    score = 0
    max_score = 100
    
    # Sharpe contribution (30 points max)
    sharpe = analysis['sharpe']['sharpe_ratio']
    if sharpe > 2:
        score += 30
    elif sharpe > 1:
        score += 20
    elif sharpe > 0.5:
        score += 10
    
    # Drawdown contribution (20 points max)
    dd = abs(analysis['drawdown']['max_drawdown'])
    if dd < 0.1:
        score += 20
    elif dd < 0.2:
        score += 15
    elif dd < 0.3:
        score += 10
    
    # Consistency (30 points max)
    if 'monthly' in analysis:
        monthly_wr = analysis['monthly']['monthly_win_rate']
        if monthly_wr > 0.7:
            score += 30
        elif monthly_wr > 0.6:
            score += 20
        elif monthly_wr > 0.5:
            score += 10
    
    # Trade stats (20 points max)
    if 'trades' in analysis:
        if analysis['trades']['total_trades'] > 100:
            score += 10
        if 1.5 <= analysis['trades']['profit_factor'] <= 2.5:
            score += 10
    
    recommendation = "APPROVED" if score >= 70 and not red_flags else "NEEDS WORK"
    
    return {
        'score': score,
        'max_score': max_score,
        'red_flags': red_flags,
        'recommendation': recommendation
    }
```

## My Personal Dashboard

```python
def create_metrics_dashboard(analysis):
    """Visual metrics dashboard"""
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Equity curve
    ax1.plot(analysis['equity_curve'])
    ax1.set_title('Equity Curve')
    ax1.grid(True)
    
    # 2. Drawdown
    dd = analysis['drawdown']['drawdown_series']
    ax2.fill_between(range(len(dd)), dd, 0, alpha=0.3, color='red')
    ax2.set_title(f'Drawdown (Max: {analysis["drawdown"]["max_drawdown"]:.2%})')
    ax2.grid(True)
    
    # 3. Monthly returns
    if 'monthly' in analysis:
        monthly_rets = analysis['monthly']['monthly_returns']
        colors = ['green' if x > 0 else 'red' for x in monthly_rets]
        ax3.bar(range(len(monthly_rets)), monthly_rets, color=colors, alpha=0.7)
        ax3.set_title(f'Monthly Returns (WR: {analysis["monthly"]["monthly_win_rate"]:.1%})')
        ax3.grid(True)
    
    # 4. Key metrics
    metrics_text = f"""
Sharpe: {analysis['sharpe']['sharpe_ratio']:.2f}
Calmar: {analysis['calmar']['calmar_ratio']:.2f}
CAGR: {analysis['returns']['cagr']:.1%}
Max DD: {analysis['drawdown']['max_drawdown']:.1%}
"""
    if 'trades' in analysis:
        metrics_text += f"""
Win Rate: {analysis['trades']['win_rate']:.1%}
Profit Factor: {analysis['trades']['profit_factor']:.2f}
Total Trades: {analysis['trades']['total_trades']}
"""
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Key Metrics')
    
    plt.tight_layout()
    plt.show()
```

## Practical Example: Evaluating a Strategy

```python
import pandas as pd
import numpy as np

# Example using real data
def evaluate_strategy_example():
    """Complete strategy evaluation example"""
    
    # Simulate equity curve data for a strategy
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Simulate returns with slight positive drift
    daily_returns = np.random.normal(0.0008, 0.02, len(dates))  # 0.08% daily return, 2% vol
    equity_curve = 100000 * (1 + daily_returns).cumprod()
    
    # Create simulated DataFrame
    strategy_data = pd.DataFrame({
        'Close': equity_curve
    }, index=dates)
    
    print("=== COMPLETE STRATEGY EVALUATION ===\n")
    
    # 1. CAGR using both methods
    cagr_optimized = CAGR(strategy_data, calculo_optimizado=True)
    cagr_returns = CAGR(strategy_data, calculo_optimizado=False)
    
    print(f"CAGR (Optimized Method): {cagr_optimized:.2%}")
    print(f"CAGR (Returns Method): {cagr_returns:.2%}")
    
    # 2. Sharpe Ratio
    daily_rets = strategy_data['Close'].pct_change().dropna()
    sharpe_result = coef_sharpe(strategy_data, tasa_lr=0.03)
    
    print(f"\nSharpe Coefficient: {sharpe_result:.2f}")
    if sharpe_result > 0:
        print("   Return exceeds the risk-free rate")
    else:
        print("   Better to invest in risk-free assets")
    
    # 3. Sortino Ratio
    sortino_result = coef_Sortino(strategy_data, tasa_lr=0.03)
    
    print(f"\nSortino Coefficient: {sortino_result:.2f}")
    print("   (Only penalizes downside volatility)")
    
    # 4. Maximum Drawdown
    max_drawdown = max_dd(strategy_data)
    
    print(f"\nMaximum Drawdown: {max_drawdown:.2%}")
    print(f"   Maximum loss from peak: ${100000 * max_drawdown:,.2f}")
    
    # 5. Complete analysis using modern framework
    analyzer = PerformanceAnalyzer(
        equity_curve=equity_curve.values,
        returns=daily_rets
    )
    
    print("\n" + "="*50)
    print(analyzer.generate_report())
    
    # 6. Backtest validation
    analysis = analyzer.full_analysis()
    validation = validate_backtest(analysis)
    
    print(f"\nFINAL SCORE: {validation['score']}/{validation['max_score']}")
    print(f"RECOMMENDATION: {validation['recommendation']}")
    
    if validation['red_flags']:
        print("\nRED FLAGS DETECTED:")
        for flag in validation['red_flags']:
            print(f"   {flag}")

# Run example
if __name__ == "__main__":
    evaluate_strategy_example()
```

## Metrics Best Practices

### Do's

1. **Use multiple metrics**: Never rely on a single metric
2. **Compare with benchmark**: Always evaluate vs SPY or relevant index
3. **Analyze drawdown**: A strategy with 50% DD is not viable
4. **Validate statistically**: Minimum 30-50 trades for significance
5. **Consider period consistency**: Metrics should be stable across periods

### Don'ts

1. **Don't ignore transaction costs**: Include commissions and slippage
2. **Don't optimize only Sharpe**: Can lead to overfitting
3. **Don't use future data**: Avoid look-ahead bias
4. **Don't ignore red flags**: "Perfect" metrics are suspicious
5. **Don't trade without out-of-sample**: Always reserve data for validation

### Realistic Targets for Small Caps

```python
# Realistic benchmarks for small cap strategies
REALISTIC_METRICS = {
    'sharpe_ratio': {
        'excellent': '>1.5',
        'good': '1.0-1.5',
        'acceptable': '0.7-1.0',
        'poor': '<0.7'
    },
    'max_drawdown': {
        'excellent': '<15%',
        'good': '15-25%',
        'acceptable': '25-35%',
        'poor': '>35%'
    },
    'cagr': {
        'excellent': '>25%',
        'good': '15-25%',
        'acceptable': '10-15%',
        'poor': '<10%'
    },
    'win_rate': {
        'excellent': '>60%',
        'good': '50-60%',
        'acceptable': '45-50%',
        'poor': '<45%'
    }
}
```

### Validation Checklist

Before going live with a strategy:

- [ ] Sharpe > 1.0
- [ ] Max Drawdown < 30%
- [ ] Minimum 50 trades in backtest
- [ ] Profit Factor between 1.3-2.5
- [ ] Win Rate 45-70% (not extreme)
- [ ] Consistency in rolling windows
- [ ] Out-of-sample testing passed
- [ ] No red flags in metrics

## Next Step

With metrics mastered and reference formulas integrated, let's move on to [How to Avoid Overfitting](overfitting.md) to ensure your results are real.
