# Métricas Clave de Backtesting

## Los Números Que Realmente Importan

Puedes tener un backtest que muestre +500% de retorno, pero si no entiendes las métricas correctas, estás viendo espejismos. Estas son las métricas que uso para evaluar si una estrategia es real o fantasía.

## Métricas de Rentabilidad

### 1. CAGR (Compound Annual Growth Rate)

**Definición**: Mide la tasa de crecimiento anualizada de una inversión durante un periodo de tiempo específico.

**Fórmula Matemática**:
```
CAGR = (Valor_Final / Valor_Inicial)^(1/n) - 1
```
Donde n = número de años (días_totales / 252)

```python
def CAGR(datos: pd.DataFrame, calculo_optimizado: bool = True, columna: str = "Close") -> float:
    """
    Tasa de Crecimiento Anual Compuesta - Implementación de referencia optimizada
    
    Parámetros
    ----------
    datos : pd.DataFrame
        Datos históricos de un activo financiero
    calculo_optimizado : bool, default True
        Si usar método directo (True) o basado en retornos (False)
    columna : str, default "Close"
        Columna a utilizar para el cálculo
    
    Returns
    -------
    float
        Tasa de crecimiento anualizada
    """
    # Calcular años
    n = np.ceil(datos.shape[0] / 252)
    
    if calculo_optimizado:
        # Método directo (más eficiente)
        valor_inicial = datos[columna].iloc[0]
        valor_final = datos[columna].iloc[-1]
        return (valor_final / valor_inicial) ** (1 / n) - 1
    else:
        # Método usando retornos diarios
        retornos_diarios = datos[columna].pct_change()
        retornos_acumulados = (1 + retornos_diarios).cumprod()
        return retornos_acumulados.iloc[-1] ** (1 / n) - 1

def calculate_returns(equity_curve):
    """Calcular diferentes tipos de retorno"""
    start_value = equity_curve[0]
    end_value = equity_curve[-1]
    num_years = len(equity_curve) / 252  # Assuming daily data
    
    # Total return
    total_return = (end_value - start_value) / start_value
    
    # CAGR usando fórmula de referencia
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
    """Comparar contra benchmark (SPY)"""
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

## Métricas de Riesgo

### 1. Sharpe Ratio - El Rey de las Métricas

**Definición**: Mide la rentabilidad que ofrece una inversión por cada unidad de riesgo que se asume.

**Fórmula Matemática**:
```
Sharpe = (Retorno_Activo - Tasa_Libre_Riesgo) / Desviación_Estándar_Anualizada
```

**Interpretación**:
- **> 0**: Rendimiento superior a la tasa libre de riesgo
- **< 0**: Conviene más invertir en activos libres de riesgo

```python
def coef_sharpe(datos: pd.DataFrame, tasa_lr: float = 0.03, columna: str = "Close") -> float:
    """
    Coeficiente de Sharpe - Implementación de referencia exacta
    
    Parámetros
    ----------
    datos : pd.DataFrame
        Datos históricos de un activo financiero
    tasa_lr : float, default 0.03
        Tasa libre de riesgo (3% por defecto)
    columna : str, default "Close"
        Columna que usaremos para realizar el cálculo
    
    Returns
    -------
    float
        Coeficiente de Sharpe
    """
    # Calcular retorno anualizado del activo
    retorno_activo = (datos[columna].iloc[-1] / datos[columna].iloc[0]) ** (1 / np.ceil(datos.shape[0] / 252)) - 1
    
    # Desviación estándar anualizada
    desviacion_estandar_anualizada = datos[columna].pct_change().std() * np.sqrt(252)
    
    return (retorno_activo - tasa_lr) / desviacion_estandar_anualizada

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Sharpe Ratio: Return por unidad de riesgo - Versión moderna"""
    excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    sharpe = excess_returns / volatility if volatility > 0 else 0
    
    # Interpretación mejorada
    if sharpe > 2:
        quality = "Excelente"
        interpretation = "Estrategia excepcional - revisar por overfitting"
    elif sharpe > 1:
        quality = "Buena"
        interpretation = "Estrategia sólida con buen ajuste riesgo-retorno"
    elif sharpe > 0.5:
        quality = "Aceptable"
        interpretation = "Estrategia viable pero mejorable"
    elif sharpe > 0:
        quality = "Pobre"
        interpretation = "Apenas supera tasa libre de riesgo"
    else:
        quality = "Negativo"
        interpretation = "Pérdidas - mejor invertir en bonos"
    
    return {
        'sharpe_ratio': sharpe,
        'quality': quality,
        'interpretation': interpretation,
        'excess_return': excess_returns,
        'volatility': volatility
    }
```

### 2. Maximum Drawdown - Tu Peor Pesadilla

**Definición**: Mide la peor pérdida sufrida por una inversión desde un máximo histórico, permitiendo evaluar el riesgo de pérdida.

**Fórmula Matemática**:
```
Drawdown = (Mayor_Rendimiento_Acumulado - Rendimiento_Actual) / Mayor_Rendimiento_Acumulado
Max_Drawdown = MAX(Drawdown_Series)
```

```python
def max_dd(datos: pd.DataFrame, columna: str = "Close") -> float:
    """
    Máxima Reducción (Maximum Drawdown) - Implementación de referencia
    
    Parámetros
    ----------
    datos : pd.DataFrame
        Datos históricos de un instrumento financiero
    columna : str, default "Close"
        Columna a utilizar para realizar el cálculo
    
    Returns
    -------
    float
        Máxima reducción (como decimal, ej: 0.15 = 15%)
    """
    # Calcular rendimientos diarios
    rendimientos_diarios = datos[columna].pct_change()
    
    # Rendimientos acumulados
    rendimientos_acumulados = (1 + rendimientos_diarios).cumprod()
    
    # Mayor rendimiento acumulado hasta cada punto
    mayor_rendimiento_acumulado = rendimientos_acumulados.cummax()
    
    # Diferencia entre el máximo y el valor actual
    diferencia = mayor_rendimiento_acumulado - rendimientos_acumulados
    
    # Convertir a porcentaje
    diferencia_porcentaje = diferencia / mayor_rendimiento_acumulado
    
    # Retroceso máximo
    retroceso_maximo = diferencia_porcentaje.max()
    
    return retroceso_maximo

def calculate_drawdown(equity_curve):
    """Drawdown: Tu peor pérdida desde el peak - Versión extendida"""
    equity_series = pd.Series(equity_curve)
    
    # Running maximum
    peak = equity_series.cummax()
    
    # Drawdown en cada punto
    drawdown = (equity_series - peak) / peak
    
    # Máximo drawdown
    max_drawdown = drawdown.min()
    
    # Duración del drawdown
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
    
    # Interpretación del drawdown
    dd_abs = abs(max_drawdown)
    if dd_abs < 0.05:
        risk_level = "Muy Bajo (¿Demasiado bueno?)"
    elif dd_abs < 0.10:
        risk_level = "Bajo"
    elif dd_abs < 0.20:
        risk_level = "Moderado"
    elif dd_abs < 0.30:
        risk_level = "Alto"
    else:
        risk_level = "Muy Alto"
    
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
        'interpretation': 'Excelente' if calmar > 1 else 'Buena' if calmar > 0.5 else 'Pobre'
    }
```

## Métricas de Trading

### 1. Win Rate y Profit Factor
```python
def calculate_trade_metrics(trades_df):
    """Métricas específicas de trades"""
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
    """Analizar rachas ganadoras y perdedoras"""
    if trades_df.empty:
        return {}
    
    # Crear serie de wins/losses
    wins_losses = (trades_df['pnl'] > 0).astype(int)
    
    # Calcular streaks
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
    
    # Último streak
    streaks.append({
        'type': 'win' if current_type else 'loss',
        'length': current_streak
    })
    
    # Estadísticas
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

## Métricas de Consistencia

### 1. Monthly Returns Analysis
```python
def analyze_monthly_returns(equity_curve, timestamps):
    """Analizar retornos mensuales"""
    equity_df = pd.DataFrame({
        'timestamp': timestamps,
        'equity': equity_curve
    })
    equity_df.set_index('timestamp', inplace=True)
    
    # Retornos mensuales
    monthly_equity = equity_df.resample('M').last()
    monthly_returns = monthly_equity['equity'].pct_change().dropna()
    
    # Métricas
    positive_months = (monthly_returns > 0).sum()
    total_months = len(monthly_returns)
    monthly_win_rate = positive_months / total_months
    
    # Mejor y peor mes
    best_month = monthly_returns.max()
    worst_month = monthly_returns.min()
    
    # Consistencia (std de retornos mensuales)
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
    """Performance en ventanas móviles"""
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

## Métricas Avanzadas

### 1. Value at Risk (VaR)
```python
def calculate_var(returns, confidence_level=0.05):
    """Value at Risk: Pérdida máxima esperada"""
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

### 2. Sortino Ratio - Penalizando Solo las Pérdidas

**Definición**: Mide el rendimiento ajustado al riesgo considerando solo la volatilidad negativa, siendo más sensible a las pérdidas que el Sharpe Ratio.

**Fórmula Matemática**:
```
Sortino = (Rendimiento_Activo - Tasa_Libre_Riesgo) / Desviación_Estándar_Negativos
```

**Ventaja sobre Sharpe**: Solo penaliza la volatilidad negativa (pérdidas no deseadas), no las ganancias.

```python
def coef_Sortino(datos: pd.DataFrame, tasa_lr: float = 0.03, columna: str = "Close") -> float:
    """
    Coeficiente de Sortino - Implementación de referencia exacta
    
    Parámetros
    ----------
    datos : pd.DataFrame
        Datos históricos de un activo financiero
    tasa_lr : float, default 0.03
        Tasa libre de riesgo (3% por defecto)
    columna : str, default "Close"
        Columna que se utilizará para realizar el cálculo
    
    Returns
    -------
    float
        Coeficiente de Sortino
    """
    # Calcular rendimiento anualizado del activo
    rendimiento_activo = (datos[columna].iloc[-1] / datos[columna].iloc[0]) ** (1 / np.ceil(datos.shape[0] / 252)) - 1
    
    # Rendimientos diarios
    rendimientos_diarios = datos[columna].pct_change()
    
    # Solo rendimientos negativos
    rendimientos_diarios_negativos = rendimientos_diarios[rendimientos_diarios < 0]
    
    # Desviación estándar de rendimientos negativos, anualizada
    desviacion_estandar_negativos = rendimientos_diarios_negativos.std() * np.sqrt(252)
    
    return (rendimiento_activo - tasa_lr) / desviacion_estandar_negativos

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Sortino: Como Sharpe pero solo penaliza downside - Versión moderna"""
    excess_returns = returns.mean() * 252 - risk_free_rate
    
    # Downside deviation (solo retornos negativos)
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252)
    
    sortino = excess_returns / downside_deviation if downside_deviation > 0 else 0
    
    # Interpretación
    if sortino > 2:
        quality = "Excelente"
        interpretation = "Muy buena gestión del riesgo de pérdidas"
    elif sortino > 1:
        quality = "Buena"
        interpretation = "Buena gestión del downside risk"
    elif sortino > 0.5:
        quality = "Aceptable"
        interpretation = "Gestión moderada del riesgo de pérdidas"
    elif sortino > 0:
        quality = "Pobre"
        interpretation = "Rendimiento superior pero mal control de pérdidas"
    else:
        quality = "Negativo"
        interpretation = "Rendimiento inferior con pérdidas significativas"
    
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
    """Framework completo para análisis de performance"""
    
    def __init__(self, equity_curve, returns, trades_df=None, benchmark_returns=None):
        self.equity_curve = equity_curve
        self.returns = returns
        self.trades_df = trades_df
        self.benchmark_returns = benchmark_returns
        
    def full_analysis(self):
        """Análisis completo"""
        analysis = {}
        
        # Métricas básicas
        analysis['returns'] = calculate_returns(self.equity_curve)
        analysis['sharpe'] = calculate_sharpe_ratio(self.returns)
        analysis['drawdown'] = calculate_drawdown(self.equity_curve)
        analysis['calmar'] = calculate_calmar_ratio(self.returns, self.equity_curve)
        
        # Métricas de trading
        if self.trades_df is not None:
            analysis['trades'] = calculate_trade_metrics(self.trades_df)
            analysis['streaks'] = analyze_streaks(self.trades_df)
        
        # Métricas avanzadas
        analysis['var'] = calculate_var(self.returns)
        analysis['sortino'] = calculate_sortino_ratio(self.returns)
        
        # Benchmark comparison
        if self.benchmark_returns is not None:
            analysis['vs_benchmark'] = compare_to_benchmark(
                self.returns, self.benchmark_returns
            )
        
        # Consistencia
        timestamps = pd.date_range(start='2023-01-01', periods=len(self.equity_curve), freq='D')
        analysis['monthly'] = analyze_monthly_returns(self.equity_curve, timestamps)
        
        return analysis
    
    def generate_report(self):
        """Generar reporte readable"""
        analysis = self.full_analysis()
        
        report = f"""
📊 BACKTEST PERFORMANCE REPORT
{'='*50}

💰 RENTABILIDAD
Total Return: {analysis['returns']['total_return']:.2%}
CAGR: {analysis['returns']['cagr']:.2%}
Profit: ${analysis['returns']['absolute_profit']:,.2f}

⚖️ RIESGO
Sharpe Ratio: {analysis['sharpe']['sharpe_ratio']:.2f} ({analysis['sharpe']['quality']})
Max Drawdown: {analysis['drawdown']['max_drawdown']:.2%}
Calmar Ratio: {analysis['calmar']['calmar_ratio']:.2f}
Volatility: {analysis['sharpe']['volatility']:.2%}

📈 TRADING METRICS
"""
        
        if 'trades' in analysis:
            trades = analysis['trades']
            report += f"""Total Trades: {trades['total_trades']}
Win Rate: {trades['win_rate']:.2%}
Profit Factor: {trades['profit_factor']:.2f}
Expectancy: ${trades['expectancy']:.2f}
Avg Win: ${trades['avg_win']:.2f}
Avg Loss: ${trades['avg_loss']:.2f}

🔥 STREAKS
Max Consecutive Wins: {analysis['streaks']['max_consecutive_wins']}
Max Consecutive Losses: {analysis['streaks']['max_consecutive_losses']}
"""
        
        report += f"""
📊 ADVANCED METRICS
Sortino Ratio: {analysis['sortino']['sortino_ratio']:.2f}
VaR (95%): {analysis['var']['var_historical']:.2%}
Expected Shortfall: {analysis['var']['expected_shortfall']:.2%}

📅 CONSISTENCIA
Monthly Win Rate: {analysis['monthly']['monthly_win_rate']:.2%}
Best Month: {analysis['monthly']['best_month']:.2%}
Worst Month: {analysis['monthly']['worst_month']:.2%}
"""
        
        return report
```

## Red Flags en Métricas

```python
def identify_red_flags(analysis):
    """Identificar señales de alerta en métricas"""
    red_flags = []
    
    # Returns too good to be true
    if analysis['returns']['cagr'] > 1.0:  # >100% CAGR
        red_flags.append("🚨 CAGR demasiado alto - posible overfitting")
    
    # Win rate too high
    if 'trades' in analysis and analysis['trades']['win_rate'] > 0.8:
        red_flags.append("🚨 Win rate demasiado alto - revisar look-ahead bias")
    
    # Drawdown too low
    if abs(analysis['drawdown']['max_drawdown']) < 0.05:
        red_flags.append("🚨 Drawdown demasiado bajo - no realista")
    
    # Too few trades
    if 'trades' in analysis and analysis['trades']['total_trades'] < 30:
        red_flags.append("⚠️ Muy pocos trades - falta significancia estadística")
    
    # Profit factor too high
    if 'trades' in analysis and analysis['trades']['profit_factor'] > 3:
        red_flags.append("🚨 Profit factor demasiado alto - posible curve fitting")
    
    # Sharpe too high
    if analysis['sharpe']['sharpe_ratio'] > 3:
        red_flags.append("🚨 Sharpe ratio demasiado alto - revisar data quality")
    
    return red_flags

def validate_backtest(analysis):
    """Validación completa del backtest"""
    red_flags = identify_red_flags(analysis)
    
    # Score general
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

## Mi Dashboard Personal

```python
def create_metrics_dashboard(analysis):
    """Dashboard visual de métricas"""
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
    
    # 4. Métricas key
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

## Ejemplo Práctico: Evaluando una Estrategia

```python
import pandas as pd
import numpy as np

# Ejemplo usando datos reales
def evaluate_strategy_example():
    """Ejemplo completo de evaluación de estrategia"""
    
    # Simular datos de equity curve de una estrategia
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Simular returns con ligero drift positivo
    daily_returns = np.random.normal(0.0008, 0.02, len(dates))  # 0.08% daily return, 2% vol
    equity_curve = 100000 * (1 + daily_returns).cumprod()
    
    # Crear DataFrame simulado
    strategy_data = pd.DataFrame({
        'Close': equity_curve
    }, index=dates)
    
    print("=== EVALUACIÓN COMPLETA DE ESTRATEGIA ===\n")
    
    # 1. CAGR usando ambos métodos
    cagr_optimized = CAGR(strategy_data, calculo_optimizado=True)
    cagr_returns = CAGR(strategy_data, calculo_optimizado=False)
    
    print(f"📈 CAGR (Método Optimizado): {cagr_optimized:.2%}")
    print(f"📈 CAGR (Método Retornos): {cagr_returns:.2%}")
    
    # 2. Sharpe Ratio
    daily_rets = strategy_data['Close'].pct_change().dropna()
    sharpe_result = coef_sharpe(strategy_data, tasa_lr=0.03)
    
    print(f"\n⚖️ Coeficiente de Sharpe: {sharpe_result:.2f}")
    if sharpe_result > 0:
        print("   ✅ Rendimiento superior a la tasa libre de riesgo")
    else:
        print("   ❌ Mejor invertir en activos libres de riesgo")
    
    # 3. Sortino Ratio
    sortino_result = coef_Sortino(strategy_data, tasa_lr=0.03)
    
    print(f"\n📉 Coeficiente de Sortino: {sortino_result:.2f}")
    print("   (Solo penaliza volatilidad negativa)")
    
    # 4. Maximum Drawdown
    max_drawdown = max_dd(strategy_data)
    
    print(f"\n💥 Máximo Drawdown: {max_drawdown:.2%}")
    print(f"   Pérdida máxima desde peak: ${100000 * max_drawdown:,.2f}")
    
    # 5. Análisis completo usando framework moderno
    analyzer = PerformanceAnalyzer(
        equity_curve=equity_curve.values,
        returns=daily_rets
    )
    
    print("\n" + "="*50)
    print(analyzer.generate_report())
    
    # 6. Validación de backtest
    analysis = analyzer.full_analysis()
    validation = validate_backtest(analysis)
    
    print(f"\n🏆 SCORE FINAL: {validation['score']}/{validation['max_score']}")
    print(f"📊 RECOMENDACIÓN: {validation['recommendation']}")
    
    if validation['red_flags']:
        print("\n🚨 RED FLAGS DETECTADAS:")
        for flag in validation['red_flags']:
            print(f"   {flag}")

# Ejecutar ejemplo
if __name__ == "__main__":
    evaluate_strategy_example()
```

## Mejores Prácticas de Métricas

### ✅ **Do's (Hacer)**

1. **Usa múltiples métricas**: Nunca te bases en una sola métrica
2. **Compara con benchmark**: Siempre evalúa vs SPY o índice relevante
3. **Analiza drawdown**: Una estrategia con 50% DD no es viable
4. **Valida estadísticamente**: Mínimo 30-50 trades para significancia
5. **Considera period consistency**: Métricas estables entre períodos

### ❌ **Don'ts (No Hacer)**

1. **No ignores transaction costs**: Incluye comisiones y slippage
2. **No optimices solo Sharpe**: Puede llevar a overfitting
3. **No uses datos futuros**: Evita look-ahead bias
4. **No ignores red flags**: Métricas "perfectas" son sospechosas
5. **No trade sin out-of-sample**: Siempre reserva datos para validación

### 🎯 **Targets Realistas para Small Caps**

```python
# Benchmarks realistas para estrategias de small caps
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

### 🔍 **Checklist de Validación**

Antes de hacer live una estrategia:

- [ ] Sharpe > 1.0
- [ ] Max Drawdown < 30%
- [ ] Mínimo 50 trades en backtest
- [ ] Profit Factor entre 1.3-2.5
- [ ] Win Rate 45-70% (no extremos)
- [ ] Consistencia en rolling windows
- [ ] Out-of-sample testing passed
- [ ] Sin red flags en métricas

## Siguiente Paso

Con las métricas dominadas y las fórmulas de referencia integradas, vamos a [Cómo Evitar Overfitting](overfitting.md) para asegurar que tus resultados sean reales.