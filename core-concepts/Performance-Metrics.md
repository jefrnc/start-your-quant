> 🇪🇸 [Leer en Español](Performance-Metrics.es.md) | 🇺🇸 **English**

# Performance Metrics for Quantitative Trading

## Philosophy: Measurement-Driven Improvement

> "What gets measured gets improved" - Peter Drucker

In quantitative trading, metrics are our **navigation system**. They tell us if we're heading in the right direction, how fast we're progressing, and when we need to adjust course.

### Principles of Effective Measurement

1. **Actionable Metrics**: Each metric must guide specific decisions
2. **Risk-Adjusted**: Performance without considering risk is meaningless
3. **Temporal Awareness**: Consider different time horizons
4. **Benchmark Relative**: Always compare with relevant alternatives
5. **Robust to Outliers**: Don't let a few trades dominate the metrics

## Fundamental Metrics

### 1. **Sharpe Ratio - The King Metric**

```python
def calculate_sharpe_ratio(returns: pd.Series,
                          risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Sharpe = (Expected Return - Risk Free Rate) / Standard Deviation

    Interpretation:
    > 2.0  : Exceptional
    1.0-2.0: Very Good
    0.5-1.0: Good
    0.0-0.5: Poor
    < 0.0  : Destroying Value
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
```

**Why Sharpe Matters for Small Caps**:
- Small caps are inherently more volatile
- Sharpe ratio "penalizes" excessive volatility
- Forces focus on risk-adjusted returns
- Allows comparison with other strategies

### 2. **Maximum Drawdown - Your Worst-Case Scenario**

```python
def calculate_max_drawdown(equity_curve: pd.Series) -> dict:
    """
    Max DD = Maximum loss from previous peak

    Critical for small cap trading because:
    - Drawdowns can be brutal (30%+ possible)
    - Psychology impact is severe
    - Capital preservation is paramount
    """
    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown series
    drawdown = (equity_curve - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdown.min()

    # Find when it occurred
    max_dd_date = drawdown.idxmin()

    # Calculate drawdown duration
    dd_start = running_max[running_max == running_max.loc[max_dd_date]].index[0]

    # Recovery time (if applicable)
    recovery_mask = equity_curve[max_dd_date:] >= running_max.loc[max_dd_date]
    if recovery_mask.any():
        recovery_date = recovery_mask[recovery_mask].index[0]
        recovery_days = (recovery_date - dd_start).days
    else:
        recovery_days = None  # Still in drawdown

    return {
        'max_drawdown': max_dd,
        'max_dd_date': max_dd_date,
        'drawdown_start': dd_start,
        'recovery_days': recovery_days,
        'dollars_lost': equity_curve.loc[dd_start] - equity_curve.loc[max_dd_date]
    }
```

### 3. **Win Rate vs Profit Factor Balance**

```python
def calculate_win_metrics(trades_df: pd.DataFrame) -> dict:
    """
    Win Rate = % of profitable trades
    Profit Factor = Gross Profit / Gross Loss

    For small caps, we typically see:
    - Win Rate: 50-70% (higher is better but not everything)
    - Profit Factor: 1.2-3.0 (>1.5 is good, >2.0 is excellent)
    """
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    win_rate = len(winning_trades) / len(trades_df)

    gross_profit = winning_trades['pnl'].sum()
    gross_loss = abs(losing_trades['pnl'].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy_per_trade': expectancy,
        'total_trades': len(trades_df)
    }
```

## Small Cap-Specific Metrics

### 1. **Gap Efficiency Ratio**

```python
def calculate_gap_efficiency(trades_df: pd.DataFrame) -> dict:
    """
    Gap trading-specific metric:
    - Gap Capture Rate: % of initial gap captured
    - Gap Fade Rate: % of gaps that reversed
    - Optimal Gap Range: most profitable gap range
    """
    # Assume we have a 'gap_percent' column in trades
    gap_trades = trades_df[trades_df['gap_percent'].notna()]

    # Gap capture (how much of the gap we captured as profit)
    gap_trades['gap_capture'] = (gap_trades['pnl'] / gap_trades['entry_price']) / (gap_trades['gap_percent'] / 100)

    avg_gap_capture = gap_trades['gap_capture'].mean()

    # Gap fade analysis
    negative_pnl_gaps = gap_trades[gap_trades['pnl'] < 0]
    gap_fade_rate = len(negative_pnl_gaps) / len(gap_trades)

    # Optimal gap range
    gap_performance = gap_trades.groupby(
        pd.cut(gap_trades['gap_percent'], bins=[0, 5, 10, 15, 25, 100])
    )['pnl'].agg(['mean', 'count', 'sum'])

    return {
        'avg_gap_capture': avg_gap_capture,
        'gap_fade_rate': gap_fade_rate,
        'gap_performance_by_range': gap_performance,
        'best_gap_range': gap_performance['mean'].idxmax()
    }
```

### 2. **Hold Time Efficiency**

```python
def calculate_hold_time_metrics(trades_df: pd.DataFrame) -> dict:
    """
    For small caps, hold time is critical:
    - Too short: Might miss the move
    - Too long: Exposed to reversal risk
    """
    # Calculate hold time in minutes
    trades_df['hold_time_minutes'] = (
        pd.to_datetime(trades_df['exit_time']) -
        pd.to_datetime(trades_df['entry_time'])
    ).dt.total_seconds() / 60

    # Performance by hold time range
    hold_time_bins = [0, 15, 30, 60, 120, 1000]
    hold_time_labels = ['0-15min', '15-30min', '30-60min', '60-120min', '120min+']

    trades_df['hold_time_bucket'] = pd.cut(
        trades_df['hold_time_minutes'],
        bins=hold_time_bins,
        labels=hold_time_labels
    )

    performance_by_hold_time = trades_df.groupby('hold_time_bucket')['pnl'].agg([
        'mean', 'count', 'sum', 'std'
    ])

    # Sharpe by hold time bucket
    sharpe_by_hold_time = {}
    for bucket in hold_time_labels:
        bucket_trades = trades_df[trades_df['hold_time_bucket'] == bucket]
        if len(bucket_trades) > 1:
            sharpe_by_hold_time[bucket] = bucket_trades['pnl'].mean() / bucket_trades['pnl'].std()

    return {
        'avg_hold_time': trades_df['hold_time_minutes'].mean(),
        'median_hold_time': trades_df['hold_time_minutes'].median(),
        'performance_by_hold_time': performance_by_hold_time,
        'sharpe_by_hold_time': sharpe_by_hold_time,
        'optimal_hold_time_range': max(sharpe_by_hold_time, key=sharpe_by_hold_time.get)
    }
```

### 3. **Position Recycling Efficiency**

```python
def analyze_position_recycling(trades_df: pd.DataFrame) -> dict:
    """
    Analyzes the effectiveness of the position recycling approach.
    Multiple trades of the same symbol on the same day = campaign
    """
    # Group by symbol and date
    daily_campaigns = trades_df.groupby(['symbol', trades_df['entry_time'].dt.date])

    campaign_metrics = []

    for (symbol, date), campaign_trades in daily_campaigns:
        if len(campaign_trades) > 1:  # Multiple trades = recycling

            # Calculate average price improvement
            weighted_avg_entry = (
                campaign_trades['entry_price'] * campaign_trades['quantity']
            ).sum() / campaign_trades['quantity'].sum()

            first_entry_price = campaign_trades.iloc[0]['entry_price']
            price_improvement = (weighted_avg_entry - first_entry_price) / first_entry_price

            total_pnl = campaign_trades['pnl'].sum()
            total_quantity = campaign_trades['quantity'].sum()

            campaign_metrics.append({
                'symbol': symbol,
                'date': date,
                'num_trades': len(campaign_trades),
                'total_pnl': total_pnl,
                'avg_price_improvement': price_improvement,
                'total_quantity': total_quantity
            })

    campaign_df = pd.DataFrame(campaign_metrics)

    if len(campaign_df) > 0:
        return {
            'total_campaigns': len(campaign_df),
            'avg_trades_per_campaign': campaign_df['num_trades'].mean(),
            'avg_price_improvement': campaign_df['avg_price_improvement'].mean(),
            'campaign_success_rate': (campaign_df['total_pnl'] > 0).mean(),
            'avg_campaign_pnl': campaign_df['total_pnl'].mean()
        }
    else:
        return {'total_campaigns': 0}
```

## Advanced Metrics

### 1. **Sortino Ratio - Downside Focus**

```python
def calculate_sortino_ratio(returns: pd.Series,
                           target_return: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """
    Sortino = (Expected Return - Target) / Downside Deviation

    Better than Sharpe for asymmetric strategies because:
    - Only penalizes downside volatility
    - Upside volatility is good (we want big winners)
    - More relevant for gap trading
    """
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf

    downside_deviation = downside_returns.std()

    return excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year)
```

### 2. **Calmar Ratio - Drawdown Efficiency**

```python
def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar = Annualized Return / Maximum Drawdown

    Excellent for small cap evaluation:
    - Measures return per unit of worst-case risk
    - DD is often the limiting factor in small caps
    - Helps compare strategies with different volatility profiles
    """
    equity_curve = (1 + returns).cumprod()

    annualized_return = (equity_curve.iloc[-1] ** (periods_per_year / len(equity_curve))) - 1

    max_dd = calculate_max_drawdown(equity_curve)['max_drawdown']

    return annualized_return / abs(max_dd) if max_dd != 0 else np.inf
```

### 3. **Information Ratio - Benchmark Beating**

```python
def calculate_information_ratio(strategy_returns: pd.Series,
                               benchmark_returns: pd.Series,
                               periods_per_year: int = 252) -> float:
    """
    Information Ratio = Excess Return / Tracking Error

    For small cap strategies, benchmark might be:
    - IWM (Russell 2000)
    - IJR (iShares Core S&P Small-Cap)
    - VB (Vanguard Small-Cap)
    """
    # Align periods
    aligned_data = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    excess_returns = aligned_data['strategy'] - aligned_data['benchmark']

    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
    excess_return = excess_returns.mean() * periods_per_year

    return excess_return / tracking_error if tracking_error != 0 else np.inf
```

## Time-Based Performance Analysis

### 1. **Performance by Time of Day**

```python
def analyze_performance_by_time(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Critical for premarket trading:
    - 5:30-6:00 AM: Early gap reaction
    - 6:00-7:00 AM: Volume building
    - 7:00-8:00 AM: Institutional participation
    - 8:00-9:30 AM: Pre-open positioning
    """
    trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
    trades_df['entry_minute'] = pd.to_datetime(trades_df['entry_time']).dt.minute

    # Create time buckets
    def time_bucket(hour, minute):
        if hour < 6:
            return "5:30-6:00"
        elif hour < 7:
            return "6:00-7:00"
        elif hour < 8:
            return "7:00-8:00"
        elif hour < 9:
            return "8:00-9:00"
        else:
            return "9:00+"

    trades_df['time_bucket'] = trades_df.apply(
        lambda x: time_bucket(x['entry_hour'], x['entry_minute']), axis=1
    )

    time_performance = trades_df.groupby('time_bucket').agg({
        'pnl': ['mean', 'sum', 'count', 'std'],
        'gap_percent': 'mean',
        'hold_time_minutes': 'mean'
    }).round(2)

    return time_performance
```

### 2. **Rolling Performance Metrics**

```python
def calculate_rolling_metrics(trades_df: pd.DataFrame,
                             window_days: int = 30) -> pd.DataFrame:
    """
    Track metrics over time to detect strategy degradation
    """
    # Sort by date
    trades_df = trades_df.sort_values('entry_time')
    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date

    # Daily P&L
    daily_pnl = trades_df.groupby('date')['pnl'].sum()

    # Rolling metrics
    rolling_metrics = pd.DataFrame(index=daily_pnl.index)

    rolling_metrics['rolling_pnl'] = daily_pnl.rolling(window_days).sum()
    rolling_metrics['rolling_sharpe'] = daily_pnl.rolling(window_days).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    rolling_metrics['rolling_max_dd'] = daily_pnl.rolling(window_days).apply(
        lambda x: calculate_max_drawdown((1 + x).cumprod())['max_drawdown']
    )
    rolling_metrics['rolling_win_rate'] = trades_df.set_index('date')['pnl'].rolling(
        window_days
    ).apply(lambda x: (x > 0).mean())

    return rolling_metrics
```

## Benchmarking and Comparison

### Relevant Small Cap Benchmarks

```python
SMALL_CAP_BENCHMARKS = {
    'IWM': 'iShares Russell 2000 ETF',           # Most liquid small cap
    'IJR': 'iShares Core S&P Small-Cap ETF',    # S&P 600 focus
    'VB': 'Vanguard Small-Cap ETF',             # Broad small cap
    'VTWO': 'Vanguard Russell 2000 ETF',        # Russell 2000 exposure
    'SLY': 'SPDR S&P 600 Small Cap ETF'         # S&P 600 specific
}

def benchmark_comparison(strategy_returns: pd.Series,
                        benchmark_symbol: str = 'IWM') -> dict:
    """
    Compare strategy vs relevant small cap benchmark
    """
    # Get benchmark data (placeholder - integrate with your data provider)
    benchmark_returns = get_benchmark_returns(benchmark_symbol)

    # Align periods
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    strategy_metrics = {
        'total_return': (1 + aligned['strategy']).prod() - 1,
        'sharpe': calculate_sharpe_ratio(aligned['strategy']),
        'max_dd': calculate_max_drawdown((1 + aligned['strategy']).cumprod())['max_drawdown'],
        'volatility': aligned['strategy'].std() * np.sqrt(252)
    }

    benchmark_metrics = {
        'total_return': (1 + aligned['benchmark']).prod() - 1,
        'sharpe': calculate_sharpe_ratio(aligned['benchmark']),
        'max_dd': calculate_max_drawdown((1 + aligned['benchmark']).cumprod())['max_drawdown'],
        'volatility': aligned['benchmark'].std() * np.sqrt(252)
    }

    return {
        'strategy': strategy_metrics,
        'benchmark': benchmark_metrics,
        'excess_return': strategy_metrics['total_return'] - benchmark_metrics['total_return'],
        'information_ratio': calculate_information_ratio(aligned['strategy'], aligned['benchmark'])
    }
```

## Metrics Dashboard Template

### Key Performance Indicators (KPIs)

```python
def generate_performance_dashboard(trades_df: pd.DataFrame) -> dict:
    """
    One-stop shop for all key metrics
    """
    # Basic setup
    equity_curve = trades_df['pnl'].cumsum()
    daily_pnl = trades_df.groupby(trades_df['entry_time'].dt.date)['pnl'].sum()

    # Core metrics
    total_pnl = trades_df['pnl'].sum()
    total_trades = len(trades_df)
    win_metrics = calculate_win_metrics(trades_df)
    drawdown_metrics = calculate_max_drawdown(equity_curve)

    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(daily_pnl)
    sortino = calculate_sortino_ratio(daily_pnl)
    calmar = calculate_calmar_ratio(daily_pnl)

    # Small cap specific
    if 'gap_percent' in trades_df.columns:
        gap_metrics = calculate_gap_efficiency(trades_df)
    else:
        gap_metrics = {}

    if 'hold_time_minutes' in trades_df.columns:
        hold_time_metrics = calculate_hold_time_metrics(trades_df)
    else:
        hold_time_metrics = {}

    return {
        'summary': {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'trading_days': len(daily_pnl),
            'avg_daily_pnl': daily_pnl.mean(),
            'best_day': daily_pnl.max(),
            'worst_day': daily_pnl.min()
        },
        'risk_metrics': {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'current_drawdown': equity_curve.iloc[-1] - equity_curve.max(),
            'volatility': daily_pnl.std() * np.sqrt(252)
        },
        'win_metrics': win_metrics,
        'gap_metrics': gap_metrics,
        'hold_time_metrics': hold_time_metrics,
        'recent_performance': {
            'last_30_days': daily_pnl.tail(30).sum(),
            'last_7_days': daily_pnl.tail(7).sum(),
            'yesterday': daily_pnl.iloc[-1] if len(daily_pnl) > 0 else 0
        }
    }
```

## Interpretation and Action Items

### Performance Ranges for Small Cap Strategies

| Metric | Poor | Good | Excellent | Elite |
|---------|------|------|-----------|-------|
| **Sharpe Ratio** | < 0.5 | 0.5-1.0 | 1.0-2.0 | > 2.0 |
| **Max Drawdown** | > -20% | -10% to -20% | -5% to -10% | < -5% |
| **Win Rate** | < 50% | 50-60% | 60-70% | > 70% |
| **Profit Factor** | < 1.2 | 1.2-1.5 | 1.5-2.5 | > 2.5 |
| **Calmar Ratio** | < 1.0 | 1.0-2.0 | 2.0-4.0 | > 4.0 |

### Action Triggers

```python
def performance_alerts(metrics: dict) -> list:
    """
    Generate alerts based on performance degradation
    """
    alerts = []

    if metrics['risk_metrics']['sharpe_ratio'] < 0.5:
        alerts.append("LOW SHARPE: Review strategy parameters")

    if metrics['risk_metrics']['max_drawdown'] < -0.15:
        alerts.append("HIGH DRAWDOWN: Consider reducing position sizes")

    if metrics['win_metrics']['win_rate'] < 0.45:
        alerts.append("LOW WIN RATE: Review entry criteria")

    if metrics['win_metrics']['profit_factor'] < 1.1:
        alerts.append("LOW PROFIT FACTOR: Review exit strategy")

    if metrics['recent_performance']['last_7_days'] < -50:
        alerts.append("RECENT UNDERPERFORMANCE: Consider pause")

    return alerts
```

## Integration with Trading System

### Real-Time Monitoring

```python
class PerformanceMonitor:
    def __init__(self, trades_file: str):
        self.trades_file = trades_file
        self.last_update = None

    def update_metrics(self):
        """Update metrics when new trade is added"""
        trades_df = pd.read_csv(self.trades_file)

        if len(trades_df) == 0:
            return

        # Generate dashboard
        dashboard = generate_performance_dashboard(trades_df)

        # Check for alerts
        alerts = performance_alerts(dashboard)

        # Send notifications if needed
        if alerts:
            self.send_alerts(alerts)

        return dashboard

    def send_alerts(self, alerts: list):
        """Send alerts via Telegram/Discord/Email"""
        for alert in alerts:
            print(f"ALERT: {alert}")
            # Integrate with your notification system
```

---

**Remember**: Metrics are tools for improvement, not endpoints. The goal is not to optimize metrics in isolation, but to build a sustainably profitable trading system.

**Next Steps**:
- Implement [Sharpe Calculator](../scripts/strategy-metrics/sharpe-calculator/)
- Set up [Performance Monitoring Dashboard](../templates/monitoring/grafana-dashboards/)
- Study [Market Microstructure](./Market-Microstructure.md) for deeper insights
