# ¿Qué es un Backtest?

## La Máquina del Tiempo del Trading

Un backtest es básicamente una máquina del tiempo. Tomas tu estrategia y la ejecutas en datos históricos para ver cómo habría funcionado. Es la diferencia entre apostar y invertir.

## Por Qué Importa

### La Cruda Realidad
- 90% de los traders pierden dinero
- La mayoría operan sin testing previo
- "Se ve bien en el gráfico" ≠ Estrategia rentable
- Un backtest te salva de perder dinero real

### Lo Que Realmente Mides
```python
# No es solo: "¿Gané dinero?"
# Es: "¿Puedo repetir esto consistentemente?"

backtest_questions = {
    'profitability': '¿Es rentable?',
    'consistency': '¿Funciona en diferentes períodos?',
    'risk': '¿Cuánto puedo perder?',
    'frequency': '¿Cuántas oportunidades hay?',
    'drawdown': '¿Puedo psicológicamente manejar las pérdidas?',
    'market_conditions': '¿Funciona en bull y bear markets?'
}
```

## Anatomía de un Backtest

### 1. Datos Históricos
```python
# Calidad de datos = Calidad de resultados
data_requirements = {
    'timeframe': '2+ años mínimo',
    'resolution': 'Acorde a tu estrategia (1min para day trading)',
    'quality': 'Adjusted for splits, dividends',
    'survivorship_bias': 'Incluir stocks delistados',
    'universe': 'Representativo de donde tradearás'
}
```

### 2. Reglas de Trading
```python
def example_strategy_rules():
    """Ejemplo de reglas claras y testeable"""
    entry_rules = {
        'signal': 'close > vwap AND rvol > 2',
        'timing': 'Market hours only',
        'size': '1% risk per trade',
        'max_positions': '3 concurrent'
    }
    
    exit_rules = {
        'stop_loss': '2% below entry',
        'take_profit': '4% above entry (2:1 R/R)',
        'time_stop': 'End of day',
        'market_stop': 'VIX > 30'
    }
    
    return {'entry': entry_rules, 'exit': exit_rules}
```

### 3. Costos y Fricciones
```python
def realistic_costs():
    """Incluir todos los costos reales"""
    return {
        'commission': 0.005,  # $5 per 1000 shares
        'spread': 0.0002,     # 2 basis points
        'slippage': 0.0001,   # 1 basis point promedio
        'borrowing_costs': 0.0003,  # Para shorts
        'platform_fees': 50,  # Mensual
        'data_fees': 100      # Mensual
    }
```

## Ejemplo: Mi Primer Backtest

```python
import pandas as pd
import numpy as np
import yfinance as yf

def simple_vwap_backtest(ticker, start_date, end_date):
    """Backtest simple de VWAP strategy"""
    
    # 1. Obtener datos
    data = yf.download(ticker, start=start_date, end=end_date, interval='5m')
    
    # 2. Calcular indicadores
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['above_vwap'] = data['Close'] > data['vwap']
    
    # 3. Generar señales
    data['signal'] = data['above_vwap'] & ~data['above_vwap'].shift(1)  # Cross above
    
    # 4. Simular trades
    initial_capital = 10000
    position = 0
    cash = initial_capital
    trades = []
    
    for i in range(len(data)):
        if data['signal'].iloc[i] and position == 0:
            # Entry
            shares = int(cash * 0.95 / data['Close'].iloc[i])
            position = shares
            cash -= shares * data['Close'].iloc[i]
            entry_price = data['Close'].iloc[i]
            
        elif position > 0:
            # Check exits
            current_price = data['Close'].iloc[i]
            
            # Stop loss: 2%
            if current_price < entry_price * 0.98:
                cash += position * current_price
                trades.append(current_price - entry_price)
                position = 0
                
            # Take profit: 4%
            elif current_price > entry_price * 1.04:
                cash += position * current_price
                trades.append(current_price - entry_price)
                position = 0
    
    # 5. Calcular métricas
    if trades:
        win_rate = len([t for t in trades if t > 0]) / len(trades)
        avg_win = np.mean([t for t in trades if t > 0])
        avg_loss = np.mean([t for t in trades if t < 0])
        profit_factor = abs(sum([t for t in trades if t > 0]) / sum([t for t in trades if t < 0]))
        
        final_value = cash + (position * data['Close'].iloc[-1] if position > 0 else 0)
        total_return = (final_value - initial_capital) / initial_capital
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'final_value': final_value
        }
    else:
        return {'error': 'No trades generated'}

# Ejecutar
results = simple_vwap_backtest('AAPL', '2023-01-01', '2023-12-31')
print(results)
```

## Tipos de Backtesting

### 1. Vectorized Backtesting
```python
# Rápido pero menos realista
def vectorized_backtest(data, signals):
    """Toda la serie de tiempo a la vez"""
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = signals.shift(1) * data['returns']
    
    cumulative_returns = (1 + data['strategy_returns']).cumprod()
    return cumulative_returns
```

### 2. Event-Driven Backtesting
```python
# Más lento pero más realista
class EventDrivenBacktest:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        
    def process_bar(self, bar):
        """Procesar cada barra individualmente"""
        # Check signals
        # Manage positions
        # Execute trades
        pass
```

### 3. Monte Carlo Simulation
```python
def monte_carlo_backtest(strategy, num_simulations=1000):
    """Múltiples simulaciones con datos alterados"""
    results = []
    
    for i in range(num_simulations):
        # Shuffle or resample data
        shuffled_data = shuffle_returns(original_data)
        result = run_backtest(strategy, shuffled_data)
        results.append(result)
    
    return analyze_distribution(results)
```

## Errores Comunes en Backtesting

### 1. Look-Ahead Bias
```python
# ❌ MALO: Usar información del futuro
data['signal'] = data['close'] > data['close'].shift(-1)  # Peek into future

# ✅ BUENO: Solo información disponible en el momento
data['signal'] = data['close'] > data['close'].shift(1)
```

### 2. Survivorship Bias
```python
# ❌ MALO: Solo stocks que sobrevivieron
universe = ['AAPL', 'MSFT', 'GOOGL']  # Solo winners

# ✅ BUENO: Incluir stocks delistados
universe = get_historical_universe('Russell3000', start_date)
```

### 3. Data Mining Bias
```python
# ❌ MALO: Optimizar hasta que funcione
for sma in range(5, 100):
    for rsi_threshold in range(20, 80):
        if backtest_return > 0.3:  # Cherry picking
            print(f"Found winning combo: SMA={sma}, RSI={rsi_threshold}")
```

### 4. Overfitting
```python
# ❌ MALO: Demasiados parámetros
def overfitted_strategy(data, p1, p2, p3, p4, p5, p6, p7, p8):
    # 8 parámetros = muy específico para data histórica
    pass

# ✅ BUENO: Mantener simple
def simple_strategy(data, short_ma=9, long_ma=20):
    # 2 parámetros = más generalizable
    pass
```

## In-Sample vs Out-of-Sample

```python
def proper_backtesting_workflow(data):
    """Workflow correcto para evitar overfitting"""
    
    # Split data
    total_length = len(data)
    in_sample_end = int(total_length * 0.7)  # 70% para desarrollo
    
    in_sample = data.iloc[:in_sample_end]
    out_sample = data.iloc[in_sample_end:]
    
    # 1. Desarrollar estrategia en in-sample
    strategy = develop_strategy(in_sample)
    
    # 2. Una sola vez: test en out-of-sample
    out_sample_results = test_strategy(strategy, out_sample)
    
    # 3. Si falla out-of-sample, volver al paso 1
    if out_sample_results['sharpe'] < 1.0:
        return "Strategy needs work"
    else:
        return "Strategy ready for paper trading"
```

## Walk-Forward Analysis

```python
def walk_forward_backtest(data, window_size=252, rebalance_freq=21):
    """Backtest con re-optimización periódica"""
    results = []
    
    for start in range(0, len(data) - window_size, rebalance_freq):
        # Training window
        train_end = start + window_size
        train_data = data.iloc[start:train_end]
        
        # Test period
        test_start = train_end
        test_end = min(test_start + rebalance_freq, len(data))
        test_data = data.iloc[test_start:test_end]
        
        # Optimize strategy on training data
        best_params = optimize_strategy(train_data)
        
        # Test on out-of-sample period
        period_result = test_strategy(best_params, test_data)
        results.append(period_result)
    
    return combine_results(results)
```

## Red Flags en Resultados

```python
def validate_backtest_results(results):
    """Identificar resultados sospechosos"""
    red_flags = []
    
    # Demasiado bueno para ser verdad
    if results['annual_return'] > 0.5:  # +50% anual
        red_flags.append("Returns too high - likely overfitted")
    
    # Win rate irreal
    if results['win_rate'] > 0.8:  # 80%+ win rate
        red_flags.append("Win rate too high - check for look-ahead bias")
    
    # Drawdown demasiado bajo
    if results['max_drawdown'] < 0.05:  # Menos de 5%
        red_flags.append("Drawdown too low - not realistic")
    
    # Pocos trades
    if results['total_trades'] < 100:
        red_flags.append("Not enough trades for statistical significance")
    
    # Profit factor irreal
    if results['profit_factor'] > 3:
        red_flags.append("Profit factor too high - likely curve-fitted")
    
    return red_flags
```

## Paper Trading: El Paso Siguiente

```python
def transition_to_paper_trading(backtest_results):
    """Cómo pasar de backtest a paper trading"""
    
    if backtest_results['sharpe_ratio'] > 1.5:
        return {
            'recommendation': 'Start paper trading',
            'position_size': 'Use 1/4 of planned size initially',
            'duration': 'Paper trade for 2-3 months minimum',
            'success_criteria': {
                'correlation_with_backtest': '>0.7',
                'sharpe_ratio': '>1.0',
                'max_drawdown': '<15%'
            }
        }
    else:
        return {
            'recommendation': 'Improve strategy first',
            'issues_to_address': analyze_weaknesses(backtest_results)
        }
```

## Herramientas para Backtesting

```python
# Frameworks populares
backtesting_tools = {
    'basic': 'pandas + numpy (custom)',
    'intermediate': 'backtrader, zipline',
    'advanced': 'vectorbt, quantconnect',
    'professional': 'QuantLib, custom C++'
}
```

## Siguiente Paso

Ahora que entiendes qué es un backtest, vamos a [Motor de Backtest Simple](simple_engine.md) para construir uno desde cero.