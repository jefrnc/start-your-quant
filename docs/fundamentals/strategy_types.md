# Tipos de Estrategias Cuantitativas

## Las 4 Categorías Principales

### 1. Momentum / Trend Following
**Filosofía**: "La tendencia es tu amiga"

```python
# Ejemplo simple: Breakout de 20 días
def momentum_strategy(data):
    data['20_day_high'] = data['High'].rolling(20).max()
    data['signal'] = data['Close'] > data['20_day_high'].shift(1)
    return data
```

**Características**:
- Win rate bajo (35-45%) pero alto reward/risk
- Funcionan mejor en mercados trending
- Fáciles de programar y backtest
- Sufren en mercados laterales

**Variantes comunes**:
- Breakout de rangos
- Cruces de medias móviles  
- Relative strength (RS) vs índice
- Gap and Go (para small caps)

### 2. Mean Reversion
**Filosofía**: "Lo que sube, baja"

```python
# Ejemplo: Bollinger Bands reversal
def mean_reversion_strategy(data):
    data['SMA'] = data['Close'].rolling(20).mean()
    data['STD'] = data['Close'].rolling(20).std()
    data['Lower_Band'] = data['SMA'] - (2 * data['STD'])
    data['Upper_Band'] = data['SMA'] + (2 * data['STD'])
    
    # Comprar en sobreventa
    data['signal'] = data['Close'] < data['Lower_Band']
    return data
```

**Características**:
- Win rate alto (60-70%) pero menor reward
- Mejor en mercados laterales
- Requieren buenos stops
- Peligrosas en trends fuertes

**Variantes comunes**:
- RSI oversold/overbought
- Bollinger Bands squeeze
- Pairs trading
- VWAP reversion

### 3. Arbitraje Estadístico
**Filosofía**: "Explotar ineficiencias temporales"

```python
# Ejemplo: Pairs trading
def pairs_trading(stock_a, stock_b, window=20):
    # Calcular spread
    ratio = stock_a / stock_b
    mean_ratio = ratio.rolling(window).mean()
    std_ratio = ratio.rolling(window).std()
    
    # Z-score del spread
    z_score = (ratio - mean_ratio) / std_ratio
    
    # Señales
    signals = pd.DataFrame()
    signals['long_a_short_b'] = z_score < -2
    signals['short_a_long_b'] = z_score > 2
    return signals
```

**Características**:
- Market neutral (no depende de dirección)
- Requiere mucho capital
- Competencia con HFT
- Margins pequeños, alto volumen

**Variantes comunes**:
- ETF arbitrage
- Index arbitrage
- Cross-exchange crypto arb
- Options arbitrage

### 4. Market Making / Liquidez
**Filosofía**: "Proveer liquidez y capturar el spread"

```python
# Ejemplo conceptual (simplificado)
def simple_market_making(ticker, spread_target=0.02):
    mid_price = (bid + ask) / 2
    
    # Colocar órdenes a ambos lados
    place_buy_order(price=mid_price - spread_target/2)
    place_sell_order(price=mid_price + spread_target/2)
    
    # Ajustar inventario si se desbalancea
    if inventory > max_inventory:
        adjust_prices_to_reduce_inventory()
```

**Características**:
- Miles de pequeñas ganancias
- Requiere baja latencia
- Gestión de inventario crítica
- Riesgo en movimientos direccionales

## Estrategias Específicas para Small Caps

### Gap and Go
```python
def gap_and_go_scanner(universe):
    candidates = []
    
    for ticker in universe:
        gap_pct = (ticker.open - ticker.prev_close) / ticker.prev_close
        
        if (gap_pct > 0.10 and  # Gap >10%
            ticker.float < 50_000_000 and  # Low float
            ticker.premarket_vol > 1_000_000):  # Volume
            
            candidates.append({
                'ticker': ticker.symbol,
                'gap': gap_pct,
                'rvol': ticker.volume / ticker.avg_volume
            })
    
    return sorted(candidates, key=lambda x: x['rvol'], reverse=True)
```

### VWAP Reclaim
```python
def vwap_reclaim_setup(data):
    # Calcular VWAP
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # Detectar reclaim
    data['below_vwap'] = data['Low'] < data['VWAP']
    data['reclaim'] = (data['below_vwap'].shift(1) & 
                      (data['Close'] > data['VWAP']))
    
    # Confirmar con volumen
    data['volume_spike'] = data['Volume'] > data['Volume'].rolling(20).mean() * 2
    data['signal'] = data['reclaim'] & data['volume_spike']
    
    return data
```

### Parabolic Short
```python
def parabolic_exhaustion(data, lookback=10):
    # Detectar movimiento parabólico
    returns = data['Close'].pct_change()
    cumulative = (1 + returns).cumprod()
    
    # Condiciones de exhaustion
    data['parabolic'] = (
        (cumulative > cumulative.shift(lookback) * 1.5) &  # +50% en 10 días
        (data['RSI'] > 80) &  # RSI extremo
        (data['Volume'] > data['Volume'].rolling(20).mean() * 3)  # Climax volume
    )
    
    return data['parabolic']
```

## Cómo Elegir Tu Estrategia

### Según tu personalidad:
- **Paciente + Analítico** → Mean Reversion
- **Agresivo + Rápido** → Momentum
- **Matemático + Detallista** → Arbitraje
- **Multitasking + Tech** → Market Making

### Según tu capital:
- **< $25k**: Momentum small caps (PDT rule)
- **$25k - $100k**: Mix momentum + mean reversion
- **$100k - $500k**: Agregar pairs trading
- **> $500k**: Todas las estrategias

### Según tu tiempo:
- **Part-time**: Swing momentum
- **Full-time**: Day trading + scalping
- **Automatizado**: Market making + arbitraje

## Framework para Evaluar Estrategias

```python
def evaluate_strategy(backtest_results):
    metrics = {
        'win_rate': backtest_results['wins'] / backtest_results['total_trades'],
        'avg_win': backtest_results['gross_profits'] / backtest_results['wins'],
        'avg_loss': backtest_results['gross_losses'] / backtest_results['losses'],
        'profit_factor': backtest_results['gross_profits'] / abs(backtest_results['gross_losses']),
        'sharpe_ratio': calculate_sharpe(backtest_results['returns']),
        'max_drawdown': calculate_max_drawdown(backtest_results['equity_curve']),
        'expectancy': calculate_expectancy(backtest_results)
    }
    
    # Score final
    if (metrics['profit_factor'] > 1.5 and 
        metrics['sharpe_ratio'] > 1.0 and 
        metrics['max_drawdown'] < 0.20):
        return "VIABLE"
    else:
        return "NEEDS WORK"
```

## Mi Mix Personal

```python
# Portfolio de estrategias
STRATEGIES = {
    'morning_momentum': {
        'allocation': 0.40,
        'timeframe': '9:30-10:30',
        'instruments': ['small_cap_gappers']
    },
    'vwap_reversion': {
        'allocation': 0.30,
        'timeframe': '10:30-15:30',
        'instruments': ['liquid_stocks']
    },
    'pairs_trading': {
        'allocation': 0.20,
        'timeframe': 'all_day',
        'instruments': ['sector_etfs']
    },
    'overnight_swing': {
        'allocation': 0.10,
        'timeframe': 'multi_day',
        'instruments': ['large_caps']
    }
}
```

## Errores Comunes

1. **Over-optimizar una estrategia** en vez de diversificar
2. **Ignorar costos** de transacción en backtests
3. **No considerar liquidez** en small caps
4. **Asumir fills perfectos** en estrategias de alta frecuencia
5. **No tener circuit breakers** para condiciones adversas

## Recursos para Profundizar

- **Momentum**: "Following the Trend" - Andreas Clenow
- **Mean Reversion**: "Mean Reversion Trading" - Ernest Chan  
- **Arbitraje**: "Statistical Arbitrage" - Andrew Pole
- **Market Making**: Papers de Avellaneda & Stoikov

## Siguiente Paso

Ya conoces los tipos de estrategias. Ahora vamos a [Fuentes de Datos](../data/data_sources.md) para ver dónde conseguir la materia prima para tus backtests.