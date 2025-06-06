# Gap % y Float

## Los Fundamentos del Small Cap Trading

Gap % y Float son los dos criterios más importantes para filtrar small caps con potencial explosivo. Un stock con float bajo y gap alto es dinamita.

## Cálculo de Gap %

```python
def calculate_gap_metrics(df):
    """Calcular todas las métricas de gap"""
    # Gap básico
    df['prev_close'] = df['close'].shift(1)
    df['gap_dollar'] = df['open'] - df['prev_close']
    df['gap_pct'] = (df['gap_dollar'] / df['prev_close']) * 100
    
    # Clasificar gaps
    df['gap_type'] = pd.cut(
        df['gap_pct'],
        bins=[-np.inf, -10, -5, -2, 2, 5, 10, 20, np.inf],
        labels=['large_gap_down', 'medium_gap_down', 'small_gap_down', 
                'no_gap', 'small_gap_up', 'medium_gap_up', 
                'large_gap_up', 'massive_gap_up']
    )
    
    # Gap vs average range
    df['avg_range'] = ((df['high'] - df['low']) / df['low']).rolling(20).mean()
    df['gap_vs_range'] = abs(df['gap_pct']) / (df['avg_range'] * 100)
    
    # Gap fill analysis
    df['gap_high'] = np.where(df['gap_pct'] > 0, df['prev_close'], df['open'])
    df['gap_low'] = np.where(df['gap_pct'] > 0, df['open'], df['prev_close'])
    
    # Check gap fill
    df['gap_filled'] = np.where(
        df['gap_pct'] > 0,
        df['low'] <= df['prev_close'],  # Gap up filled
        df['high'] >= df['prev_close']   # Gap down filled
    )
    
    return df
```

## Float Analysis

```python
def get_float_data(ticker):
    """Obtener datos de float y shares outstanding"""
    # Esto requiere conexión a API de fundamentales
    # Ejemplo con yfinance (no siempre confiable para float)
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    info = stock.info
    
    float_data = {
        'shares_outstanding': info.get('sharesOutstanding', None),
        'float_shares': info.get('floatShares', None),
        'held_by_insiders': info.get('heldPercentInsiders', None),
        'held_by_institutions': info.get('heldPercentInstitutions', None),
        'short_ratio': info.get('shortRatio', None),
        'short_percent': info.get('shortPercentOfFloat', None)
    }
    
    # Calcular float real si no está disponible
    if float_data['float_shares'] is None and float_data['shares_outstanding']:
        insider_held = float_data['held_by_insiders'] or 0
        institutional_held = float_data['held_by_institutions'] or 0
        locked_up = (insider_held + institutional_held) / 100
        float_data['float_shares'] = float_data['shares_outstanding'] * (1 - locked_up)
    
    return float_data

def classify_float_size(float_shares):
    """Clasificar tamaño de float"""
    if float_shares is None:
        return 'unknown'
    elif float_shares < 10_000_000:
        return 'micro_float'
    elif float_shares < 25_000_000:
        return 'low_float'
    elif float_shares < 50_000_000:
        return 'medium_float'
    elif float_shares < 100_000_000:
        return 'large_float'
    else:
        return 'institutional'
```

## Gap & Float Scanner

```python
def gap_float_scanner(universe, gap_threshold=15, max_float=50_000_000):
    """Scanner para gap + low float combo"""
    candidates = []
    
    for ticker in universe:
        try:
            # Obtener datos de precio
            data = get_latest_data(ticker)
            gap_pct = calculate_gap_pct(data)
            
            # Filtros básicos
            if abs(gap_pct) < gap_threshold:
                continue
                
            # Obtener float
            float_data = get_float_data(ticker)
            float_shares = float_data['float_shares']
            
            if float_shares is None or float_shares > max_float:
                continue
            
            # Calcular métricas adicionales
            price = data['close'].iloc[-1]
            volume = data['volume'].iloc[-1]
            dollar_volume = price * volume
            
            # News/catalysts check (simplified)
            has_news = check_recent_news(ticker)
            
            candidate = {
                'ticker': ticker,
                'gap_pct': gap_pct,
                'float_shares': float_shares,
                'float_category': classify_float_size(float_shares),
                'price': price,
                'volume': volume,
                'dollar_volume': dollar_volume,
                'rvol': calculate_current_rvol(data),
                'has_news': has_news,
                'short_interest': float_data['short_percent'],
                'risk_score': calculate_risk_score(gap_pct, float_shares, volume)
            }
            
            candidates.append(candidate)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    # Ordenar por potencial
    candidates_df = pd.DataFrame(candidates)
    candidates_df['rank_score'] = (
        candidates_df['gap_pct'].abs() * 0.3 +
        (50_000_000 / candidates_df['float_shares']) * 0.4 +
        candidates_df['rvol'] * 0.3
    )
    
    return candidates_df.sort_values('rank_score', ascending=False)
```

## Gap Fill Probability

```python
def gap_fill_analysis(df, lookback_period=252):
    """Analizar probabilidad de que se llene el gap"""
    df = calculate_gap_metrics(df)
    
    # Análisis histórico de gap fills
    gap_fill_stats = {}
    
    for gap_type in df['gap_type'].unique():
        if pd.isna(gap_type):
            continue
            
        gaps = df[df['gap_type'] == gap_type].copy()
        
        # Para cada gap, ver si se llenó en N días
        fill_rates = {}
        for days in [1, 3, 5, 10, 20]:
            fills = 0
            total = 0
            
            for idx in gaps.index:
                if idx + days < len(df):
                    future_data = df.loc[idx:idx+days]
                    gap_row = df.loc[idx]
                    
                    if gap_row['gap_pct'] > 0:  # Gap up
                        filled = future_data['low'].min() <= gap_row['prev_close']
                    else:  # Gap down
                        filled = future_data['high'].max() >= gap_row['prev_close']
                    
                    if filled:
                        fills += 1
                    total += 1
            
            fill_rates[f'{days}_days'] = fills / total if total > 0 else 0
        
        gap_fill_stats[gap_type] = fill_rates
    
    return gap_fill_stats
```

## Float Rotation Analysis

```python
def float_rotation_analysis(df, float_shares):
    """Analizar cuántas veces rota el float"""
    if float_shares is None:
        return None
    
    # Volumen acumulado vs float
    df['daily_rotation'] = df['volume'] / float_shares
    df['cumulative_rotation'] = df['daily_rotation'].cumsum()
    
    # Ventana móvil de rotación
    df['rotation_5d'] = df['daily_rotation'].rolling(5).sum()
    df['rotation_20d'] = df['daily_rotation'].rolling(20).sum()
    
    # Velocidad de rotación
    df['rotation_velocity'] = df['daily_rotation'].rolling(5).mean()
    
    # Alerta de alta rotación
    df['high_rotation'] = df['daily_rotation'] > 0.5  # 50% del float en un día
    df['extreme_rotation'] = df['daily_rotation'] > 1.0  # Float completo
    
    return df
```

## Gap Fade vs Follow Strategy

```python
def gap_strategy_signals(df, float_shares):
    """Determinar si fade o follow el gap"""
    df = calculate_gap_metrics(df)
    
    # Factores para la decisión
    gap_size = abs(df['gap_pct'])
    float_category = classify_float_size(float_shares)
    
    # Reglas generales (simplificadas)
    df['strategy'] = 'hold'
    
    # Gap pequeño en large float = probable fill (fade)
    fade_conditions = (
        (gap_size < 5) & (float_category in ['large_float', 'institutional'])
    )
    
    # Gap grande en low float = probable momentum (follow)
    follow_conditions = (
        (gap_size > 15) & (float_category in ['micro_float', 'low_float'])
    )
    
    df.loc[fade_conditions, 'strategy'] = 'fade'
    df.loc[follow_conditions, 'strategy'] = 'follow'
    
    # Ajustar por volumen
    if 'rvol' in df.columns:
        # Alto volumen favorece follow
        df.loc[(df['strategy'] == 'fade') & (df['rvol'] > 3), 'strategy'] = 'follow'
        # Bajo volumen favorece fade
        df.loc[(df['strategy'] == 'follow') & (df['rvol'] < 1), 'strategy'] = 'fade'
    
    return df
```

## Insider/Institution Impact

```python
def analyze_ownership_impact(ticker, float_data):
    """Analizar impacto de ownership en volatilidad"""
    insider_pct = float_data.get('held_by_insiders', 0)
    institution_pct = float_data.get('held_by_institutions', 0)
    float_shares = float_data.get('float_shares', 0)
    
    # Free float real
    locked_shares = (insider_pct + institution_pct) / 100
    truly_free_float = float_shares * (1 - locked_shares)
    
    # Volatility multiplier basado en ownership
    ownership_factor = 1 + (locked_shares * 2)  # Más locked = más volátil
    
    analysis = {
        'insider_locked_pct': insider_pct,
        'institution_locked_pct': institution_pct,
        'total_locked_pct': insider_pct + institution_pct,
        'truly_free_float': truly_free_float,
        'volatility_multiplier': ownership_factor,
        'risk_level': 'extreme' if locked_shares > 0.8 else 
                     'high' if locked_shares > 0.6 else 
                     'medium' if locked_shares > 0.4 else 'low'
    }
    
    return analysis
```

## Historical Gap Performance

```python
def historical_gap_performance(ticker, lookback_days=252):
    """Performance histórico por tipo de gap"""
    df = get_historical_data(ticker, lookback_days)
    df = calculate_gap_metrics(df)
    
    performance_by_gap = {}
    
    for gap_type in df['gap_type'].unique():
        if pd.isna(gap_type):
            continue
            
        gap_days = df[df['gap_type'] == gap_type].copy()
        
        if len(gap_days) == 0:
            continue
        
        # Performance metrics
        gap_days['day_return'] = (gap_days['close'] - gap_days['open']) / gap_days['open'] * 100
        gap_days['intraday_high'] = (gap_days['high'] - gap_days['open']) / gap_days['open'] * 100
        gap_days['intraday_low'] = (gap_days['low'] - gap_days['open']) / gap_days['open'] * 100
        
        performance_by_gap[gap_type] = {
            'count': len(gap_days),
            'avg_day_return': gap_days['day_return'].mean(),
            'win_rate': (gap_days['day_return'] > 0).mean(),
            'avg_intraday_high': gap_days['intraday_high'].mean(),
            'avg_intraday_low': gap_days['intraday_low'].mean(),
            'max_gain': gap_days['intraday_high'].max(),
            'max_loss': gap_days['intraday_low'].min()
        }
    
    return performance_by_gap
```

## Real-Time Gap Monitor

```python
class GapFloatMonitor:
    def __init__(self, gap_threshold=10, max_float=50_000_000):
        self.gap_threshold = gap_threshold
        self.max_float = max_float
        self.watchlist = []
        
    def scan_premarket(self):
        """Scan pre-market para gaps"""
        candidates = []
        
        # Obtener movers pre-market
        premarket_movers = get_premarket_movers()
        
        for ticker in premarket_movers:
            try:
                # Validar criterios
                gap_pct = calculate_premarket_gap(ticker)
                float_data = get_float_data(ticker)
                
                if (abs(gap_pct) >= self.gap_threshold and 
                    float_data['float_shares'] <= self.max_float):
                    
                    candidates.append({
                        'ticker': ticker,
                        'gap_pct': gap_pct,
                        'float': float_data['float_shares'],
                        'added_time': pd.Timestamp.now()
                    })
                    
            except Exception as e:
                continue
        
        self.watchlist.extend(candidates)
        return candidates
    
    def monitor_intraday(self):
        """Monitorear durante el día"""
        alerts = []
        
        for item in self.watchlist:
            ticker = item['ticker']
            current_data = get_current_data(ticker)
            
            # Check key levels
            if 'gap_filled' in current_data and current_data['gap_filled']:
                alerts.append(f"🔄 {ticker}: Gap filled @ ${current_data['price']:.2f}")
            
            # Check momentum
            if current_data['rvol'] > 5:
                alerts.append(f"🚀 {ticker}: Explosive volume - RVol {current_data['rvol']:.1f}x")
        
        return alerts
```

## Tips de Trading Real

### 1. Float Categories Strategy
```python
FLOAT_STRATEGIES = {
    'micro_float': {
        'max_position': 0.05,  # 5% max position
        'stop_loss': 0.15,     # 15% stop
        'take_profit': 0.50,   # 50% target
        'time_limit': 30       # 30 min max hold
    },
    'low_float': {
        'max_position': 0.10,
        'stop_loss': 0.10,
        'take_profit': 0.30,
        'time_limit': 60
    },
    'medium_float': {
        'max_position': 0.20,
        'stop_loss': 0.08,
        'take_profit': 0.20,
        'time_limit': 120
    }
}
```

### 2. Gap Size Rules
```python
def gap_trading_rules(gap_pct):
    """Reglas según tamaño de gap"""
    if abs(gap_pct) > 50:
        return "AVOID - Too risky"
    elif abs(gap_pct) > 30:
        return "SCALP ONLY - Quick in/out"
    elif abs(gap_pct) > 15:
        return "MOMENTUM PLAY - Follow with tight stops"
    elif abs(gap_pct) > 5:
        return "FADE CANDIDATE - Look for mean reversion"
    else:
        return "NORMAL GAP - No special strategy"
```

## Alertas Críticas

```python
def critical_gap_float_alerts(ticker, gap_pct, float_shares):
    """Alertas para combinaciones extremas"""
    alerts = []
    
    # Micro float + large gap = nuclear
    if float_shares < 5_000_000 and abs(gap_pct) > 25:
        alerts.append(f"☢️ {ticker}: NUCLEAR SETUP - {gap_pct:.1f}% gap on {float_shares/1_000_000:.1f}M float")
    
    # High short interest + gap up
    short_data = get_short_interest(ticker)
    if short_data and short_data > 30 and gap_pct > 15:
        alerts.append(f"🔥 {ticker}: SQUEEZE POTENTIAL - {gap_pct:.1f}% gap + {short_data:.1f}% SI")
    
    return alerts
```

He completado toda la sección de indicadores técnicos para small caps. ¿Quieres que continúe con la siguiente sección (backtesting) o necesitas revisar alguno de los indicadores?