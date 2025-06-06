# SuperTendencia - El Seguidor de Tendencias Definitivo

## Definición

SuperTendencia es un indicador de seguimiento de tendencia que dibuja una línea en el gráfico de velas. Dependiendo del color, indica si está en tendencia negativa (línea roja sobre las velas) o tendencia positiva (línea verde debajo de las velas), funcionando como soporte dinámico o resistencia.

## Filosofía del Indicador

### ¿Por Qué Funciona?
- **Basado en ATR**: Utiliza volatilidad real para ajustar distancias
- **Soporte/Resistencia Dinámico**: Se adapta a las condiciones del mercado
- **Señales Claras**: Color verde = alcista, color rojo = bajista
- **Filtro de Ruido**: Reduce señales falsas comparado con MA simples

### Componentes Clave
```
ATR = Average True Range (volatilidad)
Precio_Medio = (High + Low) / 2
Banda_Superior = Precio_Medio + (Factor × ATR)
Banda_Inferior = Precio_Medio - (Factor × ATR)
```

## Implementación de Referencia

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def SuperTendencia(df: pd.DataFrame, longitud: int = 14, factor: float = 3.0) -> pd.DataFrame:
    """
    SuperTendencia - Implementación de referencia exacta
    
    Indicador de seguimiento de tendencia que dibuja líneas de soporte/resistencia
    dinámicas basadas en ATR (Average True Range).
    
    Parámetros
    ----------
    df : pd.DataFrame
        Datos históricos del activo (debe incluir High, Low, Close)
    longitud : int, default 14
        Ventana para el cálculo del ATR
    factor : float, default 3.0
        Multiplicador del ATR para calcular las bandas
    
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: FinalUpperB, FinalLowerB, SuperTendencia
        
    Cómo Operarlo
    -------------
    - COMPRA: Línea verde (debajo del precio) - Tendencia alcista
    - VENTA: Línea roja (encima del precio) - Tendencia bajista
    - La línea actúa como soporte dinámico (verde) o resistencia (roja)
    
    Ejemplo de Uso
    --------------
    >>> df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
    >>> st = SuperTendencia(df, longitud=14, factor=3.0)
    >>> print(st.head())
    """
    # Calcular True Range (TR)
    High, Low = df["High"], df["Low"]
    
    # Componentes del True Range
    H_minus_L = High - Low
    prev_close = df["Close"].shift(periods=1)
    H_minus_PC = abs(High - prev_close)
    L_minus_PC = abs(Low - prev_close)
    
    # True Range = max(H-L, |H-PC|, |L-PC|)
    TR = pd.Series(np.max([H_minus_L, H_minus_PC, L_minus_PC], axis=0), 
                   index=df.index, name="TR")
    
    # Calcular ATR usando suavizado exponencial
    ATR = TR.ewm(alpha=1 / longitud).mean()
    
    # Calcular precio medio y bandas básicas
    medio = (High + Low) / 2
    FinalUpperB = medio + factor * ATR
    FinalLowerB = medio - factor * ATR
    
    # Inicializar SuperTendencia
    Supertendencia = np.zeros(ATR.shape[0], dtype=bool)
    close = df["Close"]
    
    # Calcular SuperTendencia punto por punto
    for i in range(1, ATR.shape[0]):
        # Determinar dirección de tendencia
        if close[i] > FinalUpperB[i - 1]:
            # Precio rompe banda superior -> tendencia alcista
            Supertendencia[i] = True
        elif close[i] < FinalLowerB[i - 1]:
            # Precio rompe banda inferior -> tendencia bajista
            Supertendencia[i] = False
        else:
            # Mantener tendencia anterior
            Supertendencia[i] = Supertendencia[i - 1]
            
            # Ajustar bandas para evitar cambios prematuros
            if Supertendencia[i] == True and FinalLowerB[i] < FinalLowerB[i - 1]:
                # En tendencia alcista, banda inferior no puede bajar
                FinalLowerB[i] = FinalLowerB[i - 1]
            elif Supertendencia[i] == False and FinalUpperB[i] > FinalUpperB[i - 1]:
                # En tendencia bajista, banda superior no puede subir
                FinalUpperB[i] = FinalUpperB[i - 1]
        
        # Eliminar banda inactiva según dirección de tendencia
        if Supertendencia[i] == True:
            # Tendencia alcista: eliminar banda superior
            FinalUpperB[i] = np.nan
        else:
            # Tendencia bajista: eliminar banda inferior
            FinalLowerB[i] = np.nan
    
    # Ajustar primer valor
    if Supertendencia[1] == False:
        FinalLowerB[0] = np.nan
    else:
        FinalUpperB[0] = np.nan
    
    # Preparar datos finales (eliminar período de calentamiento)
    FU = FinalUpperB[longitud - 1:]
    FL = FinalLowerB[longitud - 1:]
    
    # Crear SuperTendencia final combinando ambas bandas
    ST_array = np.nansum([FU, FL], axis=0)
    ST_array[0] = np.nan  # Primer valor siempre NaN
    
    # Crear DataFrame resultado
    ST_df = pd.concat([FU, FL], axis=1)
    ST_df["SuperTendencia"] = ST_array
    ST_df.columns = ["FinalUpperB", "FinalLowerB", "SuperTendencia"]
    
    return ST_df

def analyze_supertrend_signals(df: pd.DataFrame, st_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analizar señales de trading del SuperTendencia
    
    Parámetros
    ----------
    df : pd.DataFrame
        Datos históricos originales
    st_data : pd.DataFrame
        Datos del SuperTendencia (output de SuperTendencia)
    
    Returns
    -------
    pd.DataFrame
        DataFrame con señales y análisis
    """
    # Alinear índices (ST tiene menos datos por período de calentamiento)
    aligned_df = df.loc[st_data.index]
    
    signals = pd.DataFrame(index=st_data.index)
    signals['price'] = aligned_df['Close']
    signals['high'] = aligned_df['High']
    signals['low'] = aligned_df['Low']
    signals['volume'] = aligned_df['Volume']
    signals['supertrend'] = st_data['SuperTendencia']
    signals['upper_band'] = st_data['FinalUpperB']
    signals['lower_band'] = st_data['FinalLowerB']
    
    # Determinar tendencia actual
    signals['trend'] = np.where(~pd.isna(st_data['FinalLowerB']), 1, -1)  # 1=alcista, -1=bajista
    signals['trend_change'] = signals['trend'].diff().fillna(0)
    
    # Señales de cambio de tendencia
    signals['buy_signal'] = signals['trend_change'] == 2    # De bajista a alcista
    signals['sell_signal'] = signals['trend_change'] == -2  # De alcista a bajista
    
    # Distancia del precio al SuperTendencia (fuerza de tendencia)
    signals['price_st_distance'] = np.where(
        signals['trend'] == 1,
        (signals['price'] - signals['supertrend']) / signals['supertrend'],  # Alcista
        (signals['supertrend'] - signals['price']) / signals['supertrend']   # Bajista
    )
    
    # Duración de la tendencia
    trend_groups = (signals['trend'] != signals['trend'].shift()).cumsum()
    signals['trend_duration'] = signals.groupby(trend_groups).cumcount() + 1
    
    # Volatilidad del período (usando SuperTendencia como proxy)
    st_changes = signals['supertrend'].pct_change().abs()
    signals['st_volatility'] = st_changes.rolling(10).mean()
    
    # Calidad de señales
    signals['signal_strength'] = 'NONE'
    
    # Señales fuertes (con confirmaciones)
    strong_buy = (
        signals['buy_signal'] &
        (signals['trend_duration'].shift(1) > 3) &  # Tendencia bajista duradera
        (signals['volume'] > signals['volume'].rolling(20).mean()) &  # Volumen confirmatorio
        (signals['price'] > signals['price'].shift(1))  # Momentum alcista
    )
    
    strong_sell = (
        signals['sell_signal'] &
        (signals['trend_duration'].shift(1) > 3) &  # Tendencia alcista duradera
        (signals['volume'] > signals['volume'].rolling(20).mean()) &
        (signals['price'] < signals['price'].shift(1))  # Momentum bajista
    )
    
    # Señales de pullback (retrocesos en tendencia)
    pullback_buy = (
        (signals['trend'] == 1) &  # Tendencia alcista
        (signals['low'] <= signals['supertrend'] * 1.005) &  # Precio cerca del ST
        (signals['close'] > signals['supertrend']) &  # Pero cierra arriba
        (signals['trend_duration'] > 5)  # Tendencia establecida
    )
    
    pullback_sell = (
        (signals['trend'] == -1) &  # Tendencia bajista
        (signals['high'] >= signals['supertrend'] * 0.995) &  # Precio cerca del ST
        (signals['close'] < signals['supertrend']) &  # Pero cierra abajo
        (signals['trend_duration'] > 5)  # Tendencia establecida
    )
    
    # Asignar calidades
    signals.loc[strong_buy, 'signal_strength'] = 'STRONG_BUY'
    signals.loc[strong_sell, 'signal_strength'] = 'STRONG_SELL'
    signals.loc[pullback_buy, 'signal_strength'] = 'PULLBACK_BUY'
    signals.loc[pullback_sell, 'signal_strength'] = 'PULLBACK_SELL'
    signals.loc[signals['buy_signal'] & ~strong_buy, 'signal_strength'] = 'WEAK_BUY'
    signals.loc[signals['sell_signal'] & ~strong_sell, 'signal_strength'] = 'WEAK_SELL'
    
    return signals
```

## Estrategias de Trading con SuperTendencia

### 1. Trend Following Strategy
```python
def supertrend_following_strategy(df: pd.DataFrame, st_length: int = 14, st_factor: float = 3.0):
    """
    Estrategia de seguimiento de tendencia usando SuperTendencia
    """
    # Calcular SuperTendencia
    st = SuperTendencia(df, longitud=st_length, factor=st_factor)
    signals = analyze_supertrend_signals(df, st)
    
    # Filtros adicionales
    sma_200 = df['Close'].rolling(200).mean()
    aligned_sma = sma_200.loc[signals.index]
    
    entry_signals = pd.Series(0, index=signals.index)
    
    # Long entries (solo en mercado alcista general)
    long_entry = (
        signals['buy_signal'] &
        (signals['signal_strength'].isin(['STRONG_BUY'])) &
        (signals['price'] > aligned_sma)  # Precio sobre SMA 200
    )
    
    # Short entries (solo en mercado bajista general)
    short_entry = (
        signals['sell_signal'] &
        (signals['signal_strength'].isin(['STRONG_SELL'])) &
        (signals['price'] < aligned_sma)  # Precio bajo SMA 200
    )
    
    # Pullback entries (entradas en retrocesos)
    pullback_long = signals['signal_strength'] == 'PULLBACK_BUY'
    pullback_short = signals['signal_strength'] == 'PULLBACK_SELL'
    
    entry_signals[long_entry] = 1
    entry_signals[short_entry] = -1
    entry_signals[pullback_long] = 0.5   # Posición reducida en pullbacks
    entry_signals[pullback_short] = -0.5
    
    return {
        'signals': entry_signals,
        'st_data': st,
        'analysis': signals,
        'strategy_type': 'trend_following'
    }

def supertrend_rsi_strategy(df: pd.DataFrame, st_length: int = 14, st_factor: float = 3.0):
    """
    Combinar SuperTendencia con RSI - estrategia mencionada en referencia
    """
    # SuperTendencia
    st = SuperTendencia(df, longitud=st_length, factor=st_factor)
    signals = analyze_supertrend_signals(df, st)
    
    # RSI
    rsi = calculate_rsi(df['Close'], period=14)
    aligned_rsi = rsi.loc[signals.index]
    
    entry_signals = pd.Series(0, index=signals.index)
    
    # Long setup: ST alcista + RSI cruza arriba de 50
    long_setup = (
        signals['buy_signal'] &  # Cambio a tendencia alcista en ST
        (aligned_rsi > 50) &     # RSI por encima de 50
        (aligned_rsi.shift(1) <= 50)  # RSI estaba por debajo de 50
    )
    
    # Short setup: ST bajista + RSI cruza abajo de 50  
    short_setup = (
        signals['sell_signal'] &  # Cambio a tendencia bajista en ST
        (aligned_rsi < 50) &      # RSI por debajo de 50
        (aligned_rsi.shift(1) >= 50)  # RSI estaba por encima de 50
    )
    
    # Entries adicionales en pullbacks con RSI
    rsi_pullback_long = (
        (signals['trend'] == 1) &  # ST en tendencia alcista
        (aligned_rsi < 40) &       # RSI oversold
        (aligned_rsi > aligned_rsi.shift(1))  # RSI empezando a recuperar
    )
    
    rsi_pullback_short = (
        (signals['trend'] == -1) &  # ST en tendencia bajista
        (aligned_rsi > 60) &        # RSI overbought
        (aligned_rsi < aligned_rsi.shift(1))  # RSI empezando a caer
    )
    
    entry_signals[long_setup] = 1
    entry_signals[short_setup] = -1
    entry_signals[rsi_pullback_long] = 0.5
    entry_signals[rsi_pullback_short] = -0.5
    
    return {
        'signals': entry_signals,
        'st_data': st,
        'analysis': signals,
        'rsi': aligned_rsi,
        'strategy_type': 'st_rsi_combo'
    }

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Helper function para calcular RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### 2. Small Cap Specific Strategy
```python
def small_cap_supertrend_strategy(df: pd.DataFrame, gap_threshold: float = 0.03):
    """
    Estrategia SuperTendencia específica para small caps
    """
    # SuperTendencia más sensible para small caps
    st = SuperTendencia(df, longitud=10, factor=2.5)  # Más reactivo
    signals = analyze_supertrend_signals(df, st)
    
    # Detectar gaps
    gap_up = (df['Open'] / df['Close'].shift(1) - 1) > gap_threshold
    gap_down = (df['Open'] / df['Close'].shift(1) - 1) < -gap_threshold
    aligned_gap_up = gap_up.loc[signals.index]
    aligned_gap_down = gap_down.loc[signals.index]
    
    # RVOL (Relative Volume)
    avg_volume = df['Volume'].rolling(20).mean()
    rvol = df['Volume'] / avg_volume
    aligned_rvol = rvol.loc[signals.index]
    
    entry_signals = pd.Series(0, index=signals.index)
    
    # Gap & Go con SuperTendencia
    gap_and_go_long = (
        aligned_gap_up &
        (signals['trend'] == 1) &  # ST confirma alcista
        (aligned_rvol > 3) &       # Alto volumen relativo
        (signals['price'] > signals['price'].shift(1))  # Momentum continúa
    )
    
    # Gap fill con SuperTendencia
    gap_fill_long = (
        aligned_gap_down &
        signals['buy_signal'] &    # ST cambia a alcista
        (aligned_rvol > 2) &       # Volumen confirmatorio
        (signals['price'] > df['Open'].loc[signals.index])  # Precio recupera sobre apertura
    )
    
    # Breakout con SuperTendencia
    high_20 = df['High'].rolling(20).max()
    aligned_high_20 = high_20.loc[signals.index]
    
    breakout_long = (
        signals['buy_signal'] &
        (signals['high'] >= aligned_high_20.shift(1)) &  # Nuevo high 20 días
        (aligned_rvol > 2.5) &
        (signals['trend_duration'] <= 2)  # Cambio reciente de tendencia
    )
    
    # Short setups
    gap_fade_short = (
        aligned_gap_up &
        signals['sell_signal'] &   # ST cambia a bajista
        (aligned_rvol > 2) &
        (signals['price'] < df['Open'].loc[signals.index])  # Precio bajo apertura
    )
    
    entry_signals[gap_and_go_long] = 1
    entry_signals[gap_fill_long] = 1
    entry_signals[breakout_long] = 1
    entry_signals[gap_fade_short] = -1
    
    return {
        'signals': entry_signals,
        'st_data': st,
        'analysis': signals,
        'gap_signals': {
            'gap_up': aligned_gap_up,
            'gap_down': aligned_gap_down,
            'rvol': aligned_rvol
        },
        'strategy_type': 'small_cap_st'
    }

def adaptive_supertrend_parameters(df: pd.DataFrame, volatility_period: int = 20):
    """
    Parámetros adaptativos basados en volatilidad
    """
    # Medir volatilidad actual
    atr = calculate_atr_simple(df, period=14)
    current_atr = atr.iloc[-1]
    price = df['Close'].iloc[-1]
    atr_pct = current_atr / price
    
    # Parámetros base
    base_length = 14
    base_factor = 3.0
    
    # Ajustar según volatilidad
    if atr_pct > 0.03:  # Alta volatilidad (>3%)
        length = base_length + 3  # Más suavizado
        factor = base_factor * 1.2  # Bandas más amplias
        regime = "HIGH_VOLATILITY"
    elif atr_pct < 0.015:  # Baja volatilidad (<1.5%)
        length = base_length - 2  # Más reactivo
        factor = base_factor * 0.8  # Bandas más estrechas
        regime = "LOW_VOLATILITY"
    else:
        length = base_length
        factor = base_factor
        regime = "NORMAL_VOLATILITY"
    
    return {
        'length': max(length, 5),  # Mínimo 5 períodos
        'factor': max(factor, 1.5),  # Mínimo factor 1.5
        'atr_pct': atr_pct,
        'regime': regime
    }

def calculate_atr_simple(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Helper function para calcular ATR simple"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(period).mean()
```

## Optimización y Multi-Timeframe

```python
def multi_timeframe_supertrend(symbol: str, primary_tf: str = '1d', secondary_tf: str = '4h'):
    """
    Análisis SuperTendencia en múltiples timeframes
    """
    import yfinance as yf
    
    # Obtener datos
    df_primary = yf.download(symbol, period="6mo", interval=primary_tf)
    df_secondary = yf.download(symbol, period="2mo", interval=secondary_tf)
    
    # SuperTendencia en cada timeframe
    st_primary = SuperTendencia(df_primary, longitud=14, factor=3.0)
    st_secondary = SuperTendencia(df_secondary, longitud=14, factor=2.5)  # Más sensible en TF menor
    
    signals_primary = analyze_supertrend_signals(df_primary, st_primary)
    signals_secondary = analyze_supertrend_signals(df_secondary, st_secondary)
    
    # Estado actual
    current_primary = signals_primary.iloc[-1]
    current_secondary = signals_secondary.iloc[-1]
    
    analysis = {
        'primary_trend': current_primary['trend'],
        'secondary_trend': current_secondary['trend'],
        'primary_duration': current_primary['trend_duration'],
        'secondary_duration': current_secondary['trend_duration'],
        'primary_distance': current_primary['price_st_distance'],
        'secondary_distance': current_secondary['price_st_distance'],
        'confluence': None,
        'setup_quality': None
    }
    
    # Análisis de confluencia
    if current_primary['trend'] == current_secondary['trend']:
        if current_primary['trend'] == 1:
            analysis['confluence'] = 'BULLISH_ALIGNMENT'
            if (current_primary['trend_duration'] > 5 and 
                current_secondary['trend_duration'] > 3):
                analysis['setup_quality'] = 'STRONG_BULLISH'
        else:
            analysis['confluence'] = 'BEARISH_ALIGNMENT'
            if (current_primary['trend_duration'] > 5 and 
                current_secondary['trend_duration'] > 3):
                analysis['setup_quality'] = 'STRONG_BEARISH'
    else:
        analysis['confluence'] = 'MIXED_SIGNALS'
        analysis['setup_quality'] = 'CONFLICTED'
    
    # Detectar setups de alta probabilidad
    if (current_primary['buy_signal'] and 
        current_secondary['trend'] == 1 and 
        current_secondary['trend_duration'] > 2):
        analysis['setup_quality'] = 'HIGH_PROB_LONG'
    elif (current_primary['sell_signal'] and 
          current_secondary['trend'] == -1 and 
          current_secondary['trend_duration'] > 2):
        analysis['setup_quality'] = 'HIGH_PROB_SHORT'
    
    return analysis

def supertrend_parameter_optimization(df: pd.DataFrame, length_range: tuple = (10, 20), factor_range: tuple = (2.0, 4.0)):
    """
    Optimización simple de parámetros SuperTendencia
    """
    import itertools
    
    # Rangos de parámetros a probar
    lengths = range(length_range[0], length_range[1] + 1, 2)
    factors = np.arange(factor_range[0], factor_range[1] + 0.1, 0.5)
    
    results = []
    
    for length, factor in itertools.product(lengths, factors):
        try:
            # Calcular SuperTendencia con parámetros
            st = SuperTendencia(df, longitud=length, factor=factor)
            signals = analyze_supertrend_signals(df, st)
            
            # Métricas simples de evaluación
            total_signals = signals['buy_signal'].sum() + signals['sell_signal'].sum()
            if total_signals == 0:
                continue
                
            # Simular returns simples
            position = 0
            returns = []
            
            for i in range(1, len(signals)):
                if signals['buy_signal'].iloc[i]:
                    position = 1
                elif signals['sell_signal'].iloc[i]:
                    position = -1
                
                if position != 0:
                    ret = position * (signals['price'].iloc[i] / signals['price'].iloc[i-1] - 1)
                    returns.append(ret)
            
            if len(returns) > 0:
                total_return = np.sum(returns)
                win_rate = np.mean([r > 0 for r in returns])
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                
                results.append({
                    'length': length,
                    'factor': factor,
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'sharpe': sharpe,
                    'total_signals': total_signals,
                    'score': sharpe * total_return  # Score compuesto
                })
        
        except Exception as e:
            continue
    
    # Encontrar mejores parámetros
    if results:
        best_result = max(results, key=lambda x: x['score'])
        return best_result, results
    else:
        return None, []
```

## Visualización Completa

```python
def plot_supertrend_analysis(df: pd.DataFrame, st_data: pd.DataFrame, signals: pd.DataFrame, title: str = "SuperTendencia Analysis"):
    """
    Crear gráfico completo de análisis SuperTendencia
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Alinear datos
    aligned_df = df.loc[st_data.index]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12),
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Chart 1: Precio + SuperTendencia
    ax1.plot(aligned_df.index, aligned_df['Close'], 'k-', linewidth=2, label='Price', zorder=1)
    
    # SuperTendencia líneas
    bullish_mask = ~pd.isna(st_data['FinalLowerB'])
    bearish_mask = ~pd.isna(st_data['FinalUpperB'])
    
    ax1.plot(st_data.index[bullish_mask], st_data['FinalLowerB'][bullish_mask], 
             'g-', linewidth=3, label='SuperTrend (Bullish)', zorder=2)
    ax1.plot(st_data.index[bearish_mask], st_data['FinalUpperB'][bearish_mask], 
             'r-', linewidth=3, label='SuperTrend (Bearish)', zorder=2)
    
    # Marcar señales
    buy_signals = signals.index[signals['buy_signal']]
    sell_signals = signals.index[signals['sell_signal']]
    
    ax1.scatter(buy_signals, signals.loc[buy_signals, 'price'], 
               color='lime', marker='^', s=150, label='Buy Signal', zorder=5)
    ax1.scatter(sell_signals, signals.loc[sell_signals, 'price'], 
               color='red', marker='v', s=150, label='Sell Signal', zorder=5)
    
    # Highlight strong signals
    strong_buys = signals.index[signals['signal_strength'] == 'STRONG_BUY']
    strong_sells = signals.index[signals['signal_strength'] == 'STRONG_SELL']
    
    ax1.scatter(strong_buys, signals.loc[strong_buys, 'price'], 
               color='darkgreen', marker='^', s=200, label='Strong Buy', zorder=6, edgecolors='white')
    ax1.scatter(strong_sells, signals.loc[strong_sells, 'price'], 
               color='darkred', marker='v', s=200, label='Strong Sell', zorder=6, edgecolors='white')
    
    ax1.set_title(f'{title} - Price & SuperTrend')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Trend Duration
    colors = ['red' if x == -1 else 'green' for x in signals['trend']]
    ax2.bar(signals.index, signals['trend_duration'], color=colors, alpha=0.7, width=1)
    ax2.set_title('Trend Duration (Days)')
    ax2.set_ylabel('Duration')
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Price-SuperTrend Distance
    ax3.plot(signals.index, signals['price_st_distance'] * 100, 'purple', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.fill_between(signals.index, 0, signals['price_st_distance'] * 100,
                    where=(signals['price_st_distance'] > 0), alpha=0.3, color='green')
    ax3.fill_between(signals.index, 0, signals['price_st_distance'] * 100,
                    where=(signals['price_st_distance'] < 0), alpha=0.3, color='red')
    ax3.set_title('Price-SuperTrend Distance (%)')
    ax3.set_ylabel('Distance %')
    ax3.grid(True, alpha=0.3)
    
    # Formato fechas
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def supertrend_complete_example():
    """
    Ejemplo completo de análisis con SuperTendencia
    """
    import yfinance as yf
    
    # Obtener datos
    ticker = "AAPL"
    df = yf.download(ticker, start="2023-01-01", end="2024-01-01", interval="1d")
    
    print(f"=== ANÁLISIS SUPERTENDENCIA: {ticker} ===\n")
    
    # Parámetros adaptativos
    adaptive_params = adaptive_supertrend_parameters(df)
    print(f"📊 PARÁMETROS ADAPTATIVOS:")
    print(f"   Régimen de Volatilidad: {adaptive_params['regime']}")
    print(f"   ATR %: {adaptive_params['atr_pct']:.2%}")
    print(f"   Longitud: {adaptive_params['length']}")
    print(f"   Factor: {adaptive_params['factor']:.1f}")
    
    # Calcular SuperTendencia
    st = SuperTendencia(df, 
                       longitud=adaptive_params['length'],
                       factor=adaptive_params['factor'])
    signals = analyze_supertrend_signals(df, st)
    
    # Estadísticas
    buy_signals_count = signals['buy_signal'].sum()
    sell_signals_count = signals['sell_signal'].sum()
    avg_trend_duration = signals['trend_duration'].mean()
    strong_signals = signals['signal_strength'].str.contains('STRONG').sum()
    
    print(f"\n📈 ESTADÍSTICAS DEL PERÍODO:")
    print(f"   Señales de Compra: {buy_signals_count}")
    print(f"   Señales de Venta: {sell_signals_count}")
    print(f"   Señales Fuertes: {strong_signals}")
    print(f"   Duración Promedio de Tendencia: {avg_trend_duration:.1f} días")
    
    # Análisis actual
    current = signals.iloc[-1]
    trend_name = "ALCISTA" if current['trend'] == 1 else "BAJISTA"
    
    print(f"\n🎯 ANÁLISIS ACTUAL:")
    print(f"   Precio: ${current['price']:.2f}")
    print(f"   SuperTrend: ${current['supertrend']:.2f}")
    print(f"   Tendencia: {trend_name}")
    print(f"   Duración Tendencia: {current['trend_duration']} días")
    print(f"   Distancia al ST: {current['price_st_distance']:.2%}")
    print(f"   Fuerza de Señal: {current['signal_strength']}")
    
    if current['buy_signal']:
        print("   🟢 SEÑAL: BUY - Cambio a tendencia alcista")
    elif current['sell_signal']:
        print("   🔴 SEÑAL: SELL - Cambio a tendencia bajista")
    elif current['trend'] == 1:
        print(f"   🟢 MANTENER LONG - Soporte en ${current['supertrend']:.2f}")
    else:
        print(f"   🔴 MANTENER SHORT - Resistencia en ${current['supertrend']:.2f}")
    
    # Multi-timeframe
    mtf_analysis = multi_timeframe_supertrend(ticker)
    print(f"\n🔄 ANÁLISIS MULTI-TIMEFRAME:")
    print(f"   Confluencia: {mtf_analysis['confluence']}")
    print(f"   Calidad Setup: {mtf_analysis['setup_quality']}")
    
    # Optimización
    best_params, all_results = supertrend_parameter_optimization(df)
    if best_params:
        print(f"\n⚙️ PARÁMETROS ÓPTIMOS:")
        print(f"   Longitud: {best_params['length']}")
        print(f"   Factor: {best_params['factor']}")
        print(f"   Sharpe: {best_params['sharpe']:.2f}")
        print(f"   Win Rate: {best_params['win_rate']:.1%}")
    
    # Crear gráfico
    plot_supertrend_analysis(df, st, signals, f"SuperTrend Analysis - {ticker}")
    
    return st, signals

# Ejecutar ejemplo
if __name__ == "__main__":
    supertrend_complete_example()
```

## Mejores Prácticas

### ✅ **Do's (Hacer)**

1. **Úsalo como soporte/resistencia dinámico**: ST excelente para trailing stops
2. **Combina con RSI**: La combinación ST + RSI es muy efectiva
3. **Ajusta parámetros según volatilidad**: Factor más alto para activos volátiles
4. **Filtra señales con volumen**: Confirma cambios de tendencia con volumen

### ❌ **Don'ts (No Hacer)**

1. **No uses en mercados muy choppy**: ST genera whipsaws en rangos estrechos
2. **No ignores la tendencia mayor**: Confirma con timeframes superiores
3. **No uses factor muy bajo**: Puede generar señales excesivas
4. **No trades contra contexto**: Una línea verde no garantiza rally

### 🎯 **Parámetros Recomendados**

```python
SUPERTREND_SETTINGS = {
    'conservative': {'length': 21, 'factor': 4.0},    # Menos señales, más confiables
    'standard': {'length': 14, 'factor': 3.0},       # Parámetros clásicos
    'aggressive': {'length': 10, 'factor': 2.5},     # Más señales, mayor riesgo
    'small_caps': {'length': 12, 'factor': 2.8},     # Adaptado a volatilidad
    'scalping': {'length': 7, 'factor': 2.0}         # Para timeframes cortos
}
```

## Siguiente Paso

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Analizar estructura y contenido de la carpeta referencia", "status": "completed", "priority": "high"}, {"id": "2", "content": "Identificar gaps en documentaci\u00f3n actual vs referencia", "status": "completed", "priority": "high"}, {"id": "3", "content": "Proponer mejoras espec\u00edficas para integrar conocimiento", "status": "completed", "priority": "medium"}, {"id": "4", "content": "Analizar como mejorar documentaci\u00f3n existente", "status": "completed", "priority": "high"}, {"id": "5", "content": "Mejorar docs/backtesting/metrics.md con f\u00f3rmulas espec\u00edficas", "status": "completed", "priority": "high"}, {"id": "6", "content": "Agregar indicadores t\u00e9cnicos faltantes a docs/indicators/", "status": "completed", "priority": "high"}, {"id": "7", "content": "Crear docs/strategies/strategy_framework.md", "status": "pending", "priority": "high"}, {"id": "8", "content": "Agregar docs/analysis/fundamental_analysis.md", "status": "pending", "priority": "medium"}]