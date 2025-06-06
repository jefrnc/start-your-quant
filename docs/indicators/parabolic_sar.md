# Parabolic SAR - El Sistema de Stop y Reversa

## Definición

El Parabolic SAR (Stop and Reverse) es un indicador que determina la dirección de la tendencia y posibles reversiones usando un sistema de puntos que siguen el precio, funcionando como trailing stops dinámicos.

## Filosofía del Indicador

### ¿Por Qué Funciona?
- **Trailing Stop Dinámico**: Se ajusta automáticamente según el momentum
- **Aceleración Progresiva**: Factor que aumenta con el tiempo en tendencia
- **Señales Claras**: Cambio de posición = cambio de tendencia

### Conceptos Clave
- **SAR**: Valor del stop actual
- **Acceleration Factor (AF)**: Incremento progresivo (0.02 por defecto)
- **Max Step**: Límite máximo del AF (0.20 por defecto)

## Implementación de Referencia

```python
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def Parabolic_SAR(df: pd.DataFrame, incremento: float = 0.02, max_paso: float = 0.20) -> pd.DataFrame:
    """
    Parabolic SAR - Implementación de referencia exacta
    
    El indicador SAR utiliza un sistema de parada y reversa para identificar
    puntos de entrada y salida basados en la aceleración del precio.
    
    Parámetros
    ----------
    df : pd.DataFrame
        Datos históricos del activo (debe incluir High, Low, Close)
    incremento : float, default 0.02
        Incremento inicial del factor de aceleración (Alpha)
    max_paso : float, default 0.20
        Paso máximo que puede alcanzar el factor de aceleración
    
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: PSAR, UpTrend, DownTrend
        
    Cómo Operarlo
    -------------
    - Señal de COMPRA: Puntos cambian de arriba a abajo del precio
    - Señal de VENTA: Puntos cambian de abajo a arriba del precio
    - Trailing Stop: Usar PSAR como stop loss dinámico
    
    Ejemplo de Uso
    --------------
    >>> df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
    >>> psar = Parabolic_SAR(df, incremento=0.02, max_paso=0.20)
    >>> print(psar.head())
    """
    # Trabajar con copia para no modificar original
    data = deepcopy(df)
    High, Low, Close = data["High"].values, data["Low"].values, data["Close"].values
    
    # Inicializar arrays para tendencias
    psar_up = np.repeat(np.nan, Close.shape[0])
    psar_down = np.repeat(np.nan, Close.shape[0])
    
    # Variables de estado inicial
    up_trend = True  # Comenzar en tendencia alcista
    up_trend_high = High[0]  # Máximo en tendencia alcista
    down_trend_low = Low[0]  # Mínimo en tendencia bajista
    acc_factor = incremento  # Factor de aceleración inicial
    
    # Calcular PSAR para cada punto
    for i in range(2, Close.shape[0]):
        reversal = False
        max_high = High[i]
        min_low = Low[i]
        
        # === TENDENCIA ALCISTA ===
        if up_trend:
            # Calcular nuevo PSAR para tendencia alcista
            # PSAR = PSAR_anterior + AF * (EP - PSAR_anterior)
            Close[i] = Close[i - 1] + (acc_factor * (up_trend_high - Close[i - 1]))
            
            # Verificar si se produce reversión (precio rompe PSAR)
            if min_low < Close[i]:
                reversal = True
                Close[i] = up_trend_high  # PSAR se convierte en el máximo previo
                down_trend_low = min_low  # Nuevo punto extremo para tendencia bajista
                acc_factor = incremento  # Reiniciar factor de aceleración
            else:
                # Actualizar máximo y acelerar si hay nuevo high
                if max_high > up_trend_high:
                    up_trend_high = max_high
                    acc_factor = min(acc_factor + incremento, max_paso)
                
                # Regla SAR: No puede ser superior a low de períodos anteriores
                low1 = Low[i - 1]
                low2 = Low[i - 2]
                if low2 < Close[i]:
                    Close[i] = low2
                elif low1 < Close[i]:
                    Close[i] = low1
        
        # === TENDENCIA BAJISTA ===
        else:
            # Calcular nuevo PSAR para tendencia bajista
            Close[i] = Close[i - 1] - (acc_factor * (Close[i - 1] - down_trend_low))
            
            # Verificar si se produce reversión (precio rompe PSAR)
            if max_high > Close[i]:
                reversal = True
                Close[i] = down_trend_low  # PSAR se convierte en el mínimo previo
                up_trend_high = max_high  # Nuevo punto extremo para tendencia alcista
                acc_factor = incremento  # Reiniciar factor de aceleración
            else:
                # Actualizar mínimo y acelerar si hay nuevo low
                if min_low < down_trend_low:
                    down_trend_low = min_low
                    acc_factor = min(acc_factor + incremento, max_paso)
                
                # Regla SAR: No puede ser inferior a high de períodos anteriores
                high1 = High[i - 1]
                high2 = High[i - 2]
                if high2 > Close[i]:
                    Close[i] = high2
                elif high1 > Close[i]:
                    Close[i] = high1
        
        # Actualizar dirección de tendencia
        up_trend = up_trend != reversal  # XOR logic para cambio de estado
        
        # Asignar puntos según tendencia
        if up_trend:
            psar_up[i] = Close[i]
        else:
            psar_down[i] = Close[i]
    
    # Crear DataFrame resultado
    data["PSAR"] = Close
    data["UpTrend"] = psar_up
    data["DownTrend"] = psar_down
    
    return data[["PSAR", "UpTrend", "DownTrend"]]

def analyze_psar_signals(df: pd.DataFrame, psar_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analizar señales de trading del Parabolic SAR
    
    Parámetros
    ----------
    df : pd.DataFrame
        Datos históricos originales
    psar_data : pd.DataFrame
        Datos del PSAR (output de Parabolic_SAR)
    
    Returns
    -------
    pd.DataFrame
        DataFrame con señales y análisis
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['high'] = df['High']
    signals['low'] = df['Low']
    signals['psar'] = psar_data['PSAR']
    signals['up_trend'] = psar_data['UpTrend']
    signals['down_trend'] = psar_data['DownTrend']
    
    # Detectar cambios de tendencia
    signals['trend'] = np.where(~pd.isna(psar_data['UpTrend']), 1, -1)
    signals['trend_change'] = signals['trend'].diff().fillna(0)
    
    # Señales de entrada
    signals['buy_signal'] = signals['trend_change'] == 2    # De bajista a alcista
    signals['sell_signal'] = signals['trend_change'] == -2  # De alcista a bajista
    
    # Distancia del precio al SAR (momentum indicator)
    signals['price_sar_distance'] = np.where(
        signals['trend'] == 1,
        (signals['price'] - signals['psar']) / signals['price'],  # Alcista: precio sobre SAR
        (signals['psar'] - signals['price']) / signals['price']   # Bajista: SAR sobre precio
    )
    
    # Fuerza de la tendencia (basada en duración)
    trend_length = signals.groupby((signals['trend'] != signals['trend'].shift()).cumsum()).cumcount() + 1
    signals['trend_strength'] = trend_length
    
    # Calidad de la señal
    signals['signal_quality'] = 'NONE'
    
    # Señales de alta calidad
    high_quality_buy = (
        signals['buy_signal'] &
        (signals['trend_strength'].shift(1) > 5) &  # Tendencia bajista previa duradera
        (df['Volume'] > df['Volume'].rolling(20).mean())  # Volumen confirmatorio
    )
    
    high_quality_sell = (
        signals['sell_signal'] &
        (signals['trend_strength'].shift(1) > 5) &  # Tendencia alcista previa duradera
        (df['Volume'] > df['Volume'].rolling(20).mean())
    )
    
    signals.loc[high_quality_buy, 'signal_quality'] = 'HIGH_QUALITY_BUY'
    signals.loc[high_quality_sell, 'signal_quality'] = 'HIGH_QUALITY_SELL'
    signals.loc[signals['buy_signal'] & ~high_quality_buy, 'signal_quality'] = 'MEDIUM_BUY'
    signals.loc[signals['sell_signal'] & ~high_quality_sell, 'signal_quality'] = 'MEDIUM_SELL'
    
    return signals
```

## Estrategias de Trading con PSAR

### 1. Trending Strategy
```python
def psar_trending_strategy(df: pd.DataFrame, af_increment: float = 0.02, af_max: float = 0.20):
    """
    Estrategia de seguimiento de tendencia usando PSAR
    """
    # Calcular PSAR
    psar = Parabolic_SAR(df, incremento=af_increment, max_paso=af_max)
    signals = analyze_psar_signals(df, psar)
    
    # Filtros adicionales para mejorar señales
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    
    # Solo trades en dirección de tendencia mayor
    bullish_context = sma_50 > sma_200
    bearish_context = sma_50 < sma_200
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Long entries
    long_entry = (
        signals['buy_signal'] &
        bullish_context &
        (signals['signal_quality'].isin(['HIGH_QUALITY_BUY', 'MEDIUM_BUY'])) &
        (df['Close'] > sma_50)  # Precio sobre media móvil
    )
    
    # Short entries  
    short_entry = (
        signals['sell_signal'] &
        bearish_context &
        (signals['signal_quality'].isin(['HIGH_QUALITY_SELL', 'MEDIUM_SELL'])) &
        (df['Close'] < sma_50)  # Precio bajo media móvil
    )
    
    entry_signals[long_entry] = 1
    entry_signals[short_entry] = -1
    
    return {
        'signals': entry_signals,
        'psar_data': psar,
        'analysis': signals,
        'strategy_type': 'trending'
    }

def psar_scalping_strategy(df: pd.DataFrame, timeframe: str = '15min'):
    """
    Estrategia de scalping usando PSAR para small caps
    """
    # PSAR más sensible para scalping
    psar = Parabolic_SAR(df, incremento=0.01, max_paso=0.10)  # Más conservador
    signals = analyze_psar_signals(df, psar)
    
    # Filtros específicos para scalping
    atr = calculate_atr(df, period=14)
    bb = Bollinger_Bands(df, longitud=20, desviacion_std=2.0)
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Long scalping setup
    long_scalp = (
        signals['buy_signal'] &
        (atr > atr.rolling(20).mean() * 1.2) &  # Volatilidad elevada
        (df['Close'] > bb['MA']) &  # Sobre media de Bollinger
        (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5)  # Volumen fuerte
    )
    
    # Short scalping setup
    short_scalp = (
        signals['sell_signal'] &
        (atr > atr.rolling(20).mean() * 1.2) &
        (df['Close'] < bb['MA']) &
        (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5)
    )
    
    entry_signals[long_scalp] = 1
    entry_signals[short_scalp] = -1
    
    return {
        'signals': entry_signals,
        'psar_data': psar,
        'analysis': signals,
        'strategy_type': 'scalping',
        'stop_loss': signals['psar']  # PSAR como stop dinámico
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Helper function para calcular ATR"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(period).mean()
```

### 2. Small Cap Gap & Go con PSAR
```python
def gap_and_go_psar_strategy(df: pd.DataFrame, gap_threshold: float = 0.05):
    """
    Combinar Gap & Go con PSAR para small caps
    """
    # Detectar gaps
    gap_up = (df['Open'] / df['Close'].shift(1) - 1) > gap_threshold
    gap_down = (df['Open'] / df['Close'].shift(1) - 1) < -gap_threshold
    
    # PSAR adaptativo para gaps
    psar = Parabolic_SAR(df, incremento=0.03, max_paso=0.25)  # Más agresivo
    signals = analyze_psar_signals(df, psar)
    
    # Pre-market high/low simulation (usando datos intraday si disponible)
    premarket_high = df['High'].rolling(3).max()  # Approximation
    premarket_low = df['Low'].rolling(3).min()
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Gap up continuation con PSAR
    gap_up_continuation = (
        gap_up &
        signals['buy_signal'] &
        (df['Close'] > premarket_high) &  # Break of pre-market high
        (df['Volume'] > df['Volume'].rolling(20).mean() * 3)  # Heavy volume
    )
    
    # Gap down reversal con PSAR
    gap_down_reversal = (
        gap_down &
        signals['buy_signal'] &
        (df['Close'] > df['Open']) &  # Green candle after gap down
        (df['Volume'] > df['Volume'].rolling(20).mean() * 2)
    )
    
    # Gap fade con PSAR
    gap_fade = (
        gap_up &
        signals['sell_signal'] &
        (df['Close'] < df['Open']) &  # Red candle after gap up
        (signals['trend_strength'].shift(1) <= 3)  # Short uptrend before reversal
    )
    
    entry_signals[gap_up_continuation] = 1
    entry_signals[gap_down_reversal] = 1
    entry_signals[gap_fade] = -1
    
    return {
        'signals': entry_signals,
        'psar_data': psar,
        'analysis': signals,
        'gap_signals': {
            'gap_up': gap_up,
            'gap_down': gap_down,
            'gap_up_continuation': gap_up_continuation,
            'gap_down_reversal': gap_down_reversal,
            'gap_fade': gap_fade
        },
        'strategy_type': 'gap_and_go_psar'
    }
```

## Optimización de Parámetros

### Parámetros Adaptativos
```python
def adaptive_psar_parameters(df: pd.DataFrame, volatility_period: int = 20):
    """
    Calcular parámetros PSAR adaptativos basados en volatilidad
    """
    # Medir volatilidad actual
    returns = df['Close'].pct_change()
    rolling_vol = returns.rolling(volatility_period).std() * np.sqrt(252)
    current_vol = rolling_vol.iloc[-1]
    
    # Parámetros base
    base_increment = 0.02
    base_max = 0.20
    
    # Ajustar según volatilidad
    if current_vol > 0.5:  # Alta volatilidad
        increment = base_increment * 0.5  # Más conservador
        max_step = base_max * 0.75
        regime = "HIGH_VOLATILITY"
    elif current_vol < 0.2:  # Baja volatilidad
        increment = base_increment * 1.5  # Más agresivo
        max_step = base_max * 1.25
        regime = "LOW_VOLATILITY"
    else:  # Volatilidad normal
        increment = base_increment
        max_step = base_max
        regime = "NORMAL_VOLATILITY"
    
    return {
        'increment': increment,
        'max_step': max_step,
        'volatility': current_vol,
        'regime': regime
    }

def multi_timeframe_psar(symbol: str, primary_tf: str = '1d', secondary_tf: str = '1h'):
    """
    Análisis PSAR en múltiples timeframes
    """
    import yfinance as yf
    
    # Obtener datos
    df_primary = yf.download(symbol, period="3mo", interval=primary_tf)
    df_secondary = yf.download(symbol, period="1mo", interval=secondary_tf)
    
    # PSAR en cada timeframe
    psar_primary = Parabolic_SAR(df_primary)
    psar_secondary = Parabolic_SAR(df_secondary)
    
    signals_primary = analyze_psar_signals(df_primary, psar_primary)
    signals_secondary = analyze_psar_signals(df_secondary, psar_secondary)
    
    # Estado actual en ambos timeframes
    current_primary = signals_primary.iloc[-1]
    current_secondary = signals_secondary.iloc[-1]
    
    # Análisis de confluencia
    analysis = {
        'primary_trend': current_primary['trend'],
        'secondary_trend': current_secondary['trend'],
        'primary_strength': current_primary['trend_strength'],
        'secondary_strength': current_secondary['trend_strength'],
        'confluence': None
    }
    
    # Detectar confluencia de tendencias
    if current_primary['trend'] == current_secondary['trend']:
        if current_primary['trend'] == 1:
            analysis['confluence'] = 'BULLISH_CONFLUENCE'
        else:
            analysis['confluence'] = 'BEARISH_CONFLUENCE'
    else:
        analysis['confluence'] = 'MIXED_SIGNALS'
    
    # Detectar posibles reversiones
    if (current_primary['buy_signal'] and 
        current_secondary['trend'] == 1 and 
        current_secondary['trend_strength'] > 3):
        analysis['setup'] = 'STRONG_BUY_SETUP'
    elif (current_primary['sell_signal'] and 
          current_secondary['trend'] == -1 and 
          current_secondary['trend_strength'] > 3):
        analysis['setup'] = 'STRONG_SELL_SETUP'
    else:
        analysis['setup'] = 'NO_CLEAR_SETUP'
    
    return analysis
```

## Visualización y Análisis

```python
def plot_psar_analysis(df: pd.DataFrame, psar_data: pd.DataFrame, signals: pd.DataFrame, title: str = "Parabolic SAR Analysis"):
    """
    Crear gráfico completo de análisis PSAR
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12),
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Chart 1: Precio + PSAR
    ax1.plot(df.index, df['Close'], 'k-', linewidth=2, label='Price', zorder=1)
    
    # PSAR points - diferentes colores para up/down trend
    up_trend_mask = ~pd.isna(psar_data['UpTrend'])
    down_trend_mask = ~pd.isna(psar_data['DownTrend'])
    
    ax1.scatter(df.index[up_trend_mask], psar_data['UpTrend'][up_trend_mask], 
               color='green', s=20, label='PSAR (Bullish)', zorder=3)
    ax1.scatter(df.index[down_trend_mask], psar_data['DownTrend'][down_trend_mask], 
               color='red', s=20, label='PSAR (Bearish)', zorder=3)
    
    # Marcar señales de cambio de tendencia
    buy_signals = df.index[signals['buy_signal']]
    sell_signals = df.index[signals['sell_signal']]
    
    ax1.scatter(buy_signals, df.loc[buy_signals, 'Close'], 
               color='lime', marker='^', s=100, label='Buy Signal', zorder=5)
    ax1.scatter(sell_signals, df.loc[sell_signals, 'Close'], 
               color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_title(f'{title} - Price & PSAR Points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Trend Strength
    colors = ['red' if x == -1 else 'green' for x in signals['trend']]
    ax2.bar(signals.index, signals['trend_strength'], color=colors, alpha=0.7)
    ax2.set_title('Trend Strength (Days in Current Trend)')
    ax2.set_ylabel('Days')
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Price-SAR Distance (Momentum)
    ax3.plot(signals.index, signals['price_sar_distance'] * 100, 'purple', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.fill_between(signals.index, 0, signals['price_sar_distance'] * 100, 
                    where=(signals['price_sar_distance'] > 0), alpha=0.3, color='green')
    ax3.fill_between(signals.index, 0, signals['price_sar_distance'] * 100,
                    where=(signals['price_sar_distance'] < 0), alpha=0.3, color='red')
    ax3.set_title('Price-SAR Distance (Momentum %)')
    ax3.set_ylabel('Distance %')
    ax3.grid(True, alpha=0.3)
    
    # Formato fechas
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def psar_complete_example():
    """
    Ejemplo completo de análisis con Parabolic SAR
    """
    import yfinance as yf
    
    # Obtener datos
    ticker = "AAPL"
    df = yf.download(ticker, start="2023-01-01", end="2024-01-01", interval="1d")
    
    print(f"=== ANÁLISIS PARABOLIC SAR: {ticker} ===\n")
    
    # Parámetros adaptativos
    adaptive_params = adaptive_psar_parameters(df)
    print(f"📊 PARÁMETROS ADAPTATIVOS:")
    print(f"   Régimen de Volatilidad: {adaptive_params['regime']}")
    print(f"   Volatilidad Actual: {adaptive_params['volatility']:.1%}")
    print(f"   Incremento AF: {adaptive_params['increment']:.3f}")
    print(f"   Máximo AF: {adaptive_params['max_step']:.2f}")
    
    # Calcular PSAR con parámetros adaptativos
    psar = Parabolic_SAR(df, 
                        incremento=adaptive_params['increment'],
                        max_paso=adaptive_params['max_step'])
    signals = analyze_psar_signals(df, psar)
    
    # Estadísticas del período
    buy_signals_count = signals['buy_signal'].sum()
    sell_signals_count = signals['sell_signal'].sum()
    avg_trend_length = signals['trend_strength'].mean()
    
    print(f"\n📈 ESTADÍSTICAS DEL PERÍODO:")
    print(f"   Señales de Compra: {buy_signals_count}")
    print(f"   Señales de Venta: {sell_signals_count}")
    print(f"   Duración Promedio de Tendencia: {avg_trend_length:.1f} días")
    
    # Análisis actual
    current = signals.iloc[-1]
    print(f"\n🎯 ANÁLISIS ACTUAL:")
    print(f"   Precio: ${current['price']:.2f}")
    print(f"   PSAR: ${current['psar']:.2f}")
    print(f"   Tendencia: {'ALCISTA' if current['trend'] == 1 else 'BAJISTA'}")
    print(f"   Fuerza de Tendencia: {current['trend_strength']} días")
    print(f"   Distancia Precio-SAR: {current['price_sar_distance']:.2%}")
    
    if current['buy_signal']:
        print("   🟢 SEÑAL: BUY - Cambio a tendencia alcista")
    elif current['sell_signal']:
        print("   🔴 SEÑAL: SELL - Cambio a tendencia bajista")
    elif current['trend'] == 1:
        print(f"   🟢 MANTENER LONG - SAR en ${current['psar']:.2f}")
    else:
        print(f"   🔴 MANTENER SHORT - SAR en ${current['psar']:.2f}")
    
    # Multi-timeframe analysis
    mtf_analysis = multi_timeframe_psar(ticker)
    print(f"\n🔄 ANÁLISIS MULTI-TIMEFRAME:")
    print(f"   Confluencia: {mtf_analysis['confluence']}")
    print(f"   Setup: {mtf_analysis['setup']}")
    
    # Crear gráfico
    plot_psar_analysis(df, psar, signals, f"Parabolic SAR Analysis - {ticker}")
    
    return psar, signals

# Ejecutar ejemplo
if __name__ == "__main__":
    psar_complete_example()
```

## Mejores Prácticas

### ✅ **Do's (Hacer)**

1. **Usa como trailing stop**: PSAR excelente para proteger ganancias
2. **Combina con tendencia**: Confirma dirección con MA o ADX
3. **Ajusta parámetros**: Adapta AF según volatilidad del activo
4. **Filtra con volumen**: Señales más confiables con volumen confirmatorio

### ❌ **Don'ts (No Hacer)**

1. **No uses en mercados laterales**: PSAR genera muchas señales falsas
2. **No ignores el contexto**: Una señal no es suficiente para entrar
3. **No uses AF muy altos**: Puede generar whipsaws en volatilidad
4. **No trades todas las señales**: Filtra calidad según contexto

### 🎯 **Parámetros para Diferentes Estilos**

```python
PSAR_PARAMETERS = {
    'conservative': {'increment': 0.01, 'max_step': 0.15},    # Menos señales, más confiables
    'standard': {'increment': 0.02, 'max_step': 0.20},       # Parámetros clásicos
    'aggressive': {'increment': 0.03, 'max_step': 0.25},     # Más señales, mayor riesgo
    'scalping': {'increment': 0.005, 'max_step': 0.10},      # Muy sensible para scalping
    'small_caps': {'increment': 0.025, 'max_step': 0.30}     # Adaptado a volatilidad alta
}
```

## Siguiente Paso

Con Parabolic SAR dominado, continuemos con [SuperTendencia](super_tendencia.md) para análisis de tendencias avanzado.