# Bollinger Bands - El Oscilador de Volatilidad

## Definición

Las Bandas de Bollinger son una herramienta de análisis técnico para generar señales de sobrecompra o sobreventa. Están compuestas por tres líneas: una media móvil simple (banda media) y una banda superior e inferior a +/- 2 desviaciones estándar.

## Filosofía del Indicador

### ¿Por Qué Funcionan?
- **Reversión a la Media**: Los precios tienden a regresar al promedio tras desviaciones extremas
- **Medición de Volatilidad**: Las bandas se expanden/contraen según la volatilidad del mercado
- **Niveles Dinámicos**: A diferencia de soportes/resistencias fijos, se adaptan al precio

### Componentes
```
Banda Superior = MA + (Desviación_Estándar × Factor)
Banda Media    = Media Móvil Simple
Banda Inferior = MA - (Desviación_Estándar × Factor)
```

**Parámetros estándar**:
- **Período**: 20 (media móvil)
- **Factor**: 2.0 (desviaciones estándar)

## Implementación de Referencia

```python
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def Bollinger_Bands(df: pd.DataFrame, longitud: int = 20, desviacion_std: float = 2.0, columna: str = "Close") -> pd.DataFrame:
    """
    Bandas de Bollinger - Implementación de referencia exacta
    
    Las Bandas de Bollinger identifican niveles de sobrecompra/sobreventa mediante
    la desviación del precio respecto a su media móvil.
    
    Parámetros
    ----------
    df : pd.DataFrame
        Datos históricos del activo (debe incluir columna Close)
    longitud : int, default 20
        Ventana para el cálculo de la media móvil y desviación estándar
    desviacion_std : float, default 2.0
        Número de desviaciones estándar para las bandas superior e inferior
    columna : str, default "Close"
        Columna a utilizar en el cálculo
    
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: BB_Up, MA, BB_Lw
        
    Ejemplo de Uso
    --------------
    >>> import yfinance as yf
    >>> df = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
    >>> bb = Bollinger_Bands(df, longitud=20, desviacion_std=2.0)
    >>> print(bb.head())
    """
    # Calcular usando copia para no modificar original
    datos = deepcopy(df)
    
    # Rolling window para media y desviación estándar
    rolling = datos[columna].rolling(window=longitud)
    
    # Media móvil (banda media)
    datos["MA"] = rolling.mean()
    
    # Desviación estándar con ddof=0 (población completa)
    std_bandas = desviacion_std * rolling.std(ddof=0)
    
    # Bandas superior e inferior
    datos["BB_Up"] = datos["MA"] + std_bandas
    datos["BB_Lw"] = datos["MA"] - std_bandas
    
    return datos[["BB_Up", "MA", "BB_Lw"]]

def calculate_bollinger_signals(df: pd.DataFrame, bb_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generar señales de trading usando Bollinger Bands
    
    Parámetros
    ----------
    df : pd.DataFrame
        Datos históricos con precio
    bb_data : pd.DataFrame
        Datos de Bollinger Bands (output de Bollinger_Bands)
    
    Returns
    -------
    pd.DataFrame
        DataFrame con señales y estadísticas
    """
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['bb_upper'] = bb_data['BB_Up']
    signals['bb_middle'] = bb_data['MA']
    signals['bb_lower'] = bb_data['BB_Lw']
    
    # Calcular posición relativa dentro de las bandas
    bb_width = bb_data['BB_Up'] - bb_data['BB_Lw']
    signals['bb_position'] = (df['Close'] - bb_data['BB_Lw']) / bb_width
    
    # Señales básicas
    signals['oversold'] = df['Close'] < bb_data['BB_Lw']  # Precio bajo banda inferior
    signals['overbought'] = df['Close'] > bb_data['BB_Up']  # Precio sobre banda superior
    signals['middle_cross'] = np.where(
        (df['Close'] > bb_data['MA']) & (df['Close'].shift(1) <= bb_data['MA'].shift(1)), 1,
        np.where((df['Close'] < bb_data['MA']) & (df['Close'].shift(1) >= bb_data['MA'].shift(1)), -1, 0)
    )
    
    # Squeeze detection (bandas contraídas = baja volatilidad)
    signals['bb_width'] = bb_width
    signals['bb_squeeze'] = bb_width < bb_width.rolling(20).mean() * 0.8
    
    # Breakout signals
    signals['upper_breakout'] = (df['Close'] > bb_data['BB_Up']) & (df['Close'].shift(1) <= bb_data['BB_Up'].shift(1))
    signals['lower_breakout'] = (df['Close'] < bb_data['BB_Lw']) & (df['Close'].shift(1) >= bb_data['BB_Lw'].shift(1))
    
    return signals
```

## Estrategias de Trading con Bollinger Bands

### 1. Mean Reversion Strategy
```python
def bollinger_mean_reversion(df: pd.DataFrame, bb_period: int = 20, std_factor: float = 2.0):
    """
    Estrategia de reversión a la media usando Bollinger Bands
    """
    # Calcular bandas
    bb = Bollinger_Bands(df, longitud=bb_period, desviacion_std=std_factor)
    signals = calculate_bollinger_signals(df, bb)
    
    # Señales de entrada
    entry_signals = pd.Series(0, index=df.index)
    
    # Long cuando precio toca banda inferior (oversold)
    long_entry = (
        signals['oversold'] & 
        (df['Volume'] > df['Volume'].rolling(20).mean()) &  # Confirmar con volumen
        (df['Close'] > df['Close'].shift(1))  # Precio empezando a recuperar
    )
    
    # Short cuando precio toca banda superior (overbought)
    short_entry = (
        signals['overbought'] &
        (df['Volume'] > df['Volume'].rolling(20).mean()) &
        (df['Close'] < df['Close'].shift(1))  # Precio empezando a caer
    )
    
    entry_signals[long_entry] = 1
    entry_signals[short_entry] = -1
    
    return {
        'signals': entry_signals,
        'bb_data': bb,
        'analysis': signals,
        'strategy_type': 'mean_reversion'
    }

def bollinger_breakout_strategy(df: pd.DataFrame, bb_period: int = 20, std_factor: float = 2.0):
    """
    Estrategia de breakout usando Bollinger Bands
    """
    bb = Bollinger_Bands(df, longitud=bb_period, desviacion_std=std_factor)
    signals = calculate_bollinger_signals(df, bb)
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Long en breakout alcista tras squeeze
    long_breakout = (
        signals['upper_breakout'] &
        signals['bb_squeeze'].shift(1) &  # Había squeeze previo
        (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5)  # Volumen fuerte
    )
    
    # Short en breakout bajista tras squeeze
    short_breakout = (
        signals['lower_breakout'] &
        signals['bb_squeeze'].shift(1) &
        (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5)
    )
    
    entry_signals[long_breakout] = 1
    entry_signals[short_breakout] = -1
    
    return {
        'signals': entry_signals,
        'bb_data': bb,
        'analysis': signals,
        'strategy_type': 'breakout'
    }
```

### 2. Small Caps Specific Strategy
```python
def small_cap_bollinger_strategy(df: pd.DataFrame, volume_data: pd.DataFrame = None):
    """
    Estrategia específica para small caps usando Bollinger Bands
    """
    # Parámetros ajustados para small caps (mayor volatilidad)
    bb = Bollinger_Bands(df, longitud=15, desviacion_std=2.5)  # Bandas más amplias
    signals = calculate_bollinger_signals(df, bb)
    
    # Agregar filtros específicos para small caps
    if volume_data is not None:
        # RVOL filter
        avg_volume = df['Volume'].rolling(20).mean()
        rvol = df['Volume'] / avg_volume
        
        # Gap detection
        gap_up = (df['Open'] / df['Close'].shift(1)) > 1.02  # 2% gap up
        gap_down = (df['Open'] / df['Close'].shift(1)) < 0.98  # 2% gap down
        
        signals['rvol'] = rvol
        signals['gap_up'] = gap_up
        signals['gap_down'] = gap_down
    
    entry_signals = pd.Series(0, index=df.index)
    
    # Long setup: Oversold + High RVOL + Gap fill potential
    long_setup = (
        signals['oversold'] &
        (signals.get('rvol', 1) > 2) &  # High volume
        signals.get('gap_down', False)  # Gap down being filled
    )
    
    # Short setup: Overbought + High RVOL + Gap fade
    short_setup = (
        signals['overbought'] &
        (signals.get('rvol', 1) > 2) &
        signals.get('gap_up', False)  # Gap up fading
    )
    
    entry_signals[long_setup] = 1
    entry_signals[short_setup] = -1
    
    return {
        'signals': entry_signals,
        'bb_data': bb,
        'analysis': signals,
        'strategy_type': 'small_cap_specialized'
    }
```

## Interpretación y Uso

### Señales Principales

1. **Oversold (Sobreventa)**
   - Precio toca o cruza banda inferior
   - Potencial reversión alcista
   - ⚠️ Confirmar con volumen y momentum

2. **Overbought (Sobrecompra)**
   - Precio toca o cruza banda superior  
   - Potencial reversión bajista
   - ⚠️ En tendencia fuerte puede extenderse

3. **Squeeze (Compresión)**
   - Bandas muy juntas = baja volatilidad
   - Precede movimientos explosivos
   - 🎯 Setup ideal para breakouts

### Combinaciones Poderosas

```python
def bollinger_multi_timeframe(symbol: str, primary_tf: str = '1d', confirmation_tf: str = '1h'):
    """
    Análisis multi-timeframe con Bollinger Bands
    """
    import yfinance as yf
    
    # Datos en múltiples timeframes
    df_daily = yf.download(symbol, period="6mo", interval=primary_tf)
    df_hourly = yf.download(symbol, period="1mo", interval=confirmation_tf)
    
    # Bollinger en cada timeframe
    bb_daily = Bollinger_Bands(df_daily, longitud=20, desviacion_std=2.0)
    bb_hourly = Bollinger_Bands(df_hourly, longitud=20, desviacion_std=2.0)
    
    # Señales combinadas
    daily_signals = calculate_bollinger_signals(df_daily, bb_daily)
    hourly_signals = calculate_bollinger_signals(df_hourly, bb_hourly)
    
    # Setup multi-timeframe
    current_daily = daily_signals.iloc[-1]
    current_hourly = hourly_signals.iloc[-1]
    
    analysis = {
        'daily_position': current_daily['bb_position'],
        'hourly_position': current_hourly['bb_position'],
        'daily_squeeze': current_daily['bb_squeeze'],
        'hourly_squeeze': current_hourly['bb_squeeze'],
        'confluence': None
    }
    
    # Detectar confluencia
    if current_daily['oversold'] and current_hourly['oversold']:
        analysis['confluence'] = 'STRONG_OVERSOLD'
    elif current_daily['overbought'] and current_hourly['overbought']:
        analysis['confluence'] = 'STRONG_OVERBOUGHT'
    elif current_daily['bb_squeeze'] and current_hourly['bb_squeeze']:
        analysis['confluence'] = 'MULTI_TF_SQUEEZE'
    
    return analysis
```

## Visualización Avanzada

```python
def plot_bollinger_analysis(df: pd.DataFrame, bb_data: pd.DataFrame, signals: pd.DataFrame, title: str = "Bollinger Bands Analysis"):
    """
    Crear gráfico completo de análisis Bollinger
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), 
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Chart 1: Precio + Bollinger Bands
    ax1.plot(df.index, df['Close'], 'k-', linewidth=2, label='Price')
    ax1.plot(bb_data.index, bb_data['BB_Up'], 'r--', label='Upper Band')
    ax1.plot(bb_data.index, bb_data['MA'], 'b-', label='Middle Band (MA)')
    ax1.plot(bb_data.index, bb_data['BB_Lw'], 'g--', label='Lower Band')
    ax1.fill_between(bb_data.index, bb_data['BB_Up'], bb_data['BB_Lw'], alpha=0.1, color='gray')
    
    # Marcar señales
    oversold_points = df.index[signals['oversold']]
    overbought_points = df.index[signals['overbought']]
    
    ax1.scatter(oversold_points, df.loc[oversold_points, 'Close'], 
               color='green', marker='^', s=100, label='Oversold', zorder=5)
    ax1.scatter(overbought_points, df.loc[overbought_points, 'Close'], 
               color='red', marker='v', s=100, label='Overbought', zorder=5)
    
    ax1.set_title(f'{title} - Price & Bollinger Bands')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: BB Position (posición dentro de las bandas)
    ax2.plot(signals.index, signals['bb_position'], 'purple', linewidth=2)
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Lower Band')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Upper Band')
    ax2.axhline(y=0.5, color='blue', linestyle='-', alpha=0.7, label='Middle')
    ax2.fill_between(signals.index, 0, 0.2, alpha=0.2, color='green', label='Oversold Zone')
    ax2.fill_between(signals.index, 0.8, 1, alpha=0.2, color='red', label='Overbought Zone')
    ax2.set_title('BB Position (0=Lower Band, 1=Upper Band)')
    ax2.set_ylabel('Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: BB Width (volatilidad)
    bb_width_norm = signals['bb_width'] / signals['bb_width'].rolling(50).mean()
    ax3.plot(signals.index, bb_width_norm, 'orange', linewidth=2)
    ax3.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Average')
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Squeeze Threshold')
    ax3.fill_between(signals.index, 0, 0.8, where=(bb_width_norm < 0.8), 
                    alpha=0.3, color='yellow', label='Squeeze Zone')
    ax3.set_title('BB Width (Normalized) - Volatility Measure')
    ax3.set_ylabel('Width Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Formato de fechas
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso completo
def bollinger_complete_example():
    """
    Ejemplo completo de análisis con Bollinger Bands
    """
    import yfinance as yf
    
    # Obtener datos
    ticker = "AAPL"
    df = yf.download(ticker, start="2023-01-01", end="2024-01-01", interval="1d")
    
    print(f"=== ANÁLISIS BOLLINGER BANDS: {ticker} ===\n")
    
    # Calcular Bollinger Bands
    bb = Bollinger_Bands(df, longitud=20, desviacion_std=2.0)
    signals = calculate_bollinger_signals(df, bb)
    
    # Estadísticas
    oversold_count = signals['oversold'].sum()
    overbought_count = signals['overbought'].sum()
    squeeze_count = signals['bb_squeeze'].sum()
    
    print(f"📊 ESTADÍSTICAS DEL PERÍODO:")
    print(f"   Señales Oversold: {oversold_count}")
    print(f"   Señales Overbought: {overbought_count}")
    print(f"   Días en Squeeze: {squeeze_count}")
    print(f"   % Tiempo en Squeeze: {squeeze_count/len(signals)*100:.1f}%")
    
    # Análisis actual
    current = signals.iloc[-1]
    print(f"\n🎯 ANÁLISIS ACTUAL:")
    print(f"   Precio: ${current['price']:.2f}")
    print(f"   Banda Superior: ${current['bb_upper']:.2f}")
    print(f"   Banda Media: ${current['bb_middle']:.2f}")
    print(f"   Banda Inferior: ${current['bb_lower']:.2f}")
    print(f"   Posición BB: {current['bb_position']:.2f} (0=Inferior, 1=Superior)")
    
    if current['bb_position'] < 0.2:
        print("   🟢 Estado: OVERSOLD - Posible rebote")
    elif current['bb_position'] > 0.8:
        print("   🔴 Estado: OVERBOUGHT - Posible corrección")
    elif current['bb_squeeze']:
        print("   🟡 Estado: SQUEEZE - Preparándose para movimiento")
    else:
        print("   ⚪ Estado: NEUTRAL")
    
    # Crear gráfico
    plot_bollinger_analysis(df, bb, signals, f"Bollinger Analysis - {ticker}")
    
    return bb, signals

# Ejecutar ejemplo si se ejecuta directamente
if __name__ == "__main__":
    bollinger_complete_example()
```

## Mejores Prácticas

### ✅ **Do's (Hacer)**

1. **Confirma con volumen**: Señales más confiables con volumen elevado
2. **Usa múltiples timeframes**: Confluencia aumenta probabilidad
3. **Aprovecha squeeze**: Baja volatilidad precede alta volatilidad
4. **Combina con trend**: En tendencia fuerte, oversold/overbought menos confiables

### ❌ **Don'ts (No Hacer)**

1. **No trades solo en touch**: Espera confirmación de reversión
2. **No ignores el contexto**: Una banda no es soporte/resistencia absoluta
3. **No uses parámetros fijos**: Ajusta según volatilidad del activo
4. **No trades contra tendencia fuerte**: BB funciona mejor en rangos

### 🎯 **Parámetros para Small Caps**

```python
SMALL_CAP_BB_PARAMS = {
    'gap_and_go': {'period': 15, 'std': 2.5},      # Más ancho para gaps
    'momentum': {'period': 20, 'std': 2.0},        # Estándar
    'reversal': {'period': 25, 'std': 1.8},        # Más sensible
    'breakout': {'period': 10, 'std': 3.0}         # Muy ancho para evitar whipsaws
}
```

## Siguiente Paso

Con Bollinger Bands dominado, continuemos con [Parabolic SAR](parabolic_sar.md) para identificar cambios de tendencia.