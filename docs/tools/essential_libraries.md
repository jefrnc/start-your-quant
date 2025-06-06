# Librerías y Herramientas Esenciales

## Core Data Science Stack

### Pandas - Manipulación de Datos
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuraciones esenciales para trading
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

class PandasTradingUtils:
    """Utilidades específicas de trading con Pandas"""
    
    @staticmethod
    def resample_ohlc(df, timeframe='5T'):
        """Resample a diferentes timeframes"""
        ohlc_dict = {
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        resampled = df.resample(timeframe).agg(ohlc_dict)
        return resampled.dropna()
    
    @staticmethod
    def calculate_returns(df, price_col='close'):
        """Calcular diferentes tipos de returns"""
        returns_df = df.copy()
        
        # Simple returns
        returns_df['simple_return'] = df[price_col].pct_change()
        
        # Log returns
        returns_df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Cumulative returns
        returns_df['cumulative_return'] = (1 + returns_df['simple_return']).cumprod() - 1
        
        return returns_df
    
    @staticmethod
    def add_trading_session_flags(df):
        """Agregar flags de sesiones de trading"""
        df = df.copy()
        
        # Convertir a ET si no está ya
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
        
        # Pre-market: 4:00 AM - 9:30 AM ET
        df['is_premarket'] = (df.index.time >= pd.Timestamp('04:00').time()) & \
                            (df.index.time < pd.Timestamp('09:30').time())
        
        # Regular hours: 9:30 AM - 4:00 PM ET
        df['is_regular_hours'] = (df.index.time >= pd.Timestamp('09:30').time()) & \
                                (df.index.time < pd.Timestamp('16:00').time())
        
        # After hours: 4:00 PM - 8:00 PM ET
        df['is_afterhours'] = (df.index.time >= pd.Timestamp('16:00').time()) & \
                             (df.index.time < pd.Timestamp('20:00').time())
        
        return df
    
    @staticmethod
    def filter_trading_hours(df, session='regular'):
        """Filtrar por sesión de trading"""
        df_with_flags = PandasTradingUtils.add_trading_session_flags(df)
        
        if session == 'regular':
            return df_with_flags[df_with_flags['is_regular_hours']]
        elif session == 'extended':
            return df_with_flags[df_with_flags['is_premarket'] | 
                               df_with_flags['is_regular_hours'] | 
                               df_with_flags['is_afterhours']]
        elif session == 'premarket':
            return df_with_flags[df_with_flags['is_premarket']]
        elif session == 'afterhours':
            return df_with_flags[df_with_flags['is_afterhours']]
        
        return df_with_flags

# Ejemplo de uso
def demo_pandas_utils():
    """Demo de utilidades de Pandas"""
    
    # Crear datos de ejemplo
    dates = pd.date_range('2024-01-01 09:30:00', '2024-01-01 16:00:00', freq='1T')
    sample_data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Resample a 5 minutos
    ohlc_5min = PandasTradingUtils.resample_ohlc(sample_data, '5T')
    print(f"📊 Original: {len(sample_data)} bars, 5min: {len(ohlc_5min)} bars")
    
    # Calcular returns
    returns_data = PandasTradingUtils.calculate_returns(sample_data)
    print(f"💰 Average return: {returns_data['simple_return'].mean():.4f}")
    
    return sample_data
```

### NumPy - Computación Numérica
```python
import numpy as np
from numba import jit
import scipy.stats as stats

class NumPyTradingUtils:
    """Utilidades de NumPy optimizadas para trading"""
    
    @staticmethod
    @jit(nopython=True)
    def fast_rolling_max(arr, window):
        """Rolling max optimizado con Numba"""
        result = np.empty(len(arr))
        result[:window-1] = np.nan
        
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def fast_rolling_min(arr, window):
        """Rolling min optimizado con Numba"""
        result = np.empty(len(arr))
        result[:window-1] = np.nan
        
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def calculate_rsi_fast(prices, period=14):
        """RSI optimizado con Numba"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(gains))
        avg_losses = np.zeros(len(losses))
        
        # Initial SMA
        avg_gains[period-1] = np.mean(gains[:period])
        avg_losses[period-1] = np.mean(losses[:period])
        
        # EMA calculation
        for i in range(period, len(gains)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calcular Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        """Calcular maximum drawdown"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)
    
    @staticmethod
    def calculate_var(returns, confidence_level=0.05):
        """Calcular Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns, confidence_level=0.05):
        """Calcular Conditional Value at Risk (Expected Shortfall)"""
        var = NumPyTradingUtils.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])

# Ejemplo de uso
def demo_numpy_utils():
    """Demo de utilidades NumPy"""
    
    # Generar returns sintéticos
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 1 año de returns diarios
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Calcular métricas
    sharpe = NumPyTradingUtils.calculate_sharpe_ratio(returns)
    max_dd = NumPyTradingUtils.calculate_max_drawdown(np.cumsum(returns))
    var_5 = NumPyTradingUtils.calculate_var(returns, 0.05)
    cvar_5 = NumPyTradingUtils.calculate_cvar(returns, 0.05)
    
    print(f"📊 Métricas de Performance:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"VaR (5%): {var_5:.2%}")
    print(f"CVaR (5%): {cvar_5:.2%}")
    
    # RSI rápido
    rsi = NumPyTradingUtils.calculate_rsi_fast(prices)
    print(f"RSI actual: {rsi[-1]:.1f}")
```

## Technical Analysis Libraries

### TA-Lib - Indicadores Técnicos
```python
import talib

class TALibIntegration:
    """Integración completa con TA-Lib"""
    
    @staticmethod
    def comprehensive_analysis(df):
        """Análisis técnico comprehensivo"""
        
        # Preparar datos
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        analysis = {}
        
        # Trend Indicators
        analysis['sma_20'] = talib.SMA(close, timeperiod=20)
        analysis['ema_20'] = talib.EMA(close, timeperiod=20)
        analysis['bbands_upper'], analysis['bbands_middle'], analysis['bbands_lower'] = \
            talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # Momentum Indicators
        analysis['rsi'] = talib.RSI(close, timeperiod=14)
        analysis['macd'], analysis['macd_signal'], analysis['macd_hist'] = \
            talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        analysis['stoch_k'], analysis['stoch_d'] = \
            talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        
        # Volume Indicators
        analysis['obv'] = talib.OBV(close, volume)
        analysis['ad'] = talib.AD(high, low, close, volume)
        analysis['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        # Volatility Indicators
        analysis['atr'] = talib.ATR(high, low, close, timeperiod=14)
        analysis['natr'] = talib.NATR(high, low, close, timeperiod=14)
        
        # Pattern Recognition
        analysis['doji'] = talib.CDLDOJI(df['open'].values, high, low, close)
        analysis['hammer'] = talib.CDLHAMMER(df['open'].values, high, low, close)
        analysis['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'].values, high, low, close)
        analysis['engulfing'] = talib.CDLENGULFING(df['open'].values, high, low, close)
        
        # Cycle Indicators
        analysis['ht_trendmode'] = talib.HT_TRENDMODE(close)
        
        return analysis
    
    @staticmethod
    def get_all_patterns(df):
        """Obtener todos los patrones de candlestick"""
        
        open_prices = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        patterns = {}
        
        # Lista de todos los patrones disponibles en TA-Lib
        pattern_functions = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
            'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
            'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
            'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
            'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
            'CDLXSIDEGAP3METHODS'
        ]
        
        for pattern_name in pattern_functions:
            try:
                pattern_func = getattr(talib, pattern_name)
                patterns[pattern_name] = pattern_func(open_prices, high, low, close)
            except:
                continue
        
        return patterns
    
    @staticmethod
    def detect_active_patterns(df, min_strength=80):
        """Detectar patrones activos con fuerza mínima"""
        
        patterns = TALibIntegration.get_all_patterns(df)
        active_patterns = {}
        
        for pattern_name, pattern_values in patterns.items():
            if len(pattern_values) > 0:
                current_value = pattern_values[-1]
                if abs(current_value) >= min_strength:
                    active_patterns[pattern_name] = {
                        'strength': current_value,
                        'bullish': current_value > 0,
                        'bearish': current_value < 0
                    }
        
        return active_patterns

# Ejemplo de uso
def demo_talib_integration():
    """Demo de integración TA-Lib"""
    
    # Crear datos de ejemplo
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
    
    # Análisis comprehensivo
    analysis = TALibIntegration.comprehensive_analysis(sample_data)
    
    print("📊 Análisis Técnico:")
    print(f"RSI actual: {analysis['rsi'][-1]:.1f}")
    print(f"MACD: {analysis['macd'][-1]:.3f}")
    print(f"ATR: {analysis['atr'][-1]:.2f}")
    
    # Detectar patrones activos
    active_patterns = TALibIntegration.detect_active_patterns(sample_data)
    if active_patterns:
        print(f"\n🕯️ Patrones Activos:")
        for pattern, info in active_patterns.items():
            direction = "📈 Bullish" if info['bullish'] else "📉 Bearish"
            print(f"{pattern}: {direction} (Strength: {info['strength']})")
```

### Custom Indicators
```python
class CustomIndicators:
    """Indicadores personalizados para small cap trading"""
    
    @staticmethod
    def vwap(df):
        """Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_volume = (typical_price * df['volume']).cumsum()
        cumulative_volume = df['volume'].cumsum()
        
        return cumulative_tp_volume / cumulative_volume
    
    @staticmethod
    def anchored_vwap(df, anchor_date):
        """VWAP anclado desde fecha específica"""
        anchor_index = df.index.get_loc(anchor_date, method='nearest')
        anchor_df = df.iloc[anchor_index:]
        
        return CustomIndicators.vwap(anchor_df)
    
    @staticmethod
    def relative_volume(df, lookback_periods=20):
        """Volumen relativo vs promedio"""
        avg_volume = df['volume'].rolling(window=lookback_periods).mean()
        return df['volume'] / avg_volume
    
    @staticmethod
    def gap_percentage(df):
        """Porcentaje de gap vs cierre anterior"""
        prev_close = df['close'].shift(1)
        gap_pct = (df['open'] - prev_close) / prev_close
        return gap_pct
    
    @staticmethod
    def daily_range_percentage(df):
        """Rango diario como porcentaje"""
        return (df['high'] - df['low']) / df['low']
    
    @staticmethod
    def money_flow_index(df, period=14):
        """Money Flow Index - RSI basado en volumen"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and negative money flow
        positive_flow = pd.Series(index=df.index, dtype=float)
        negative_flow = pd.Series(index=df.index, dtype=float)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
                negative_flow.iloc[i] = 0
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
                positive_flow.iloc[i] = 0
            else:
                positive_flow.iloc[i] = 0
                negative_flow.iloc[i] = 0
        
        # Calculate MFI
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    def squeeze_indicator(df, bb_period=20, kc_period=20, kc_mult=1.5):
        """Bollinger Bands / Keltner Channel Squeeze"""
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Keltner Channels
        kc_middle = df['close'].rolling(window=kc_period).mean()
        atr = CustomIndicators.true_range(df).rolling(window=kc_period).mean()
        kc_upper = kc_middle + (atr * kc_mult)
        kc_lower = kc_middle - (atr * kc_mult)
        
        # Squeeze condition
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        return squeeze
    
    @staticmethod
    def true_range(df):
        """True Range calculation"""
        prev_close = df['close'].shift(1)
        
        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - prev_close)
        tr3 = np.abs(df['low'] - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    @staticmethod
    def consecutive_bars(df, direction='up'):
        """Contar barras consecutivas en una dirección"""
        if direction == 'up':
            condition = df['close'] > df['open']
        else:
            condition = df['close'] < df['open']
        
        # Reset counter when condition changes
        groups = condition.ne(condition.shift()).cumsum()
        consecutive = condition.groupby(groups).cumsum()
        
        # Only count when condition is true
        consecutive = consecutive.where(condition, 0)
        
        return consecutive
    
    @staticmethod
    def pivot_points(df, method='traditional'):
        """Calcular pivot points"""
        
        if method == 'traditional':
            pivot = (df['high'] + df['low'] + df['close']) / 3
            
            resistance_1 = 2 * pivot - df['low']
            support_1 = 2 * pivot - df['high']
            
            resistance_2 = pivot + (df['high'] - df['low'])
            support_2 = pivot - (df['high'] - df['low'])
            
            resistance_3 = df['high'] + 2 * (pivot - df['low'])
            support_3 = df['low'] - 2 * (df['high'] - pivot)
            
            return {
                'pivot': pivot,
                'r1': resistance_1, 'r2': resistance_2, 'r3': resistance_3,
                's1': support_1, 's2': support_2, 's3': support_3
            }
        
        elif method == 'fibonacci':
            pivot = (df['high'] + df['low'] + df['close']) / 3
            range_hl = df['high'] - df['low']
            
            resistance_1 = pivot + 0.382 * range_hl
            resistance_2 = pivot + 0.618 * range_hl
            resistance_3 = pivot + range_hl
            
            support_1 = pivot - 0.382 * range_hl
            support_2 = pivot - 0.618 * range_hl
            support_3 = pivot - range_hl
            
            return {
                'pivot': pivot,
                'r1': resistance_1, 'r2': resistance_2, 'r3': resistance_3,
                's1': support_1, 's2': support_2, 's3': support_3
            }

# Demo de indicadores personalizados
def demo_custom_indicators():
    """Demo de indicadores personalizados"""
    
    # Datos de ejemplo
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.randn(50).cumsum() + 100,
        'high': np.random.randn(50).cumsum() + 102,
        'low': np.random.randn(50).cumsum() + 98,
        'close': np.random.randn(50).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, 50)
    }, index=dates)
    
    # Calcular indicadores
    vwap = CustomIndicators.vwap(sample_data)
    rvol = CustomIndicators.relative_volume(sample_data)
    gap_pct = CustomIndicators.gap_percentage(sample_data)
    mfi = CustomIndicators.money_flow_index(sample_data)
    squeeze = CustomIndicators.squeeze_indicator(sample_data)
    
    print("🔧 Indicadores Personalizados:")
    print(f"VWAP actual: ${vwap.iloc[-1]:.2f}")
    print(f"Relative Volume: {rvol.iloc[-1]:.2f}x")
    print(f"Gap %: {gap_pct.iloc[-1]:.2%}")
    print(f"MFI: {mfi.iloc[-1]:.1f}")
    print(f"En squeeze: {'Sí' if squeeze.iloc[-1] else 'No'}")
```

## Visualization Libraries

### Matplotlib & Seaborn
```python
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Configuración de estilo
plt.style.use('dark_background')
sns.set_palette("husl")

class TradingVisualization:
    """Visualizaciones específicas para trading"""
    
    @staticmethod
    def setup_plot_style():
        """Configurar estilo de plots"""
        plt.rcParams['figure.figsize'] = (15, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    @staticmethod
    def candlestick_chart(df, indicators=None, title=""):
        """Gráfico de candlestick con indicadores"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Candlestick chart
        for i in range(len(df)):
            date = df.index[i]
            open_price = df['open'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i] 
            close_price = df['close'].iloc[i]
            
            color = 'green' if close_price > open_price else 'red'
            
            # Wick
            ax1.plot([i, i], [low_price, high_price], color='gray', linewidth=1)
            
            # Body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, alpha=0.7)
            ax1.add_patch(rect)
        
        # Add indicators if provided
        if indicators:
            if 'sma_20' in indicators:
                ax1.plot(range(len(df)), indicators['sma_20'], 
                        label='SMA 20', alpha=0.7)
            if 'vwap' in indicators:
                ax1.plot(range(len(df)), indicators['vwap'], 
                        label='VWAP', alpha=0.7)
        
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Volume chart
        colors = ['green' if df['close'].iloc[i] > df['open'].iloc[i] 
                 else 'red' for i in range(len(df))]
        ax2.bar(range(len(df)), df['volume'], color=colors, alpha=0.7)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time')
        
        # Format x-axis
        if len(df) < 50:
            step = 1
        elif len(df) < 200:
            step = 5
        else:
            step = len(df) // 20
            
        ax1.set_xticks(range(0, len(df), step))
        ax1.set_xticklabels([df.index[i].strftime('%m/%d') 
                            for i in range(0, len(df), step)], rotation=45)
        
        ax2.set_xticks(range(0, len(df), step))
        ax2.set_xticklabels([df.index[i].strftime('%m/%d') 
                            for i in range(0, len(df), step)], rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def performance_dashboard(returns, title="Performance Dashboard"):
        """Dashboard de performance comprehensivo"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Cumulative returns
        ax1 = plt.subplot(2, 3, 1)
        cumulative_returns = (1 + returns).cumprod()
        ax1.plot(cumulative_returns.index, cumulative_returns.values)
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        
        # Returns distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {returns.mean():.4f}')
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Daily Return')
        ax2.legend()
        
        # Rolling Sharpe ratio
        ax3 = plt.subplot(2, 3, 3)
        rolling_sharpe = returns.rolling(60).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std()
        )
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax3.set_title('Rolling 60-Day Sharpe Ratio')
        ax3.set_ylabel('Sharpe Ratio')
        
        # Drawdown
        ax4 = plt.subplot(2, 3, 4)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ax4.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color='red')
        ax4.set_title('Drawdown')
        ax4.set_ylabel('Drawdown %')
        
        # Monthly returns heatmap
        ax5 = plt.subplot(2, 3, 5)
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pct = monthly_returns * 100
        
        # Create heatmap data
        years = monthly_returns.index.year.unique()
        months = range(1, 13)
        heatmap_data = np.full((len(years), 12), np.nan)
        
        for i, year in enumerate(years):
            year_data = monthly_returns_pct[monthly_returns_pct.index.year == year]
            for month_ret in year_data:
                month = year_data[year_data == month_ret].index[0].month
                heatmap_data[i, month-1] = month_ret
        
        sns.heatmap(heatmap_data, 
                   xticklabels=['Jan','Feb','Mar','Apr','May','Jun',
                               'Jul','Aug','Sep','Oct','Nov','Dec'],
                   yticklabels=years,
                   annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   ax=ax5)
        ax5.set_title('Monthly Returns Heatmap (%)')
        
        # Risk/Return scatter
        ax6 = plt.subplot(2, 3, 6)
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        ax6.scatter(annual_vol, annual_return, s=100, alpha=0.7)
        ax6.set_xlabel('Annual Volatility')
        ax6.set_ylabel('Annual Return')
        ax6.set_title('Risk/Return Profile')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig

# Demo de visualizaciones
def demo_trading_visualization():
    """Demo de visualizaciones de trading"""
    
    # Configurar estilo
    TradingVisualization.setup_plot_style()
    
    # Datos de ejemplo
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.randn(60).cumsum() + 100,
        'high': np.random.randn(60).cumsum() + 102,
        'low': np.random.randn(60).cumsum() + 98,
        'close': np.random.randn(60).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, 60)
    }, index=dates)
    
    # Indicadores
    indicators = {
        'sma_20': sample_data['close'].rolling(20).mean(),
        'vwap': CustomIndicators.vwap(sample_data)
    }
    
    # Gráfico de candlestick
    fig1 = TradingVisualization.candlestick_chart(
        sample_data, indicators, "Sample Stock - Candlestick Chart"
    )
    plt.show()
    
    # Dashboard de performance
    returns = sample_data['close'].pct_change().dropna()
    fig2 = TradingVisualization.performance_dashboard(
        returns, "Sample Strategy Performance"
    )
    plt.show()
```

Estas librerías forman el core stack para análisis cuantitativo y visualización de datos de trading, optimizadas específicamente para small cap trading strategies.