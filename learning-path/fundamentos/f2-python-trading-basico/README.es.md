> 🇺🇸 [Read in English](README.md) | 🇪🇸 **Español**

# F2: Python Trading Básico 🐍

**Módulo Fundamental 2 - Duración: 3-4 horas**

## 🎯 Objetivos del Módulo

Al completar este módulo podrás:
- ✅ Instalar y configurar tu entorno de trading
- ✅ Descargar datos reales de cualquier acción
- ✅ Crear gráficos profesionales de precios
- ✅ Calcular indicadores técnicos básicos
- ✅ Escribir tu primer script de análisis

## 🛠️ Setup Completo (Una Sola Vez)

### Paso 1: Verificar Python
```bash
# Abrir terminal/cmd y verificar versión
python --version
# Debe mostrar: Python 3.8 o superior

# Si no tienes Python:
# Windows: Descargar de python.org
# Mac: brew install python
# Linux: sudo apt install python3 python3-pip
```

### Paso 2: Instalar Librerías Esenciales
```bash
# Instalar todas las librerías de una vez
pip install yfinance pandas matplotlib seaborn numpy jupyter

# Verificar instalación
python -c "import yfinance, pandas, matplotlib; print('✅ Todo instalado correctamente!')"
```

### Paso 3: Configurar Entorno
```bash
# Crear carpeta para tus proyectos
mkdir my-quant-trading
cd my-quant-trading

# Opcional: crear entorno virtual
python -m venv quant_env
# Activar: quant_env\\Scripts\\activate (Windows) o source quant_env/bin/activate (Mac/Linux)
```

## 📊 Tu Primera Descarga de Datos

### Script 1: Datos Básicos

```python
# archivo: descarga_basica.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def descargar_accion(symbol, periodo='1y'):
    """
    Descarga datos de una acción de Yahoo Finance

    Args:
        symbol (str): Símbolo de la acción (ej: 'AAPL', 'MSFT')
        periodo (str): Periodo de datos ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

    Returns:
        DataFrame: Datos OHLCV de la acción
    """
    try:
        # Descargar datos
        stock = yf.Ticker(symbol)
        data = stock.history(period=periodo)

        if data.empty:
            print(f"❌ No se encontraron datos para {symbol}")
            return None

        print(f"✅ Descargados {len(data)} días de datos para {symbol}")
        return data

    except Exception as e:
        print(f"❌ Error descargando {symbol}: {e}")
        return None

def mostrar_info_basica(data, symbol):
    """Muestra información básica de los datos"""

    print(f"\\n📊 INFORMACIÓN BÁSICA DE {symbol}")
    print("=" * 40)

    # Información general
    print(f"Período: {data.index[0].date()} a {data.index[-1].date()}")
    print(f"Total de días: {len(data)}")

    # Precios
    precio_actual = data['Close'].iloc[-1]
    precio_inicial = data['Close'].iloc[0]
    cambio_total = (precio_actual / precio_inicial - 1) * 100

    print(f"\\n💰 PRECIOS:")
    print(f"  Precio inicial: ${precio_inicial:.2f}")
    print(f"  Precio actual: ${precio_actual:.2f}")
    print(f"  Cambio total: {cambio_total:+.2f}%")

    # Estadísticas
    print(f"\\n📈 ESTADÍSTICAS:")
    print(f"  Precio máximo: ${data['High'].max():.2f}")
    print(f"  Precio mínimo: ${data['Low'].min():.2f}")
    print(f"  Volumen promedio: {data['Volume'].mean():,.0f}")

    # Volatilidad
    rendimientos_diarios = data['Close'].pct_change().dropna()
    volatilidad_diaria = rendimientos_diarios.std() * 100
    volatilidad_anual = volatilidad_diaria * (252 ** 0.5)  # 252 días de trading al año

    print(f"  Volatilidad diaria: {volatilidad_diaria:.2f}%")
    print(f"  Volatilidad anual: {volatilidad_anual:.2f}%")

# Probar con Apple
if __name__ == "__main__":
    symbol = 'AAPL'
    data = descargar_accion(symbol, '1y')

    if data is not None:
        mostrar_info_basica(data, symbol)
```

**🎯 Ejercicio 2.1**: Ejecuta este script y luego:
1. Cambia el símbolo a 'TSLA' y compara las estadísticas
2. Prueba con diferentes períodos ('6mo', '2y')
3. Anota qué acción es más volátil

## 📈 Crear Gráficos Profesionales

### Script 2: Visualizaciones

```python
# archivo: graficos_trading.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def grafico_precio_basico(data, symbol):
    """Crea gráfico básico de precio"""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Graficar precio de cierre
    ax.plot(data.index, data['Close'], linewidth=2, label=f'{symbol} Close Price')

    # Personalizar
    ax.set_title(f'{symbol} - Precio de Cierre', fontsize=16, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Precio ($)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Formato de fechas en eje X
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def grafico_candlestick_simple(data, symbol, dias=60):
    """Crea gráfico tipo candlestick simplificado"""

    # Solo mostrar últimos X días
    data_reciente = data.tail(dias)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Gráfico de precios (simulando candlesticks con líneas)
    for i, (date, row) in enumerate(data_reciente.iterrows()):
        color = 'green' if row['Close'] > row['Open'] else 'red'

        # Línea high-low
        ax1.plot([date, date], [row['Low'], row['High']], color='black', linewidth=0.5)

        # "Cuerpo" de la vela
        ax1.plot([date, date], [row['Open'], row['Close']], color=color, linewidth=3)

    ax1.set_title(f'{symbol} - Candlestick (Últimos {dias} días)', fontsize=16)
    ax1.set_ylabel('Precio ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Gráfico de volumen
    colors = ['green' if close > open else 'red'
              for close, open in zip(data_reciente['Close'], data_reciente['Open'])]

    ax2.bar(data_reciente.index, data_reciente['Volume'], color=colors, alpha=0.7)
    ax2.set_ylabel('Volumen', fontsize=12)
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Formato de fechas
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def grafico_comparacion_multiple(symbols, periodo='6mo'):
    """Compara múltiples acciones normalizadas"""

    fig, ax = plt.subplots(figsize=(12, 8))

    for symbol in symbols:
        try:
            data = yf.download(symbol, period=periodo)
            if not data.empty:
                # Normalizar a 100 al inicio
                precio_normalizado = (data['Close'] / data['Close'].iloc[0]) * 100
                ax.plot(precio_normalizado.index, precio_normalizado,
                       linewidth=2, label=symbol)
        except:
            print(f"Error con {symbol}")

    ax.set_title('Comparación de Rendimientos (Base 100)', fontsize=16)
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Valor Normalizado', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def heatmap_correlaciones(symbols, periodo='1y'):
    """Crea heatmap de correlaciones entre acciones"""

    # Descargar datos de todos los símbolos
    portfolio_data = {}

    for symbol in symbols:
        try:
            data = yf.download(symbol, period=periodo)
            if not data.empty:
                portfolio_data[symbol] = data['Close'].pct_change().dropna()
        except:
            print(f"Error descargando {symbol}")

    # Crear DataFrame de rendimientos
    df_rendimientos = pd.DataFrame(portfolio_data)

    # Calcular matriz de correlación
    correlacion = df_rendimientos.corr()

    # Crear heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacion, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})

    plt.title('Matriz de Correlación de Rendimientos', fontsize=16)
    plt.tight_layout()
    plt.show()

    return correlacion

# Probar visualizaciones
if __name__ == "__main__":
    # Descargar datos
    data = yf.download('AAPL', period='1y')

    # Crear gráficos
    print("Creando gráfico básico...")
    grafico_precio_basico(data, 'AAPL')

    print("Creando candlestick...")
    grafico_candlestick_simple(data, 'AAPL')

    print("Comparando múltiples acciones...")
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    grafico_comparacion_multiple(tech_stocks)

    print("Creando heatmap de correlaciones...")
    correlaciones = heatmap_correlaciones(tech_stocks)
    print("\\nCorrelaciones:")
    print(correlaciones.round(2))
```

**🎯 Ejercicio 2.2**:
1. Ejecuta todos los gráficos
2. Cambia los símbolos por acciones que te interesen
3. ¿Qué observas en las correlaciones?

## 🔢 Indicadores Técnicos Básicos

### Script 3: Indicadores Esenciales

```python
# archivo: indicadores_basicos.py

def calcular_medias_moviles(data, periodos=[20, 50, 200]):
    """Calcula medias móviles simples"""

    for periodo in periodos:
        columna = f'SMA_{periodo}'
        data[columna] = data['Close'].rolling(window=periodo).mean()

    return data

def calcular_rsi(data, periodo=14):
    """Calcula Relative Strength Index"""

    delta = data['Close'].diff()

    # Separar ganancias y pérdidas
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)

    # Calcular medias móviles de ganancias y pérdidas
    avg_ganancia = ganancia.rolling(window=periodo).mean()
    avg_perdida = perdida.rolling(window=periodo).mean()

    # RSI
    rs = avg_ganancia / avg_perdida
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calcular_bandas_bollinger(data, periodo=20, std_dev=2):
    """Calcula Bandas de Bollinger"""

    sma = data['Close'].rolling(window=periodo).mean()
    std = data['Close'].rolling(window=periodo).std()

    data['BB_Superior'] = sma + (std * std_dev)
    data['BB_Inferior'] = sma - (std * std_dev)
    data['BB_Media'] = sma

    # Posición relativa dentro de las bandas
    data['BB_Posicion'] = (data['Close'] - data['BB_Inferior']) / (data['BB_Superior'] - data['BB_Inferior'])

    return data

def calcular_macd(data, rapido=12, lento=26, señal=9):
    """Calcula MACD (Moving Average Convergence Divergence)"""

    ema_rapido = data['Close'].ewm(span=rapido).mean()
    ema_lento = data['Close'].ewm(span=lento).mean()

    data['MACD'] = ema_rapido - ema_lento
    data['MACD_Signal'] = data['MACD'].ewm(span=señal).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

    return data

def calcular_volatilidad(data, periodo=20):
    """Calcula volatilidad realizada"""

    rendimientos = data['Close'].pct_change()
    data['Volatilidad'] = rendimientos.rolling(window=periodo).std() * np.sqrt(252) * 100

    return data

def analisis_tecnico_completo(symbol, periodo='1y'):
    """Análisis técnico completo de una acción"""

    # Descargar datos
    data = yf.download(symbol, period=periodo)

    if data.empty:
        print(f"No se encontraron datos para {symbol}")
        return None

    # Calcular todos los indicadores
    data = calcular_medias_moviles(data)
    data['RSI'] = calcular_rsi(data)
    data = calcular_bandas_bollinger(data)
    data = calcular_macd(data)
    data = calcular_volatilidad(data)

    # Crear dashboard de gráficos
    crear_dashboard_tecnico(data, symbol)

    # Análisis actual
    analizar_situacion_actual(data, symbol)

    return data

def crear_dashboard_tecnico(data, symbol):
    """Crea dashboard con múltiples indicadores"""

    fig, axes = plt.subplots(4, 1, figsize=(15, 16))

    # 1. Precio con medias móviles y Bollinger
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='Precio', linewidth=2)
    ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
    ax1.fill_between(data.index, data['BB_Superior'], data['BB_Inferior'],
                     alpha=0.2, label='Bandas Bollinger')
    ax1.set_title(f'{symbol} - Precio y Medias Móviles', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. RSI
    ax2 = axes[1]
    ax2.plot(data.index, data['RSI'], color='purple', linewidth=2)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobreventa')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title('RSI (Relative Strength Index)', fontsize=14)
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # 3. MACD
    ax3 = axes[2]
    ax3.plot(data.index, data['MACD'], label='MACD', linewidth=2)
    ax3.plot(data.index, data['MACD_Signal'], label='Signal', linewidth=2)
    ax3.bar(data.index, data['MACD_Histogram'], alpha=0.3, label='Histogram')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('MACD', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Volatilidad
    ax4 = axes[3]
    ax4.plot(data.index, data['Volatilidad'], color='orange', linewidth=2)
    ax4.set_title('Volatilidad Realizada (Anualizada)', fontsize=14)
    ax4.set_ylabel('Volatilidad (%)')
    ax4.set_xlabel('Fecha')
    ax4.grid(True, alpha=0.3)

    # Formato de fechas
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analizar_situacion_actual(data, symbol):
    """Analiza la situación técnica actual"""

    # Datos más recientes
    ultima_fila = data.iloc[-1]
    precio_actual = ultima_fila['Close']

    print(f"\\n📊 ANÁLISIS TÉCNICO ACTUAL - {symbol}")
    print("=" * 50)

    # Precio vs Medias Móviles
    print("📈 TENDENCIA:")
    if precio_actual > ultima_fila['SMA_20'] > ultima_fila['SMA_50']:
        print("  ✅ Tendencia alcista fuerte (precio > SMA20 > SMA50)")
    elif precio_actual > ultima_fila['SMA_20']:
        print("  🟡 Tendencia alcista débil (precio > SMA20)")
    elif precio_actual < ultima_fila['SMA_20'] < ultima_fila['SMA_50']:
        print("  ❌ Tendencia bajista fuerte (precio < SMA20 < SMA50)")
    else:
        print("  🔄 Tendencia lateral o cambiando")

    # RSI
    rsi_actual = ultima_fila['RSI']
    print(f"\\n📊 RSI: {rsi_actual:.1f}")
    if rsi_actual > 70:
        print("  ⚠️ Zona de sobrecompra - posible corrección")
    elif rsi_actual < 30:
        print("  🔔 Zona de sobreventa - posible rebote")
    else:
        print("  ✅ RSI en zona neutral")

    # Bandas de Bollinger
    bb_pos = ultima_fila['BB_Posicion']
    print(f"\\n📊 Posición en Bandas Bollinger: {bb_pos:.2f}")
    if bb_pos > 0.8:
        print("  ⚠️ Cerca de banda superior - posible resistencia")
    elif bb_pos < 0.2:
        print("  🔔 Cerca de banda inferior - posible soporte")
    else:
        print("  ✅ En rango normal de las bandas")

    # MACD
    macd_actual = ultima_fila['MACD']
    signal_actual = ultima_fila['MACD_Signal']
    print(f"\\n📊 MACD: {macd_actual:.4f}")
    if macd_actual > signal_actual:
        print("  ✅ MACD por encima de señal - momentum positivo")
    else:
        print("  ❌ MACD por debajo de señal - momentum negativo")

    # Volatilidad
    vol_actual = ultima_fila['Volatilidad']
    vol_promedio = data['Volatilidad'].tail(60).mean()
    print(f"\\n📊 Volatilidad: {vol_actual:.1f}% (Promedio 60d: {vol_promedio:.1f}%)")
    if vol_actual > vol_promedio * 1.5:
        print("  ⚠️ Volatilidad alta - mayor riesgo")
    elif vol_actual < vol_promedio * 0.7:
        print("  😴 Volatilidad baja - mercado tranquilo")
    else:
        print("  ✅ Volatilidad normal")

# Ejecutar análisis completo
if __name__ == "__main__":
    symbol = 'AAPL'
    print(f"Analizando {symbol}...")

    data = analisis_tecnico_completo(symbol)

    if data is not None:
        print("\\n✅ Análisis completo terminado!")
        print(f"Datos disponibles desde {data.index[0].date()} hasta {data.index[-1].date()}")
```

**🎯 Ejercicio 2.3**:
1. Ejecuta el análisis completo para AAPL
2. Cambia a otra acción (TSLA, MSFT, etc.)
3. Compara los análisis técnicos actuales
4. ¿Cuál parece más "comprable" según los indicadores?

## 🏆 Proyecto Final del Módulo

### Script 4: Tu Primer Sistema de Análisis

```python
# archivo: mi_primer_sistema.py

class AnalizadorAcciones:
    """
    Tu primer sistema de análisis cuantitativo
    """

    def __init__(self):
        self.resultados = {}

    def analizar_accion(self, symbol, periodo='6mo'):
        """Analiza una acción completamente"""

        print(f"\\nAnalizando {symbol}...")

        # Descargar datos
        data = yf.download(symbol, period=periodo)

        if data.empty:
            print(f"❌ Sin datos para {symbol}")
            return None

        # Calcular indicadores
        data = calcular_medias_moviles(data)
        data['RSI'] = calcular_rsi(data)
        data = calcular_bandas_bollinger(data)
        data = calcular_macd(data)
        data = calcular_volatilidad(data)

        # Calcular métricas
        resultado = self.calcular_metricas(data, symbol)
        self.resultados[symbol] = resultado

        return resultado

    def calcular_metricas(self, data, symbol):
        """Calcula métricas clave"""

        # Datos actuales
        actual = data.iloc[-1]

        # Rendimiento
        rendimiento_total = (actual['Close'] / data['Close'].iloc[0] - 1) * 100

        # Tendencia (score 0-100)
        score_tendencia = 0
        if actual['Close'] > actual['SMA_20']:
            score_tendencia += 25
        if actual['SMA_20'] > actual['SMA_50']:
            score_tendencia += 25
        if actual['Close'] > actual['SMA_50']:
            score_tendencia += 25
        if data['Close'].tail(5).mean() > data['Close'].tail(10).mean():
            score_tendencia += 25

        # Score RSI (50 = neutral, 0 = sobreventa extrema, 100 = sobrecompra extrema)
        rsi_score = min(100, max(0, actual['RSI']))

        # Score Momentum (MACD)
        momentum_score = 50  # Base neutral
        if actual['MACD'] > actual['MACD_Signal']:
            momentum_score += 25
        if actual['MACD'] > 0:
            momentum_score += 15
        if data['MACD'].tail(3).mean() > data['MACD'].tail(6).mean():
            momentum_score += 10
        momentum_score = min(100, momentum_score)

        # Score de calidad general
        score_general = (score_tendencia * 0.4 +
                        (100 - abs(rsi_score - 50)) * 0.3 +
                        momentum_score * 0.3)

        return {
            'simbolo': symbol,
            'precio_actual': actual['Close'],
            'rendimiento_periodo': rendimiento_total,
            'score_tendencia': score_tendencia,
            'score_momentum': momentum_score,
            'rsi_actual': actual['RSI'],
            'volatilidad_actual': actual['Volatilidad'],
            'score_general': score_general,
            'recomendacion': self.generar_recomendacion(score_general, actual['RSI'])
        }

    def generar_recomendacion(self, score_general, rsi):
        """Genera recomendación basada en scores"""

        if score_general > 75 and 30 < rsi < 70:
            return "🟢 COMPRA FUERTE"
        elif score_general > 60 and 25 < rsi < 75:
            return "🟡 COMPRA DÉBIL"
        elif score_general < 25 or rsi > 80 or rsi < 20:
            return "🔴 EVITAR"
        else:
            return "⚪ NEUTRAL"

    def analizar_portfolio(self, symbols):
        """Analiza múltiples acciones"""

        print("🚀 Iniciando análisis de portfolio...")

        for symbol in symbols:
            self.analizar_accion(symbol)

        # Crear reporte
        self.generar_reporte()

    def generar_reporte(self):
        """Genera reporte final"""

        if not self.resultados:
            print("❌ No hay resultados para reportar")
            return

        # Convertir a DataFrame
        df = pd.DataFrame(self.resultados).T

        # Ordenar por score general
        df = df.sort_values('score_general', ascending=False)

        print("\\n" + "="*80)
        print("📊 REPORTE FINAL DE ANÁLISIS")
        print("="*80)

        print(f"\\n🏆 TOP 3 RECOMENDACIONES:")
        for i, (symbol, row) in enumerate(df.head(3).iterrows(), 1):
            print(f"{i}. {symbol}: {row['recomendacion']} (Score: {row['score_general']:.1f})")

        print(f"\\n📈 RENDIMIENTOS EN EL PERÍODO:")
        for symbol, row in df.iterrows():
            print(f"{symbol}: {row['rendimiento_periodo']:+.2f}%")

        print(f"\\n⚡ ANÁLISIS DE RIESGO (Volatilidad):")
        for symbol, row in df.iterrows():
            nivel_riesgo = "ALTO" if row['volatilidad_actual'] > 30 else "MEDIO" if row['volatilidad_actual'] > 20 else "BAJO"
            print(f"{symbol}: {row['volatilidad_actual']:.1f}% ({nivel_riesgo})")

        # Crear gráfico de comparación
        self.grafico_comparacion()

        return df

    def grafico_comparacion(self):
        """Crea gráfico comparativo"""

        df = pd.DataFrame(self.resultados).T

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Score General
        axes[0,0].bar(df.index, df['score_general'],
                     color=['green' if x > 60 else 'orange' if x > 40 else 'red' for x in df['score_general']])
        axes[0,0].set_title('Score General')
        axes[0,0].set_ylabel('Score (0-100)')
        axes[0,0].tick_params(axis='x', rotation=45)

        # Rendimientos
        colors = ['green' if x > 0 else 'red' for x in df['rendimiento_periodo']]
        axes[0,1].bar(df.index, df['rendimiento_periodo'], color=colors)
        axes[0,1].set_title('Rendimiento del Período')
        axes[0,1].set_ylabel('Rendimiento (%)')
        axes[0,1].tick_params(axis='x', rotation=45)

        # RSI
        axes[1,0].bar(df.index, df['rsi_actual'])
        axes[1,0].axhline(y=70, color='red', linestyle='--', alpha=0.7)
        axes[1,0].axhline(y=30, color='green', linestyle='--', alpha=0.7)
        axes[1,0].set_title('RSI Actual')
        axes[1,0].set_ylabel('RSI')
        axes[1,0].tick_params(axis='x', rotation=45)

        # Volatilidad
        axes[1,1].bar(df.index, df['volatilidad_actual'])
        axes[1,1].set_title('Volatilidad')
        axes[1,1].set_ylabel('Volatilidad (%)')
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

# Ejecutar sistema completo
if __name__ == "__main__":
    # Crear analizador
    analizador = AnalizadorAcciones()

    # Lista de acciones a analizar
    portfolio = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']

    # Analizar portfolio
    resultados = analizador.analizar_portfolio(portfolio)

    print("\\n🎉 ¡Análisis completo terminado!")
    print("Ya tienes tu primer sistema de análisis cuantitativo funcionando!")
```

**🎯 Ejercicio Final 2.4**:
1. Ejecuta el sistema completo con las acciones por defecto
2. Cambia la lista por acciones que te interesen
3. Analiza los resultados: ¿confías en las recomendaciones?
4. ¿Qué cambiarías en la lógica de scoring?

## ✅ Checkpoint del Módulo

### Instalación ✅
- [ ] Python funcionando correctamente
- [ ] Todas las librerías instaladas
- [ ] Puedes descargar datos sin errores

### Habilidades ✅
- [ ] Descargar datos de cualquier acción
- [ ] Crear gráficos de precios profesionales
- [ ] Calcular indicadores técnicos básicos
- [ ] Interpretar RSI, MACD, Bollinger Bands

### Código ✅
- [ ] Todos los scripts funcionan sin errores
- [ ] Tu sistema de análisis produce resultados
- [ ] Entiendes la lógica básica del código
- [ ] Puedes modificar símbolos y parámetros

### Mentalidad ✅
- [ ] Te sientes cómodo ejecutando código
- [ ] Entiendes que los gráficos cuentan historias
- [ ] Ves el valor de automatizar análisis
- [ ] Estás emocionado de crear estrategias

## 🚀 Próximo Módulo

**F3: Indicadores Técnicos Avanzados**
- Más indicadores profesionales
- Interpretación avanzada
- Combinación de señales
- Filtros de calidad

**¡Ya puedes llamarte oficialmente "Python Trader"! 🎉**

---

🎯 **¿Listo para indicadores más avanzados?** → [F3: Indicadores Técnicos](../f3-indicadores-tecnicos/README.md)