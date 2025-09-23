# F3: Indicadores Técnicos 📊

**Módulo Fundamental 3 - Duración: 2-3 horas**

## 🎯 Objetivos del Módulo

Al completar este módulo podrás:
- ✅ Entender qué son y para qué sirven los indicadores técnicos
- ✅ Calcular e interpretar los indicadores más importantes
- ✅ Combinar múltiples indicadores para señales más fuertes
- ✅ Evitar los errores comunes al usar indicadores

## 📚 ¿Qué son los Indicadores Técnicos?

Los **indicadores técnicos** son cálculos matemáticos basados en precio y volumen que ayudan a:
- 📈 **Identificar tendencias** (¿está subiendo o bajando?)
- 🔄 **Detectar reversiones** (¿va a cambiar de dirección?)
- 💪 **Medir momentum** (¿qué tan fuerte es el movimiento?)
- 🎯 **Encontrar niveles** (¿dónde comprar/vender?)

### Tipos de Indicadores

| Tipo | Función | Ejemplos |
|------|---------|----------|
| **Tendencia** | Identificar dirección | SMA, EMA, MACD |
| **Momentum** | Medir fuerza/velocidad | RSI, Stochastic |
| **Volatilidad** | Medir variabilidad | Bollinger Bands, ATR |
| **Volumen** | Confirmar movimientos | OBV, Volume Profile |

## 🧮 Los 5 Indicadores Esenciales

### 1️⃣ Media Móvil Simple (SMA)

**¿Qué es?** El promedio de precio de los últimos N días.

```python
# Código para Google Colab
!pip install yfinance matplotlib pandas

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Descargar datos
symbol = 'AAPL'
data = yf.download(symbol, period='6mo')

# Calcular SMA de diferentes períodos
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Visualizar
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Precio', linewidth=2)
plt.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.8)
plt.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.8)
plt.plot(data.index, data['SMA_200'], label='SMA 200', alpha=0.8)
plt.title(f'{symbol} - Medias Móviles Simples')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Interpretación
precio_actual = data['Close'].iloc[-1]
sma20_actual = data['SMA_20'].iloc[-1]

if precio_actual > sma20_actual:
    print(f"✅ Precio ({precio_actual:.2f}) > SMA20 ({sma20_actual:.2f}) = TENDENCIA ALCISTA")
else:
    print(f"❌ Precio ({precio_actual:.2f}) < SMA20 ({sma20_actual:.2f}) = TENDENCIA BAJISTA")
```

**📊 Interpretación:**
- Precio > SMA = Tendencia alcista
- Precio < SMA = Tendencia bajista
- SMA corta > SMA larga = Golden Cross (muy alcista)
- SMA corta < SMA larga = Death Cross (muy bajista)

### 2️⃣ RSI (Relative Strength Index)

**¿Qué es?** Mide si una acción está sobrecomprada o sobrevendida (escala 0-100).

```python
def calcular_rsi(data, periodo=14):
    """Calcula el RSI"""
    delta = data['Close'].diff()
    ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()

    rs = ganancia / perdida
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcular RSI
data['RSI'] = calcular_rsi(data)

# Visualizar
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Precio
ax1.plot(data.index, data['Close'], label='Precio')
ax1.set_title(f'{symbol} - Precio y RSI')
ax1.legend()
ax1.grid(True, alpha=0.3)

# RSI
ax2.plot(data.index, data['RSI'], color='purple', linewidth=2)
ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecompra (70)')
ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobreventa (30)')
ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
ax2.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretación
rsi_actual = data['RSI'].iloc[-1]
if rsi_actual > 70:
    print(f"⚠️ RSI = {rsi_actual:.1f} - SOBRECOMPRA (posible corrección)")
elif rsi_actual < 30:
    print(f"🔔 RSI = {rsi_actual:.1f} - SOBREVENTA (posible rebote)")
else:
    print(f"➖ RSI = {rsi_actual:.1f} - ZONA NEUTRAL")
```

**📊 Interpretación:**
- RSI > 70 = Sobrecompra (cuidado, puede bajar)
- RSI < 30 = Sobreventa (oportunidad, puede subir)
- RSI = 50 = Equilibrio
- Divergencias = Cuando precio y RSI van en direcciones opuestas

### 3️⃣ MACD (Moving Average Convergence Divergence)

**¿Qué es?** Muestra la relación entre dos medias móviles y el momentum.

```python
def calcular_macd(data, rapido=12, lento=26, signal=9):
    """Calcula MACD, Signal y Histograma"""
    ema_rapido = data['Close'].ewm(span=rapido, adjust=False).mean()
    ema_lento = data['Close'].ewm(span=lento, adjust=False).mean()

    macd_line = ema_rapido - ema_lento
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histograma = macd_line - signal_line

    return macd_line, signal_line, histograma

# Calcular MACD
data['MACD'], data['Signal'], data['Histogram'] = calcular_macd(data)

# Visualizar
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Precio
ax1.plot(data.index, data['Close'], label='Precio')
ax1.set_title(f'{symbol} - MACD Analysis')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MACD
ax2.plot(data.index, data['MACD'], label='MACD', linewidth=2)
ax2.plot(data.index, data['Signal'], label='Signal', linewidth=2)
ax2.bar(data.index, data['Histogram'], label='Histogram', alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretación
if data['MACD'].iloc[-1] > data['Signal'].iloc[-1]:
    print("✅ MACD > Signal = MOMENTUM ALCISTA")
    if data['Histogram'].iloc[-1] > data['Histogram'].iloc[-2]:
        print("   📈 Histograma creciendo = Momentum acelerando")
else:
    print("❌ MACD < Signal = MOMENTUM BAJISTA")
    if data['Histogram'].iloc[-1] < data['Histogram'].iloc[-2]:
        print("   📉 Histograma decreciendo = Momentum debilitándose")
```

**📊 Interpretación:**
- MACD cruza Signal hacia arriba = Señal de compra
- MACD cruza Signal hacia abajo = Señal de venta
- Histograma positivo y creciente = Momentum alcista fuerte
- Histograma negativo y decreciente = Momentum bajista fuerte

### 4️⃣ Bandas de Bollinger

**¿Qué es?** Muestra un rango de precios "normal" basado en volatilidad.

```python
def calcular_bollinger_bands(data, periodo=20, std_dev=2):
    """Calcula Bandas de Bollinger"""
    sma = data['Close'].rolling(window=periodo).mean()
    std = data['Close'].rolling(window=periodo).std()

    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return upper_band, sma, lower_band

# Calcular Bandas
data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calcular_bollinger_bands(data)

# Calcular ancho de bandas (volatilidad)
data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

# Visualizar
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Precio y Bandas
ax1.plot(data.index, data['Close'], label='Precio', linewidth=2, color='black')
ax1.plot(data.index, data['BB_Upper'], label='Banda Superior', alpha=0.7)
ax1.plot(data.index, data['BB_Middle'], label='SMA 20', alpha=0.7)
ax1.plot(data.index, data['BB_Lower'], label='Banda Inferior', alpha=0.7)
ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1, color='gray')
ax1.set_title(f'{symbol} - Bandas de Bollinger')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Ancho de Bandas (Volatilidad)
ax2.plot(data.index, data['BB_Width'], color='orange', linewidth=2)
ax2.set_ylabel('Ancho de Bandas')
ax2.set_title('Volatilidad (Ancho de Bandas)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretación
posicion = data['BB_Position'].iloc[-1]
if posicion > 1:
    print(f"⚠️ Precio SOBRE banda superior ({posicion:.2f}) - Posible resistencia")
elif posicion < 0:
    print(f"🔔 Precio BAJO banda inferior ({posicion:.2f}) - Posible soporte")
elif posicion > 0.8:
    print(f"📈 Precio cerca de banda superior ({posicion:.2f}) - Tendencia fuerte")
elif posicion < 0.2:
    print(f"📉 Precio cerca de banda inferior ({posicion:.2f}) - Presión bajista")
else:
    print(f"➖ Precio en zona media ({posicion:.2f}) - Rango normal")
```

**📊 Interpretación:**
- Precio toca banda superior = Posible resistencia
- Precio toca banda inferior = Posible soporte
- Bandas estrechándose = Volatilidad baja (calma antes de tormenta)
- Bandas expandiéndose = Volatilidad alta (movimiento fuerte)

### 5️⃣ Volumen (El Confirmador)

**¿Qué es?** Muestra cuánto interés hay en un movimiento de precio.

```python
# Análisis de Volumen
data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
data['Price_Change'] = data['Close'].pct_change()

# Visualizar
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Precio
ax1.plot(data.index, data['Close'], label='Precio')
ax1.set_title(f'{symbol} - Análisis de Volumen')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Volumen
colors = ['green' if c > 0 else 'red' for c in data['Price_Change']]
ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7, label='Volumen')
ax2.plot(data.index, data['Volume_SMA'], color='blue', linewidth=2, label='Vol. Promedio 20d')
ax2.set_ylabel('Volumen')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Ratio de Volumen
ax3.bar(data.index, data['Volume_Ratio'], color='purple', alpha=0.7)
ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Volumen Alto (2x)')
ax3.set_ylabel('Ratio Volumen')
ax3.set_ylim(0, 4)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretación
vol_ratio = data['Volume_Ratio'].iloc[-1]
price_change = data['Price_Change'].iloc[-1] * 100

if vol_ratio > 2 and price_change > 0:
    print(f"🚀 Volumen ALTO ({vol_ratio:.1f}x) + Precio SUBIENDO = SEÑAL FUERTE DE COMPRA")
elif vol_ratio > 2 and price_change < 0:
    print(f"⚠️ Volumen ALTO ({vol_ratio:.1f}x) + Precio BAJANDO = SEÑAL FUERTE DE VENTA")
elif vol_ratio < 0.5:
    print(f"😴 Volumen BAJO ({vol_ratio:.1f}x) = Movimiento débil, no confiable")
else:
    print(f"➖ Volumen normal ({vol_ratio:.1f}x)")
```

**📊 Interpretación:**
- Precio sube + Volumen alto = Movimiento fuerte y confiable
- Precio sube + Volumen bajo = Movimiento débil, cuidado
- Precio baja + Volumen alto = Venta masiva, alejarse
- Volumen spike = Algo importante está pasando

## 🎯 Combinando Indicadores (El Poder Real)

### Sistema de Señales Multi-Indicador

```python
def generar_señales_combinadas(data):
    """Combina múltiples indicadores para señales más confiables"""

    señales = []

    # Calcular todos los indicadores
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['RSI'] = calcular_rsi(data)
    data['MACD'], data['Signal'], _ = calcular_macd(data)
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calcular_bollinger_bands(data)
    data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

    # Sistema de puntuación
    score = 0

    # 1. Tendencia (SMA)
    if data['Close'].iloc[-1] > data['SMA_20'].iloc[-1]:
        score += 1
        señales.append("✅ Precio > SMA20")
    if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]:
        score += 1
        señales.append("✅ SMA20 > SMA50")

    # 2. Momentum (RSI)
    if 30 < data['RSI'].iloc[-1] < 70:
        score += 1
        señales.append(f"✅ RSI en zona neutral ({data['RSI'].iloc[-1]:.1f})")
    elif data['RSI'].iloc[-1] < 30:
        score += 2
        señales.append(f"🔔 RSI sobreventa ({data['RSI'].iloc[-1]:.1f})")

    # 3. MACD
    if data['MACD'].iloc[-1] > data['Signal'].iloc[-1]:
        score += 1
        señales.append("✅ MACD > Signal")

    # 4. Bollinger Bands
    bb_pos = (data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1])
    if 0.2 < bb_pos < 0.8:
        score += 1
        señales.append(f"✅ Precio en rango normal BB ({bb_pos:.2f})")

    # 5. Volumen
    if data['Volume_Ratio'].iloc[-1] > 1.5:
        score += 1
        señales.append(f"✅ Volumen alto ({data['Volume_Ratio'].iloc[-1]:.1f}x)")

    return score, señales

# Analizar
score, señales = generar_señales_combinadas(data)

print(f"\n{'='*50}")
print(f"ANÁLISIS MULTI-INDICADOR: {symbol}")
print(f"{'='*50}")
print(f"Score Total: {score}/7")
print("\nSeñales Detectadas:")
for señal in señales:
    print(f"  {señal}")

print(f"\n🎯 RECOMENDACIÓN:")
if score >= 6:
    print("  🟢🟢🟢 COMPRA FUERTE - Múltiples confirmaciones")
elif score >= 4:
    print("  🟢 COMPRA MODERADA - Señales positivas")
elif score >= 2:
    print("  ⚪ NEUTRAL - Señales mixtas")
else:
    print("  🔴 EVITAR - Pocas señales positivas")
```

## ⚠️ Errores Comunes a Evitar

### ❌ Error #1: Usar un solo indicador
**Problema**: Los indicadores fallan, especialmente solos.
**Solución**: Siempre combina 2-3 indicadores de diferentes tipos.

### ❌ Error #2: Ignorar el contexto del mercado
**Problema**: RSI puede estar en sobrecompra por semanas en tendencia fuerte.
**Solución**: Considera la tendencia general antes de actuar en señales.

### ❌ Error #3: No ajustar parámetros
**Problema**: RSI(14) funciona diferente en crypto vs. blue chips.
**Solución**: Prueba diferentes períodos y ajusta según el activo.

### ❌ Error #4: Reaccionar a cada señal
**Problema**: Demasiadas señales = overtrading.
**Solución**: Solo actúa cuando múltiples indicadores coinciden.

### ❌ Error #5: Olvidar el volumen
**Problema**: Movimientos sin volumen son falsos.
**Solución**: SIEMPRE confirma con volumen.

## 🏆 Proyecto Final: Tu Dashboard de Indicadores

```python
def crear_dashboard_completo(symbol='AAPL'):
    """Crea un dashboard completo con todos los indicadores"""

    # Descargar datos
    data = yf.download(symbol, period='6mo')

    # Calcular todos los indicadores
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['RSI'] = calcular_rsi(data)
    data['MACD'], data['Signal'], data['Histogram'] = calcular_macd(data)
    data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calcular_bollinger_bands(data)

    # Crear visualización
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)

    # 1. Precio y SMA
    axes[0].plot(data.index, data['Close'], label='Precio', linewidth=2)
    axes[0].plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
    axes[0].set_title(f'{symbol} - Dashboard Completo de Indicadores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. RSI
    axes[1].plot(data.index, data['RSI'], color='purple', linewidth=2)
    axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('RSI')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    # 3. MACD
    axes[2].plot(data.index, data['MACD'], label='MACD')
    axes[2].plot(data.index, data['Signal'], label='Signal')
    axes[2].bar(data.index, data['Histogram'], alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_ylabel('MACD')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 4. Bollinger Bands
    axes[3].plot(data.index, data['Close'], label='Precio', color='black')
    axes[3].plot(data.index, data['BB_Upper'], 'r--', alpha=0.7)
    axes[3].plot(data.index, data['BB_Lower'], 'g--', alpha=0.7)
    axes[3].fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1)
    axes[3].set_ylabel('Bollinger')
    axes[3].grid(True, alpha=0.3)

    # 5. Volumen
    colors = ['green' if c > o else 'red' for c, o in zip(data['Close'], data['Open'])]
    axes[4].bar(data.index, data['Volume'], color=colors, alpha=0.7)
    axes[4].set_ylabel('Volumen')
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Análisis final
    score, señales = generar_señales_combinadas(data)
    return score, señales

# Ejecutar dashboard
score, señales = crear_dashboard_completo('AAPL')
```

## ✅ Checkpoint del Módulo

### Conocimientos ✅
- [ ] Entiendo los 5 tipos principales de indicadores
- [ ] Sé calcular SMA, RSI, MACD, Bollinger Bands
- [ ] Puedo interpretar señales de cada indicador
- [ ] Entiendo por qué combinar indicadores es crucial

### Habilidades ✅
- [ ] Puedo codificar cualquier indicador desde cero
- [ ] Sé crear visualizaciones claras de indicadores
- [ ] Puedo combinar múltiples indicadores para señales
- [ ] Evito los errores comunes de principiantes

### Proyecto ✅
- [ ] Mi dashboard multi-indicador funciona
- [ ] Genera señales combinadas correctamente
- [ ] Puedo aplicarlo a cualquier acción

## 🚀 Próximo Módulo

**F4: Tu Primera Estrategia Completa**
- Diseño de estrategia sistemática
- Reglas de entrada y salida
- Gestión de riesgo
- Backtesting básico

**¡Ya dominas los indicadores! Hora de crear tu primera estrategia real 🎯**

---

🎯 **¿Listo para tu primera estrategia?** → [F4: Primera Estrategia](../f4-primera-estrategia/)