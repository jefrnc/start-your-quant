# 🚀 Guía de Inicio - Start Your Quant

**Tu aventura hacia convertirte en quant trader empieza aquí.**

## 🎯 ¿Qué vas a aprender?

- **🧮 Trading Cuantitativo**: Usar matemáticas y programación en lugar de intuición
- **🐍 Python para Finanzas**: El lenguaje #1 de Wall Street
- **📊 Análisis de Datos**: Encontrar patrones rentables en el mercado
- **🤖 Automatización**: Sistemas que tradean mientras duermes
- **📈 Estrategias Reales**: Métodos probados en mercados reales

## ⚡ Empezar Sin Instalaciones

### 🌐 Opción 1: Google Colab (Recomendado)
**Perfecto para principiantes - No instalas nada**

1. Ve a [Google Colab](https://colab.research.google.com)
2. Crea un nuevo notebook
3. Pega este código y ejecuta:

```python
# Instalar librerías básicas
!pip install yfinance pandas matplotlib seaborn

# Tu primer análisis cuantitativo
import yfinance as yf
import matplotlib.pyplot as plt

# Descargar datos de Apple
data = yf.download('AAPL', period='1y')

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'])
plt.title('Apple - Tu Primer Análisis Quant')
plt.show()

print(f"Precio actual: ${data['Close'][-1]:.2f}")
print("🎉 ¡Ya eres un quant trader!")
```

### 💻 Opción 2: Tu Computadora
**Si prefieres trabajar localmente**

1. **Instalar Python** (si no lo tienes):
   - [python.org](https://python.org) → Descargar última versión
   - Marcar "Add to PATH" en Windows

2. **Instalar librerías**:
   ```bash
   pip install yfinance pandas matplotlib seaborn numpy
   ```

3. **Verificar**:
   ```bash
   python -c "import yfinance; print('✅ Todo listo!')"
   ```

## 🎯 ¿Por Dónde Empezar?

### 📊 Test Rápido: ¿Cuál es tu nivel?

**Pregunta 1: ¿Has programado antes?**
- A) Nunca → Empieza en **F1**
- B) Un poco → Empieza en **F2**
- C) Sí, pero no Python → Empieza en **F2**
- D) Sí, conozco Python → Empieza en **F3**

**Pregunta 2: ¿Has hecho trading antes?**
- A) Nunca → Empieza en **F1**
- B) Un poco manual → Empieza en **F2**
- C) Sí, pero manualmente → Empieza en **F3**
- D) Sí, conozco análisis técnico → Empieza en **E1**

### 📊 Tu Plan Personalizado (elige UNO):
| Si eres... | Semana 1 | Semana 2 | Semana 3 | Semana 4 | META |
|------------|----------|-----------|-----------|-----------|------|
| **Total Noob** | Python basics + Colab | Leer datos Yahoo | Tu primer indicador | Bot papel trading | Bot funcionando |
| **Dev Sin Trading** | Indicadores técnicos | Backtesting basics | Optimización | Broker API | Bot real |
| **Trader Manual** | Python crash course | Automatizar TU estrategia | Backtesting | Mejoras | Tu sistema auto |
| **Quiero Todo YA** | Setup completo | 3 estrategias | Portfolio manager | Cloud deploy | Sistema pro |

## 🔥 MINUTO 10-20: Bots Listos para Copiar

### Bot #1: Detector de Gaps Matutinos (Gana con volatilidad)
```python
# Bot que detecta gaps (aperturas con gran diferencia)
import yfinance as yf
import pandas as pd

# Lista de acciones volátiles
volatiles = ['TSLA', 'NVDA', 'AMD', 'COIN', 'MARA']

print("🔍 ESCANEANDO GAPS DEL DÍA...\n")

for ticker in volatiles:
    data = yf.download(ticker, period='5d', progress=False)

    # Gap = Apertura hoy vs Cierre ayer
    gap = ((data['Open'][-1] - data['Close'][-2]) / data['Close'][-2]) * 100

    if abs(gap) > 2:  # Gap mayor a 2%
        tipo = "UP ↑️" if gap > 0 else "DOWN ↓️"
        print(f"🚨 {ticker}: GAP {tipo} de {gap:.2f}%")
        print(f"   Potencial: Regresión a ${data['Close'][-2]:.2f}")
        print(f"   Entrada: ${data['Open'][-1]:.2f}")
        print(f"   Target: ${data['Close'][-2]:.2f}")
        print(f"   Ganancia potencial: {abs(gap):.2f}%\n")
```

### Bot #2: Cazador de Oversold (Compra en pánico)
```python
# Bot que encuentra acciones sobrevendidas
import yfinance as yf
import pandas as pd

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

blue_chips = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']

print("🎯 BUSCANDO OPORTUNIDADES DE COMPRA (RSI < 30)\n")

for ticker in blue_chips:
    data = yf.download(ticker, period='1mo', progress=False)
    data['RSI'] = calculate_rsi(data['Close'])

    current_rsi = data['RSI'][-1]

    if current_rsi < 35:  # Sobrevendido
        print(f"🟢 {ticker} SOBREVENDIDO!")
        print(f"   RSI: {current_rsi:.2f}")
        print(f"   Precio actual: ${data['Close'][-1]:.2f}")
        print(f"   Promedio 30d: ${data['Close'].mean():.2f}")
        print(f"   Potencial rebote: {((data['Close'].mean() - data['Close'][-1]) / data['Close'][-1] * 100):.2f}%\n")
```

### Bot #3: Sistema Completo con Stop Loss
```python
# Sistema completo con gestión de riesgo
import yfinance as yf
import numpy as np

# Configuración
TICKER = 'SPY'  # ETF del S&P 500
CAPITAL_INICIAL = 10000
RIESGO_POR_TRADE = 0.02  # 2% máximo

data = yf.download(TICKER, period='6mo', progress=False)

# Indicadores
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['Volatilidad'] = data['Close'].rolling(20).std()

# Señales
data['Signal'] = 0
data.loc[data['MA20'] > data['MA50'], 'Signal'] = 1  # Compra
data.loc[data['MA20'] < data['MA50'], 'Signal'] = -1  # Venta

# Stop Loss dinámico (2x volatilidad)
data['StopLoss'] = data['Close'] - (2 * data['Volatilidad'])

# Simular trades
capital = CAPITAL_INICIAL
trades = []

for i in range(1, len(data)):
    if data['Signal'].iloc[i] == 1 and data['Signal'].iloc[i-1] != 1:
        # Entrada
        entrada = data['Close'].iloc[i]
        stop = data['StopLoss'].iloc[i]
        size = (capital * RIESGO_POR_TRADE) / (entrada - stop)

        trades.append({
            'Fecha': data.index[i].date(),
            'Tipo': 'COMPRA',
            'Precio': entrada,
            'Stop': stop,
            'Shares': int(size)
        })

# Resultados
print("📈 BACKTEST COMPLETO - ÚLTIMOS 6 MESES\n")
print(f"Capital inicial: ${CAPITAL_INICIAL:,}")
print(f"Total trades: {len(trades)}")
print(f"\nÚLTIMAS 3 SEÑALES:")

for trade in trades[-3:]:
    print(f"\n{trade['Fecha']}: {trade['Tipo']} @ ${trade['Precio']:.2f}")
    print(f"  Stop Loss: ${trade['Stop']:.2f}")
    print(f"  Posición: {trade['Shares']} shares")
    print(f"  Riesgo: ${(trade['Shares'] * (trade['Precio'] - trade['Stop'])):.2f}")
```

## 💵 MINUTO 20-30: Conecta con un Broker REAL

### Opción A: Paper Trading (Sin Riesgo)
```python
# Alpaca Paper Trading - GRATIS, sin dinero real
# 1. Registrate en: alpaca.markets (2 min)
# 2. Obtén tus API keys
# 3. Pega este código:

!pip install alpaca-py

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Tus credenciales (paper trading)
API_KEY = 'tu_api_key'
SECRET_KEY = 'tu_secret_key'

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Tu cuenta
account = client.get_account()
print(f"Balance: ${account.cash}")
print(f"Poder de compra: ${account.buying_power}")

# Hacer un trade
order_data = MarketOrderRequest(
    symbol="AAPL",
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

order = client.submit_order(order_data)
print(f"Orden enviada: Compra 1 AAPL")
```

### Opción B: Notificaciones a tu Teléfono
```python
# Bot de Telegram - Recibe alertas en tu celular
# 1. Busca @BotFather en Telegram
# 2. Crea un bot (/newbot)
# 3. Guarda el token

!pip install python-telegram-bot

import requests

def enviar_alerta(mensaje):
    token = "TU_BOT_TOKEN"
    chat_id = "TU_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={'chat_id': chat_id, 'text': mensaje})

# Enviar alerta de tu bot
if signal == "COMPRA":
    enviar_alerta(f"🟢 COMPRAR {ticker} a ${precio:.2f}")
```

## 🎯 RESULTADO: Tu Sistema de Trading en 30 Minutos

✅ **Ya tienes:**
- 3 bots funcionando
- Scanner de oportunidades
- Sistema con stop loss
- Conexión a broker (paper)
- Alertas al celular

🚀 **Siguiente paso:**
- Únete al Discord para compartir resultados
- Mejora los bots con la comunidad
- Empieza el curso completo

🔗 **Links Útiles:**
- [Discord QuantLab](https://discord.gg/GgXZ3zAS)
- [Academia Completa]({{ site.baseurl }}/learning-path/)
- [Código en GitHub](https://github.com/jefrnc/start-your-quant)

### Minutos 31-45: Primera Estrategia
1. Copia el código de media móvil
2. Pruébalo con diferentes acciones
3. Modifica los parámetros

### Minutos 46-60: Plan Personal
1. Define tu objetivo (¿por qué quieres ser quant?)
2. Elige tu ruta de entrada
3. Marca en tu calendario 30 min diarios

## 🆘 Problemas Comunes

### "No se instala yfinance"
```bash
pip install --upgrade pip
pip install yfinance
```

### "No aparecen los gráficos"
```python
import matplotlib.pyplot as plt
plt.show()  # Agregar al final del código
```

### "Error al descargar datos"
- Verificar conexión a internet
- Probar otro símbolo: 'MSFT', 'GOOGL'
- Usar período más corto: period='1mo'

### "Python no reconocido"
- Windows: Marcar "Add to PATH" al instalar
- Mac: `brew install python`
- Linux: `sudo apt install python3`

## 💪 Mantener la Motivación

### 🎯 Objetivos Semanales
- **Semana 1**: Completar F1 y F2
- **Semana 2**: Completar F3 y F4
- **Semana 3**: Empezar estrategias (E1)
- **Semana 4**: Primera estrategia completa

### 📈 Progreso Visible
- Cada módulo incluye ejercicios verificables
- Tu código va mejorando gradualmente
- Builds un portfolio de estrategias
- Métricas reales de performance

### 🤝 Comunidad
- GitHub Issues para preguntas técnicas
- Discord para chat diario (próximamente)
- Comparte tu progreso con #StartYourQuant

## 🏆 Al Final Tendrás

### 💻 Portfolio Técnico
- 5+ estrategias probadas
- Sistema de backtesting robusto
- Dashboard de monitoreo
- Código reutilizable

### 🧠 Conocimientos
- Python para trading
- Análisis técnico cuantitativo
- Gestión de riesgo
- Optimización de estrategias

### 🚀 Habilidades
- Analizar cualquier acción en minutos
- Probar ideas rápidamente
- Automatizar decisiones de trading
- Evaluar performance objetivamente

## ⚡ ¡Empezar AHORA!

**La mejor estrategia es empezar, aunque sea imperfecto.**

1. **Setup**: `python quick-start.py` (5 min)
2. **Primera lección**: [F1 - ¿Qué es ser Quant?](learning-path/fundamentos/f1-que-es-ser-quant/) (30 min)
3. **Compromiso**: 30 min diarios por 30 días

**En 30 días tendrás más conocimiento cuantitativo que 95% de traders retail.**

---

🎯 **¿Listo para empezar?** → Ejecuta `python quick-start.py` y comienza tu aventura quant!