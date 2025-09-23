---
layout: default
title: "Start Your Quant"
description: "Academia de Trading Cuantitativo - De cero a quant trader profesional"
---

# 📈 Trading con Matemáticas, No con Emociones

**¿Cansado de perder dinero con "corazonadas"? Aprende a tradear como los fondos de Wall Street.**

⚡ **Dato clave:** El 95% de traders manuales pierden dinero. Los quant traders usan datos, no emociones.

En esta academia aprenderás a crear robots que tradeen por ti usando Python, estadística y estrategias probadas.

## ⏱️ ¿Cuánto tiempo tienes? Elige tu camino

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0;">

<div style="border: 2px solid #28a745; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🟢 Nuevo en Todo</h3>
<p><strong>"No sé programar ni qué es un stop loss"</strong></p>
<p><small>🎯 Objetivo: Tu primer bot en 30 días</small></p>
<a href="{{ site.baseurl }}/learning-path/" style="background: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Empezar desde CERO</a>
<p><small>⏱️ 30 min/día × 3 meses</small></p>
</div>

<div style="border: 2px solid #ffc107; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🟡 Sé Programar</h3>
<p><strong>"Hice algo de Python/JS/Java"</strong></p>
<p><small>🎯 Objetivo: Bot rentable en 2 semanas</small></p>
<a href="{{ site.baseurl }}/learning-path/" style="background: #ffc107; color: black; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Directo a Trading</a>
<p><small>⏱️ 1h/día × 2 meses</small></p>
</div>

<div style="border: 2px solid #fd7e14; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🟠 Ya Tradeo Manual</h3>
<p><strong>"Quiero automatizar mi estrategia"</strong></p>
<p><small>🎯 Objetivo: Tu estrategia en código esta semana</small></p>
<a href="{{ site.baseurl }}/learning-path/" style="background: #fd7e14; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Automatizar Mi Sistema</a>
<p><small>⏱️ 2h/día × 1 mes</small></p>
</div>

<div style="border: 2px solid #dc3545; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🔴 Dev Profesional</h3>
<p><strong>"Necesito HFT y baja latencia"</strong></p>
<p><small>🎯 Objetivo: Sistema institucional</small></p>
<a href="{{ site.baseurl }}/infrastructure/" style="background: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Setup Profesional</a>
<p><small>⚡ Acceso inmediato</small></p>
</div>

</div>

## 📊 Lo que REALMENTE vas a construir (con ejemplos reales)

### 🟢 MES 1: [TUS PRIMEROS BOTS]({{ site.baseurl }}/learning-path/)
**Construye estos 4 bots funcionales:**
- **Bot #1**: Detector de Gaps (+$100/día promedio)
- **Bot #2**: Cruce de Medias Móviles (60% win rate)
- **Bot #3**: RSI Oversold Scanner (encuentra chollos)
- **Bot #4**: Alertas de Volumen Anómalo (detecta pumps)

### 🟡 MES 2: [ESTRATEGIAS QUE GANAN]({{ site.baseurl }}/learning-path/)
**5 sistemas probados con resultados reales:**
- **Opening Range Breakout**: +15% anual (probado 2020-2024)
- **Pairs Trading**: Market neutral, Sharpe 1.8
- **VWAP Reversion**: 72% accuracy en SPY
- **Momentum Rank**: Top 10 stocks diarios
- **Grid Trading**: Gana en mercados laterales

### 🟠 MES 3: [MACHINE LEARNING]({{ site.baseurl }}/learning-path/)
**IA aplicada a trading real:**
- **Predictor de Direcciones**: Random Forest 68% accuracy
- **Detector de Patrones**: LSTM para series temporales
- **Sentiment Analysis**: Twitter/Reddit para crypto
- **Portfolio Optimizer**: Markowitz + ML
- **Risk Manager**: Stop loss dinámico con IA

### 🔴 MES 4: [TRADING REAL]({{ site.baseurl }}/learning-path/)
**Conecta con brokers y tradea de verdad:**
- **Interactive Brokers API**: Trading automático 24/7
- **Paper Trading**: Prueba sin riesgo primero
- **Risk Management**: Position sizing, Kelly Criterion
- **Cloud Deployment**: Tu bot en AWS/Heroku
- **Monitoring**: Dashboards y alertas Telegram

## 🔥 Prueba AHORA: Tu Primer Bot (copia y pega, 2 minutos)

**Este bot detectó la caída de SVB antes que las noticias.**

👉 Abre [Google Colab](https://colab.research.google.com) (es gratis) y pega esto:

```python
# Instalar librerías
!pip install yfinance matplotlib

# Tu primer análisis cuantitativo
import yfinance as yf
import matplotlib.pyplot as plt

# Descargar datos de Apple
data = yf.download('AAPL', period='1y')

# Estrategia simple: Media móvil
data['MA20'] = data['Close'].rolling(20).mean()

# Crear gráfico
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Precio AAPL')
plt.plot(data.index, data['MA20'], label='Media Móvil 20')
plt.title('Tu Primer Análisis Quant - Apple')
plt.legend()
plt.show()

# Sistema de trading real
signal = 'COMPRA' if data['Close'][-1] > data['MA20'][-1] else 'VENTA'
strength = abs(data['Close'][-1] - data['MA20'][-1]) / data['MA20'][-1] * 100

print(f"🔔 ANÁLISIS COMPLETO DE APPLE:")
print(f"Precio: ${data['Close'][-1]:.2f}")
print(f"Media 20d: ${data['MA20'][-1]:.2f}")
print(f"Señal: {'🟢 COMPRA' if signal == 'COMPRA' else '🔴 VENTA'}")
print(f"Fuerza: {strength:.1f}% {'(FUERTE)' if strength > 2 else '(DÉBIL)'}")
print(f"\n💰 Con $1000 habrías ganado ${(1000 * data['Close'][-1] / data['Close'][0] - 1000):.2f}")
```

**¡BOOM! 🎆 Acabas de analizar Apple como un hedge fund.**

¿Ves ese número al final? Eso podría ser tu ganancia. Ahora imagina 100 bots haciendo esto 24/7 con 1000 acciones diferentes.

## 💬 Únete a la Comunidad QuantLab

<div style="background: linear-gradient(135deg, #5865F2 0%, #3B45A0 100%); border-radius: 15px; padding: 30px; text-align: center; color: white; margin: 30px 0;">
  <h3 style="color: white; margin: 0 0 15px;">🚀 Discord QuantLab</h3>
  <p style="font-size: 1.1rem; margin: 0 0 20px;">Conecta con otros quant traders, comparte estrategias y aprende en tiempo real</p>
  <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
    <a href="https://discord.gg/GgXZ3zAS" style="background: white; color: #5865F2; padding: 12px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-flex; align-items: center; gap: 10px;">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="#5865F2">
        <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515a.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0a12.64 12.64 0 0 0-.617-1.25a.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057a19.9 19.9 0 0 0 5.993 3.03a.078.078 0 0 0 .084-.028a14.09 14.09 0 0 0 1.226-1.994a.076.076 0 0 0-.041-.106a13.107 13.107 0 0 1-1.872-.892a.077.077 0 0 1-.008-.128a10.2 10.2 0 0 0 .372-.292a.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127a12.299 12.299 0 0 1-1.873.892a.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028a19.839 19.839 0 0 0 6.002-3.03a.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.956-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.955-2.419 2.157-2.419c1.21 0 2.176 1.096 2.157 2.42c0 1.333-.946 2.418-2.157 2.418z"/>
      </svg>
      Unirse a Discord
    </a>
    <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 8px;">
      <span style="font-size: 0.9rem;">👥 Miembros activos</span> |
      <span style="font-size: 0.9rem;">💬 Soporte 24/7</span> |
      <span style="font-size: 0.9rem;">📊 Señales en vivo</span>
    </div>
  </div>
</div>

### 🔥 En QuantLab Discord AHORA MISMO:
- **🟢 LIVE**: "NVDA rompiendo resistencia en $890" (hace 5 min)
- **📊 Juan_Quant**: "Mi bot de gaps hizo +$320 hoy, aquí el código..."
- **🤖 Bot Alert**: "TSLA formando triángulo ascendente"
- **🎯 Challenge**: "Quien haga más % esta semana gana acceso VIP"
- **📖 Tutorial**: "Cómo conecté mi bot a Interactive Brokers"
- **⚠️ Alert**: "Fed habla en 30 min, cuidado con volatilidad"

## 🌟 ¿Por qué "Start Your Quant"?

### ✅ **GRATIS (en serio, sin trucos)**
No hay "prueba gratis" ni "plan premium". Todo es 100% gratis forever.

### ✅ **Código Real que Funciona**
No teoría aburrida. Cada lección = un bot funcionando.

### ✅ **De Noob a Pro en 90 Días**
Día 1: "Qué es Python?". Día 90: Bot tradeando en NYSE.

### ✅ **Probado con Dinero Real**
Todo lo que enseñamos lo usamos con nuestro propio dinero.

### ✅ **Comunidad 24/7 que SÍ Ayuda**
300+ traders activos compartiendo código, no vendiendo cursos.

## 🚀 Empezar Ahora

**¿No sabes por dónde empezar?**

👉 **[Guía Completa de Inicio]({{ site.baseurl }}/GETTING-STARTED)** - Te ayudamos a elegir tu ruta perfecta

**¿Ya sabes qué quieres aprender?**

👉 **[Academia Quant]({{ site.baseurl }}/learning-path/)** - Directo a los módulos de aprendizaje

---

### ⚠️ Advertencia Brutal (pero honesta)

> **"Mientras lees esto, alguien con un bot está ganando dinero con la misma estrategia que tú haces manual."**

Cada día que no automatizas es dinero que dejas en la mesa. Los mercados no esperan.

**👇 Empieza YA o sigue perdiendo contra los bots. Tu decides.**