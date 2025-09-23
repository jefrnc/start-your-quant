---
layout: home
title: "Start Your Quant - Academia de Trading Cuantitativo"
---

# 🎓 Conviértete en Quant Trader

**De cero a profesional con módulos prácticos y progresivos.**

Aprende trading cuantitativo usando **matemáticas y programación** en lugar de intuición. Desde conceptos básicos hasta estrategias institucionales.

## 🚀 Empieza Tu Aventura

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0;">

<div style="border: 2px solid #28a745; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🟢 Total Principiante</h3>
<p>Nunca programé ni hice trading</p>
<a href="learning-path/fundamentos/f1-que-es-ser-quant/" style="background: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">¿Qué es ser Quant?</a>
<p><small>⏱️ 2-4 meses</small></p>
</div>

<div style="border: 2px solid #ffc107; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🟡 Sé algo de Python</h3>
<p>Tengo experiencia básica programando</p>
<a href="learning-path/fundamentos/f2-python-trading-basico/" style="background: #ffc107; color: black; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Python Trading</a>
<p><small>⏱️ 1-3 meses</small></p>
</div>

<div style="border: 2px solid #fd7e14; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🟠 Ya tradeo manualmente</h3>
<p>Conozco trading pero quiero automatizar</p>
<a href="learning-path/estrategias/e1-momentum-trading/" style="background: #fd7e14; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Estrategias Quant</a>
<p><small>⏱️ 1-2 meses</small></p>
</div>

<div style="border: 2px solid #dc3545; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🔴 Desarrollador Avanzado</h3>
<p>Quiero infraestructura profesional</p>
<a href="infrastructure/" style="background: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Infraestructura</a>
<p><small>⏱️ Inmediato</small></p>
</div>

</div>

## 📚 Ruta de Aprendizaje Completa

### 🟢 [FUNDAMENTOS](learning-path/fundamentos/)
**Aprende las bases del trading cuantitativo**
- ¿Qué es ser Quant? (1h)
- Python Trading Básico (3h)
- Indicadores Técnicos (2h)
- Primera Estrategia (2h)

### 🟡 [ESTRATEGIAS](learning-path/estrategias/)
**Desarrolla estrategias rentables**
- Momentum Trading (3h)
- Mean Reversion (3h)
- Backtesting Robusto (4h)
- Optimización (3h)
- Multi-Estrategia (5h)

### 🟠 [ANÁLISIS AVANZADO](learning-path/analisis/)
**Herramientas profesionales**
- Gestión de Riesgo (3h)
- Performance Metrics (2h)
- Datos Alternativos (4h)
- Machine Learning (5h)

### 🔴 [TRADING PROFESIONAL](learning-path/profesional/)
**Del papel a la realidad**
- Conexión con Broker (3h)
- Automatización (4h)
- Scaling Profesional (3h)

## 🎯 Tu Primer Análisis (5 minutos)

**¿Quieres ver el poder del trading cuantitativo?**

Ve a [Google Colab](https://colab.research.google.com), crea un nuevo notebook y pega:

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

# Señal simple
if data['Close'][-1] > data['MA20'][-1]:
    print("🟢 SEÑAL DE COMPRA")
else:
    print("🔴 SEÑAL DE VENTA")

print(f"Precio actual: ${data['Close'][-1]:.2f}")
```

**¡Felicitaciones! Ya usaste un método cuantitativo para generar una señal de trading.**

## 🌟 ¿Por qué "Start Your Quant"?

### ✅ **100% Gratis y Open Source**
Todo el contenido es gratuito y está disponible en GitHub.

### ✅ **Enfoque Práctico**
80% práctica, 20% teoría. Aprendes haciendo.

### ✅ **Progresión Estructurada**
Desde conceptos básicos hasta niveles institucionales.

### ✅ **Ejemplos Reales**
Estrategias probadas con datos reales de mercado.

### ✅ **Comunidad Activa**
GitHub Issues para preguntas y Discord para chat.

## 🚀 Empezar Ahora

**¿No sabes por dónde empezar?**

👉 **[Guía Completa de Inicio](GETTING-STARTED.md)** - Te ayudamos a elegir tu ruta perfecta

**¿Ya sabes qué quieres aprender?**

👉 **[Academia Quant](learning-path/)** - Directo a los módulos de aprendizaje

---

### 💡 Recuerda

> **"El mejor momento para empezar fue ayer. El segundo mejor momento es ahora."**

Los mercados generan nuevos datos cada segundo. Cada día que esperas es un día menos de datos para analizar y mejorar tus estrategias.

**Tu aventura quant empieza con un click.**