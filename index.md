---
layout: default
title: "Start Your Quant"
description: "Academia de Trading Cuantitativo - De cero a quant trader profesional"
---

# 🎓 Conviértete en Quant Trader

**De cero a profesional con módulos prácticos y progresivos.**

Aprende trading cuantitativo usando **matemáticas y programación** en lugar de intuición. Desde conceptos básicos hasta estrategias institucionales.

## 🚀 Empieza Tu Aventura

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0;">

<div style="border: 2px solid #28a745; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🟢 Total Principiante</h3>
<p>Nunca programé ni hice trading</p>
<a href="{{ site.baseurl }}/learning-path/" style="background: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">¿Qué es ser Quant?</a>
<p><small>⏱️ 2-4 meses</small></p>
</div>

<div style="border: 2px solid #ffc107; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🟡 Sé algo de Python</h3>
<p>Tengo experiencia básica programando</p>
<a href="{{ site.baseurl }}/learning-path/" style="background: #ffc107; color: black; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Python Trading</a>
<p><small>⏱️ 1-3 meses</small></p>
</div>

<div style="border: 2px solid #fd7e14; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🟠 Ya tradeo manualmente</h3>
<p>Conozco trading pero quiero automatizar</p>
<a href="{{ site.baseurl }}/learning-path/" style="background: #fd7e14; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Estrategias Quant</a>
<p><small>⏱️ 1-2 meses</small></p>
</div>

<div style="border: 2px solid #dc3545; border-radius: 10px; padding: 20px; text-align: center;">
<h3>🔴 Desarrollador Avanzado</h3>
<p>Quiero infraestructura profesional</p>
<a href="{{ site.baseurl }}/infrastructure/" style="background: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Infraestructura</a>
<p><small>⏱️ Inmediato</small></p>
</div>

</div>

## 📚 Ruta de Aprendizaje Completa

### 🟢 [FUNDAMENTOS]({{ site.baseurl }}/learning-path/)
**Aprende las bases del trading cuantitativo**
- ¿Qué es ser Quant? (1h)
- Python Trading Básico (3h)
- Indicadores Técnicos (2h)
- Primera Estrategia (2h)

### 🟡 [ESTRATEGIAS]({{ site.baseurl }}/learning-path/)
**Desarrolla estrategias rentables**
- Momentum Trading (3h)
- Mean Reversion (3h)
- Backtesting Robusto (4h)
- Optimización (3h)
- Multi-Estrategia (5h)

### 🟠 [ANÁLISIS AVANZADO]({{ site.baseurl }}/learning-path/)
**Herramientas profesionales**
- Gestión de Riesgo (3h)
- Performance Metrics (2h)
- Datos Alternativos (4h)
- Machine Learning (5h)

### 🔴 [TRADING PROFESIONAL]({{ site.baseurl }}/learning-path/)
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

### Lo que encontrarás en QuantLab Discord:
- **📊 Análisis en Tiempo Real** - Comparte y discute oportunidades del mercado
- **🤝 Mentorías** - Aprende de traders experimentados
- **💻 Code Reviews** - Mejora tu código con feedback de la comunidad
- **🎯 Challenges Semanales** - Compite y mejora tus habilidades
- **📚 Recursos Exclusivos** - Accede a materiales y herramientas premium
- **🔔 Alertas de Mercado** - Notificaciones de oportunidades

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
Discord QuantLab para chat en tiempo real y GitHub para código.

## 🚀 Empezar Ahora

**¿No sabes por dónde empezar?**

👉 **[Guía Completa de Inicio]({{ site.baseurl }}/GETTING-STARTED)** - Te ayudamos a elegir tu ruta perfecta

**¿Ya sabes qué quieres aprender?**

👉 **[Academia Quant]({{ site.baseurl }}/learning-path/)** - Directo a los módulos de aprendizaje

---

### 💡 Recuerda

> **"El mejor momento para empezar fue ayer. El segundo mejor momento es ahora."**

Los mercados generan nuevos datos cada segundo. Cada día que esperas es un día menos de datos para analizar y mejorar tus estrategias.

**Tu aventura quant empieza con un click.**