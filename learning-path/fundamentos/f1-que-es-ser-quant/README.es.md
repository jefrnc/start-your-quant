> 🇺🇸 [Read in English](README.md) | 🇪🇸 **Español**

# F1: ¿Qué es ser Quant? 🧮

**Módulo Fundamental 1 - Duración: 1-2 horas**

## 🎯 Objetivos del Módulo

Al completar este módulo podrás:
- ✅ Explicar qué hace un quant trader vs. trader tradicional
- ✅ Entender por qué los números funcionan mejor que la intuición
- ✅ Identificar oportunidades donde aplicar métodos cuantitativos
- ✅ Tener claridad sobre tu camino hacia convertirte en quant

## 📚 ¿Qué es un Quant Trader?

### Definición Simple
Un **quant trader** es alguien que usa **matemáticas, estadísticas y programación** para tomar decisiones de trading en lugar de intuición o "feeling".

### Ejemplo Real: Dos Enfoques

**👤 Trader Tradicional**:
```
"Apple está bajando mucho, creo que va a rebotar.
Voy a comprar porque 'se siente' barato."
```

**🧮 Quant Trader**:
```python
# Descarga 5 años de datos de AAPL
data = get_market_data('AAPL', '5y')

# Calcula estadísticamente si "bajas fuertes" predicen rebotes
strong_drops = data[data['daily_change'] < -3%]
next_day_returns = strong_drops['next_day_change']

# Resultado: 67% de rebotes al día siguiente
# Probabilidad histórica > 50% → Estrategia viable
if probability > 0.6:
    place_order('BUY', 'AAPL', shares=100)
```

### ¿Cuál es Mejor?

**Trader Tradicional**:
- ❌ Emociones afectan decisiones
- ❌ No puede manejar muchas acciones
- ❌ Difícil mejorar sistemáticamente
- ❌ Resultados inconsistentes

**Quant Trader**:
- ✅ Decisiones basadas en datos
- ✅ Puede analizar 1000+ acciones
- ✅ Mejora continua con más datos
- ✅ Resultados medibles y repetibles

## 💰 ¿Por Qué Ser Quant?

### 1. **Escalabilidad**
- Un algoritmo puede monitorear **todo el mercado** 24/7
- No hay límite humano de atención
- Puedes tradear múltiples estrategias simultáneamente

### 2. **Consistencia**
- Las mismas reglas **siempre**
- No hay "días malos" donde decides diferente
- Elimina FOMO, miedo, codicia

### 3. **Mejora Continua**
- Cada día tienes **más datos** para mejorar
- Puedes probar miles de variaciones
- A/B testing de estrategias

### 4. **Ventaja Competitiva**
- Mayoría de retail traders usan intuición
- Acceso a **herramientas institucionales** (gratis)
- Puedes automatizar mientras otros trabajan

## 🔍 Ejemplos de Estrategias Quant

### 1. **Gap Trading** (Principiante)
```
REGLA: Si una acción abre 5%+ arriba del cierre anterior
Y el volumen es 2x el promedio
ENTONCES: Comprar en la apertura, vender en 30 minutos

RESULTADO HISTÓRICO: 62% win rate, 1.3 profit factor
```

### 2. **Mean Reversion** (Intermedio)
```
REGLA: Si el precio está 2 desviaciones estándar debajo de la media de 20 días
Y el RSI < 30
ENTONCES: Comprar y mantener hasta que regrese a la media

RESULTADO: 71% win rate en small caps
```

### 3. **Momentum + Volume** (Avanzado)
```
REGLA: Si precio rompe máximo de 52 semanas
Y volumen > 300% del promedio
Y sector está en tendencia alcista
ENTONCES: Comprar con stop loss dinámico

RESULTADO: 58% win rate, pero trades ganadores 3x más grandes
```

## 🛠️ Herramientas del Quant

### **Software (Todo Gratis)**
- **Python**: Lenguaje principal
- **Pandas**: Manipulación de datos
- **YFinance**: Datos de mercado gratuitos
- **Matplotlib**: Gráficos
- **Jupyter**: Análisis interactivo

### **Datos (Muchos Gratuitos)**
- **Yahoo Finance**: Precios históricos
- **Alpha Vantage**: API gratuita
- **Federal Reserve**: Datos económicos
- **SEC Filings**: Información fundamental

### **Brokers con APIs**
- **Interactive Brokers**: API profesional
- **Alpaca**: Commission-free con API
- **TD Ameritrade**: thinkorswim API

## 📊 Tu Primer Análisis Cuantitativo

### Ejercicio 1.1: Análisis de Gap
Vamos a probar si los "gaps" realmente funcionan:

```python
# No te preocupes si no entiendes todo el código aún
# Solo observa la METODOLOGÍA

import yfinance as yf
import pandas as pd

# 1. HIPÓTESIS: Las acciones que abren +5% tienden a seguir subiendo
symbol = 'AAPL'
data = yf.download(symbol, start='2020-01-01', end='2024-01-01')

# 2. IDENTIFICAR gaps
data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100

# 3. FILTRAR gaps grandes (>5%)
big_gaps = data[data['Gap'] > 5]

# 4. MEDIR qué pasó después
big_gaps['Next_Hour_Return'] = (data['High'] - data['Open']) / data['Open'] * 100

# 5. ESTADÍSTICAS
print(f"Total gaps >5%: {len(big_gaps)}")
print(f"Promedio retorno primera hora: {big_gaps['Next_Hour_Return'].mean():.2f}%")
print(f"% de gaps que siguieron subiendo: {(big_gaps['Next_Hour_Return'] > 0).mean()*100:.1f}%")
```

**Tu Tarea**: Ejecuta este código y reflexiona:
1. ¿Los números apoyan la estrategia de gaps?
2. ¿Qué cambiarías para mejorar la estrategia?
3. ¿Cómo es diferente esto a "intuir" que va a subir?

### Ejercicio 1.2: Define Tu "Por Qué"
Escribe 2-3 párrafos respondiendo:
1. **¿Por qué quieres ser quant trader?**
2. **¿Qué te atrae más: la programación, las matemáticas, o el trading?**
3. **¿Cuál es tu objetivo en 6 meses?**

## 🎯 Tipos de Quant Traders

### 1. **Retail Quant** (Objetivo inicial)
- Administra capital propio ($1K - $100K)
- Estrategias simples pero efectivas
- Automatización básica
- Trabajo de medio tiempo o hobby lucrativo

### 2. **Hedge Fund Quant** (Objetivo a largo plazo)
- Administra capital institucional ($1M+)
- Estrategias complejas con ML/AI
- Infraestructura avanzada
- Trabajo de tiempo completo, salarios $200K+

### 3. **Prop Trading Quant**
- Trading con capital de la firma
- Enfoque en high-frequency o arbitraje
- Tecnología de punta
- Compartir ganancias con la firma

### ¿Cuál te atrae más?

## 🚧 Obstáculos Comunes (Y Cómo Superarlos)

### **"No sé programar"**
**Solución**: Solo necesitas Python básico. Este curso te enseña todo lo necesario.

### **"No tengo capital"**
**Solución**: Puedes empezar con $100 en paper trading y generar track record.

### **"Es muy complicado"**
**Solución**: Estrategias simples funcionan. No necesitas ser Einstein.

### **"Ya hay mucha competencia"**
**Solución**: Retail quants compiten contra retail traders, no contra Goldman Sachs.

### **"No tengo tiempo"**
**Solución**: Una vez programado, el sistema trabaja mientras duermes.

## 🎓 Perfiles de Éxito

### **Jim Simons** (Renaissance Technologies)
- Matemático convertido en trader
- Retornos anuales promedio: 35%+
- Usa modelos estadísticos complejos
- Patrimonio neto: $25B+

### **Ray Dalio** (Bridgewater)
- Creó "All Weather" portfolio usando correlaciones
- Gestiona $150B+ usando principios cuantitativos
- Enfoque en datos macro-económicos

### **Andreas Clenow** (Retail Quant)
- Empezó como retail trader
- Escribió libros sobre trading cuantitativo
- Gestiona fondos usando estrategias "simples"
- Demostró que no necesitas PhD para ser exitoso

## ✅ Checkpoint del Módulo

Antes de continuar al siguiente módulo, debes poder:

### Conocimiento ✅
- [ ] Explicar la diferencia entre quant y trader tradicional
- [ ] Dar 3 ventajas del trading cuantitativo
- [ ] Entender qué significa "probar una hipótesis con datos"
- [ ] Identificar qué tipo de quant quieres ser

### Mentalidad ✅
- [ ] Estás convencido de que datos > intuición
- [ ] Tienes claridad sobre tu "por qué"
- [ ] Entiendes que es un proceso de aprendizaje gradual
- [ ] Estás emocionado de empezar a programar

### Ejercicios ✅
- [ ] Ejecutaste el análisis de gaps (aunque no entiendas todo)
- [ ] Escribiste tu "por qué" personal
- [ ] Identificaste qué tipo de quant quieres ser

## 🚀 Próximo Módulo

**F2: Python Trading Básico**
- Instalación y setup completo
- Descarga de datos reales
- Tus primeros gráficos
- Calculando indicadores simples

**Tiempo estimado**: 3-4 horas

---

## 💡 Reflexión Final

> **"El mejor momento para plantar un árbol fue hace 20 años. El segundo mejor momento es ahora."**

Cada día que pasa sin empezar es un día menos de datos históricos para analizar y mejorar. Los mercados generan nuevos datos cada segundo.

**Tu ventana de oportunidad es AHORA.**

🎯 **¿Listo para el siguiente módulo?** → [F2: Python Trading Básico](../f2-python-trading-basico/README.md)