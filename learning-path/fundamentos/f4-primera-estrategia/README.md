# F4: Tu Primera Estrategia Completa 🚀

**Módulo Fundamental 4 - Duración: 2-3 horas**

## 🎯 Objetivos del Módulo

Al completar este módulo tendrás:
- ✅ Una estrategia de trading completa y funcional
- ✅ Reglas claras de entrada, salida y gestión de riesgo
- ✅ Un backtest que prueba tu estrategia con datos reales
- ✅ Métricas para evaluar si tu estrategia es rentable

## 📊 La Estrategia: "Golden Cross con Filtros"

Vamos a construir una estrategia **real y probada** paso a paso.

### ¿Por qué esta estrategia?
- ✅ **Simple pero efectiva**: Fácil de entender y programar
- ✅ **Probada históricamente**: Usada por fondos institucionales
- ✅ **Adaptable**: Funciona en diferentes mercados
- ✅ **Buena para aprender**: Incluye todos los componentes esenciales

### Componentes de la Estrategia

| Componente | Descripción |
|------------|-------------|
| **Señal Principal** | Golden Cross (SMA 50 cruza SMA 200) |
| **Filtro 1** | RSI no en sobrecompra (< 70) |
| **Filtro 2** | Volumen > promedio 20 días |
| **Stop Loss** | 5% desde entrada |
| **Take Profit** | 15% desde entrada |
| **Gestión de Riesgo** | Máximo 2% del capital por trade |

## 🔨 Paso 1: Construir la Estrategia

### Código Base de la Estrategia

```python
# Para Google Colab
!pip install yfinance pandas numpy matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class GoldenCrossStrategy:
    """
    Estrategia Golden Cross con Filtros de Calidad
    """

    def __init__(self, symbol, start_date, end_date, capital_inicial=10000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.capital_inicial = capital_inicial
        self.capital = capital_inicial

        # Parámetros de la estrategia
        self.sma_corta = 50
        self.sma_larga = 200
        self.rsi_periodo = 14
        self.rsi_sobrecompra = 70
        self.volumen_filtro = 1.2  # 20% sobre promedio
        self.stop_loss_pct = 0.05   # 5%
        self.take_profit_pct = 0.15 # 15%
        self.riesgo_por_trade = 0.02 # 2% del capital

        # Datos para tracking
        self.trades = []
        self.posicion_actual = None

    def descargar_datos(self):
        """Descarga y prepara los datos"""
        print(f"📊 Descargando datos de {self.symbol}...")

        # Descargar con margen extra para calcular indicadores
        fecha_inicio_ext = pd.to_datetime(self.start_date) - timedelta(days=300)
        self.data = yf.download(self.symbol, start=fecha_inicio_ext, end=self.end_date)

        if self.data.empty:
            raise ValueError(f"No se encontraron datos para {self.symbol}")

        print(f"✅ Descargados {len(self.data)} días de datos")

    def calcular_indicadores(self):
        """Calcula todos los indicadores necesarios"""
        print("🧮 Calculando indicadores...")

        # Medias móviles
        self.data['SMA_50'] = self.data['Close'].rolling(window=self.sma_corta).mean()
        self.data['SMA_200'] = self.data['Close'].rolling(window=self.sma_larga).mean()

        # RSI
        delta = self.data['Close'].diff()
        ganancia = (delta.where(delta > 0, 0)).rolling(window=self.rsi_periodo).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_periodo).mean()
        rs = ganancia / perdida
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # Volumen promedio
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']

        # Señales de cruce
        self.data['Golden_Cross'] = (
            (self.data['SMA_50'] > self.data['SMA_200']) &
            (self.data['SMA_50'].shift(1) <= self.data['SMA_200'].shift(1))
        )

        self.data['Death_Cross'] = (
            (self.data['SMA_50'] < self.data['SMA_200']) &
            (self.data['SMA_50'].shift(1) >= self.data['SMA_200'].shift(1))
        )

        # Limpiar NaN del inicio
        self.data = self.data[self.data.index >= self.start_date].dropna()

        print("✅ Indicadores calculados")

    def generar_señales(self):
        """Genera señales de compra/venta basadas en la estrategia"""
        print("🎯 Generando señales de trading...")

        self.data['Señal'] = 0  # 0: Sin posición, 1: Compra, -1: Venta

        for i in range(len(self.data)):
            fecha = self.data.index[i]
            row = self.data.iloc[i]

            # SEÑAL DE COMPRA
            if (row['Golden_Cross'] and
                row['RSI'] < self.rsi_sobrecompra and
                row['Volume_Ratio'] > self.volumen_filtro and
                self.posicion_actual is None):

                self.data.loc[fecha, 'Señal'] = 1
                self.abrir_posicion(fecha, row['Close'], 'LONG')

            # VERIFICAR STOPS SI HAY POSICIÓN
            elif self.posicion_actual is not None:
                precio_actual = row['Close']
                precio_entrada = self.posicion_actual['precio_entrada']

                # Stop Loss
                if precio_actual <= precio_entrada * (1 - self.stop_loss_pct):
                    self.data.loc[fecha, 'Señal'] = -1
                    self.cerrar_posicion(fecha, precio_actual, 'STOP_LOSS')

                # Take Profit
                elif precio_actual >= precio_entrada * (1 + self.take_profit_pct):
                    self.data.loc[fecha, 'Señal'] = -1
                    self.cerrar_posicion(fecha, precio_actual, 'TAKE_PROFIT')

                # Death Cross (salida por señal)
                elif row['Death_Cross']:
                    self.data.loc[fecha, 'Señal'] = -1
                    self.cerrar_posicion(fecha, precio_actual, 'DEATH_CROSS')

        # Cerrar posición al final si queda abierta
        if self.posicion_actual is not None:
            ultimo_precio = self.data['Close'].iloc[-1]
            self.cerrar_posicion(self.data.index[-1], ultimo_precio, 'FIN_PERIODO')

        print(f"✅ Generadas {len(self.trades)} operaciones")

    def abrir_posicion(self, fecha, precio, tipo):
        """Abre una nueva posición"""

        # Calcular tamaño de posición basado en riesgo
        capital_a_riesgo = self.capital * self.riesgo_por_trade
        stop_loss_precio = precio * (1 - self.stop_loss_pct)
        riesgo_por_accion = precio - stop_loss_precio
        num_acciones = int(capital_a_riesgo / riesgo_por_accion)

        # Limitar al capital disponible
        max_acciones = int(self.capital * 0.95 / precio)  # Usar máximo 95% del capital
        num_acciones = min(num_acciones, max_acciones)

        self.posicion_actual = {
            'fecha_entrada': fecha,
            'precio_entrada': precio,
            'num_acciones': num_acciones,
            'tipo': tipo,
            'stop_loss': stop_loss_precio,
            'take_profit': precio * (1 + self.take_profit_pct)
        }

    def cerrar_posicion(self, fecha, precio, razon):
        """Cierra la posición actual y registra el trade"""

        if self.posicion_actual is None:
            return

        # Calcular resultado
        precio_entrada = self.posicion_actual['precio_entrada']
        num_acciones = self.posicion_actual['num_acciones']
        ganancia_perdida = (precio - precio_entrada) * num_acciones
        retorno_pct = ((precio - precio_entrada) / precio_entrada) * 100

        # Actualizar capital
        self.capital += ganancia_perdida

        # Registrar trade
        trade = {
            'fecha_entrada': self.posicion_actual['fecha_entrada'],
            'fecha_salida': fecha,
            'precio_entrada': precio_entrada,
            'precio_salida': precio,
            'num_acciones': num_acciones,
            'ganancia_perdida': ganancia_perdida,
            'retorno_pct': retorno_pct,
            'razon_salida': razon,
            'capital_despues': self.capital
        }
        self.trades.append(trade)

        # Limpiar posición
        self.posicion_actual = None

    def calcular_metricas(self):
        """Calcula métricas de performance de la estrategia"""

        if not self.trades:
            print("❌ No hay trades para analizar")
            return None

        df_trades = pd.DataFrame(self.trades)

        # Métricas básicas
        total_trades = len(df_trades)
        trades_ganadores = len(df_trades[df_trades['ganancia_perdida'] > 0])
        trades_perdedores = len(df_trades[df_trades['ganancia_perdida'] < 0])
        win_rate = (trades_ganadores / total_trades) * 100

        # Ganancias y pérdidas
        total_ganancia = df_trades[df_trades['ganancia_perdida'] > 0]['ganancia_perdida'].sum()
        total_perdida = abs(df_trades[df_trades['ganancia_perdida'] < 0]['ganancia_perdida'].sum())
        profit_factor = total_ganancia / total_perdida if total_perdida > 0 else np.inf

        # Retornos
        retorno_total = ((self.capital - self.capital_inicial) / self.capital_inicial) * 100
        promedio_ganancia = df_trades[df_trades['ganancia_perdida'] > 0]['retorno_pct'].mean()
        promedio_perdida = df_trades[df_trades['ganancia_perdida'] < 0]['retorno_pct'].mean()

        # Drawdown máximo
        capital_maximo = self.capital_inicial
        max_drawdown = 0
        for trade in self.trades:
            capital_maximo = max(capital_maximo, trade['capital_despues'])
            drawdown = ((capital_maximo - trade['capital_despues']) / capital_maximo) * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Buy & Hold comparison
        precio_inicial = self.data['Close'].iloc[0]
        precio_final = self.data['Close'].iloc[-1]
        buy_hold_return = ((precio_final - precio_inicial) / precio_inicial) * 100

        metricas = {
            'total_trades': total_trades,
            'trades_ganadores': trades_ganadores,
            'trades_perdedores': trades_perdedores,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'retorno_total': retorno_total,
            'promedio_ganancia': promedio_ganancia,
            'promedio_perdida': promedio_perdida,
            'max_drawdown': max_drawdown,
            'capital_final': self.capital,
            'buy_hold_return': buy_hold_return
        }

        return metricas

    def visualizar_resultados(self):
        """Crea visualizaciones de los resultados"""

        fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

        # 1. Precio y Señales
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Precio', linewidth=1, alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_50'], label='SMA 50', linewidth=1, alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_200'], label='SMA 200', linewidth=1, alpha=0.7)

        # Marcar entradas y salidas
        entradas = self.data[self.data['Señal'] == 1]
        salidas = self.data[self.data['Señal'] == -1]

        ax1.scatter(entradas.index, entradas['Close'], color='green', marker='^',
                   s=100, label=f'Compras ({len(entradas)})', zorder=5)
        ax1.scatter(salidas.index, salidas['Close'], color='red', marker='v',
                   s=100, label=f'Ventas ({len(salidas)})', zorder=5)

        ax1.set_title(f'{self.symbol} - Estrategia Golden Cross con Filtros', fontsize=14)
        ax1.set_ylabel('Precio ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. RSI
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['RSI'], color='purple', linewidth=1)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        # 3. Volumen
        ax3 = axes[2]
        colors = ['green' if s == 1 else 'red' if s == -1 else 'gray' for s in self.data['Señal']]
        ax3.bar(self.data.index, self.data['Volume'], color=colors, alpha=0.5)
        ax3.plot(self.data.index, self.data['Volume_MA'], color='blue', linewidth=1)
        ax3.set_ylabel('Volumen')
        ax3.grid(True, alpha=0.3)

        # 4. Equity Curve
        ax4 = axes[3]
        if self.trades:
            equity_data = [self.capital_inicial]
            equity_dates = [self.data.index[0]]

            for trade in self.trades:
                equity_dates.append(trade['fecha_salida'])
                equity_data.append(trade['capital_despues'])

            ax4.plot(equity_dates, equity_data, color='blue', linewidth=2, label='Estrategia')

            # Comparar con Buy & Hold
            buy_hold_values = self.capital_inicial * (self.data['Close'] / self.data['Close'].iloc[0])
            ax4.plot(self.data.index, buy_hold_values, color='gray', linewidth=1,
                    alpha=0.7, label='Buy & Hold')

            ax4.set_ylabel('Capital ($)')
            ax4.set_xlabel('Fecha')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def ejecutar_backtest(self):
        """Ejecuta el backtest completo"""
        print("\\n" + "="*60)
        print(f"🚀 BACKTESTING: {self.symbol}")
        print(f"📅 Período: {self.start_date} a {self.end_date}")
        print(f"💰 Capital Inicial: ${self.capital_inicial:,.2f}")
        print("="*60)

        # Ejecutar pasos
        self.descargar_datos()
        self.calcular_indicadores()
        self.generar_señales()

        # Calcular métricas
        metricas = self.calcular_metricas()

        if metricas:
            print("\\n📊 RESULTADOS DEL BACKTEST:")
            print("="*60)
            print(f"Total de Operaciones: {metricas['total_trades']}")
            print(f"Operaciones Ganadoras: {metricas['trades_ganadores']}")
            print(f"Operaciones Perdedoras: {metricas['trades_perdedores']}")
            print(f"\\n✅ Win Rate: {metricas['win_rate']:.2f}%")
            print(f"📈 Profit Factor: {metricas['profit_factor']:.2f}")
            print(f"\\n💰 Capital Final: ${metricas['capital_final']:,.2f}")
            print(f"📊 Retorno Total: {metricas['retorno_total']:.2f}%")
            print(f"📉 Max Drawdown: {metricas['max_drawdown']:.2f}%")
            print(f"\\n🔄 Buy & Hold Return: {metricas['buy_hold_return']:.2f}%")

            if metricas['retorno_total'] > metricas['buy_hold_return']:
                print("\\n✅ ¡La estrategia SUPERA a Buy & Hold!")
            else:
                print("\\n❌ La estrategia NO supera a Buy & Hold")

        # Visualizar
        self.visualizar_resultados()

        return metricas

# EJECUTAR LA ESTRATEGIA
if __name__ == "__main__":
    # Configurar parámetros
    SYMBOL = 'AAPL'  # Puedes cambiar el símbolo
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    CAPITAL_INICIAL = 10000

    # Crear y ejecutar estrategia
    estrategia = GoldenCrossStrategy(SYMBOL, START_DATE, END_DATE, CAPITAL_INICIAL)
    resultados = estrategia.ejecutar_backtest()
```

## 📈 Paso 2: Optimizar la Estrategia

### Probando Diferentes Parámetros

```python
def optimizar_parametros(symbol='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """Prueba diferentes combinaciones de parámetros"""

    print("🔧 OPTIMIZACIÓN DE PARÁMETROS")
    print("="*60)

    resultados_optimizacion = []

    # Parámetros a probar
    sma_cortas = [20, 30, 50]
    sma_largas = [100, 150, 200]
    stop_losses = [0.03, 0.05, 0.07]
    take_profits = [0.10, 0.15, 0.20]

    mejor_resultado = None
    mejor_retorno = -np.inf

    for sma_c in sma_cortas:
        for sma_l in sma_largas:
            if sma_c >= sma_l:
                continue

            for sl in stop_losses:
                for tp in take_profits:
                    # Crear estrategia con estos parámetros
                    estrategia = GoldenCrossStrategy(symbol, start_date, end_date)
                    estrategia.sma_corta = sma_c
                    estrategia.sma_larga = sma_l
                    estrategia.stop_loss_pct = sl
                    estrategia.take_profit_pct = tp

                    # Ejecutar backtest silenciosamente
                    try:
                        estrategia.descargar_datos()
                        estrategia.calcular_indicadores()
                        estrategia.generar_señales()
                        metricas = estrategia.calcular_metricas()

                        if metricas and metricas['total_trades'] > 0:
                            resultado = {
                                'sma_corta': sma_c,
                                'sma_larga': sma_l,
                                'stop_loss': sl,
                                'take_profit': tp,
                                'retorno': metricas['retorno_total'],
                                'win_rate': metricas['win_rate'],
                                'profit_factor': metricas['profit_factor'],
                                'max_drawdown': metricas['max_drawdown'],
                                'num_trades': metricas['total_trades']
                            }

                            resultados_optimizacion.append(resultado)

                            if metricas['retorno_total'] > mejor_retorno:
                                mejor_retorno = metricas['retorno_total']
                                mejor_resultado = resultado

                    except:
                        continue

    # Mostrar resultados
    if mejor_resultado:
        print("\\n🏆 MEJORES PARÁMETROS ENCONTRADOS:")
        print("="*60)
        print(f"SMA Corta: {mejor_resultado['sma_corta']}")
        print(f"SMA Larga: {mejor_resultado['sma_larga']}")
        print(f"Stop Loss: {mejor_resultado['stop_loss']*100:.1f}%")
        print(f"Take Profit: {mejor_resultado['take_profit']*100:.1f}%")
        print(f"\\nRetorno: {mejor_resultado['retorno']:.2f}%")
        print(f"Win Rate: {mejor_resultado['win_rate']:.2f}%")
        print(f"Profit Factor: {mejor_resultado['profit_factor']:.2f}")
        print(f"Max Drawdown: {mejor_resultado['max_drawdown']:.2f}%")

    return pd.DataFrame(resultados_optimizacion)

# Ejecutar optimización
df_optimizacion = optimizar_parametros('AAPL')

# Ver top 5 combinaciones
print("\\n📊 TOP 5 COMBINACIONES:")
print(df_optimizacion.nlargest(5, 'retorno')[['sma_corta', 'sma_larga', 'stop_loss', 'take_profit', 'retorno', 'win_rate']])
```

## 🎯 Paso 3: Validación y Mejoras

### Analizando Debilidades

```python
def analizar_trades_detallado(estrategia):
    """Análisis detallado de cada trade"""

    if not estrategia.trades:
        print("No hay trades para analizar")
        return

    df_trades = pd.DataFrame(estrategia.trades)

    print("\\n📋 ANÁLISIS DETALLADO DE TRADES:")
    print("="*60)

    # Análisis por razón de salida
    print("\\n📊 Distribución de Salidas:")
    for razon in df_trades['razon_salida'].unique():
        trades_razon = df_trades[df_trades['razon_salida'] == razon]
        avg_return = trades_razon['retorno_pct'].mean()
        count = len(trades_razon)
        print(f"{razon}: {count} trades, Retorno promedio: {avg_return:.2f}%")

    # Duración de trades
    df_trades['duracion'] = (pd.to_datetime(df_trades['fecha_salida']) -
                            pd.to_datetime(df_trades['fecha_entrada'])).dt.days

    print(f"\\n⏱️ Duración promedio: {df_trades['duracion'].mean():.1f} días")
    print(f"Duración máxima: {df_trades['duracion'].max()} días")
    print(f"Duración mínima: {df_trades['duracion'].min()} días")

    # Mejor y peor trade
    mejor_trade = df_trades.loc[df_trades['ganancia_perdida'].idxmax()]
    peor_trade = df_trades.loc[df_trades['ganancia_perdida'].idxmin()]

    print(f"\\n🏆 Mejor Trade:")
    print(f"  Entrada: {mejor_trade['fecha_entrada']}")
    print(f"  Ganancia: ${mejor_trade['ganancia_perdida']:.2f} ({mejor_trade['retorno_pct']:.2f}%)")

    print(f"\\n😞 Peor Trade:")
    print(f"  Entrada: {peor_trade['fecha_entrada']}")
    print(f"  Pérdida: ${peor_trade['ganancia_perdida']:.2f} ({peor_trade['retorno_pct']:.2f}%)")

    return df_trades

# Analizar trades de la última estrategia
df_trades = analizar_trades_detallado(estrategia)
```

## 🏆 Proyecto Final: Multi-Estrategia

### Comparando Múltiples Estrategias

```python
def comparar_estrategias():
    """Compara diferentes estrategias en el mismo período"""

    estrategias_config = [
        {'nombre': 'Golden Cross Original', 'sma_c': 50, 'sma_l': 200, 'sl': 0.05, 'tp': 0.15},
        {'nombre': 'Golden Cross Agresivo', 'sma_c': 20, 'sma_l': 50, 'sl': 0.03, 'tp': 0.10},
        {'nombre': 'Golden Cross Conservador', 'sma_c': 50, 'sma_l': 200, 'sl': 0.07, 'tp': 0.20},
        {'nombre': 'Golden Cross Rápido', 'sma_c': 10, 'sma_l': 30, 'sl': 0.02, 'tp': 0.05},
    ]

    resultados_comparacion = []

    for config in estrategias_config:
        print(f"\\nProbando: {config['nombre']}...")

        estrategia = GoldenCrossStrategy('SPY', '2020-01-01', '2024-01-01', 10000)
        estrategia.sma_corta = config['sma_c']
        estrategia.sma_larga = config['sma_l']
        estrategia.stop_loss_pct = config['sl']
        estrategia.take_profit_pct = config['tp']

        estrategia.descargar_datos()
        estrategia.calcular_indicadores()
        estrategia.generar_señales()
        metricas = estrategia.calcular_metricas()

        if metricas:
            resultados_comparacion.append({
                'Estrategia': config['nombre'],
                'Retorno (%)': metricas['retorno_total'],
                'Win Rate (%)': metricas['win_rate'],
                'Profit Factor': metricas['profit_factor'],
                'Max DD (%)': metricas['max_drawdown'],
                'Trades': metricas['total_trades']
            })

    # Crear tabla comparativa
    df_comparacion = pd.DataFrame(resultados_comparacion)
    df_comparacion = df_comparacion.round(2)

    print("\\n" + "="*80)
    print("📊 COMPARACIÓN DE ESTRATEGIAS")
    print("="*80)
    print(df_comparacion.to_string(index=False))

    # Identificar ganadora
    mejor_estrategia = df_comparacion.loc[df_comparacion['Retorno (%)'].idxmax()]
    print(f"\\n🏆 MEJOR ESTRATEGIA: {mejor_estrategia['Estrategia']}")
    print(f"   Retorno: {mejor_estrategia['Retorno (%)']}%")
    print(f"   Profit Factor: {mejor_estrategia['Profit Factor']}")

    return df_comparacion

# Ejecutar comparación
df_comparacion = comparar_estrategias()
```

## ✅ Checkpoint del Módulo

### Estrategia Completa ✅
- [ ] Mi estrategia tiene reglas claras de entrada
- [ ] Implementé stop loss y take profit
- [ ] Uso gestión de riesgo (2% por trade)
- [ ] Combino múltiples indicadores

### Backtest Funcional ✅
- [ ] Puedo probar con datos históricos reales
- [ ] Calculo métricas de performance
- [ ] Comparo con Buy & Hold
- [ ] Visualizo resultados claramente

### Optimización ✅
- [ ] Probé diferentes parámetros
- [ ] Identifiqué mejores configuraciones
- [ ] Analicé trades individuales
- [ ] Comparé múltiples variaciones

### Aprendizajes ✅
- [ ] Entiendo por qué algunas estrategias fallan
- [ ] Sé la importancia del stop loss
- [ ] Veo el impacto de los parámetros
- [ ] Puedo mejorar la estrategia

## 🎓 ¡FELICITACIONES!

**Has completado los FUNDAMENTOS del trading cuantitativo.**

Ya tienes:
- ✅ Conocimiento de qué es ser quant
- ✅ Habilidades de Python para trading
- ✅ Dominio de indicadores técnicos
- ✅ Tu primera estrategia completa y probada

## 🚀 Próximos Pasos

### Continúa con ESTRATEGIAS (Nivel 2)

**E1: Momentum Trading**
- Gap & Go para small caps
- Breakout strategies
- Momentum scanning

**E2: Mean Reversion**
- VWAP reclaim
- Oversold bounces
- Pairs trading

**E3: Backtesting Avanzado**
- Walk-forward analysis
- Monte Carlo simulation
- Stress testing

---

### 💡 Reflexión Final

> **"Una estrategia mediocre bien ejecutada es mejor que una estrategia perfecta mal ejecutada."**

Tu primera estrategia no será perfecta, pero ya tienes las herramientas para mejorarla continuamente. Cada día de trading genera nuevos datos para aprender.

**¡Ya eres oficialmente un Quant Trader en formación! 🎉**

---

🎯 **¿Listo para estrategias más avanzadas?** → [E1: Momentum Trading](../../estrategias/e1-momentum-trading/)