# Calculadora de Sharpe Ratio

Script completo para calcular Sharpe ratio y métricas relacionadas de estrategias de trading cuantitativo.

## Características

✅ **Múltiples formatos de entrada**: Returns directos o trades con P&L
✅ **Análisis completo**: Sharpe, Sortino, Calmar, Information Ratio
✅ **Comparación con benchmarks**: SPY, QQQ, sector ETFs
✅ **Sharpe ratio rodante**: Para análisis temporal
✅ **Validaciones robustas**: Manejo de datos faltantes y outliers

## Uso Rápido

```bash
# Análisis básico desde returns
python calculate_sharpe.py --returns daily_returns.csv

# Análisis desde trades
python calculate_sharpe.py --trades my_trades.csv --capital 10000

# Comparación con benchmark
python calculate_sharpe.py --returns strategy.csv --benchmark spy.csv
```

## Formato de Datos

### Returns CSV
```csv
date,returns
2024-01-01,0.025
2024-01-02,-0.012
2024-01-03,0.018
```

### Trades CSV
```csv
date,pnl
2024-01-01,150.50
2024-01-02,-75.25
2024-01-03,200.00
```

## Métricas Calculadas

| Métrica | Descripción | Interpretación |
|---------|-------------|----------------|
| **Sharpe Ratio** | Return ajustado por riesgo | > 1.0 = Bueno, > 2.0 = Excelente |
| **Sortino Ratio** | Solo considera downside risk | Mejor para estrategias asimétricas |
| **Calmar Ratio** | Return anual / Max Drawdown | Eficiencia vs pérdida máxima |
| **Information Ratio** | Excess return vs benchmark | Skill del trader vs mercado |

## Interpretación Sharpe Ratio

```
> 2.0   : Excelente 🚀
1.0-2.0 : Muy bueno ✅
0.5-1.0 : Bueno 👍
0.0-0.5 : Pobre ⚠️
< 0.0   : Destruye valor ❌
```

## Ejemplos Prácticos

### Gap & Go Strategy Analysis
```bash
python calculate_sharpe.py \
  --trades gap_go_trades.csv \
  --benchmark spy_returns.csv \
  --capital 10000 \
  --rf-rate 0.045
```

### Rolling Sharpe para Monitoring
```bash
python calculate_sharpe.py \
  --returns daily_returns.csv \
  --rolling-window 30 \
  --period daily
```

## Consideraciones Importantes

⚠️ **Limitaciones del Sharpe Ratio**:
- Asume distribución normal de returns
- Sensible a outliers extremos
- No captura tail risk
- Período de medición afecta el resultado

💡 **Mejores Prácticas**:
- Usar múltiples métricas (Sharpe + Sortino + Calmar)
- Comparar siempre con benchmarks relevantes
- Analizar Sharpe rolling para detectar degradación
- Considerar régimen de mercado (bull vs bear)

## Configuración Avanzada

```python
from calculate_sharpe import SharpeCalculator

# Personalizar risk-free rate
calc = SharpeCalculator(risk_free_rate=0.045)

# Análisis detallado
analysis = calc.detailed_analysis(returns, period='daily')
print(f"Sharpe: {analysis['sharpe_ratio']:.4f}")
print(f"Max DD: {analysis['max_drawdown']:.2%}")
```

## Integración con Otros Scripts

Este script es parte del **Quant Playbook** y se integra con:

- `../max-drawdown/`: Análisis de drawdown detallado
- `../profit-factor/`: Cálculo de profit factor
- `../../backtesting/`: Validación de estrategias
- `../../data-collection/`: Pipelines de datos

## Troubleshooting

### Error: "Datos insuficientes"
- Mínimo 2 observaciones válidas
- Verificar formato de fechas
- Remover filas con NaN

### Sharpe ratio = ∞
- Volatilidad = 0 (todos los returns iguales)
- Usar período más largo
- Verificar calidad de datos

### Comparación con benchmark falla
- Alinear fechas entre strategy y benchmark
- Usar mismo período de tiempo
- Verificar formato de columnas