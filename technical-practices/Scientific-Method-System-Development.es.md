> 🇺🇸 [Read in English](Scientific-Method-System-Development.md) | 🇪🇸 **Español**

# Método Científico Aplicado al Desarrollo de Sistemas

Desarrollar un sistema de trading no es "probar cosas hasta que algo funcione". Es un proceso científico con protocolo, aislamiento de variables, y evaluación estadística. Si no seguís un método estructurado, estás haciendo data snooping disfrazado de investigación.

## Los Tres Pilares del Método Científico en Trading

### 1. Objeto de Estudio

Tu sistema de trading — o una de sus partes. Cada componente debe ser observable de forma objetiva, sin depender de tu juicio subjetivo.

Esto implica que todo lo que evalúes debe poder expresarse en números: win rate, profit factor, drawdown, Sharpe ratio. "Me parece que funciona bien" no es una evaluación — es una opinión.

### 2. Procedimiento Estandarizado

Un método ordenado para evaluar el comportamiento del sistema. El procedimiento debe ser **reproducible**: si vos lo hacés o lo hace otra persona con los mismos datos y el mismo protocolo, los resultados deben ser compatibles.

Si tu proceso de evaluación da resultados distintos según quién lo ejecute, hay subjetividad infiltrada en alguna fase.

### 3. Evaluación Estadística

Los resultados se evalúan con herramientas estadísticas, no con intuición. Una racha de 15 trades ganadores no prueba que un sistema funcione, y una racha de 10 pérdidas no prueba que esté roto. Lo que importa es la significancia estadística sobre una muestra suficiente.

## El Protocolo: La Palabra Clave

El método científico aplicado al trading se resume en una palabra: **protocolo**. Todos los procesos que seguís al desarrollar un sistema deben estar protocolizados. No cambiás las reglas a mitad de camino, no hacés excepciones "porque esta vez es diferente", no ajustás el proceso después de ver los resultados.

```python
class SystemDevelopmentProtocol:
    """
    Framework para desarrollo de sistemas siguiendo el método científico.
    Cada fase tiene inputs definidos, proceso estandarizado, y outputs medibles.
    """

    def __init__(self):
        self.phases = [
            "1_hypothesis",      # idea con lógica de mercado
            "2_data_preparation", # datos limpios, período definido
            "3_in_sample_test",   # backtest en datos de entrenamiento
            "4_parameter_selection",  # optimización con protocolo
            "5_out_of_sample",    # validación en datos NO vistos
            "6_walk_forward",     # validación temporal progresiva
            "7_robustness_tests", # Monte Carlo, sensibilidad
            "8_paper_trading",    # ejecución sin capital real
            "9_live_deployment",  # capital real, tamaño reducido
        ]
        self.log = []

    def execute_phase(self, phase_name, inputs, process, outputs):
        """
        Cada fase se documenta ANTES de ejecutarse.
        No se puede cambiar el criterio de éxito después de ver los resultados.
        """
        record = {
            'phase': phase_name,
            'inputs': inputs,
            'process': process,
            'expected_outputs': outputs,
            'status': 'pending'
        }
        self.log.append(record)
        return record
```

### Aislamiento de Variables

Cuando evaluás un sistema, cambiá **una cosa a la vez**. Si modificás el indicador de entrada, el stop loss, y el filtro de mercado simultáneamente, no sabés cuál de los tres causó el cambio en los resultados.

```python
def isolated_test(base_system, variable_name, variable_values, data):
    """
    Testear el efecto de UNA variable manteniendo todo lo demás fijo.
    Esto es el método científico: aislar para entender causalidad.
    """
    results = {}
    for value in variable_values:
        # Modificar solo la variable bajo estudio
        test_system = base_system.copy()
        test_system[variable_name] = value

        # Ejecutar backtest con todo lo demás idéntico
        metrics = run_backtest(test_system, data)
        results[value] = {
            'profit_factor': metrics['profit_factor'],
            'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_drawdown'],
            'trades': metrics['total_trades']
        }

    return results

# Ejemplo: testear distintos períodos de media móvil,
# manteniendo fijo el stop loss, el take profit, y el filtro de volumen
# results = isolated_test(my_system, 'ma_period', range(10, 60, 5), data)
```

### Pruebas Estandarizadas

Para que las comparaciones sean válidas, las condiciones del test deben ser idénticas:

- **Mismo período de datos** para todos los tests de una comparación
- **Mismos costos de transacción** (comisiones, slippage)
- **Mismo capital inicial** y reglas de position sizing
- **Misma semilla** si hay componentes aleatorios

Si comparás el Sistema A testeado en 2020-2023 con el Sistema B testeado en 2018-2024, la comparación no es válida. Los períodos de mercado son diferentes.

## De la Hipótesis al Sistema: El Flujo Completo

### Fase 1: Hipótesis con Lógica de Mercado

Toda idea de sistema debe partir de una razón lógica de por qué debería funcionar. "Compro cuando el RSI cruza 30 porque el backtest da bien" no es una hipótesis — es overfitting esperando a ocurrir.

Una hipótesis válida se basa en una ineficiencia de mercado explicable:

```
BIEN: "Las acciones small cap que abren con gap >10% con volumen alto
tienden a continuar en la dirección del gap durante los primeros 30 minutos
porque los traders retail entran tarde persiguiendo el movimiento."

MAL: "Si la EMA de 13 cruza la EMA de 34 cuando el RSI está entre
42 y 58 y el MACD es positivo, el precio sube."
```

La primera tiene una razón de mercado (comportamiento de participantes). La segunda es una combinación arbitraria de indicadores que probablemente sea ruido.

### Fase 2: División de Datos

Antes de tocar un solo parámetro, dividí tus datos:

```python
def split_data(data, in_sample_pct=0.60, validation_pct=0.20):
    """
    División de datos ANTES de cualquier optimización.
    Una vez divididos, no se tocan los límites.
    """
    n = len(data)
    is_end = int(n * in_sample_pct)
    val_end = int(n * (in_sample_pct + validation_pct))

    return {
        'in_sample': data[:is_end],           # para desarrollar y optimizar
        'validation': data[is_end:val_end],    # para validar candidatos
        'out_of_sample': data[val_end:]         # NUNCA se toca hasta el final
    }

# REGLA INQUEBRANTABLE: el out-of-sample se usa UNA VEZ.
# Si lo usás para ajustar y volvés a testear, ya no es out-of-sample.
```

### Fase 3: Optimización In-Sample

Buscás los mejores parámetros usando solo los datos in-sample. Pero "mejor" no significa "más rentable" — significa más **robusto**.

Un sistema robusto mantiene resultados aceptables en un rango amplio de parámetros. Si solo funciona con MA=17 pero falla con MA=15 y MA=19, es frágil.

```python
def evaluate_robustness(optimization_results, metric='profit_factor'):
    """
    Un sistema robusto tiene una "meseta" de parámetros buenos,
    no un pico aislado. Si los vecinos del óptimo también funcionan,
    la señal es real. Si solo funciona un punto, es ruido.
    """
    values = [r[metric] for r in optimization_results]
    peak_idx = values.index(max(values))

    # Verificar que los parámetros vecinos también sean buenos
    neighbors = []
    for offset in [-2, -1, 1, 2]:
        idx = peak_idx + offset
        if 0 <= idx < len(values):
            neighbors.append(values[idx])

    if not neighbors:
        return False, "Insuficientes datos para evaluar"

    peak_value = values[peak_idx]
    avg_neighbor = sum(neighbors) / len(neighbors)
    ratio = avg_neighbor / peak_value if peak_value > 0 else 0

    # Si los vecinos retienen >70% del valor del pico, es robusto
    return ratio > 0.70, f"Ratio vecinos/pico: {ratio:.2f}"
```

### Fase 4: Validación

El sistema con los parámetros seleccionados se testea en los datos de validación. No se modifica nada. Si pasa, avanza. Si no pasa, se descarta o se vuelve a la hipótesis.

**No se reoptimiza para que pase la validación.** Eso convierte la validación en in-sample.

### Fase 5: Out-of-Sample

El test definitivo. Una sola oportunidad. Si los resultados son consistentes con el in-sample (no idénticos — consistentes), el sistema es candidato para paper trading.

### Fase 6: Paper Trading

Ejecución real sin capital. Verificás que la ejecución en tiempo real produce resultados compatibles con el backtest. Diferencias esperables: slippage, timing de ejecución, datos que difieren ligeramente del histórico.

### Fase 7: Live con Tamaño Reducido

Capital real, pero con el tamaño de posición mínimo posible. El objetivo no es ganar dinero — es validar que todo funciona en producción.

## Errores que el Protocolo Previene

| Error | Sin protocolo | Con protocolo |
|---|---|---|
| **Overfitting** | Optimizás hasta que el backtest sea perfecto | Validás en datos no vistos, evaluás robustez de parámetros |
| **Data snooping** | Probás 500 combinaciones y elegís la mejor | Definís la hipótesis antes de testear, limitás las variables |
| **Look-ahead bias** | Usás información futura sin darte cuenta | El procedimiento estandarizado fuerza el shift temporal |
| **Survivorship bias** | Testeas con las acciones que existen hoy | El protocolo exige datos con composición histórica real |
| **Cambio de criterio** | "Este sistema no pasa mi filtro, pero lo voy a usar igual" | El criterio se define antes de ver resultados, y no se cambia |

## Cuándo el Protocolo Parece Excesivo

Al principio, seguir un protocolo completo para cada idea parece lento. Y lo es — pero te ahorra meses de operar sistemas que no funcionan.

Un atajo legítimo para ideas iniciales: antes de todo el proceso formal, hacé un **test rápido** con parámetros default y sin optimización. Si la idea no muestra señales de vida con parámetros genéricos, no vale la pena dedicarle un protocolo completo.

Pero una vez que decidís avanzar con una idea, el protocolo no es negociable. "Voy a saltarme la validación porque me lleva tiempo" es exactamente cómo terminás operando sistemas overfitteados con dinero real.
