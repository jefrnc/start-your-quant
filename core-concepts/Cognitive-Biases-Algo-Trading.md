# Sesgos Cognitivos en Trading Algorítmico

"Si automatizo, las emociones no me afectan." Falso. Tu robot ejecuta, pero vos decidís cuándo activarlo, cuándo pararlo, qué datos usar para diseñarlo, y cómo reaccionar cuando pierde 12 veces seguidas. La psicología sigue siendo el componente más frágil del trading algorítmico.

## Las Tres M's de Alexander Elder

Alexander Elder, en *Trading for a Living* (1993), descompuso el trading en tres pilares:

- **Mind (Mente)**: la psicología del trader
- **Money (Dinero)**: gestión de posición y protección de capital
- **Method (Método)**: las reglas y estrategias

En trading algorítmico, tendemos a pensar que solo importa el Method (el código) y el Money (el position sizing). Pero la Mind sigue siendo el pilar más importante porque:

- Vos decidís si parar un sistema en drawdown o dejarlo correr
- Vos elegís qué datos usar y cómo validar
- Vos interpretás los resultados del backtest
- Vos podés caer en la tentación de intervenir manualmente cuando "sabés" que el mercado va a hacer algo

Sin Mind sólido, el mejor Method se sabotea solo.

## Tu Perfil de Riesgo: El Test que Revela tu Sesgo

Antes de diseñar cualquier sistema, necesitás entender cómo tu mente procesa ganancias y pérdidas. Estos dos escenarios, basados en la teoría de prospectos de Kahneman y Tversky, lo revelan.

### Escenario 1: Pérdidas

Elegí UNA opción:
- **A**: Pérdida segura de $9,000
- **B**: 95% de probabilidad de perder $10,000 + 5% de probabilidad de no perder nada

### Escenario 2: Ganancias

Elegí UNA opción:
- **A**: Ganancia segura de $9,000
- **B**: 95% de probabilidad de ganar $10,000 + 5% de probabilidad de no ganar nada

### Las Respuestas Matemáticamente Correctas

```python
def expected_value(outcomes):
    """Esperanza matemática: suma de (probabilidad × valor) para cada suceso."""
    return sum(prob * value for prob, value in outcomes)

# Escenario 1
ev_loss_sure = expected_value([(1.0, -9000)])           # = -9,000
ev_loss_game = expected_value([(0.95, -10000), (0.05, 0)])  # = -9,500
print(f"Escenario 1: Pérdida segura EV={ev_loss_sure}, Jugar EV={ev_loss_game}")
# Mejor opción: pérdida segura (-9,000 > -9,500)

# Escenario 2
ev_gain_sure = expected_value([(1.0, 9000)])             # = 9,000
ev_gain_game = expected_value([(0.95, 10000), (0.05, 0)])  # = 9,500
print(f"Escenario 2: Ganancia segura EV={ev_gain_sure}, Jugar EV={ev_gain_game}")
# Mejor opción: jugar (9,500 > 9,000)
```

**Lo que la mayoría elige**: en el Escenario 1, jugar (incorrecto). En el Escenario 2, la ganancia segura (incorrecto).

**Lo que esto significa para tu trading**:
- **Escenario 1**: nos cuesta aceptar pérdidas seguras → en el mercado, esto se traduce en no cortar pérdidas, mover el stop, o "esperar que vuelva"
- **Escenario 2**: nos cuesta dejar correr ganancias → cerramos posiciones ganadoras prematuramente por miedo a perder lo ganado

Si elegiste correctamente ambas, tenés una ventaja psicológica real. Si no, no es un problema — es la reacción normal del 70-80% de las personas. Pero ahora lo sabés, y podés diseñar tus sistemas con reglas que te protejan de vos mismo.

## Los Sesgos por Fase de Desarrollo

Los sesgos no aparecen todos juntos. Diferentes fases de tu trabajo como quant son vulnerables a diferentes sesgos.

### Fase 1: Búsqueda de Ideas

| Sesgo | Qué es | Cómo te afecta en algo trading |
|---|---|---|
| **Optimismo/Pesimismo** | Predisposición a ver todo positivo o negativo, sin base en datos | Descartás ideas viables por pesimismo o te enamorás de ideas malas por optimismo |
| **Exceso de confianza** | Sobreestimar tus predicciones y habilidades | Tomás más riesgo del necesario, subestimás drawdowns posibles |
| **Aversión a la pérdida** | Darle más peso a las pérdidas que a las ganancias equivalentes | No cortás pérdidas, cerrás ganancias prematuramente |

### Fase 2: Investigación y Análisis

| Sesgo | Qué es | Cómo te afecta en algo trading |
|---|---|---|
| **Ilusión de control** | Creer que tus decisiones influyen más de lo que realmente lo hacen | Sobreoptimizás creyendo que podés "controlar" el mercado |
| **Confirmación** | Buscar solo información que confirma lo que ya creés | Solo testeas condiciones favorables, ignorás evidencia en contra |
| **Efecto gurú** (Pygmalion) | Darle autoridad desmedida a una persona o fuente | Copiás sistemas de un "experto" sin validarlos vos mismo |
| **Disponibilidad** | Darle más importancia a lo que recordás fácilmente | Solo operás activos conocidos (AAPL, TSLA) sin evaluar si son los mejores |
| **Anclaje** | Fijarte en un número o idea específica y decidir alrededor de eso | Creés que un número redondo ($100, $50) tiene significado especial sin evidencia |
| **Efecto grupo** | Seguir la opinión mayoritaria | Elegís estrategias "de moda" en vez de las que los datos soportan |

**El sesgo de confirmación es el más peligroso en esta fase.** Es extremadamente fácil backtestear solo las condiciones que favorecen tu hipótesis e ignorar las que la contradicen. Protocolo, protocolo, protocolo.

### Fase 3: Sesgos Específicos del Trading Algorítmico

Estos sesgos son propios del trabajo con datos y backtesting. No existen (o son menos relevantes) en trading discrecional. Para el protocolo que previene estos sesgos, ver [Método Científico en Desarrollo de Sistemas](../technical-practices/Scientific-Method-System-Development.md).

**Sesgo de selección (Selection Bias)**

Elegir subconjuntos de datos de manera arbitraria. "Voy a testear solo de 2020 a 2023 porque ahí el mercado era alcista." Los datos de entrenamiento, validación y out-of-sample deben seleccionarse con protocolo, no con conveniencia.

**Sesgo de anticipación (Look-Ahead Bias)**

Usar información que no estaría disponible en tiempo real. Es el más técnico y el más traicionero.

```python
# INCORRECTO: look-ahead bias
# El RSI del día se calcula con el close del día,
# pero tu señal de compra se genera DURANTE el día
data['rsi'] = calculate_rsi(data['close'], 14)
data['signal'] = data['rsi'] < 30  # usás el RSI del día para comprar ese mismo día

# CORRECTO: usar datos disponibles al momento de la decisión
data['rsi'] = calculate_rsi(data['close'], 14)
data['signal'] = data['rsi'].shift(1) < 30  # señal basada en el RSI del día anterior
```

Algunos lenguajes de backtesting previenen esto por diseño. Python/pandas no — sos responsable de evitarlo vos.

**Data Snooping (Torturar los Datos)**

Como dijo Ronald Coase: "Si torturás los datos lo suficiente, eventualmente confiesan lo que querés." También conocido como **p-hacking** o **data mining bias** — buscar patrones exhaustivamente hasta encontrar algo que fitee, sin significancia estadística real.

Si probás 1000 combinaciones de parámetros, por puro azar ~50 van a parecer rentables. Eso no es un sistema — es ruido. La solución: validación out-of-sample rigurosa y walk-forward analysis.

**Sesgo de supervivencia (Survivorship Bias)**

Backtestear con las acciones que existen HOY, ignorando las que quebraron o fueron deslistadas. El S&P 500 de hoy no es el de hace 10 años — las empresas que quebraron fueron reemplazadas. Si tu backtest histórico solo usa las sobrevivientes, los resultados están inflados.

```python
# Si backtesteas una estrategia rotacional de acciones del S&P 500
# desde 2010, necesitás la composición del índice EN CADA MOMENTO,
# incluyendo las que fueron removidas (Lehman, Enron, etc.)
#
# Fuentes con datos libres de survivorship bias:
# - Sharadar (Nasdaq Data Link)
# - CRSP
# - Norgate Data
```

### Fase 4: Evaluación y Optimización

| Sesgo | Qué es | Cómo te afecta |
|---|---|---|
| **Validación insuficiente** | Muestras demasiado chicas para ser significativas | 30 trades no prueban nada. Necesitás cientos para conclusiones válidas |
| **Sesgo de normalidad** | Asumir que los retornos siguen una distribución normal | Subestimás cisnes negros. Las colas reales son mucho más gruesas |

El sesgo de normalidad merece atención especial. Si diseñás tu gestión de riesgo asumiendo distribución normal, estás subestimando la frecuencia de eventos extremos en un factor de 10x o más.

### Fase 5: Operativa en Vivo

| Sesgo | Qué es | Cómo te afecta |
|---|---|---|
| **Falacia del jugador** | Creer que eventos independientes están correlacionados | "Lleva 8 pérdidas seguidas, la siguiente TIENE que ser ganadora" — no, no tiene |
| **Status quo** | Resistencia a cambiar | No actualizar sistemas que ya no funcionan porque "siempre hicieron esto" |
| **Coste hundido** | Mantener algo solo porque ya invertiste mucho en ello | Seguir operando un sistema roto porque te llevó 6 meses desarrollarlo |
| **Dotación** | Sobrevalorar lo que ya tenés | Creer que tus sistemas son mejores que nuevas alternativas solo porque son tuyos |

```python
def is_gambler_fallacy(consecutive_losses, expected_loss_streaks):
    """
    Después de N pérdidas consecutivas, la probabilidad del
    siguiente trade sigue siendo la misma. Los trades son
    (generalmente) eventos independientes.
    """
    # Un sistema con 40% win rate puede tener rachas de:
    # 5 pérdidas: ~7.8% de probabilidad en cualquier secuencia de 5
    # 10 pérdidas: ~0.6% - raro pero esperable en 1000+ trades
    # 15 pérdidas: ~0.05% - muy raro pero posible

    from math import pow
    loss_rate = 0.60  # 40% win rate = 60% loss rate
    prob_streak = pow(loss_rate, consecutive_losses)

    return {
        'prob_this_streak': f"{prob_streak*100:.2f}%",
        'message': "La probabilidad del próximo trade NO cambia por los anteriores."
    }
```

## Protocolo Anti-Sesgos

No podés eliminar los sesgos — son parte de cómo funciona el cerebro humano. Pero podés construir protocolos que los neutralicen:

1. **Escribí tus reglas ANTES de ver los resultados.** Definí qué vas a testear, con qué datos, y qué criterio de éxito usás. Si lo definís después de ver el backtest, estás sesgado.

2. **Usá out-of-sample siempre.** Reservá un período de datos que NUNCA uses para optimizar. Es tu prueba de realidad.

3. **Registrá tus decisiones.** Un diario de trading algorítmico no registra trades (eso lo hace el log del sistema). Registra decisiones: "Hoy pausé el sistema X porque..." Releé esas entradas un mes después — vas a sorprenderte de cuánto cambia tu perspectiva.

4. **Definí el clip point antes de lanzar.** ¿Cuánto drawdown o cuántas pérdidas consecutivas hacen que pares el sistema? Definilo cuando NO estás en drawdown. Escribilo. No lo cambies durante la operativa.

5. **Pedí una segunda opinión sobre tus datos**, no sobre tu opinión de mercado. Mostrá tu metodología de backtest a alguien y preguntá: "¿ves algún sesgo en cómo estoy testeando esto?"

6. **Aceptá que todo sistema tiene vida útil.** No existe el sistema eterno. Diseñá desde el inicio un protocolo de supervisión con métricas rolling que te digan cuándo el edge se está degradando, antes de que sea obvio.
