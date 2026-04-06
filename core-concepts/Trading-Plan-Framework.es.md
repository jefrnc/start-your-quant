> 🇺🇸 [Read in English](Trading-Plan-Framework.md) | 🇪🇸 **Español**

# El Plan de Trading: Framework Profesional

Si no tenés un plan de trading escrito, no tenés un negocio — tenés un hobby caro. El plan de trading es el documento que define quién sos como trader, cómo operás, y qué hacés cuando las cosas salen mal. Lo escribís cuando estás tranquilo, y lo seguís cuando no lo estás.

## Por Qué un Plan Escrito

Cuando no estás operando, pensás con claridad. Definís reglas racionales, evaluás riesgo objetivamente, y tomás decisiones basadas en datos.

Cuando estás en medio de un drawdown del 12%, tu mente cambia. Las reglas que parecían obvias ahora parecen cuestionables. El stop loss que definiste te parece demasiado ajustado. El sistema que validaste con 3 años de datos "quizás no sirve para este mercado."

**El plan escrito es tu ancla.** No se modifica en caliente. Se revisa periódicamente, en frío, con datos — nunca durante una racha mala.

Dato real: traders que llevan un diario de decisiones descubren, al releerlo semanas después, que su perspectiva cambió significativamente sin que fueran conscientes. El papel no miente.

## Las 6 Secciones del Plan

### 1. Filosofía de Trading

Es tu misión y visión como trader. No es filosófico en el sentido abstracto — es concreto:

**Qué definir:**
- **Dedicación**: ¿Tiempo completo o parcial? Si es parcial, ¿cuántas horas semanales realistas?
- **Capital disponible**: ¿Cuánto capital tenés para operar? Esto condiciona todo — instrumentos, estrategias, expectativas
- **Origen del capital**: ¿Propio o de terceros? Operar con dinero ajeno cambia completamente la presión psicológica
- **Objetivos económicos necesarios vs deseados**: "Necesito $2,000/mes para vivir" es muy distinto de "quiero ganar $10,000/mes"
- **Mercados**: ¿Acciones US? ¿Futuros? ¿Forex? La elección depende de tu capital, horario y perfil

**La trampa de la necesidad**: si dependés económicamente de tu trading desde el día 1, la presión emocional compromete tus decisiones. La recomendación, siempre que sea posible, es empezar como actividad complementaria hasta que el track record y el capital justifiquen la transición.

```yaml
# Ejemplo de filosofía documentada
filosofia:
  dedicacion: parcial  # 2h/dia, mañanas antes del trabajo
  capital_inicial: 15000
  origen: propio
  objetivo_necesario: 0  # no dependo de esto para vivir
  objetivo_deseado: 500_mensual  # primer año
  mercados: [acciones_us_smallcap, futuros_indices]
  horizonte: largo_plazo  # mínimo 3 años para evaluar
  timeframes_preferidos: [diario, semanal]
  # Si mi dedicación es parcial, NO opero intradiario agresivo
  # porque no puedo monitorear constantemente
```

### 2. Psicología

La mente es el componente más frágil del trading, incluso del algorítmico. Tu robot ejecuta, pero vos decidís cuándo activarlo, cuándo pararlo, y cómo reaccionar cuando las cosas no van como el backtest prometía.

**Qué incluir:**
- **Autoevaluación**: tu perfil de riesgo, tus sesgos conocidos (ver [Sesgos Cognitivos](./Cognitive-Biases-Algo-Trading.md))
- **Protocolo emocional**: qué hacés cuando estás en drawdown máximo — ¿revisás los datos o empezás a cambiar parámetros impulsivamente?
- **Mantenimiento**: ejercicio, descanso, desconexión. 20 años operando cada día requiere cuidar el cuerpo y la mente de forma sostenida
- **Red de soporte**: ¿tenés alguien con quien hablar de trading objetivamente? ¿Un mentor, un grupo, un profesional?

**Lo que no parece importante hasta que lo es**: un drawdown del 10% en el backtest es un número. Un drawdown del 10% en tu cuenta real, cuando necesitás ese dinero, es una experiencia completamente diferente. El plan psicológico se escribe para el segundo caso, no para el primero.

### 3. Reglas y Sistemas

El corazón operativo del plan. Aquí viven tus sistemas de trading, sus especificaciones, y el portfolio.

**Para cada sistema, documentar:**

```yaml
sistema:
  nombre: "Gap_SmallCap_Long_v2"
  mercado: acciones_us
  universo: smallcap_0.50_10.99
  timeframe: premarket_5min
  tipo: momentum  # tendencial

  reglas_entrada:
    - gap_up > 10%
    - volumen_premarket > 500000
    - precio > vwap
  reglas_salida:
    - stop_loss: -3%
    - take_profit: +8%
    - tiempo_maximo: 2h

  metricas_backtest:
    periodo: "2021-01-01 a 2024-12-31"
    trades: 847
    win_rate: 0.42
    profit_factor: 1.85
    max_drawdown: -12.3%
    sharpe: 1.45
    max_consecutive_losses: 11

  gestion_riesgo:
    risk_per_trade: 1%  # del capital
    max_posiciones_simultaneas: 3
    max_exposure: 30%  # del capital total
```

**El portfolio importa más que cualquier sistema individual.** Documentá cómo se combinan tus sistemas:
- ¿Qué correlación tienen entre sí?
- ¿Operan en los mismos mercados/horarios?
- ¿Cómo se reparte el capital entre ellos?

Esta es la sección más dinámica del plan — los sistemas cambian, se agregan, se retiran.

### 4. Puesta en Marcha (Infraestructura)

Trading algorítmico requiere infraestructura. No necesitás un datacenter, pero sí un setup confiable.

**Qué definir:**

| Componente | Decisiones clave |
|---|---|
| **Hardware** | ¿Local o cloud? Para optimización pesada, servidores cloud temporales son más eficientes que comprar hardware |
| **Software** | Plataforma de backtesting, lenguaje (Python, etc.), broker API |
| **Datos** | Proveedor, calidad, costo. ¿Necesitás tick data o bastan velas de 1min? |
| **Conectividad** | Internet principal + backup. Un corte durante una posición abierta es un riesgo real |
| **Costos recurrentes** | Datos, hosting, broker, herramientas. Calculá el break-even mensual |

```yaml
# Ejemplo de análisis de costos
costos_mensuales:
  datos_polygon: 29     # USD
  vps_cloud: 40         # para ejecución 24/7
  broker_data: 0        # incluido con IBKR
  herramientas: 0       # Python + open source
  total: 69
  # Break-even: necesito generar > $69/mes solo para cubrir costos
  # Con capital de $15,000 eso es un 0.46% mensual
```

### 5. Supervisión y Reciclaje

Todo sistema tiene vida útil. El mercado cambia de régimen, la volatilidad evoluciona, nuevos participantes alteran la microestructura. Tu trabajo no termina cuando lanzás un sistema — empieza.

**Punto de Quiebre (Clip Point)**

Definí, ANTES de lanzar, en qué condiciones pausás o retirás un sistema. En la industria se lo conoce como "clip point" o "kill switch" — el umbral donde decidís que el sistema dejó de funcionar:

```python
def evaluate_system_health(rolling_metrics):
    """
    Protocolo de supervisión: evaluar mensualmente.
    Definir ANTES de lanzar, no durante un drawdown.
    """
    checks = {
        'drawdown_breach': rolling_metrics['current_dd'] > rolling_metrics['max_expected_dd'] * 1.5,
        'sharpe_degraded': rolling_metrics['sharpe_90d'] < 0.3,
        'win_rate_collapsed': rolling_metrics['win_rate_60d'] < rolling_metrics['expected_wr'] * 0.6,
        'consecutive_losses': rolling_metrics['current_streak'] > rolling_metrics['max_expected_streak'],
    }

    triggered = [k for k, v in checks.items() if v]

    if len(triggered) >= 2:
        return "PAUSAR — múltiples señales de degradación"
    elif len(triggered) == 1:
        return f"MONITOREAR — señal de alerta: {triggered[0]}"
    return "NORMAL"
```

**Formación continua**: el mercado evoluciona y vos también debés hacerlo. Nuevas técnicas, nuevos datos, nuevas regulaciones. Dedicá tiempo regular a aprender, no solo a operar.

**Reciclaje de sistemas**: cuando un sistema se retira, documentá por qué. Esa información es valiosa para el diseño de futuros sistemas.

### 6. Plan de Crisis

¿Qué hacés cuando las cosas salen mal de verdad?

**Escenarios a planificar:**

| Escenario | Protocolo |
|---|---|
| **Corte de internet** | ¿Tenés stops en el servidor del broker (no solo locales)? ¿Podés cerrar desde el celular? |
| **Caída del servidor** | ¿Hay redundancia? ¿Cuánto tiempo podés estar sin ejecución? |
| **Flash crash** | ¿Tus stops están en el mercado o son simulados? Los simulados no se ejecutan si tu software se cae |
| **Broker sin respuesta** | ¿Tenés cuenta en un segundo broker? ¿Podés hedgear la posición? |
| **Drawdown máximo** | ¿A qué porcentaje parás todo? ¿Quién decide — vos o el código? |
| **Corte de luz** | Un UPS de 30 minutos cuesta poco y salva mucho |
| **Error en el código** | ¿Cómo detectás un bug en producción? ¿Hay alertas automáticas para trades anómalos? |

## Mentalidad Empresarial

El trading algorítmico es un negocio con pocas barreras de entrada — lo cual es bueno para acceder, pero significa mucha competencia. Tener un buen sistema (producto) es condición necesaria pero no suficiente.

Lo que diferencia a un trader profesional de uno que juega:

- **Protocolos escritos** para cada fase (diseño, testing, lanzamiento, supervisión, retiro)
- **Análisis de costos** realista (no solo comisiones — también datos, hosting, tiempo)
- **Métricas de negocio**, no solo de trading: ¿cuánto tiempo invertís por dólar ganado? ¿Es escalable?
- **Plan de contingencia** para escenarios adversos
- **Revisión periódica** del plan completo, no solo de los sistemas

Independientemente de si dedicás 2 horas por semana o 12 horas por día, tratá tu trading con la seriedad de un negocio. Los mercados no distinguen entre profesionales y amateurs — pero los resultados a largo plazo sí.

## Cuándo Revisar el Plan

| Frecuencia | Qué revisar |
|---|---|
| **Semanal** | Métricas de sistemas activos, alertas del protocolo de supervisión |
| **Mensual** | Performance del portfolio completo, correlaciones entre sistemas |
| **Trimestral** | Filosofía, costos, ¿siguen vigentes los supuestos del plan? |
| **Ante cambios de vida** | Nuevo trabajo, cambio de capital, cambios familiares → revisar secciones 1 y 2 |
| **Ante crisis de mercado** | No cambiar el plan durante la crisis. Revisarlo DESPUÉS, en frío, con datos |
