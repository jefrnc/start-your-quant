# Backtesting: Del Clásico al Walk-Forward

Hay tres formas de simular un sistema en datos históricos. Cada una tiene su momento y su utilidad. No son excluyentes — son progresivas.

## Nivel 1: Backtest Clásico

Evaluar la estrategia en todo el histórico disponible, sin separar datos.

```
[========== Todo el histórico ==========]
         Optimizo y evalúo aquí
```

**Qué hace**: muestra cómo hubiera funcionado la estrategia en el pasado.

**Para qué sirve**: evaluación preliminar. Descartar ideas que no tienen un mínimo de viabilidad antes de invertir tiempo en análisis más profundos.

**Limitación**: es tremendamente fácil sobreajustar. Si optimizás todo el histórico y elegís los mejores parámetros, casi con seguridad estás ajustando el sistema a datos pasados que no se van a repetir.

**Cuándo usarlo**: fase de investigación y evaluación preliminar. Si acá ya no funciona, no tiene sentido seguir.

## Nivel 2: Forward Testing (Out-of-Sample)

Optimizar en un período y evaluar en otro que el optimizador nunca vio.

```
[==== In-Sample (optimización) ====][==== Out-of-Sample (prueba) ====]
         2000 - 2015                      2015 - 2024
```

**Qué hace**: simula lo que pasaría si hubieras desarrollado el sistema en 2015 y lo hubieras dejado correr sin tocarlo.

**Para qué sirve**: validación más realista. Si los resultados out-of-sample son consistentes con el in-sample (no idénticos — consistentes), la señal probablemente es real.

**Limitación**: una sola prueba forward no garantiza nada. Y es fácil hacerse trampas sin protocolos claros (ej: ajustar el punto de corte entre IS y OOS hasta que funcione).

**Cuándo usarlo**: después de que la evaluación preliminar muestre potencial.

## Nivel 3: Walk-Forward Testing

Múltiples ciclos de optimización + prueba, avanzando por el histórico.

```
[IS-1][OOS-1]
      [IS-2][OOS-2]
            [IS-3][OOS-3]
                  [IS-4][OOS-4]
                        [IS-5][OOS-5]
```

Cada bloque IS (In-Sample) se optimiza. Los parámetros ganadores se prueban en el OOS (Out-of-Sample) siguiente. Luego la ventana avanza y se repite.

**Qué hace**: genera una curva completa de resultados out-of-sample a lo largo de casi todo el histórico. Es la prueba más cercana a simular lo que realmente hubiera pasado si hubieras ido reoptimizando periódicamente.

**Para qué sirve**: es la prueba que mejor permite **medir objetivamente** la robustez. Si el sistema pasa un walk-forward bien hecho, la probabilidad de que funcione en real es significativamente mayor.

**Ventajas**:
- Produce muchos datos fuera de muestra, no solo uno
- Mide objetivamente la robustez
- Permite elegir los parámetros para operar en real (los del último ciclo)
- Reduce dramáticamente el riesgo de sobreoptimización

**Limitaciones**:
- Proceso largo, lento e intensivo computacionalmente
- Pocos sistemas lo pasan — es una prueba muy dura
- Aún así, no es infalible. Con mala praxis se pueden hacer trampas

**Cuándo usarlo**: cuando el sistema ya pasó la evaluación preliminar y el forward testing básico.

## Resumen: Cuándo Usar Cada Uno

| Fase | Método | Objetivo |
|---|---|---|
| Investigación / idea inicial | Backtest clásico | ¿Tiene un mínimo de viabilidad? |
| Validación inicial | Forward testing | ¿Funciona en datos no vistos? |
| Validación seria | Walk-forward | ¿Es robusto de verdad? |
| Operativa real | Paper trading → live reducido | ¿Se ejecuta como el backtest predijo? |

No tiene sentido hacer walk-forward a un sistema que no pasa ni el backtest clásico. Usá cada nivel como filtro progresivo.

## La Regla de Oro

Un backtest te dice cómo hubiera ido en el pasado. **No te dice cómo irá en el futuro.** Es condición necesaria para operar un sistema, pero nunca suficiente por sí sola. El walk-forward es lo más cercano a una garantía que podemos obtener — y aún así, no es una garantía.
