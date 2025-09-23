"""
Template de Estrategia: Gap & Go para Small Caps
==============================================

Este template proporciona una estructura base para implementar
la estrategia Gap & Go específicamente optimizada para small caps.

Uso:
1. Copia este archivo
2. Modifica los parámetros según tu backtesting
3. Implementa la lógica específica de entry/exit
4. Añade validaciones de riesgo personalizadas
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class GapAndGoConfig:
    """Configuración para estrategia Gap & Go"""

    # Filtros de Gap
    min_gap_percent: float = 3.0          # Gap mínimo para considerar la acción
    max_gap_percent: float = 25.0         # Gap máximo (evitar gaps extremos)

    # Filtros de Volumen
    min_premarket_volume: int = 50000     # Volumen mínimo premarket
    volume_multiplier: float = 2.0        # Multiplicador vs volumen promedio

    # Filtros de Precio
    min_price: float = 0.50               # Precio mínimo de la acción
    max_price: float = 10.99              # Precio máximo (small caps)

    # Gestión de Riesgo
    max_position_size: float = 70.0       # Tamaño máximo de posición ($)
    max_risk_per_trade: float = 10.0      # Riesgo máximo por trade ($)
    stop_loss_percent: float = 5.0        # Stop loss en %

    # Timing
    entry_start_time: str = "05:30"       # Inicio ventana de entrada (ET)
    entry_end_time: str = "08:00"         # Fin ventana de entrada (ET)
    max_hold_time_minutes: int = 60       # Tiempo máximo de hold


class GapAndGoStrategy:
    """
    Estrategia Gap & Go Template

    Ejemplo de uso:
    >>> config = GapAndGoConfig()
    >>> strategy = GapAndGoStrategy(config)
    >>> signal = strategy.generate_signal(market_data)
    """

    def __init__(self, config: GapAndGoConfig):
        self.config = config
        self.position = None
        self.entry_price = None
        self.entry_time = None

    def screen_candidates(self, universe: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra candidatos que cumplen criterios Gap & Go

        Args:
            universe: DataFrame con columnas [symbol, price, gap_percent,
                     premarket_volume, avg_volume, float_shares]

        Returns:
            DataFrame filtrado con candidatos válidos
        """
        candidates = universe.copy()

        # Filtro de gap
        candidates = candidates[
            (candidates['gap_percent'] >= self.config.min_gap_percent) &
            (candidates['gap_percent'] <= self.config.max_gap_percent)
        ]

        # Filtro de precio
        candidates = candidates[
            (candidates['price'] >= self.config.min_price) &
            (candidates['price'] <= self.config.max_price)
        ]

        # Filtro de volumen
        candidates = candidates[
            (candidates['premarket_volume'] >= self.config.min_premarket_volume) &
            (candidates['premarket_volume'] >=
             candidates['avg_volume'] * self.config.volume_multiplier)
        ]

        return candidates.sort_values('gap_percent', ascending=False)

    def calculate_position_size(self, price: float, atr: float) -> int:
        """
        Calcula el tamaño de posición basado en riesgo fijo

        Args:
            price: Precio actual de la acción
            atr: Average True Range para volatilidad

        Returns:
            Número de shares a comprar
        """
        # Método 1: Basado en riesgo máximo
        risk_per_share = price * (self.config.stop_loss_percent / 100)
        shares_by_risk = int(self.config.max_risk_per_trade / risk_per_share)

        # Método 2: Basado en tamaño máximo de posición
        shares_by_size = int(self.config.max_position_size / price)

        # Tomar el menor de los dos
        return min(shares_by_risk, shares_by_size)

    def generate_signal(self, data: Dict) -> Optional[Dict]:
        """
        Genera señal de trading basada en datos actuales

        Args:
            data: Diccionario con datos de mercado
                 {symbol, price, volume, time, etc.}

        Returns:
            Diccionario con señal o None si no hay señal
        """
        current_time = data['time']

        # Verificar ventana de tiempo
        if not self._is_trading_window(current_time):
            return None

        # Verificar si ya tenemos posición
        if self.position is not None:
            return self._check_exit_signal(data)

        # Buscar señal de entrada
        return self._check_entry_signal(data)

    def _check_entry_signal(self, data: Dict) -> Optional[Dict]:
        """Lógica específica para señal de entrada"""

        # Ejemplo de condiciones de entrada:
        # 1. Precio por encima del gap level
        # 2. Volumen confirmando el movimiento
        # 3. No hay resistencia técnica inmediata

        signal_strength = self._calculate_signal_strength(data)

        if signal_strength > 0.7:  # Umbral de confianza
            position_size = self.calculate_position_size(
                data['price'],
                data.get('atr', data['price'] * 0.02)  # 2% default ATR
            )

            return {
                'action': 'BUY',
                'symbol': data['symbol'],
                'quantity': position_size,
                'price': data['price'],
                'signal_strength': signal_strength,
                'stop_loss': data['price'] * (1 - self.config.stop_loss_percent / 100),
                'timestamp': data['time']
            }

        return None

    def _check_exit_signal(self, data: Dict) -> Optional[Dict]:
        """Lógica específica para señal de salida"""

        # Condiciones de salida:
        # 1. Stop loss alcanzado
        # 2. Target de ganancia alcanzado
        # 3. Tiempo máximo de hold excedido
        # 4. Cambio en momentum

        current_price = data['price']

        # Stop loss
        stop_price = self.entry_price * (1 - self.config.stop_loss_percent / 100)
        if current_price <= stop_price:
            return self._create_exit_signal(data, 'STOP_LOSS')

        # Tiempo máximo
        hold_time = (data['time'] - self.entry_time).total_seconds() / 60
        if hold_time >= self.config.max_hold_time_minutes:
            return self._create_exit_signal(data, 'TIME_EXIT')

        # Trailing stop u otros criterios técnicos aquí...

        return None

    def _calculate_signal_strength(self, data: Dict) -> float:
        """
        Calcula la fuerza de la señal (0-1)

        Factores a considerar:
        - Magnitud del gap
        - Volumen relativo
        - Momentum del precio
        - Análisis técnico adicional
        """
        strength = 0.0

        # Factor gap (peso: 30%)
        gap_strength = min(data['gap_percent'] / 10.0, 1.0)
        strength += gap_strength * 0.3

        # Factor volumen (peso: 25%)
        volume_ratio = data['volume'] / data.get('avg_volume', 1)
        volume_strength = min(volume_ratio / 3.0, 1.0)
        strength += volume_strength * 0.25

        # Factor momentum (peso: 25%)
        # Implementar basado en price action, VWAP, etc.
        momentum_strength = 0.5  # Placeholder
        strength += momentum_strength * 0.25

        # Factor técnico (peso: 20%)
        # RSI, resistencias, etc.
        technical_strength = 0.5  # Placeholder
        strength += technical_strength * 0.2

        return min(strength, 1.0)

    def _is_trading_window(self, current_time) -> bool:
        """Verifica si estamos en ventana de trading"""
        # Implementar lógica de tiempo específica
        return True  # Placeholder

    def _create_exit_signal(self, data: Dict, reason: str) -> Dict:
        """Crea señal de salida"""
        return {
            'action': 'SELL',
            'symbol': data['symbol'],
            'quantity': self.position,
            'price': data['price'],
            'reason': reason,
            'timestamp': data['time']
        }


# Ejemplo de uso del template
if __name__ == "__main__":

    # Configurar la estrategia
    config = GapAndGoConfig(
        min_gap_percent=5.0,
        max_position_size=50.0,
        entry_start_time="06:00",
        entry_end_time="07:30"
    )

    # Inicializar estrategia
    strategy = GapAndGoStrategy(config)

    # Datos de ejemplo
    sample_data = {
        'symbol': 'ABCD',
        'price': 3.50,
        'gap_percent': 8.5,
        'volume': 150000,
        'avg_volume': 75000,
        'time': pd.Timestamp.now(),
        'atr': 0.15
    }

    # Generar señal
    signal = strategy.generate_signal(sample_data)

    if signal:
        print(f"Señal generada: {signal}")
    else:
        print("No hay señal en este momento")


"""
NOTAS DE IMPLEMENTACIÓN:
========================

1. PERSONALIZACIÓN:
   - Modifica los parámetros en GapAndGoConfig según tu backtesting
   - Ajusta la lógica de _calculate_signal_strength() con tus indicadores
   - Implementa stop losses más sofisticados (trailing, ATR-based, etc.)

2. VALIDACIÓN:
   - Siempre backtestea cambios antes de usar en vivo
   - Verifica que el risk management funcione correctamente
   - Testea en diferentes condiciones de mercado

3. EXTENSIONES COMUNES:
   - Integrar con APIs de datos reales (Polygon, IBKR)
   - Añadir análisis de sentimiento de noticias
   - Implementar múltiples timeframes
   - Añadir machine learning para signal strength

4. MEJORES PRÁCTICAS:
   - Loggea todas las decisiones para análisis posterior
   - Implementa circuit breakers para evitar pérdidas excesivas
   - Mantén configuraciones separadas para paper vs live trading
   - Documenta todos los cambios y su rationale
"""