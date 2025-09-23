"""
Template de Estrategia: VWAP Reclaim para Small Caps
===================================================

La estrategia VWAP Reclaim busca oportunidades cuando el precio de una acción:
1. Es rechazado en el VWAP (Volume Weighted Average Price)
2. Posteriormente "reclaim" (recupera) el VWAP con volumen confirmatorio
3. Continúa el movimiento alcista

Esta estrategia es especialmente efectiva en small caps porque:
- VWAP actúa como soporte/resistencia psicológica
- Institutional traders usan VWAP como reference point
- Breakout del VWAP con volumen indica momentum shift

Uso:
1. Identificar rechazo inicial en VWAP
2. Esperar pullback y consolidación
3. Entry en breakout sobre VWAP con volumen
4. Exit en targets o señales de debilidad
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum


class VWAPState(Enum):
    """Estados del precio relativo al VWAP"""
    ABOVE_VWAP = "above"
    BELOW_VWAP = "below"
    AT_VWAP = "at"


class TradePhase(Enum):
    """Fases de la estrategia VWAP Reclaim"""
    SCANNING = "scanning"           # Buscando setup inicial
    REJECTION_FOUND = "rejection"   # Encontrado rechazo en VWAP
    WAITING_RECLAIM = "waiting"     # Esperando reclaim con volumen
    POSITION_OPEN = "position"      # Posición abierta


@dataclass
class VWAPReclaimConfig:
    """Configuración para estrategia VWAP Reclaim"""

    # Filtros de VWAP
    min_distance_from_vwap: float = 0.005   # Mínimo 0.5% del VWAP para considerar rejection
    max_distance_from_vwap: float = 0.03    # Máximo 3% del VWAP (evitar muy lejos)

    # Filtros de Volumen
    min_volume_spike: float = 1.5           # 1.5x volumen promedio para confirmation
    volume_lookback_minutes: int = 20       # Ventana para calcular volumen promedio

    # Filtros de Precio y Mercado
    min_price: float = 1.00                 # Precio mínimo de la acción
    max_price: float = 15.99                # Precio máximo (small caps)
    min_daily_volume: int = 100000          # Volumen diario mínimo

    # Timing de Setup
    rejection_lookback_minutes: int = 30    # Tiempo máximo desde rejection
    min_consolidation_minutes: int = 5      # Mínimo tiempo de consolidación
    max_wait_time_minutes: int = 60         # Máximo tiempo esperando reclaim

    # Gestión de Riesgo
    max_position_size: float = 60.0         # Tamaño máximo de posición ($)
    max_risk_per_trade: float = 8.0         # Riesgo máximo por trade ($)
    stop_loss_atr_multiplier: float = 1.5   # Stop loss = 1.5x ATR bajo entry

    # Targets de Profit
    first_target_ratio: float = 1.5         # 1.5:1 risk/reward first target
    second_target_ratio: float = 3.0        # 3:1 risk/reward second target
    partial_exit_percent: float = 0.50      # 50% de posición en first target

    # Timing de Trading
    entry_start_time: str = "06:00"         # Inicio ventana de entrada
    entry_end_time: str = "15:30"           # Fin ventana de entrada
    max_hold_time_minutes: int = 120        # Tiempo máximo de hold


class VWAPIndicator:
    """Calculadora de VWAP en tiempo real"""

    def __init__(self):
        self.cumulative_volume = 0
        self.cumulative_pv = 0  # price * volume
        self.reset_time = None

    def reset_daily(self):
        """Reset VWAP al inicio del día"""
        self.cumulative_volume = 0
        self.cumulative_pv = 0

    def update(self, price: float, volume: int) -> float:
        """
        Actualiza VWAP con nueva data

        Args:
            price: Precio actual
            volume: Volumen actual

        Returns:
            VWAP actualizado
        """
        self.cumulative_volume += volume
        self.cumulative_pv += (price * volume)

        if self.cumulative_volume == 0:
            return price

        return self.cumulative_pv / self.cumulative_volume

    def get_vwap_bands(self, vwap: float, atr: float) -> Dict[str, float]:
        """
        Calcula bandas alrededor del VWAP

        Args:
            vwap: VWAP actual
            atr: Average True Range

        Returns:
            Dict con upper_band, lower_band, middle (VWAP)
        """
        band_width = atr * 0.5  # Half ATR bands

        return {
            'upper_band': vwap + band_width,
            'lower_band': vwap - band_width,
            'middle': vwap
        }


class VWAPReclaimStrategy:
    """
    Estrategia VWAP Reclaim Template

    Workflow:
    1. Monitor precio relativo a VWAP
    2. Detectar rejection (precio bounces off VWAP)
    3. Esperar consolidación
    4. Entry en reclaim con volume confirmation
    5. Manage position con targets y stops

    Ejemplo de uso:
    >>> config = VWAPReclaimConfig()
    >>> strategy = VWAPReclaimStrategy(config)
    >>> signal = strategy.process_tick(market_data)
    """

    def __init__(self, config: VWAPReclaimConfig):
        self.config = config
        self.vwap_indicator = VWAPIndicator()

        # State tracking
        self.current_phase = TradePhase.SCANNING
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.stop_loss_price = None
        self.first_target_price = None
        self.second_target_price = None

        # Setup tracking
        self.rejection_time = None
        self.rejection_price = None
        self.rejection_vwap = None
        self.price_history = []
        self.volume_history = []

    def reset_daily(self):
        """Reset al inicio del día de trading"""
        self.vwap_indicator.reset_daily()
        self.current_phase = TradePhase.SCANNING
        self.position = None
        self._clear_setup_tracking()

    def _clear_setup_tracking(self):
        """Limpia tracking del setup actual"""
        self.rejection_time = None
        self.rejection_price = None
        self.rejection_vwap = None
        self.entry_price = None
        self.entry_time = None

    def process_tick(self, data: Dict) -> Optional[Dict]:
        """
        Procesa tick individual de market data

        Args:
            data: Dict con {symbol, price, volume, timestamp, atr}

        Returns:
            Signal dict o None
        """
        current_time = data['timestamp']
        current_price = data['price']
        current_volume = data['volume']

        # Actualizar VWAP
        vwap = self.vwap_indicator.update(current_price, current_volume)

        # Actualizar history
        self._update_history(current_price, current_volume, current_time)

        # Verificar ventana de trading
        if not self._is_trading_window(current_time):
            return None

        # Process según fase actual
        if self.current_phase == TradePhase.SCANNING:
            return self._scan_for_rejection(data, vwap)

        elif self.current_phase == TradePhase.REJECTION_FOUND:
            return self._wait_for_reclaim_setup(data, vwap)

        elif self.current_phase == TradePhase.WAITING_RECLAIM:
            return self._check_reclaim_entry(data, vwap)

        elif self.current_phase == TradePhase.POSITION_OPEN:
            return self._manage_position(data, vwap)

        return None

    def _scan_for_rejection(self, data: Dict, vwap: float) -> Optional[Dict]:
        """
        Escanea por rejection del VWAP

        Condiciones para rejection:
        1. Precio se acerca al VWAP (dentro de max_distance)
        2. Precio es rechazado (reverses direction)
        3. Volumen confirma el rejection
        """
        current_price = data['price']
        distance_from_vwap = abs(current_price - vwap) / vwap

        # Verificar si estamos cerca del VWAP
        if distance_from_vwap > self.config.max_distance_from_vwap:
            return None

        # Buscar rejection pattern en price history reciente
        if len(self.price_history) < 10:  # Necesitamos histórico mínimo
            return None

        recent_prices = self.price_history[-10:]
        recent_volumes = self.volume_history[-10:]

        # Detectar rejection (simplified logic)
        if self._detect_vwap_rejection(recent_prices, vwap, recent_volumes):
            self.rejection_time = data['timestamp']
            self.rejection_price = current_price
            self.rejection_vwap = vwap
            self.current_phase = TradePhase.REJECTION_FOUND

            return {
                'type': 'SETUP_DETECTED',
                'message': f"VWAP rejection detected at {current_price:.2f} (VWAP: {vwap:.2f})",
                'setup_data': {
                    'rejection_price': current_price,
                    'vwap': vwap,
                    'distance_pct': distance_from_vwap * 100
                }
            }

        return None

    def _wait_for_reclaim_setup(self, data: Dict, vwap: float) -> Optional[Dict]:
        """
        Espera por setup de reclaim después de rejection

        Condiciones:
        1. Tiempo suficiente de consolidación
        2. Precio no se aleja mucho del VWAP
        3. Volumen se mantiene elevated
        """
        current_time = data['timestamp']
        current_price = data['price']

        # Verificar timeout
        time_since_rejection = (current_time - self.rejection_time).total_seconds() / 60
        if time_since_rejection > self.config.max_wait_time_minutes:
            self._clear_setup_tracking()
            self.current_phase = TradePhase.SCANNING
            return {'type': 'SETUP_TIMEOUT', 'message': "Rejection setup timed out"}

        # Verificar que precio no se aleje mucho
        distance_from_vwap = abs(current_price - vwap) / vwap
        if distance_from_vwap > self.config.max_distance_from_vwap:
            self._clear_setup_tracking()
            self.current_phase = TradePhase.SCANNING
            return {'type': 'SETUP_INVALIDATED', 'message': "Price moved too far from VWAP"}

        # Verificar tiempo mínimo de consolidación
        if time_since_rejection >= self.config.min_consolidation_minutes:
            self.current_phase = TradePhase.WAITING_RECLAIM
            return {
                'type': 'READY_FOR_RECLAIM',
                'message': f"Ready for VWAP reclaim entry. VWAP: {vwap:.2f}"
            }

        return None

    def _check_reclaim_entry(self, data: Dict, vwap: float) -> Optional[Dict]:
        """
        Verifica condiciones para entry en VWAP reclaim

        Entry conditions:
        1. Precio breaks above VWAP
        2. Volume spike confirms breakout
        3. Risk/reward favorable
        """
        current_price = data['price']
        current_volume = data['volume']

        # Verificar breakout sobre VWAP
        if current_price <= vwap:
            return None

        # Verificar volumen spike
        avg_volume = self._calculate_average_volume()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        if volume_ratio < self.config.min_volume_spike:
            return None

        # Calcular position size y risk management
        atr = data.get('atr', current_price * 0.02)  # 2% default if no ATR
        stop_loss_price = current_price - (atr * self.config.stop_loss_atr_multiplier)

        # Verificar risk/reward
        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0:
            return None

        # Position sizing
        position_size = self._calculate_position_size(current_price, stop_loss_price)

        if position_size <= 0:
            return None

        # Calcular targets
        first_target = current_price + (risk_per_share * self.config.first_target_ratio)
        second_target = current_price + (risk_per_share * self.config.second_target_ratio)

        # Execute entry
        self.position = position_size
        self.entry_price = current_price
        self.entry_time = data['timestamp']
        self.stop_loss_price = stop_loss_price
        self.first_target_price = first_target
        self.second_target_price = second_target
        self.current_phase = TradePhase.POSITION_OPEN

        return {
            'action': 'BUY',
            'symbol': data['symbol'],
            'quantity': position_size,
            'price': current_price,
            'stop_loss': stop_loss_price,
            'first_target': first_target,
            'second_target': second_target,
            'strategy': 'VWAP_RECLAIM',
            'setup_data': {
                'vwap': vwap,
                'volume_ratio': volume_ratio,
                'risk_reward_1': self.config.first_target_ratio,
                'atr': atr
            },
            'timestamp': data['timestamp']
        }

    def _manage_position(self, data: Dict, vwap: float) -> Optional[Dict]:
        """
        Maneja posición abierta

        Exit conditions:
        1. Stop loss hit
        2. First target reached (partial exit)
        3. Second target reached (full exit)
        4. Time-based exit
        5. VWAP break (momentum failure)
        """
        current_price = data['price']
        current_time = data['timestamp']

        # Check stop loss
        if current_price <= self.stop_loss_price:
            return self._create_exit_signal(data, 'STOP_LOSS', self.position)

        # Check first target
        if current_price >= self.first_target_price and self.position == self._original_position_size():
            partial_exit_size = int(self.position * self.config.partial_exit_percent)
            self.position -= partial_exit_size

            # Move stop to breakeven
            self.stop_loss_price = self.entry_price

            return self._create_exit_signal(data, 'FIRST_TARGET', partial_exit_size)

        # Check second target
        if current_price >= self.second_target_price:
            return self._create_exit_signal(data, 'SECOND_TARGET', self.position)

        # Check time-based exit
        hold_time_minutes = (current_time - self.entry_time).total_seconds() / 60
        if hold_time_minutes >= self.config.max_hold_time_minutes:
            return self._create_exit_signal(data, 'TIME_EXIT', self.position)

        # Check VWAP break (momentum failure)
        if current_price < vwap * 0.995:  # 0.5% buffer below VWAP
            return self._create_exit_signal(data, 'VWAP_BREAK', self.position)

        return None

    def _detect_vwap_rejection(self, recent_prices: List[float],
                              vwap: float, recent_volumes: List[int]) -> bool:
        """
        Detecta rejection pattern en el VWAP

        Simplified logic - en implementación real sería más sophisticated
        """
        if len(recent_prices) < 5:
            return False

        # Buscar approach hacia VWAP seguido de rejection
        price_approached_vwap = any(abs(p - vwap) / vwap < 0.01 for p in recent_prices[-5:])
        price_rejected = recent_prices[-1] < recent_prices[-3]  # Simple rejection check
        volume_confirmed = recent_volumes[-1] > np.mean(recent_volumes[-5:])

        return price_approached_vwap and price_rejected and volume_confirmed

    def _calculate_position_size(self, entry_price: float, stop_loss_price: float) -> int:
        """
        Calcula position size basado en risk management

        Similar al método en Gap & Go template
        """
        risk_per_share = entry_price - stop_loss_price

        if risk_per_share <= 0:
            return 0

        # Method 1: Based on max risk per trade
        shares_by_risk = int(self.config.max_risk_per_trade / risk_per_share)

        # Method 2: Based on max position size
        shares_by_size = int(self.config.max_position_size / entry_price)

        return min(shares_by_risk, shares_by_size)

    def _calculate_average_volume(self) -> float:
        """Calcula volumen promedio reciente"""
        if len(self.volume_history) < self.config.volume_lookback_minutes:
            return np.mean(self.volume_history) if self.volume_history else 0

        recent_volume = self.volume_history[-self.config.volume_lookback_minutes:]
        return np.mean(recent_volume)

    def _update_history(self, price: float, volume: int, timestamp):
        """Actualiza historical data para analysis"""
        self.price_history.append(price)
        self.volume_history.append(volume)

        # Mantener solo últimos N data points
        max_history = 100
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]

    def _is_trading_window(self, current_time) -> bool:
        """Verifica si estamos en ventana de trading"""
        # Placeholder - implementar lógica de tiempo real
        return True

    def _original_position_size(self) -> int:
        """
        Retorna el tamaño original de la posición
        Útil para partial exits
        """
        # En implementación real, trackear esto properly
        return int(self.position / (1 - self.config.partial_exit_percent))

    def _create_exit_signal(self, data: Dict, reason: str, quantity: int) -> Dict:
        """Crea señal de exit"""
        exit_signal = {
            'action': 'SELL',
            'symbol': data['symbol'],
            'quantity': quantity,
            'price': data['price'],
            'reason': reason,
            'timestamp': data['timestamp'],
            'strategy': 'VWAP_RECLAIM'
        }

        # Si es exit completo, reset strategy
        if quantity == self.position:
            self.position = None
            self._clear_setup_tracking()
            self.current_phase = TradePhase.SCANNING

        return exit_signal

    def get_strategy_state(self) -> Dict:
        """
        Retorna estado actual de la estrategia para monitoring
        """
        return {
            'phase': self.current_phase.value,
            'position': self.position,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss_price,
            'first_target': self.first_target_price,
            'second_target': self.second_target_price,
            'rejection_time': self.rejection_time,
            'rejection_price': self.rejection_price,
            'current_vwap': self.vwap_indicator.cumulative_pv / self.vwap_indicator.cumulative_volume if self.vwap_indicator.cumulative_volume > 0 else None
        }


# Ejemplo de uso del template
if __name__ == "__main__":

    # Configurar la estrategia
    config = VWAPReclaimConfig(
        min_volume_spike=2.0,
        max_position_size=50.0,
        first_target_ratio=2.0,
        max_hold_time_minutes=90
    )

    # Inicializar estrategia
    strategy = VWAPReclaimStrategy(config)

    # Simulación de datos de mercado
    sample_ticks = [
        {'symbol': 'WXYZ', 'price': 5.00, 'volume': 1000, 'timestamp': pd.Timestamp.now(), 'atr': 0.12},
        {'symbol': 'WXYZ', 'price': 5.02, 'volume': 1200, 'timestamp': pd.Timestamp.now(), 'atr': 0.12},
        {'symbol': 'WXYZ', 'price': 4.98, 'volume': 1500, 'timestamp': pd.Timestamp.now(), 'atr': 0.12},
        # ... más ticks
    ]

    # Procesar cada tick
    for tick in sample_ticks:
        signal = strategy.process_tick(tick)
        if signal:
            print(f"Signal: {signal}")

        # Monitor state
        state = strategy.get_strategy_state()
        print(f"Strategy State: Phase={state['phase']}, Position={state['position']}")


"""
PERSONALIZACIÓN Y MEJORAS:
=========================

1. DETECCIÓN DE REJECTION MÁS SOFISTICADA:
   - Usar machine learning para pattern recognition
   - Incorporar order flow analysis
   - Considerar multiple timeframes

2. VWAP VARIANTS:
   - Anchored VWAP (desde gap open)
   - Rolling VWAP (últimas X horas)
   - Session VWAP vs Daily VWAP

3. FILTROS ADICIONALES:
   - Market regime (trending vs ranging)
   - Relative strength vs sector
   - News flow analysis
   - Options flow (si available)

4. EXITS MÁS DINÁMICOS:
   - Trailing stops based on VWAP distance
   - Volume-based exits
   - Time decay adjustments
   - Multiple partial exits

5. INTEGRATION FEATURES:
   - Level 2 data para mejor timing
   - Real-time news feeds
   - Social sentiment indicators
   - Correlation with broader market

BACKTESTING CONSIDERATIONS:
==========================

1. REALISTIC SLIPPAGE:
   - Small caps tienen wider spreads
   - Volume spikes pueden causar más slippage
   - Consider partial fills

2. DATA REQUIREMENTS:
   - Minute-by-minute price/volume data
   - Intraday VWAP calculations
   - Historical ATR values

3. PARAMETER OPTIMIZATION:
   - Volume spike thresholds
   - Distance from VWAP limits
   - Hold time parameters
   - Risk/reward ratios

RISK CONSIDERATIONS:
===================

1. SMALL CAP SPECIFIC RISKS:
   - Lower liquidity = higher slippage
   - More volatile moves = wider stops needed
   - News-driven gaps can invalidate setups

2. STRATEGY SPECIFIC RISKS:
   - False breakouts are common
   - VWAP can lose relevance in trending markets
   - Overfitting to specific market conditions

3. POSITION MANAGEMENT:
   - Never risk more than planned
   - Scale out profits systematically
   - Monitor correlation with other positions
"""