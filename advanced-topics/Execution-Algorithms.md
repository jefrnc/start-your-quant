# Algoritmos de Ejecución para Small Caps

Los small caps presentan desafíos únicos de ejecución debido a su liquidez limitada y spreads más amplios. Este documento proporciona algoritmos optimizados para la ejecución eficiente en estos mercados.

## Índice
1. [Desafíos de Small Caps](#desafíos-de-small-caps)
2. [Smart Order Routing](#smart-order-routing)
3. [Modelos de Impacto de Mercado](#modelos-de-impacto-de-mercado)
4. [Algoritmos de Ejecución](#algoritmos-de-ejecución)
5. [Detección de Liquidez](#detección-de-liquidez)
6. [Optimización de Timing](#optimización-de-timing)

## Desafíos de Small Caps

### Características del Mercado
- **Liquidez Limitada**: Volúmenes bajos comparados con large caps
- **Spreads Amplios**: Bid-ask spreads de 1-10 cents vs. 1 cent en large caps
- **Volatilidad Intraday**: Movimientos abruptos por órdenes grandes
- **Fragmentación**: Liquidez distribuida en múltiples venues
- **Información Asimétrica**: Mayor impacto de trading informado

### Métricas Críticas
```python
class SmallCapMetrics:
    def __init__(self):
        self.min_spread_threshold = 0.01  # 1 cent mínimo
        self.max_spread_threshold = 0.10  # 10 cents máximo
        self.min_volume_adv = 100000  # $100K ADV mínimo
        self.max_position_adv_pct = 0.10  # 10% max del ADV

    def evaluate_tradability(self, symbol_data: dict) -> dict:
        """Evalúa si un símbolo es tradeable eficientemente"""
        adv = symbol_data['average_daily_volume_usd']
        spread_pct = symbol_data['avg_spread'] / symbol_data['price']

        return {
            'tradeable': adv >= self.min_volume_adv and spread_pct <= 0.05,
            'liquidity_score': min(adv / 1000000, 10),  # Score 0-10
            'spread_score': max(0, 10 - (spread_pct * 200)),  # Score 0-10
            'execution_complexity': 'high' if adv < 500000 else 'medium'
        }
```

## Smart Order Routing

### Router Multi-Venue
```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class VenueQuote:
    venue: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    latency_ms: float

@dataclass
class ExecutionVenue:
    name: str
    fee_per_share: float
    rebate_per_share: float
    avg_latency_ms: float
    dark_pool: bool
    min_size: int
    reliability_score: float

class SmallCapRouter:
    def __init__(self):
        self.venues = {
            'NASDAQ': ExecutionVenue('NASDAQ', 0.003, 0.001, 2.1, False, 100, 0.99),
            'NYSE': ExecutionVenue('NYSE', 0.003, 0.001, 2.3, False, 100, 0.99),
            'EDGX': ExecutionVenue('EDGX', 0.002, 0.002, 2.8, False, 100, 0.97),
            'IEX': ExecutionVenue('IEX', 0.000, 0.000, 3.2, False, 100, 0.98),
            'MEMX': ExecutionVenue('MEMX', 0.001, 0.001, 2.9, False, 100, 0.96),
            'UBS_DARK': ExecutionVenue('UBS_DARK', 0.002, 0.000, 15.0, True, 500, 0.85),
            'CROSSFINDER': ExecutionVenue('CROSSFINDER', 0.002, 0.000, 20.0, True, 1000, 0.82)
        }

        self.venue_quotes: Dict[str, VenueQuote] = {}
        self.execution_history: List[dict] = []

    async def get_best_execution_plan(self,
                                    symbol: str,
                                    side: str,  # 'buy' or 'sell'
                                    quantity: int,
                                    urgency: str = 'medium') -> dict:
        """Determina el mejor plan de ejecución"""

        # Obtener quotes de todos los venues
        await self._update_venue_quotes(symbol)

        # Analizar liquidez disponible
        liquidity_analysis = self._analyze_liquidity(side, quantity)

        # Determinar estrategia óptima
        if urgency == 'high':
            return self._aggressive_execution_plan(symbol, side, quantity, liquidity_analysis)
        elif urgency == 'low':
            return self._passive_execution_plan(symbol, side, quantity, liquidity_analysis)
        else:
            return self._balanced_execution_plan(symbol, side, quantity, liquidity_analysis)

    def _analyze_liquidity(self, side: str, quantity: int) -> dict:
        """Analiza liquidez disponible en el mercado"""
        lit_liquidity = 0
        dark_liquidity_estimate = 0
        best_venues = []

        for venue_name, quote in self.venue_quotes.items():
            venue = self.venues[venue_name]

            if not venue.dark_pool:
                available = quote.ask_size if side == 'buy' else quote.bid_size
                lit_liquidity += available

                if available >= quantity * 0.1:  # Al menos 10% de la orden
                    best_venues.append({
                        'venue': venue_name,
                        'available': available,
                        'price': quote.ask if side == 'buy' else quote.bid,
                        'net_cost': self._calculate_net_cost(venue, side)
                    })
            else:
                # Estimar liquidez dark basado en ADV histórico
                dark_liquidity_estimate += quantity * 0.05  # Estimación conservadora

        return {
            'lit_liquidity': lit_liquidity,
            'dark_liquidity_estimate': dark_liquidity_estimate,
            'best_venues': sorted(best_venues, key=lambda x: x['net_cost']),
            'fragmentation_score': len([v for v in best_venues if v['available'] >= quantity * 0.2])
        }

    def _aggressive_execution_plan(self, symbol: str, side: str, quantity: int, liquidity: dict) -> dict:
        """Plan de ejecución agresivo para máxima velocidad"""
        plan = {
            'strategy': 'aggressive_sweep',
            'target_completion_time': 30,  # 30 segundos
            'orders': []
        }

        remaining_qty = quantity

        # Sweep los mejores venues lit hasta completar
        for venue_info in liquidity['best_venues']:
            if remaining_qty <= 0:
                break

            fill_qty = min(remaining_qty, venue_info['available'])

            plan['orders'].append({
                'venue': venue_info['venue'],
                'type': 'market',
                'quantity': fill_qty,
                'expected_price': venue_info['price'],
                'sequence': len(plan['orders']) + 1
            })

            remaining_qty -= fill_qty

        # Si queda cantidad, usar dark pools
        if remaining_qty > 0:
            for venue_name, venue in self.venues.items():
                if venue.dark_pool and remaining_qty > venue.min_size:
                    plan['orders'].append({
                        'venue': venue_name,
                        'type': 'market',
                        'quantity': remaining_qty,
                        'expected_price': None,  # Precio incierto en dark
                        'sequence': len(plan['orders']) + 1
                    })
                    break

        return plan

    def _passive_execution_plan(self, symbol: str, side: str, quantity: int, liquidity: dict) -> dict:
        """Plan de ejecución pasivo para minimizar impacto"""
        plan = {
            'strategy': 'passive_liquidity_seeking',
            'target_completion_time': 1800,  # 30 minutos
            'orders': []
        }

        # Dividir en chunks más pequeños
        chunk_size = max(100, quantity // 10)  # Máximo 10 chunks
        chunks = []

        remaining = quantity
        while remaining > 0:
            chunk = min(chunk_size, remaining)
            chunks.append(chunk)
            remaining -= chunk

        # Planificar ejecución escalonada
        for i, chunk in enumerate(chunks):
            # Alternar entre venues para evitar detection
            venue_options = [v for v in liquidity['best_venues'] if v['available'] >= chunk]
            selected_venue = venue_options[i % len(venue_options)] if venue_options else liquidity['best_venues'][0]

            plan['orders'].append({
                'venue': selected_venue['venue'],
                'type': 'limit',
                'quantity': chunk,
                'limit_price': selected_venue['price'],
                'time_in_force': 'IOC',  # Immediate or Cancel
                'delay_seconds': i * 180,  # 3 minutos entre órdenes
                'sequence': i + 1
            })

        return plan

    def _balanced_execution_plan(self, symbol: str, side: str, quantity: int, liquidity: dict) -> dict:
        """Plan de ejecución balanceado (velocidad vs. impacto)"""
        plan = {
            'strategy': 'balanced_twap',
            'target_completion_time': 600,  # 10 minutos
            'orders': []
        }

        # Estrategia híbrida: 60% agresivo, 40% pasivo
        aggressive_qty = int(quantity * 0.6)
        passive_qty = quantity - aggressive_qty

        # Parte agresiva: barrer mejores venues
        remaining_aggressive = aggressive_qty
        for venue_info in liquidity['best_venues'][:3]:  # Top 3 venues
            if remaining_aggressive <= 0:
                break

            fill_qty = min(remaining_aggressive, venue_info['available'] // 2)

            plan['orders'].append({
                'venue': venue_info['venue'],
                'type': 'limit',
                'quantity': fill_qty,
                'limit_price': venue_info['price'],
                'time_in_force': 'IOC',
                'sequence': len(plan['orders']) + 1,
                'phase': 'aggressive'
            })

            remaining_aggressive -= fill_qty

        # Parte pasiva: TWAP en chunks
        twap_chunks = max(2, passive_qty // 500)  # Chunks de ~500 acciones
        chunk_size = passive_qty // twap_chunks

        for i in range(twap_chunks):
            delay = 60 + (i * 300)  # 1 min inicial + 5 min entre chunks

            plan['orders'].append({
                'venue': 'IEX',  # Venue neutro para TWAP
                'type': 'limit',
                'quantity': chunk_size,
                'limit_price': None,  # Precio a determinar en el momento
                'time_in_force': 'GTC',
                'delay_seconds': delay,
                'sequence': len(plan['orders']) + 1,
                'phase': 'passive'
            })

        return plan

    def _calculate_net_cost(self, venue: ExecutionVenue, side: str) -> float:
        """Calcula costo neto considerando fees y rebates"""
        base_cost = venue.fee_per_share

        if side == 'buy':
            # Fees por tomar liquidez
            return base_cost
        else:
            # Posible rebate por proveer liquidez
            return base_cost - venue.rebate_per_share

    async def _update_venue_quotes(self, symbol: str):
        """Actualiza quotes de todos los venues (simulado)"""
        # En implementación real, conectaría a market data feeds
        base_price = 5.50  # Precio ejemplo

        for venue_name in self.venues.keys():
            venue = self.venues[venue_name]

            # Simular variación en spread y sizes
            spread = 0.02 + (0.01 if venue.dark_pool else 0)
            size_variance = 500 + (venue.reliability_score * 1000)

            self.venue_quotes[venue_name] = VenueQuote(
                venue=venue_name,
                bid=base_price - spread/2,
                ask=base_price + spread/2,
                bid_size=int(size_variance),
                ask_size=int(size_variance),
                timestamp=datetime.now(),
                latency_ms=venue.avg_latency_ms
            )
```

## Modelos de Impacto de Mercado

### Modelo de Impacto Permanente y Temporal
```python
import numpy as np
from scipy.optimize import minimize
import pandas as pd

class SmallCapImpactModel:
    def __init__(self):
        # Parámetros calibrados para small caps
        self.lambda_permanent = 0.05  # Impacto permanente más alto
        self.lambda_temporary = 0.02   # Impacto temporal
        self.gamma = 0.6              # Elasticidad de volumen
        self.decay_halflife = 300     # 5 minutos para decay temporal

    def estimate_market_impact(self,
                             quantity: int,
                             adv: float,  # Average Daily Volume
                             spread: float,
                             volatility: float,
                             time_horizon_seconds: int = 600) -> dict:
        """Estima impacto de mercado total"""

        # Normalizar cantidad por ADV
        participation_rate = quantity / adv

        # Impacto permanente (no se recupera)
        permanent_impact_bps = self.lambda_permanent * (participation_rate ** self.gamma) * 10000

        # Impacto temporal (se decay exponencialmente)
        temporary_impact_bps = self.lambda_temporary * (participation_rate ** 0.5) * 10000

        # Factor de spread (small caps tienen spreads más altos)
        spread_impact_bps = (spread * 10000) * min(1.0, participation_rate * 2)

        # Factor de volatilidad
        volatility_impact_bps = volatility * participation_rate * 5000

        # Impacto total
        total_impact_bps = (permanent_impact_bps +
                           temporary_impact_bps +
                           spread_impact_bps +
                           volatility_impact_bps)

        # Calcular decay temporal
        decay_factor = np.exp(-np.log(2) * time_horizon_seconds / self.decay_halflife)
        recoverable_impact_bps = temporary_impact_bps * decay_factor

        return {
            'total_impact_bps': total_impact_bps,
            'permanent_impact_bps': permanent_impact_bps,
            'temporary_impact_bps': temporary_impact_bps,
            'spread_impact_bps': spread_impact_bps,
            'volatility_impact_bps': volatility_impact_bps,
            'recoverable_impact_bps': recoverable_impact_bps,
            'participation_rate': participation_rate,
            'recommended_max_quantity': int(adv * 0.05)  # Máximo 5% ADV
        }

    def optimize_execution_schedule(self,
                                  total_quantity: int,
                                  adv: float,
                                  price: float,
                                  target_time_minutes: int = 30) -> dict:
        """Optimiza cronograma de ejecución para minimizar impacto"""

        # Función objetivo: minimizar impacto total
        def objective(chunk_sizes):
            total_cost = 0
            cumulative_qty = 0

            for i, chunk in enumerate(chunk_sizes):
                if chunk <= 0:
                    continue

                # Impacto marginal de este chunk
                remaining_adv = adv * (1 - cumulative_qty / total_quantity * 0.1)
                impact = self.estimate_market_impact(chunk, remaining_adv, 0.02, 0.3)

                total_cost += chunk * price * impact['total_impact_bps'] / 10000
                cumulative_qty += chunk

            return total_cost

        # Restricciones
        n_chunks = min(10, target_time_minutes // 3)  # Un chunk cada 3 minutos

        def constraint_total_quantity(chunk_sizes):
            return np.sum(chunk_sizes) - total_quantity

        def constraint_max_chunk(chunk_sizes):
            max_chunk = adv * 0.02  # Máximo 2% ADV por chunk
            return max_chunk - np.max(chunk_sizes)

        # Optimización
        x0 = np.ones(n_chunks) * (total_quantity / n_chunks)
        bounds = [(0, total_quantity) for _ in range(n_chunks)]
        constraints = [
            {'type': 'eq', 'fun': constraint_total_quantity},
            {'type': 'ineq', 'fun': constraint_max_chunk}
        ]

        result = minimize(objective, x0, bounds=bounds, constraints=constraints)

        optimal_chunks = [max(0, int(chunk)) for chunk in result.x if chunk > 0]

        return {
            'optimal_chunks': optimal_chunks,
            'total_estimated_cost_bps': result.fun / (total_quantity * price) * 10000,
            'n_chunks': len(optimal_chunks),
            'avg_chunk_size': np.mean(optimal_chunks),
            'execution_schedule_minutes': [i * target_time_minutes / len(optimal_chunks)
                                         for i in range(len(optimal_chunks))]
        }
```

## Algoritmos de Ejecución

### TWAP Adaptativo para Small Caps
```python
class AdaptiveTWAP:
    def __init__(self, symbol: str, total_quantity: int, duration_minutes: int):
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.duration_minutes = duration_minutes
        self.executed_quantity = 0
        self.remaining_quantity = total_quantity

        # Parámetros adaptativos
        self.base_chunk_size = total_quantity // (duration_minutes // 3)  # Chunk cada 3 min
        self.volatility_multiplier = 1.0
        self.liquidity_multiplier = 1.0
        self.urgency_multiplier = 1.0

        # Estado del mercado
        self.current_spread = 0.02
        self.current_volume_rate = 1.0  # Multiplicador de volumen normal
        self.market_trend = 0.0  # -1 (bajista) a 1 (alcista)

    def get_next_chunk_size(self, current_market_data: dict) -> int:
        """Calcula tamaño del próximo chunk basado en condiciones actuales"""

        # Actualizar estado del mercado
        self._update_market_state(current_market_data)

        # Chunk base ajustado por tiempo restante
        progress = self.executed_quantity / self.total_quantity
        time_progress = self._get_time_progress()

        if time_progress > progress + 0.1:  # Estamos atrasados
            urgency_adj = 1.5
        elif time_progress < progress - 0.1:  # Estamos adelantados
            urgency_adj = 0.7
        else:
            urgency_adj = 1.0

        # Ajuste por volatilidad
        if self.current_spread > 0.03:  # Spread alto = reducir chunks
            volatility_adj = 0.8
        else:
            volatility_adj = 1.2

        # Ajuste por liquidez disponible
        if current_market_data.get('volume_rate', 1.0) > 2.0:  # Alta actividad
            liquidity_adj = 1.3
        else:
            liquidity_adj = 0.9

        # Calcular chunk final
        adjusted_chunk = int(self.base_chunk_size * urgency_adj * volatility_adj * liquidity_adj)

        # Límites de seguridad
        min_chunk = max(100, self.remaining_quantity // 20)  # Mínimo razonable
        max_chunk = min(self.remaining_quantity, self.total_quantity // 5)  # Máximo 20%

        return max(min_chunk, min(adjusted_chunk, max_chunk))

    def _update_market_state(self, market_data: dict):
        """Actualiza estado del mercado para decisiones adaptativas"""
        self.current_spread = market_data.get('spread', 0.02)
        self.current_volume_rate = market_data.get('volume_rate', 1.0)

        # Calcular tendencia basada en precio reciente
        price_change = market_data.get('price_change_5min', 0.0)
        self.market_trend = np.tanh(price_change / market_data.get('price', 1.0) * 100)

    def _get_time_progress(self) -> float:
        """Calcula progreso temporal (0 a 1)"""
        # Simplificado - en realidad usaría timestamp actual
        return 0.5  # Placeholder

### Iceberg Orders para Small Caps
class IcebergOrderManager:
    def __init__(self, symbol: str, total_quantity: int, visible_size: int):
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.visible_size = visible_size
        self.executed_quantity = 0
        self.active_order_id = None

        # Configuración específica para small caps
        self.min_refresh_interval = 30  # 30 segundos mínimo entre refreshes
        self.max_visible_percentage = 0.15  # Máximo 15% visible del size total

    def calculate_optimal_visible_size(self, current_book_depth: dict) -> int:
        """Calcula tamaño visible óptimo basado en book depth"""

        total_book_size = sum(current_book_depth.get('bid_sizes', []))

        # No ser más del 20% del book depth total
        max_by_book = int(total_book_size * 0.2)

        # No ser más del 15% de nuestra orden total
        max_by_order = int(self.total_quantity * self.max_visible_percentage)

        # Usar el menor
        optimal_size = min(max_by_book, max_by_order, self.visible_size)

        return max(100, optimal_size)  # Mínimo 100 acciones

    def should_refresh_order(self, time_since_last: int, market_move: float) -> bool:
        """Determina si debe hacer refresh de la orden visible"""

        # Refresh por tiempo
        if time_since_last > 300:  # 5 minutos máximo
            return True

        # Refresh por movimiento de mercado
        if abs(market_move) > 0.01:  # 1% de movimiento
            return True

        # Refresh si muy poco fill en tiempo razonable
        if time_since_last > 120 and self.executed_quantity == 0:  # 2 min sin fills
            return True

        return False
```

## Detección de Liquidez

### Monitor de Liquidez en Tiempo Real
```python
class LiquidityMonitor:
    def __init__(self):
        self.liquidity_history = {}
        self.anomaly_threshold = 2.0  # Desviaciones estándar

    def analyze_liquidity_pattern(self, symbol: str, timeframe_minutes: int = 60) -> dict:
        """Analiza patrones de liquidez para timing óptimo"""

        # Simular datos históricos de liquidez
        historical_data = self._get_historical_liquidity(symbol, timeframe_minutes)

        current_liquidity = self._calculate_current_liquidity(symbol)

        # Detectar anomalías en liquidez
        liquidity_z_score = self._calculate_z_score(current_liquidity, historical_data)

        # Predecir ventanas de liquidez óptima
        optimal_windows = self._predict_liquidity_windows(historical_data)

        return {
            'current_liquidity_score': current_liquidity,
            'liquidity_percentile': self._get_percentile(current_liquidity, historical_data),
            'is_anomaly': abs(liquidity_z_score) > self.anomaly_threshold,
            'z_score': liquidity_z_score,
            'optimal_execution_windows': optimal_windows,
            'recommendation': self._get_execution_recommendation(current_liquidity, historical_data)
        }

    def _calculate_current_liquidity(self, symbol: str) -> float:
        """Calcula score de liquidez actual"""
        # Factores de liquidez:
        # 1. Book depth (suma de bid/ask sizes)
        # 2. Spread tightness
        # 3. Volume rate vs. normal
        # 4. Number of active market makers

        book_depth_score = 5.0  # Placeholder
        spread_score = 7.0      # Placeholder
        volume_score = 6.0      # Placeholder
        mm_score = 4.0          # Placeholder

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        scores = [book_depth_score, spread_score, volume_score, mm_score]

        return sum(w * s for w, s in zip(weights, scores))

    def _predict_liquidity_windows(self, historical_data: list) -> list:
        """Predice ventanas de liquidez óptima usando patterns históricos"""

        # Análisis por hora del día
        hourly_patterns = {}
        for hour in range(24):
            hour_data = [d for d in historical_data if d['hour'] == hour]
            if hour_data:
                hourly_patterns[hour] = np.mean([d['liquidity_score'] for d in hour_data])

        # Identificar mejores horas
        best_hours = sorted(hourly_patterns.items(), key=lambda x: x[1], reverse=True)[:3]

        return [
            {
                'hour': hour,
                'liquidity_score': score,
                'recommended': score > np.mean(list(hourly_patterns.values()))
            }
            for hour, score in best_hours
        ]

    def _get_execution_recommendation(self, current_liquidity: float, historical_data: list) -> str:
        """Recomienda timing de ejecución"""

        avg_liquidity = np.mean([d['liquidity_score'] for d in historical_data])

        if current_liquidity > avg_liquidity * 1.5:
            return "EXECUTE_NOW"  # Liquidez excepcional
        elif current_liquidity > avg_liquidity * 1.2:
            return "EXECUTE_SOON"  # Buena liquidez
        elif current_liquidity > avg_liquidity * 0.8:
            return "WAIT_FOR_BETTER"  # Liquidez promedio
        else:
            return "AVOID_EXECUTION"  # Liquidez pobre

    def _get_historical_liquidity(self, symbol: str, timeframe_minutes: int) -> list:
        """Obtiene datos históricos de liquidez (simulado)"""
        # En implementación real, consultaría base de datos
        return [
            {'hour': i % 24, 'liquidity_score': 5.0 + np.random.normal(0, 1.5)}
            for i in range(timeframe_minutes // 15)  # Data cada 15 minutos
        ]

    def _calculate_z_score(self, current: float, historical: list) -> float:
        """Calcula z-score de liquidez actual vs. histórica"""
        historical_scores = [d['liquidity_score'] for d in historical]
        mean = np.mean(historical_scores)
        std = np.std(historical_scores)

        return (current - mean) / std if std > 0 else 0

    def _get_percentile(self, current: float, historical: list) -> float:
        """Calcula percentil de liquidez actual"""
        historical_scores = [d['liquidity_score'] for d in historical]
        return (sum(1 for score in historical_scores if score <= current) /
                len(historical_scores) * 100)
```

## Optimización de Timing

### Sistema de Timing Inteligente
```python
from datetime import datetime, time
import pytz

class SmallCapTimingOptimizer:
    def __init__(self):
        self.est = pytz.timezone('US/Eastern')

        # Ventanas de tiempo optimizadas para small caps
        self.optimal_windows = {
            'market_open': (time(9, 30), time(10, 30)),    # 60 min después de apertura
            'mid_morning': (time(10, 30), time(11, 30)),   # Actividad estable
            'lunch_lull': (time(12, 0), time(14, 0)),      # EVITAR - baja liquidez
            'afternoon': (time(14, 0), time(15, 30)),      # Buena liquidez
            'close': (time(15, 30), time(16, 0))           # EVITAR - volatilidad alta
        }

        self.volatility_windows = {
            'high': ['market_open', 'close'],
            'medium': ['mid_morning', 'afternoon'],
            'low': ['lunch_lull']
        }

    def get_optimal_execution_time(self,
                                 strategy: str,
                                 urgency: str = 'medium',
                                 max_acceptable_spread: float = 0.05) -> dict:
        """Determina timing óptimo para ejecución"""

        current_time = datetime.now(self.est).time()
        current_window = self._get_current_window(current_time)

        recommendation = {
            'current_window': current_window,
            'current_time': current_time.strftime('%H:%M:%S'),
            'action': 'HOLD',
            'reason': '',
            'next_optimal_window': None,
            'estimated_wait_minutes': 0
        }

        # Lógica de decisión basada en estrategia
        if strategy == 'aggressive':
            if current_window in ['mid_morning', 'afternoon']:
                recommendation['action'] = 'EXECUTE'
                recommendation['reason'] = 'Ventana óptima para ejecución agresiva'
            else:
                recommendation['action'] = 'WAIT'
                recommendation['reason'] = 'Esperar ventana de menor volatilidad'
                recommendation['next_optimal_window'] = self._get_next_optimal_window(current_time, ['mid_morning', 'afternoon'])

        elif strategy == 'passive':
            if current_window == 'lunch_lull':
                recommendation['action'] = 'EXECUTE'
                recommendation['reason'] = 'Período de baja competencia por liquidez'
            elif current_window in ['mid_morning', 'afternoon']:
                recommendation['action'] = 'EXECUTE'
                recommendation['reason'] = 'Ventana aceptable para ejecución pasiva'
            else:
                recommendation['action'] = 'WAIT'
                recommendation['reason'] = 'Esperar menor volatilidad'

        elif strategy == 'opportunistic':
            # Ejecutar en cualquier ventana que no sea alta volatilidad
            if current_window not in ['market_open', 'close']:
                recommendation['action'] = 'EXECUTE'
                recommendation['reason'] = 'Ventana aceptable para oportunidades'
            else:
                recommendation['action'] = 'WAIT'
                recommendation['reason'] = 'Alta volatilidad - esperar'

        # Override por urgencia
        if urgency == 'high':
            recommendation['action'] = 'EXECUTE'
            recommendation['reason'] = 'Ejecutar inmediatamente por urgencia alta'

        # Calcular tiempo de espera si aplica
        if recommendation['action'] == 'WAIT' and recommendation['next_optimal_window']:
            recommendation['estimated_wait_minutes'] = self._calculate_wait_time(
                current_time, recommendation['next_optimal_window']
            )

        return recommendation

    def _get_current_window(self, current_time: time) -> str:
        """Identifica ventana de tiempo actual"""
        for window_name, (start, end) in self.optimal_windows.items():
            if start <= current_time <= end:
                return window_name
        return 'after_hours'

    def _get_next_optimal_window(self, current_time: time, preferred_windows: list) -> str:
        """Encuentra la próxima ventana óptima"""
        for window_name in preferred_windows:
            window_start, window_end = self.optimal_windows[window_name]
            if current_time < window_start:
                return window_name

        # Si no hay ventana hoy, retornar la primera del próximo día
        return preferred_windows[0]

    def _calculate_wait_time(self, current_time: time, target_window: str) -> int:
        """Calcula minutos de espera hasta ventana objetivo"""
        target_start, _ = self.optimal_windows[target_window]

        current_minutes = current_time.hour * 60 + current_time.minute
        target_minutes = target_start.hour * 60 + target_start.minute

        if target_minutes > current_minutes:
            return target_minutes - current_minutes
        else:
            # Próximo día
            return (24 * 60) - current_minutes + target_minutes

    def get_execution_quality_forecast(self, symbol: str, target_time: datetime) -> dict:
        """Predice calidad de ejecución para un timing específico"""

        target_time_et = target_time.astimezone(self.est)
        target_window = self._get_current_window(target_time_et.time())

        # Score base por ventana
        window_scores = {
            'market_open': 6.0,    # Alta liquidez pero alta volatilidad
            'mid_morning': 8.5,    # Óptimo
            'lunch_lull': 7.0,     # Baja competencia pero menos liquidez
            'afternoon': 8.0,      # Muy bueno
            'close': 5.0,          # Alta volatilidad
            'after_hours': 3.0     # Muy limitado
        }

        base_score = window_scores.get(target_window, 5.0)

        # Ajustes por día de la semana
        weekday = target_time_et.weekday()
        if weekday == 0:  # Lunes
            day_adjustment = -0.5  # Slightly worse
        elif weekday == 4:  # Viernes
            day_adjustment = -1.0  # Worse liquidity
        else:
            day_adjustment = 0.0

        final_score = max(0, min(10, base_score + day_adjustment))

        return {
            'execution_quality_score': final_score,
            'target_window': target_window,
            'expected_spread_bps': self._estimate_spread_for_window(target_window),
            'expected_impact_bps': self._estimate_impact_for_window(target_window),
            'liquidity_confidence': 'high' if final_score > 7.5 else 'medium' if final_score > 5.0 else 'low'
        }

    def _estimate_spread_for_window(self, window: str) -> float:
        """Estima spread esperado por ventana"""
        spread_estimates = {
            'market_open': 3.5,    # Spreads más amplios
            'mid_morning': 2.0,    # Spreads normales
            'lunch_lull': 3.0,     # Spreads ligeramente amplios
            'afternoon': 2.2,      # Spreads normales
            'close': 4.0,          # Spreads amplios
            'after_hours': 8.0     # Spreads muy amplios
        }
        return spread_estimates.get(window, 3.0)

    def _estimate_impact_for_window(self, window: str) -> float:
        """Estima impacto de mercado por ventana"""
        impact_estimates = {
            'market_open': 4.0,    # Alto impacto por volatilidad
            'mid_morning': 2.5,    # Impacto normal
            'lunch_lull': 3.5,     # Impacto alto por baja liquidez
            'afternoon': 2.8,      # Impacto normal
            'close': 5.0,          # Alto impacto
            'after_hours': 8.0     # Impacto muy alto
        }
        return impact_estimates.get(window, 3.0)
```

## Ejemplo de Implementación Completa

```python
async def execute_small_cap_order():
    """Ejemplo completo de ejecución optimizada para small caps"""

    # Configuración de la orden
    symbol = "ABCD"
    quantity = 5000
    side = "buy"
    urgency = "medium"

    # Inicializar componentes
    router = SmallCapRouter()
    impact_model = SmallCapImpactModel()
    timing_optimizer = SmallCapTimingOptimizer()
    liquidity_monitor = LiquidityMonitor()

    # 1. Analizar timing óptimo
    timing_rec = timing_optimizer.get_optimal_execution_time("balanced", urgency)

    if timing_rec['action'] == 'WAIT':
        print(f"Esperando {timing_rec['estimated_wait_minutes']} minutos para ventana óptima")
        return

    # 2. Analizar liquidez actual
    liquidity_analysis = liquidity_monitor.analyze_liquidity_pattern(symbol)

    if liquidity_analysis['recommendation'] == 'AVOID_EXECUTION':
        print("Liquidez insuficiente - evitar ejecución")
        return

    # 3. Estimar impacto de mercado
    adv = 250000  # $250K ADV estimado
    impact_estimate = impact_model.estimate_market_impact(
        quantity=quantity,
        adv=adv,
        spread=0.02,
        volatility=0.35
    )

    print(f"Impacto estimado: {impact_estimate['total_impact_bps']:.1f} bps")

    if impact_estimate['total_impact_bps'] > 50:  # Más de 50 bps
        print("Impacto muy alto - considerando división en chunks")

        # Optimizar cronograma
        schedule = impact_model.optimize_execution_schedule(
            quantity, adv, 5.50, target_time_minutes=30
        )

        print(f"Ejecutando en {len(schedule['optimal_chunks'])} chunks")
        return schedule

    # 4. Obtener plan de ejecución
    execution_plan = await router.get_best_execution_plan(
        symbol, side, quantity, urgency
    )

    print(f"Estrategia: {execution_plan['strategy']}")
    print(f"Órdenes planificadas: {len(execution_plan['orders'])}")

    # 5. Ejecutar plan
    for order in execution_plan['orders']:
        print(f"Ejecutando {order['quantity']} en {order['venue']}")
        # Aquí iría la lógica de envío real de órdenes

    return execution_plan

# Ejecutar ejemplo
# asyncio.run(execute_small_cap_order())
```

## Conclusiones

Los algoritmos de ejecución para small caps requieren consideraciones especiales:

1. **Fragmentación de Liquidez**: Usar smart routing para encontrar liquidez oculta
2. **Impacto de Mercado**: Modelos calibrados para spreads amplios y baja liquidez
3. **Timing Inteligente**: Evitar períodos de alta volatilidad
4. **Ejecución Adaptativa**: Ajustar estrategia basada en condiciones en tiempo real
5. **Monitoreo Continuo**: Detectar cambios en patrones de liquidez

La clave está en balancear velocidad vs. impacto, adaptándose constantemente a las condiciones únicas del mercado de small caps.