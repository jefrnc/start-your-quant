# Estrategias de Automatización

## Automatización de Estrategias de Trading

### Framework de Estrategia Automatizada
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
import numpy as np
import pandas as pd
from enum import Enum

class SignalType(Enum):
    """Tipos de señales"""
    BUY = "buy"
    SELL = "sell" 
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"

class SignalStrength(Enum):
    """Fuerza de señales"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class TradingSignal:
    """Señal de trading"""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    price: float
    quantity: int
    timestamp: datetime
    strategy_name: str
    confidence: float  # 0-1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketData:
    """Datos de mercado"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    vwap: Optional[float] = None
    indicators: Dict[str, float] = field(default_factory=dict)

class AutomatedStrategy(ABC):
    """Clase base para estrategias automatizadas"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"Strategy.{name}")
        self.is_active = True
        self.positions = {}
        self.signals_history = []
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
        
        # Risk parameters
        self.max_position_size = config.get('max_position_size', 0.05)  # 5% of portfolio
        self.max_daily_trades = config.get('max_daily_trades', 10)
        self.min_confidence = config.get('min_confidence', 0.6)
        
        # Timing parameters
        self.signal_cooldown = config.get('signal_cooldown_minutes', 5)
        self.last_signal_time = {}
    
    @abstractmethod
    async def analyze_market_data(self, data: MarketData) -> List[TradingSignal]:
        """Analizar datos de mercado y generar señales"""
        pass
    
    @abstractmethod
    async def calculate_position_size(self, signal: TradingSignal, 
                                    account_equity: float) -> int:
        """Calcular tamaño de posición"""
        pass
    
    def should_generate_signal(self, symbol: str) -> bool:
        """Verificar si debe generar señal (cooldown)"""
        if symbol not in self.last_signal_time:
            return True
        
        time_since_last = datetime.now() - self.last_signal_time[symbol]
        return time_since_last >= timedelta(minutes=self.signal_cooldown)
    
    def record_signal(self, signal: TradingSignal):
        """Registrar señal generada"""
        self.signals_history.append(signal)
        self.last_signal_time[signal.symbol] = signal.timestamp
        self.performance_metrics['total_signals'] += 1
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(days=30)
        self.signals_history = [s for s in self.signals_history 
                               if s.timestamp >= cutoff_time]
    
    def update_performance(self, signal: TradingSignal, pnl: float):
        """Actualizar métricas de performance"""
        if pnl > 0:
            self.performance_metrics['successful_signals'] += 1
        
        self.performance_metrics['total_pnl'] += pnl
        
        if self.performance_metrics['total_signals'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['successful_signals'] / 
                self.performance_metrics['total_signals']
            )
    
    def get_performance_summary(self) -> Dict:
        """Obtener resumen de performance"""
        return {
            'strategy_name': self.name,
            'is_active': self.is_active,
            'metrics': self.performance_metrics.copy(),
            'recent_signals_count': len([s for s in self.signals_history 
                                       if s.timestamp >= datetime.now() - timedelta(days=1)]),
            'avg_confidence': np.mean([s.confidence for s in self.signals_history]) if self.signals_history else 0
        }

class GapAndGoAutomated(AutomatedStrategy):
    """Estrategia Gap and Go automatizada"""
    
    def __init__(self, config: Dict):
        super().__init__("GapAndGo", config)
        
        # Strategy-specific parameters
        self.min_gap_percent = config.get('min_gap_percent', 0.05)  # 5%
        self.max_gap_percent = config.get('max_gap_percent', 0.20)  # 20%
        self.min_volume_ratio = config.get('min_volume_ratio', 2.0)  # 2x normal
        self.min_price = config.get('min_price', 5.0)
        self.max_price = config.get('max_price', 50.0)
        
        # Market context
        self.spy_trend_bullish = True
        self.vix_level = 20.0
    
    async def analyze_market_data(self, data: MarketData) -> List[TradingSignal]:
        """Analizar datos para Gap and Go"""
        
        signals = []
        
        # Skip if not in active hours (first 2 hours of market)
        market_open = data.timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        if data.timestamp > market_open + timedelta(hours=2):
            return signals
        
        # Skip if cooldown not satisfied
        if not self.should_generate_signal(data.symbol):
            return signals
        
        # Calculate gap
        prev_close = data.indicators.get('prev_close')
        if not prev_close:
            return signals
        
        gap_percent = (data.open_price - prev_close) / prev_close
        
        # Check gap criteria
        if not (self.min_gap_percent <= gap_percent <= self.max_gap_percent):
            return signals
        
        # Check price range
        if not (self.min_price <= data.close_price <= self.max_price):
            return signals
        
        # Check volume
        avg_volume = data.indicators.get('avg_volume_20d', data.volume)
        volume_ratio = data.volume / avg_volume if avg_volume > 0 else 0
        
        if volume_ratio < self.min_volume_ratio:
            return signals
        
        # Calculate signal strength and confidence
        strength, confidence = self._calculate_signal_quality(data, gap_percent, volume_ratio)
        
        if confidence >= self.min_confidence:
            # Calculate position details
            stop_loss = self._calculate_stop_loss(data)
            take_profit = self._calculate_take_profit(data, gap_percent)
            
            signal = TradingSignal(
                symbol=data.symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                price=data.close_price,
                quantity=0,  # Will be calculated later
                timestamp=data.timestamp,
                strategy_name=self.name,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'gap_percent': gap_percent,
                    'volume_ratio': volume_ratio,
                    'prev_close': prev_close,
                    'market_context': {
                        'spy_bullish': self.spy_trend_bullish,
                        'vix_level': self.vix_level
                    }
                }
            )
            
            signals.append(signal)
            self.record_signal(signal)
            
            self.logger.info(f"Gap and Go signal: {data.symbol} gap={gap_percent:.2%} vol={volume_ratio:.1f}x confidence={confidence:.2f}")
        
        return signals
    
    def _calculate_signal_quality(self, data: MarketData, 
                                gap_percent: float, volume_ratio: float) -> Tuple[SignalStrength, float]:
        """Calcular calidad de señal"""
        
        score = 0
        
        # Gap size score (optimal around 8-12%)
        if 0.08 <= gap_percent <= 0.12:
            score += 30
        elif 0.05 <= gap_percent <= 0.15:
            score += 20
        else:
            score += 10
        
        # Volume score
        if volume_ratio >= 5:
            score += 25
        elif volume_ratio >= 3:
            score += 20
        elif volume_ratio >= 2:
            score += 15
        
        # Price action score
        if data.close_price > data.open_price:  # Green candle
            score += 15
        
        # VWAP relationship
        vwap = data.vwap or data.close_price
        if data.close_price > vwap:
            score += 15
        
        # Market context
        if self.spy_trend_bullish:
            score += 10
        
        if self.vix_level < 25:  # Low volatility environment
            score += 5
        
        # Convert score to strength and confidence
        confidence = min(score / 100, 1.0)
        
        if confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        return strength, confidence
    
    def _calculate_stop_loss(self, data: MarketData) -> float:
        """Calcular stop loss"""
        # Stop loss below VWAP or previous day low
        vwap = data.vwap or data.close_price
        prev_low = data.indicators.get('prev_low', data.low_price)
        
        vwap_stop = vwap * 0.97  # 3% below VWAP
        prev_low_stop = prev_low * 0.98  # 2% below previous low
        
        return min(vwap_stop, prev_low_stop)
    
    def _calculate_take_profit(self, data: MarketData, gap_percent: float) -> float:
        """Calcular take profit"""
        # Target based on gap size
        if gap_percent >= 0.10:
            target_percent = 0.15  # 15% target for large gaps
        else:
            target_percent = 0.10  # 10% target for smaller gaps
        
        return data.close_price * (1 + target_percent)
    
    async def calculate_position_size(self, signal: TradingSignal, 
                                    account_equity: float) -> int:
        """Calcular tamaño de posición para Gap and Go"""
        
        # Base risk amount
        base_risk = account_equity * 0.02  # 2% base risk
        
        # Adjust for signal confidence
        confidence_multiplier = signal.confidence
        adjusted_risk = base_risk * confidence_multiplier
        
        # Calculate risk per share
        entry_price = signal.price
        stop_loss = signal.stop_loss or (entry_price * 0.95)
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            return 0
        
        # Calculate shares
        max_shares = int(adjusted_risk / risk_per_share)
        
        # Apply position size limit
        max_position_value = account_equity * self.max_position_size
        max_shares_by_limit = int(max_position_value / entry_price)
        
        return min(max_shares, max_shares_by_limit)

class VWAPReclaimAutomated(AutomatedStrategy):
    """Estrategia VWAP Reclaim automatizada"""
    
    def __init__(self, config: Dict):
        super().__init__("VWAPReclaim", config)
        
        # Strategy parameters
        self.min_time_below_vwap = config.get('min_time_below_vwap_minutes', 30)
        self.max_distance_from_vwap = config.get('max_distance_from_vwap_pct', 0.03)
        self.min_volume_confirmation = config.get('min_volume_confirmation', 1.5)
        
        # Track symbols below VWAP
        self.symbols_below_vwap = {}
    
    async def analyze_market_data(self, data: MarketData) -> List[TradingSignal]:
        """Analizar datos para VWAP Reclaim"""
        
        signals = []
        
        # Skip if not in active hours
        if not self._is_active_time(data.timestamp):
            return signals
        
        # Skip if cooldown not satisfied
        if not self.should_generate_signal(data.symbol):
            return signals
        
        vwap = data.vwap
        if not vwap:
            return signals
        
        current_price = data.close_price
        distance_from_vwap = (current_price - vwap) / vwap
        
        # Track time below VWAP
        if current_price < vwap:
            if data.symbol not in self.symbols_below_vwap:
                self.symbols_below_vwap[data.symbol] = data.timestamp
            return signals  # Still below VWAP
        
        # Check if was below VWAP recently
        if data.symbol not in self.symbols_below_vwap:
            return signals
        
        time_below_vwap = data.timestamp - self.symbols_below_vwap[data.symbol]
        
        # Remove from tracking since above VWAP now
        del self.symbols_below_vwap[data.symbol]
        
        # Check criteria
        if time_below_vwap.total_seconds() / 60 < self.min_time_below_vwap:
            return signals
        
        if distance_from_vwap > self.max_distance_from_vwap:
            return signals  # Too far above VWAP
        
        # Check volume confirmation
        avg_volume = data.indicators.get('avg_volume_20d', data.volume)
        volume_ratio = data.volume / avg_volume if avg_volume > 0 else 0
        
        if volume_ratio < self.min_volume_confirmation:
            return signals
        
        # Calculate signal quality
        strength, confidence = self._calculate_vwap_signal_quality(
            data, distance_from_vwap, volume_ratio, time_below_vwap
        )
        
        if confidence >= self.min_confidence:
            signal = TradingSignal(
                symbol=data.symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                price=current_price,
                quantity=0,
                timestamp=data.timestamp,
                strategy_name=self.name,
                confidence=confidence,
                stop_loss=vwap * 0.98,  # 2% below VWAP
                take_profit=current_price * 1.08,  # 8% target
                metadata={
                    'vwap': vwap,
                    'distance_from_vwap': distance_from_vwap,
                    'time_below_vwap_minutes': time_below_vwap.total_seconds() / 60,
                    'volume_ratio': volume_ratio
                }
            )
            
            signals.append(signal)
            self.record_signal(signal)
            
            self.logger.info(f"VWAP Reclaim signal: {data.symbol} distance={distance_from_vwap:.2%} vol={volume_ratio:.1f}x")
        
        return signals
    
    def _is_active_time(self, timestamp: datetime) -> bool:
        """Verificar si es tiempo activo para la estrategia"""
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = timestamp.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= timestamp <= market_close
    
    def _calculate_vwap_signal_quality(self, data: MarketData, distance: float,
                                     volume_ratio: float, time_below: timedelta) -> Tuple[SignalStrength, float]:
        """Calcular calidad de señal VWAP"""
        
        score = 0
        
        # Distance score (closer to VWAP is better)
        if abs(distance) <= 0.01:  # Within 1%
            score += 30
        elif abs(distance) <= 0.02:  # Within 2%
            score += 20
        else:
            score += 10
        
        # Volume confirmation
        if volume_ratio >= 3:
            score += 25
        elif volume_ratio >= 2:
            score += 20
        elif volume_ratio >= 1.5:
            score += 15
        
        # Time below VWAP (optimal range)
        time_minutes = time_below.total_seconds() / 60
        if 45 <= time_minutes <= 120:  # 45 min to 2 hours
            score += 25
        elif 30 <= time_minutes <= 180:
            score += 15
        else:
            score += 5
        
        # Price action
        if data.close_price > data.open_price:
            score += 15
        
        # Recent momentum
        sma_20 = data.indicators.get('sma_20')
        if sma_20 and data.close_price > sma_20:
            score += 5
        
        confidence = min(score / 100, 1.0)
        
        if confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        return strength, confidence
    
    async def calculate_position_size(self, signal: TradingSignal, 
                                    account_equity: float) -> int:
        """Calcular tamaño de posición para VWAP Reclaim"""
        
        base_risk = account_equity * 0.015  # 1.5% base risk
        
        # Adjust for signal strength
        strength_multipliers = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MODERATE: 0.75,
            SignalStrength.STRONG: 1.0,
            SignalStrength.VERY_STRONG: 1.25
        }
        
        multiplier = strength_multipliers[signal.strength]
        adjusted_risk = base_risk * multiplier * signal.confidence
        
        # Calculate position size
        entry_price = signal.price
        stop_loss = signal.stop_loss or (entry_price * 0.95)
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            return 0
        
        shares = int(adjusted_risk / risk_per_share)
        
        # Apply position size limit
        max_position_value = account_equity * self.max_position_size
        max_shares_by_limit = int(max_position_value / entry_price)
        
        return min(shares, max_shares_by_limit)

class StrategyOrchestrator:
    """Orquestador de estrategias automatizadas"""
    
    def __init__(self, account_equity: float):
        self.account_equity = account_equity
        self.strategies: Dict[str, AutomatedStrategy] = {}
        self.active_signals: List[TradingSignal] = []
        self.signal_queue: asyncio.Queue = asyncio.Queue()
        
        # Portfolio limits
        self.max_total_exposure = 0.80  # 80% max exposure
        self.max_strategies_per_symbol = 2
        self.current_exposure = 0.0
        
        self.logger = logging.getLogger("StrategyOrchestrator")
    
    def add_strategy(self, strategy: AutomatedStrategy):
        """Agregar estrategia"""
        self.strategies[strategy.name] = strategy
        self.logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str):
        """Remover estrategia"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Removed strategy: {strategy_name}")
    
    async def process_market_data(self, data: MarketData):
        """Procesar datos de mercado con todas las estrategias"""
        
        all_signals = []
        
        # Run each strategy
        for strategy_name, strategy in self.strategies.items():
            if not strategy.is_active:
                continue
            
            try:
                signals = await strategy.analyze_market_data(data)
                
                for signal in signals:
                    # Calculate position size
                    signal.quantity = await strategy.calculate_position_size(
                        signal, self.account_equity
                    )
                    
                    if signal.quantity > 0:
                        all_signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy_name}: {e}")
        
        # Filter and prioritize signals
        filtered_signals = await self._filter_and_prioritize_signals(all_signals, data.symbol)
        
        # Add to signal queue
        for signal in filtered_signals:
            await self.signal_queue.put(signal)
        
        return filtered_signals
    
    async def _filter_and_prioritize_signals(self, signals: List[TradingSignal], 
                                           symbol: str) -> List[TradingSignal]:
        """Filtrar y priorizar señales"""
        
        if not signals:
            return []
        
        # Filter by exposure limits
        filtered_signals = []
        
        for signal in signals:
            # Check if adding this position would exceed exposure limit
            signal_exposure = (signal.quantity * signal.price) / self.account_equity
            
            if self.current_exposure + signal_exposure <= self.max_total_exposure:
                # Check max strategies per symbol
                existing_signals_for_symbol = len([s for s in self.active_signals 
                                                 if s.symbol == signal.symbol])
                
                if existing_signals_for_symbol < self.max_strategies_per_symbol:
                    filtered_signals.append(signal)
        
        # Sort by confidence * strength
        strength_values = {
            SignalStrength.WEAK: 1,
            SignalStrength.MODERATE: 2,
            SignalStrength.STRONG: 3,
            SignalStrength.VERY_STRONG: 4
        }
        
        filtered_signals.sort(
            key=lambda s: s.confidence * strength_values[s.strength],
            reverse=True
        )
        
        # Take top signals based on available capacity
        final_signals = []
        temp_exposure = self.current_exposure
        
        for signal in filtered_signals:
            signal_exposure = (signal.quantity * signal.price) / self.account_equity
            
            if temp_exposure + signal_exposure <= self.max_total_exposure:
                final_signals.append(signal)
                temp_exposure += signal_exposure
            else:
                break
        
        return final_signals
    
    async def get_next_signal(self) -> Optional[TradingSignal]:
        """Obtener próxima señal de la cola"""
        try:
            signal = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
            return signal
        except asyncio.TimeoutError:
            return None
    
    def update_position_opened(self, signal: TradingSignal):
        """Actualizar cuando se abre posición"""
        self.active_signals.append(signal)
        signal_exposure = (signal.quantity * signal.price) / self.account_equity
        self.current_exposure += signal_exposure
        
        self.logger.info(f"Position opened: {signal.symbol} exposure={signal_exposure:.2%} total={self.current_exposure:.2%}")
    
    def update_position_closed(self, signal: TradingSignal, pnl: float):
        """Actualizar cuando se cierra posición"""
        # Remove from active signals
        self.active_signals = [s for s in self.active_signals if s != signal]
        
        # Update exposure
        signal_exposure = (signal.quantity * signal.price) / self.account_equity
        self.current_exposure -= signal_exposure
        self.current_exposure = max(0, self.current_exposure)  # Ensure non-negative
        
        # Update strategy performance
        strategy = self.strategies.get(signal.strategy_name)
        if strategy:
            strategy.update_performance(signal, pnl)
        
        self.logger.info(f"Position closed: {signal.symbol} PnL={pnl:.2f} exposure={self.current_exposure:.2%}")
    
    def get_portfolio_summary(self) -> Dict:
        """Obtener resumen del portfolio"""
        
        strategy_summaries = {}
        for name, strategy in self.strategies.items():
            strategy_summaries[name] = strategy.get_performance_summary()
        
        return {
            'total_exposure_pct': self.current_exposure,
            'active_signals_count': len(self.active_signals),
            'active_signals': [
                {
                    'symbol': s.symbol,
                    'strategy': s.strategy_name,
                    'confidence': s.confidence,
                    'quantity': s.quantity
                }
                for s in self.active_signals
            ],
            'strategies': strategy_summaries
        }

# Demo del sistema de automatización
async def demo_automation_system():
    """Demo del sistema de automatización"""
    
    # Configurar estrategias
    gap_config = {
        'min_gap_percent': 0.05,
        'max_gap_percent': 0.15,
        'min_volume_ratio': 2.0,
        'max_position_size': 0.10,
        'min_confidence': 0.65
    }
    
    vwap_config = {
        'min_time_below_vwap_minutes': 30,
        'max_distance_from_vwap_pct': 0.02,
        'min_volume_confirmation': 1.5,
        'max_position_size': 0.08,
        'min_confidence': 0.60
    }
    
    # Crear estrategias
    gap_strategy = GapAndGoAutomated(gap_config)
    vwap_strategy = VWAPReclaimAutomated(vwap_config)
    
    # Crear orquestador
    orchestrator = StrategyOrchestrator(account_equity=100000)
    orchestrator.add_strategy(gap_strategy)
    orchestrator.add_strategy(vwap_strategy)
    
    # Simular datos de mercado
    market_data = MarketData(
        symbol="AAPL",
        timestamp=datetime.now().replace(hour=10, minute=0),
        open_price=150.0,
        high_price=155.0,
        low_price=149.0,
        close_price=154.0,
        volume=2000000,
        vwap=152.0,
        indicators={
            'prev_close': 145.0,  # 6% gap
            'avg_volume_20d': 800000,  # 2.5x volume
            'sma_20': 148.0
        }
    )
    
    # Procesar datos
    signals = await orchestrator.process_market_data(market_data)
    
    print(f"📊 Generated {len(signals)} signals:")
    for signal in signals:
        print(f"  {signal.strategy_name}: {signal.symbol} {signal.signal_type.value} "
              f"qty={signal.quantity} confidence={signal.confidence:.2f}")
    
    # Obtener resumen
    portfolio_summary = orchestrator.get_portfolio_summary()
    print(f"\n📈 Portfolio Summary:")
    print(f"Total exposure: {portfolio_summary['total_exposure_pct']:.2%}")
    print(f"Active signals: {portfolio_summary['active_signals_count']}")
    
    print(f"\n🎯 Strategy Performance:")
    for name, summary in portfolio_summary['strategies'].items():
        metrics = summary['metrics']
        print(f"  {name}: {metrics['total_signals']} signals, "
              f"{metrics['win_rate']:.1%} win rate, "
              f"${metrics['total_pnl']:.2f} P&L")

if __name__ == "__main__":
    asyncio.run(demo_automation_system())
```

## Gestión de Riesgo Automatizada

### Sistema de Risk Management Automatizado
```python
class AutomatedRiskManager:
    """Sistema de gestión de riesgo automatizado"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("AutomatedRiskManager")
        
        # Risk limits
        self.max_daily_loss_pct = config.get('max_daily_loss_pct', 0.05)
        self.max_portfolio_exposure = config.get('max_portfolio_exposure', 0.80)
        self.max_position_size_pct = config.get('max_position_size_pct', 0.20)
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.30)
        
        # Portfolio state
        self.start_of_day_equity = 0
        self.current_equity = 0
        self.daily_pnl = 0
        self.positions = {}
        self.sector_exposure = {}
        
        # Emergency protocols
        self.emergency_stop_triggered = False
        self.trading_halted = False
        
    async def evaluate_signal_risk(self, signal: TradingSignal, 
                                 account_equity: float) -> Tuple[bool, str]:
        """Evaluar riesgo de una señal"""
        
        self.current_equity = account_equity
        
        # Check emergency stop
        if self.emergency_stop_triggered:
            return False, "Emergency stop is active"
        
        # Check trading halt
        if self.trading_halted:
            return False, "Trading is halted"
        
        # Check daily loss limit
        daily_loss_pct = self.daily_pnl / self.start_of_day_equity if self.start_of_day_equity > 0 else 0
        if daily_loss_pct <= -self.max_daily_loss_pct:
            await self._trigger_daily_loss_protection()
            return False, f"Daily loss limit exceeded: {daily_loss_pct:.2%}"
        
        # Check position size
        position_value = signal.quantity * signal.price
        position_pct = position_value / account_equity
        
        if position_pct > self.max_position_size_pct:
            return False, f"Position size too large: {position_pct:.2%}"
        
        # Check portfolio exposure
        current_exposure = sum(pos['value'] for pos in self.positions.values()) / account_equity
        if current_exposure + position_pct > self.max_portfolio_exposure:
            return False, f"Portfolio exposure limit exceeded"
        
        # Check sector concentration
        symbol_sector = await self._get_symbol_sector(signal.symbol)
        sector_exposure = self.sector_exposure.get(symbol_sector, 0) / account_equity
        
        if sector_exposure + position_pct > self.max_correlation_exposure:
            return False, f"Sector concentration limit exceeded for {symbol_sector}"
        
        return True, "Risk checks passed"
    
    async def monitor_position_risk(self, symbol: str, current_price: float):
        """Monitorear riesgo de posición existente"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        
        # Calculate unrealized P&L
        if side == 'long':
            unrealized_pnl = (current_price - entry_price) * quantity
        else:
            unrealized_pnl = (entry_price - current_price) * quantity
        
        unrealized_pnl_pct = unrealized_pnl / (entry_price * quantity)
        
        # Check stop loss
        stop_loss = position.get('stop_loss')
        if stop_loss:
            if side == 'long' and current_price <= stop_loss:
                await self._trigger_stop_loss(symbol, "Stop loss hit")
            elif side == 'short' and current_price >= stop_loss:
                await self._trigger_stop_loss(symbol, "Stop loss hit")
        
        # Check maximum loss per position
        max_loss_pct = -0.15  # 15% max loss
        if unrealized_pnl_pct <= max_loss_pct:
            await self._trigger_emergency_exit(symbol, f"Maximum loss exceeded: {unrealized_pnl_pct:.2%}")
        
        # Check profit protection
        if unrealized_pnl_pct >= 0.20:  # 20% profit
            await self._consider_profit_protection(symbol, unrealized_pnl_pct)
    
    async def _trigger_daily_loss_protection(self):
        """Activar protección por pérdida diaria"""
        
        self.trading_halted = True
        self.logger.critical("Daily loss protection triggered - halting trading")
        
        # Close all positions
        for symbol in list(self.positions.keys()):
            await self._trigger_emergency_exit(symbol, "Daily loss protection")
        
        # Send alert
        await self._send_risk_alert("DAILY LOSS PROTECTION ACTIVATED", {
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': self.daily_pnl / self.start_of_day_equity,
            'action': 'All positions closed, trading halted'
        })
    
    async def _trigger_stop_loss(self, symbol: str, reason: str):
        """Activar stop loss"""
        
        self.logger.warning(f"Stop loss triggered for {symbol}: {reason}")
        
        # Add to emergency exit queue
        await self._trigger_emergency_exit(symbol, reason)
    
    async def _trigger_emergency_exit(self, symbol: str, reason: str):
        """Activar salida de emergencia"""
        
        if symbol in self.positions:
            position = self.positions[symbol]
            
            self.logger.critical(f"Emergency exit triggered for {symbol}: {reason}")
            
            # Create emergency exit order
            exit_order = {
                'symbol': symbol,
                'side': 'sell' if position['side'] == 'long' else 'buy',
                'quantity': position['quantity'],
                'order_type': 'market',
                'urgency': 'emergency',
                'reason': reason
            }
            
            # Send to execution engine (implementation depends on your architecture)
            await self._send_emergency_order(exit_order)
    
    async def _consider_profit_protection(self, symbol: str, profit_pct: float):
        """Considerar protección de ganancias"""
        
        position = self.positions[symbol]
        
        # Move stop loss to protect profits
        entry_price = position['entry_price']
        side = position['side']
        
        if side == 'long':
            # Move stop to 10% profit
            new_stop = entry_price * 1.10
        else:
            # Move stop to 10% profit (for short)
            new_stop = entry_price * 0.90
        
        # Update position
        position['stop_loss'] = new_stop
        
        self.logger.info(f"Profit protection activated for {symbol}: stop moved to ${new_stop:.2f}")
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Obtener sector del símbolo"""
        # This would typically query a database or API
        # For demo purposes, using a simple mapping
        sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'TSLA': 'Automotive',
            'JPM': 'Financial',
            'JNJ': 'Healthcare'
        }
        
        return sector_mapping.get(symbol, 'Unknown')
    
    async def _send_risk_alert(self, title: str, data: Dict):
        """Enviar alerta de riesgo"""
        
        alert = {
            'type': 'risk_alert',
            'title': title,
            'data': data,
            'timestamp': datetime.now(),
            'severity': 'critical'
        }
        
        # Send to alert system (implementation depends on your architecture)
        self.logger.critical(f"Risk Alert: {title} - {data}")
    
    async def _send_emergency_order(self, order: Dict):
        """Enviar orden de emergencia"""
        
        self.logger.critical(f"Emergency order: {order}")
        # Implementation would send to execution engine
    
    def update_position(self, symbol: str, side: str, quantity: int, 
                       entry_price: float, stop_loss: float = None):
        """Actualizar posición"""
        
        self.positions[symbol] = {
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'value': quantity * entry_price,
            'timestamp': datetime.now()
        }
        
        # Update sector exposure
        sector = asyncio.create_task(self._get_symbol_sector(symbol))
        # This is simplified - in real implementation, you'd await this properly
    
    def remove_position(self, symbol: str, exit_price: float, pnl: float):
        """Remover posición"""
        
        if symbol in self.positions:
            del self.positions[symbol]
            self.daily_pnl += pnl
    
    def set_start_of_day_equity(self, equity: float):
        """Establecer equity al inicio del día"""
        self.start_of_day_equity = equity
        self.daily_pnl = 0
        self.emergency_stop_triggered = False
        self.trading_halted = False
    
    def get_risk_metrics(self) -> Dict:
        """Obtener métricas de riesgo"""
        
        current_exposure = 0
        position_count = len(self.positions)
        largest_position_pct = 0
        
        if self.current_equity > 0:
            total_position_value = sum(pos['value'] for pos in self.positions.values())
            current_exposure = total_position_value / self.current_equity
            
            if self.positions:
                largest_position_value = max(pos['value'] for pos in self.positions.values())
                largest_position_pct = largest_position_value / self.current_equity
        
        daily_loss_pct = self.daily_pnl / self.start_of_day_equity if self.start_of_day_equity > 0 else 0
        
        return {
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': daily_loss_pct,
            'current_exposure_pct': current_exposure,
            'position_count': position_count,
            'largest_position_pct': largest_position_pct,
            'emergency_stop_active': self.emergency_stop_triggered,
            'trading_halted': self.trading_halted,
            'risk_limits': {
                'max_daily_loss_pct': self.max_daily_loss_pct,
                'max_portfolio_exposure': self.max_portfolio_exposure,
                'max_position_size_pct': self.max_position_size_pct
            }
        }

# Demo del risk manager
async def demo_risk_manager():
    """Demo del sistema de risk management"""
    
    # Configuración
    risk_config = {
        'max_daily_loss_pct': 0.05,
        'max_portfolio_exposure': 0.80,
        'max_position_size_pct': 0.15,
        'max_correlation_exposure': 0.30
    }
    
    # Crear risk manager
    risk_manager = AutomatedRiskManager(risk_config)
    risk_manager.set_start_of_day_equity(100000)
    
    # Simular señal
    signal = TradingSignal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        price=150.0,
        quantity=500,  # $75k position (75% of portfolio)
        timestamp=datetime.now(),
        strategy_name="GapAndGo",
        confidence=0.8
    )
    
    # Evaluar riesgo
    approved, reason = await risk_manager.evaluate_signal_risk(signal, 100000)
    
    print(f"📊 Risk Evaluation:")
    print(f"Signal approved: {approved}")
    print(f"Reason: {reason}")
    
    if approved:
        # Simular apertura de posición
        risk_manager.update_position(
            signal.symbol, 'long', signal.quantity, signal.price, 140.0
        )
        
        # Simular monitoreo
        await risk_manager.monitor_position_risk("AAPL", 145.0)  # Price moving down
        
        # Obtener métricas
        metrics = risk_manager.get_risk_metrics()
        print(f"\n📈 Risk Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'pct' in key:
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(demo_risk_manager())
```

Este sistema de automatización proporciona un framework robusto para ejecutar estrategias de trading de forma completamente automatizada, con gestión de riesgo integrada y monitoreo continuo.