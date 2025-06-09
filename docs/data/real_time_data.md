# Real-Time Data Management para Trading Algorítmico

## Introducción: La Velocidad Como Ventaja Competitiva

En trading algorítmico, especialmente en small caps y estrategias intraday, la latencia de datos puede ser la diferencia entre profit y loss. Este documento cubre la arquitectura completa para manejo de datos en tiempo real, desde la ingesta hasta el procesamiento y distribución.

## Arquitectura de Real-Time Data

### Core Components

```python
import asyncio
import websockets
import threading
import queue
import time
from typing import Dict, Any, Callable, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

@dataclass
class MarketDataTick:
    """Estructura estándar para datos de mercado"""
    symbol: str
    timestamp: float
    bid: float
    ask: float
    last: float
    volume: int
    bid_size: int
    ask_size: int
    exchange: str
    data_type: str = "quote"

class DataFeedInterface(ABC):
    """Interface estándar para feeds de datos"""
    
    @abstractmethod
    async def connect(self) -> bool:
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> bool:
        pass
    
    @abstractmethod
    async def start_stream(self, callback: Callable) -> None:
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        pass

class RealTimeDataManager:
    """Manager central para datos en tiempo real"""
    
    def __init__(self):
        self.feeds = {}
        self.subscribers = {}
        self.data_buffer = queue.Queue(maxsize=10000)
        self.latency_monitor = LatencyMonitor()
        self.data_validator = DataValidator()
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'messages_dropped': 0,
            'avg_latency_ms': 0,
            'error_count': 0
        }
    
    def register_feed(self, name: str, feed: DataFeedInterface):
        """Registra un nuevo feed de datos"""
        self.feeds[name] = feed
    
    def subscribe_to_symbols(self, symbols: List[str], callback: Callable):
        """Suscribe a símbolos específicos"""
        for symbol in symbols:
            if symbol not in self.subscribers:
                self.subscribers[symbol] = []
            self.subscribers[symbol].append(callback)
    
    async def start_all_feeds(self):
        """Inicia todos los feeds registrados"""
        tasks = []
        for name, feed in self.feeds.items():
            task = asyncio.create_task(self._manage_feed(name, feed))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _manage_feed(self, name: str, feed: DataFeedInterface):
        """Maneja un feed individual con reconexión automática"""
        while True:
            try:
                await feed.connect()
                print(f"✅ Connected to {name}")
                
                # Suscribir a todos los símbolos
                symbols = list(self.subscribers.keys())
                if symbols:
                    await feed.subscribe(symbols)
                
                # Iniciar stream
                await feed.start_stream(self._process_data_tick)
                
            except Exception as e:
                print(f"❌ Error with {name}: {e}")
                await asyncio.sleep(5)  # Retry after 5 seconds
                continue
    
    def _process_data_tick(self, tick: MarketDataTick):
        """Procesa cada tick de datos"""
        try:
            # 1. Validation
            if not self.data_validator.validate_tick(tick):
                self.metrics['messages_dropped'] += 1
                return
            
            # 2. Latency monitoring
            latency = self.latency_monitor.calculate_latency(tick)
            self._update_latency_metrics(latency)
            
            # 3. Distribute to subscribers
            if tick.symbol in self.subscribers:
                for callback in self.subscribers[tick.symbol]:
                    try:
                        callback(tick)
                    except Exception as e:
                        print(f"Error in callback: {e}")
            
            self.metrics['messages_processed'] += 1
            
        except Exception as e:
            self.metrics['error_count'] += 1
            print(f"Error processing tick: {e}")
```

### High-Performance Data Feeds

```python
class AlpacaRealTimeFeed(DataFeedInterface):
    """Feed de datos real-time para Alpaca"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.websocket = None
        
    async def connect(self) -> bool:
        """Conecta al websocket de Alpaca"""
        try:
            self.websocket = await websockets.connect(
                "wss://stream.data.alpaca.markets/v2/iex",
                extra_headers={
                    "Authorization": f"Basic {self._get_auth_token()}"
                }
            )
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            await self.websocket.send(json.dumps(auth_message))
            
            response = await self.websocket.recv()
            auth_response = json.loads(response)
            
            return auth_response.get("T") == "success"
            
        except Exception as e:
            print(f"Alpaca connection error: {e}")
            return False
    
    async def subscribe(self, symbols: List[str]) -> bool:
        """Suscribe a símbolos específicos"""
        try:
            subscribe_message = {
                "action": "subscribe",
                "quotes": symbols,
                "trades": symbols
            }
            await self.websocket.send(json.dumps(subscribe_message))
            return True
        except Exception as e:
            print(f"Subscription error: {e}")
            return False
    
    async def start_stream(self, callback: Callable) -> None:
        """Inicia el stream de datos"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                # Parse different message types
                for item in data:
                    if item.get("T") == "q":  # Quote
                        tick = self._parse_quote(item)
                        callback(tick)
                    elif item.get("T") == "t":  # Trade
                        tick = self._parse_trade(item)
                        callback(tick)
                        
        except websockets.exceptions.ConnectionClosed:
            print("Alpaca connection closed")
        except Exception as e:
            print(f"Stream error: {e}")

class PolygonRealTimeFeed(DataFeedInterface):
    """Feed de datos real-time para Polygon.io"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.websocket = None
    
    async def connect(self) -> bool:
        try:
            self.websocket = await websockets.connect(
                f"wss://socket.polygon.io/stocks"
            )
            
            # Authenticate
            auth_message = {"action": "auth", "params": self.api_key}
            await self.websocket.send(json.dumps(auth_message))
            
            response = await self.websocket.recv()
            auth_data = json.loads(response)
            
            return auth_data[0].get("status") == "auth_success"
            
        except Exception as e:
            print(f"Polygon connection error: {e}")
            return False
    
    async def subscribe(self, symbols: List[str]) -> bool:
        try:
            # Subscribe to quotes and trades
            subscribe_quotes = {
                "action": "subscribe", 
                "params": f"Q.{',Q.'.join(symbols)}"
            }
            subscribe_trades = {
                "action": "subscribe", 
                "params": f"T.{',T.'.join(symbols)}"
            }
            
            await self.websocket.send(json.dumps(subscribe_quotes))
            await self.websocket.send(json.dumps(subscribe_trades))
            
            return True
        except Exception as e:
            print(f"Polygon subscription error: {e}")
            return False

class IEXRealTimeFeed(DataFeedInterface):
    """Feed de datos real-time para IEX Cloud"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.websocket = None
    
    async def connect(self) -> bool:
        try:
            self.websocket = await websockets.connect(
                f"wss://ws-api.iextrading.com/1.0/deep?token={self.api_token}"
            )
            return True
        except Exception as e:
            print(f"IEX connection error: {e}")
            return False
```

### Advanced Data Processing

```python
class MarketDataProcessor:
    """Procesador avanzado de datos de mercado"""
    
    def __init__(self):
        self.tick_storage = {}
        self.bar_builders = {}
        self.indicators = {}
        self.anomaly_detector = AnomalyDetector()
        
    def process_tick(self, tick: MarketDataTick):
        """Procesa tick individual y actualiza estructuras"""
        symbol = tick.symbol
        
        # 1. Store raw tick
        self._store_tick(tick)
        
        # 2. Update bar builders
        self._update_bars(tick)
        
        # 3. Update indicators
        self._update_indicators(tick)
        
        # 4. Detect anomalies
        anomaly = self.anomaly_detector.detect(tick)
        if anomaly:
            self._handle_anomaly(tick, anomaly)
        
        # 5. Trigger strategy callbacks
        self._trigger_strategy_updates(tick)
    
    def _update_bars(self, tick: MarketDataTick):
        """Actualiza constructores de barras"""
        symbol = tick.symbol
        
        if symbol not in self.bar_builders:
            self.bar_builders[symbol] = {
                '1min': BarBuilder(timeframe='1min'),
                '5min': BarBuilder(timeframe='5min'),
                '15min': BarBuilder(timeframe='15min')
            }
        
        for timeframe, builder in self.bar_builders[symbol].items():
            new_bar = builder.update(tick)
            if new_bar:
                self._publish_new_bar(symbol, timeframe, new_bar)
    
    def _update_indicators(self, tick: MarketDataTick):
        """Actualiza indicadores en tiempo real"""
        symbol = tick.symbol
        
        if symbol not in self.indicators:
            self.indicators[symbol] = {
                'vwap': RealTimeVWAP(),
                'ema_20': RealTimeEMA(period=20),
                'rsi': RealTimeRSI(period=14),
                'bollinger': RealTimeBollingerBands()
            }
        
        for name, indicator in self.indicators[symbol].items():
            indicator.update(tick)

class RealTimeVWAP:
    """VWAP en tiempo real"""
    
    def __init__(self):
        self.cum_volume = 0
        self.cum_pv = 0
        self.vwap = 0
        self.session_start = None
    
    def update(self, tick: MarketDataTick):
        """Actualiza VWAP con nuevo tick"""
        # Reset at market open
        current_time = pd.Timestamp.fromtimestamp(tick.timestamp)
        if self._is_new_session(current_time):
            self._reset_session()
        
        # Update VWAP
        price = tick.last
        volume = tick.volume
        
        self.cum_volume += volume
        self.cum_pv += price * volume
        
        if self.cum_volume > 0:
            self.vwap = self.cum_pv / self.cum_volume
        
        return self.vwap
    
    def _is_new_session(self, timestamp: pd.Timestamp) -> bool:
        """Detecta nueva sesión de trading"""
        if self.session_start is None:
            self.session_start = timestamp.normalize() + pd.Timedelta(hours=9, minutes=30)
            return True
        
        market_open = timestamp.normalize() + pd.Timedelta(hours=9, minutes=30)
        if timestamp >= market_open and self.session_start < market_open:
            self.session_start = market_open
            return True
        
        return False

class RealTimeEMA:
    """EMA en tiempo real"""
    
    def __init__(self, period: int):
        self.period = period
        self.alpha = 2 / (period + 1)
        self.ema = None
        self.initialized = False
    
    def update(self, tick: MarketDataTick):
        """Actualiza EMA con nuevo tick"""
        price = tick.last
        
        if not self.initialized:
            self.ema = price
            self.initialized = True
        else:
            self.ema = self.alpha * price + (1 - self.alpha) * self.ema
        
        return self.ema

class AnomalyDetector:
    """Detector de anomalías en datos de mercado"""
    
    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        self.spread_history = {}
        
    def detect(self, tick: MarketDataTick) -> Dict[str, Any]:
        """Detecta anomalías en el tick"""
        symbol = tick.symbol
        anomalies = {}
        
        # 1. Price anomaly
        price_anomaly = self._detect_price_anomaly(tick)
        if price_anomaly:
            anomalies['price'] = price_anomaly
        
        # 2. Volume anomaly
        volume_anomaly = self._detect_volume_anomaly(tick)
        if volume_anomaly:
            anomalies['volume'] = volume_anomaly
        
        # 3. Spread anomaly
        spread_anomaly = self._detect_spread_anomaly(tick)
        if spread_anomaly:
            anomalies['spread'] = spread_anomaly
        
        # 4. Stale data detection
        stale_data = self._detect_stale_data(tick)
        if stale_data:
            anomalies['stale_data'] = stale_data
        
        return anomalies if anomalies else None
    
    def _detect_price_anomaly(self, tick: MarketDataTick) -> Dict[str, Any]:
        """Detecta anomalías de precio"""
        symbol = tick.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(tick.last)
        
        # Keep only last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        if len(self.price_history[symbol]) < 10:
            return None
        
        prices = np.array(self.price_history[symbol])
        
        # Z-score based anomaly detection
        z_score = abs((tick.last - np.mean(prices)) / np.std(prices))
        
        if z_score > 3:  # 3 standard deviations
            return {
                'type': 'price_spike',
                'z_score': float(z_score),
                'current_price': tick.last,
                'mean_price': float(np.mean(prices)),
                'severity': 'high' if z_score > 5 else 'medium'
            }
        
        return None
```

### Production Deployment

```python
class ProductionDataInfrastructure:
    """Infraestructura de datos para producción"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_manager = RealTimeDataManager()
        self.backup_feeds = []
        self.circuit_breaker = CircuitBreaker()
        self.metrics_collector = MetricsCollector()
        
    async def deploy(self):
        """Deploya infraestructura completa"""
        # 1. Setup primary and backup feeds
        await self._setup_feeds()
        
        # 2. Start monitoring
        await self._start_monitoring()
        
        # 3. Start data processing
        await self._start_processing()
        
        # 4. Setup alerting
        await self._setup_alerting()
    
    async def _setup_feeds(self):
        """Configura feeds primarios y de backup"""
        # Primary feed
        primary_feed = AlpacaRealTimeFeed(
            self.config['alpaca']['api_key'],
            self.config['alpaca']['api_secret']
        )
        self.data_manager.register_feed('primary', primary_feed)
        
        # Backup feeds
        if 'polygon' in self.config:
            backup_feed1 = PolygonRealTimeFeed(self.config['polygon']['api_key'])
            self.data_manager.register_feed('backup_polygon', backup_feed1)
        
        if 'iex' in self.config:
            backup_feed2 = IEXRealTimeFeed(self.config['iex']['api_token'])
            self.data_manager.register_feed('backup_iex', backup_feed2)
    
    async def _start_monitoring(self):
        """Inicia monitoreo de infraestructura"""
        monitor = InfrastructureMonitor(self.data_manager)
        await monitor.start()
    
    def _setup_alerting(self):
        """Configura sistema de alertas"""
        alerting = AlertingSystem()
        
        # Alertas críticas
        alerting.add_alert(
            condition=lambda: self.data_manager.metrics['error_count'] > 100,
            action=self._handle_critical_error,
            severity='critical'
        )
        
        # Alertas de latencia
        alerting.add_alert(
            condition=lambda: self.data_manager.metrics['avg_latency_ms'] > 100,
            action=self._handle_latency_issue,
            severity='warning'
        )

class CircuitBreaker:
    """Circuit breaker para feeds de datos"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """Llama función con circuit breaker"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
            
            raise e

# Usage Example
async def main():
    """Ejemplo de uso completo"""
    
    # Configuration
    config = {
        'alpaca': {
            'api_key': 'your_alpaca_key',
            'api_secret': 'your_alpaca_secret'
        },
        'polygon': {
            'api_key': 'your_polygon_key'
        },
        'symbols': ['AAPL', 'TSLA', 'SPY', 'QQQ']
    }
    
    # Initialize infrastructure
    infrastructure = ProductionDataInfrastructure(config)
    
    # Deploy
    await infrastructure.deploy()
    
    # Subscribe to symbols
    symbols = config['symbols']
    
    def handle_tick(tick: MarketDataTick):
        print(f"{tick.symbol}: {tick.last} @ {tick.timestamp}")
        # Your trading strategy logic here
    
    infrastructure.data_manager.subscribe_to_symbols(symbols, handle_tick)
    
    # Start data feeds
    await infrastructure.data_manager.start_all_feeds()

if __name__ == "__main__":
    asyncio.run(main())
```

### Performance Optimization

```python
class DataPerformanceOptimizer:
    """Optimizador de performance para datos real-time"""
    
    def __init__(self):
        self.compression_enabled = True
        self.batching_enabled = True
        self.caching_enabled = True
        
    def optimize_tick_processing(self, processor_func):
        """Optimiza procesamiento de ticks"""
        
        @functools.wraps(processor_func)
        def optimized_processor(tick: MarketDataTick):
            # 1. Batch processing
            if self.batching_enabled:
                return self._batch_process(tick, processor_func)
            
            # 2. Direct processing with caching
            if self.caching_enabled:
                cache_key = f"{tick.symbol}_{tick.timestamp}"
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    return cached_result
            
            result = processor_func(tick)
            
            if self.caching_enabled:
                self._cache_result(cache_key, result)
            
            return result
        
        return optimized_processor
    
    def _batch_process(self, tick: MarketDataTick, processor_func):
        """Procesa ticks en batches para eficiencia"""
        # Implementation would accumulate ticks and process in batches
        pass
```

## Best Practices

### 1. Latency Optimization

```python
def minimize_latency():
    """Best practices para minimizar latencia"""
    return {
        'network': [
            'Use dedicated network connections',
            'Co-locate servers near exchanges',
            'Optimize TCP buffer sizes',
            'Use UDP for non-critical data'
        ],
        'processing': [
            'Pre-allocate memory',
            'Use lock-free data structures',
            'Minimize garbage collection',
            'Process on dedicated threads'
        ],
        'architecture': [
            'Separate market data from strategy logic',
            'Use message queues efficiently',
            'Implement circuit breakers',
            'Monitor and alert on latency spikes'
        ]
    }
```

### 2. Data Quality Assurance

```python
class DataQualityMonitor:
    """Monitor de calidad de datos"""
    
    def __init__(self):
        self.quality_checks = [
            self._check_price_continuity,
            self._check_volume_reasonableness,
            self._check_timestamp_consistency,
            self._check_spread_reasonableness
        ]
    
    def validate_tick(self, tick: MarketDataTick) -> bool:
        """Valida calidad del tick"""
        for check in self.quality_checks:
            if not check(tick):
                return False
        return True
    
    def _check_price_continuity(self, tick: MarketDataTick) -> bool:
        """Verifica continuidad de precios"""
        # Implementation for price continuity check
        return True
```

### 3. Disaster Recovery

```python
class DisasterRecoveryManager:
    """Manager de recuperación ante desastres"""
    
    def __init__(self):
        self.backup_systems = []
        self.recovery_procedures = {}
        
    def add_recovery_procedure(self, failure_type: str, procedure: Callable):
        """Agrega procedimiento de recuperación"""
        self.recovery_procedures[failure_type] = procedure
    
    async def handle_failure(self, failure_type: str, context: Dict):
        """Maneja falla del sistema"""
        if failure_type in self.recovery_procedures:
            await self.recovery_procedures[failure_type](context)
        else:
            await self._default_recovery_procedure(context)
```

---

*El manejo eficiente de datos en tiempo real es fundamental para el éxito en trading algorítmico. Una infraestructura robusta que combine baja latencia, alta disponibilidad y calidad de datos es esencial para strategies competitivas.*