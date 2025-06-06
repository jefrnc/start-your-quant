# Arquitectura del Sistema de Trading Automatizado

## Diseño de Arquitectura General

### Componentes del Sistema
```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import asyncio
import logging
from datetime import datetime, timedelta
import json
import threading
import queue

class SystemComponent(Enum):
    """Componentes del sistema"""
    DATA_MANAGER = "data_manager"
    STRATEGY_ENGINE = "strategy_engine"
    RISK_MANAGER = "risk_manager"
    EXECUTION_ENGINE = "execution_engine"
    PERFORMANCE_TRACKER = "performance_tracker"
    ALERT_SYSTEM = "alert_system"
    WEB_DASHBOARD = "web_dashboard"
    ORDER_MANAGER = "order_manager"

class SystemState(Enum):
    """Estados del sistema"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class SystemConfig:
    """Configuración del sistema"""
    # Trading parameters
    account_size: float = 100000
    max_positions: int = 5
    max_daily_loss_pct: float = 0.05
    max_position_size_pct: float = 0.20
    
    # System parameters
    update_frequency_ms: int = 1000
    data_retention_days: int = 30
    log_level: str = "INFO"
    
    # Strategies to run
    active_strategies: List[str] = field(default_factory=list)
    
    # Market hours
    market_open_time: str = "09:30"
    market_close_time: str = "16:00"
    premarket_start: str = "04:00"
    afterhours_end: str = "20:00"
    
    # Broker settings
    primary_broker: str = "ibkr"
    backup_broker: Optional[str] = "alpaca"
    
    # Data providers
    primary_data_provider: str = "polygon"
    backup_data_provider: str = "yahoo"

class TradingSystemCore:
    """Core del sistema de trading automatizado"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = SystemState.STOPPED
        self.components: Dict[SystemComponent, Any] = {}
        self.message_queues: Dict[str, queue.Queue] = {}
        self.running = False
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize message queues
        self._initialize_message_queues()
        
        # Initialize components
        self._initialize_components()
    
    def _setup_logging(self):
        """Configurar sistema de logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/trading_system_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('TradingSystem')
    
    def _initialize_message_queues(self):
        """Inicializar colas de mensajes"""
        queue_names = [
            'market_data',
            'trade_signals',
            'order_commands',
            'risk_alerts',
            'performance_updates',
            'system_events'
        ]
        
        for queue_name in queue_names:
            self.message_queues[queue_name] = queue.Queue(maxsize=1000)
    
    def _initialize_components(self):
        """Inicializar componentes del sistema"""
        try:
            # Data Manager
            self.components[SystemComponent.DATA_MANAGER] = DataManager(
                self.config, 
                self.message_queues['market_data']
            )
            
            # Strategy Engine
            self.components[SystemComponent.STRATEGY_ENGINE] = StrategyEngine(
                self.config,
                self.message_queues['market_data'],
                self.message_queues['trade_signals']
            )
            
            # Risk Manager
            self.components[SystemComponent.RISK_MANAGER] = RiskManager(
                self.config,
                self.message_queues['trade_signals'],
                self.message_queues['order_commands'],
                self.message_queues['risk_alerts']
            )
            
            # Execution Engine
            self.components[SystemComponent.EXECUTION_ENGINE] = ExecutionEngine(
                self.config,
                self.message_queues['order_commands']
            )
            
            # Performance Tracker
            self.components[SystemComponent.PERFORMANCE_TRACKER] = PerformanceTracker(
                self.config,
                self.message_queues['performance_updates']
            )
            
            # Alert System
            self.components[SystemComponent.ALERT_SYSTEM] = AlertSystem(
                self.config,
                self.message_queues['risk_alerts']
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.state = SystemState.ERROR
            raise
    
    async def start_system(self):
        """Iniciar sistema completo"""
        try:
            self.logger.info("Starting trading system...")
            self.state = SystemState.STARTING
            
            # Start all components
            for component_type, component in self.components.items():
                self.logger.info(f"Starting {component_type.value}...")
                await component.start()
            
            # Start main loop
            self.running = True
            self.state = SystemState.RUNNING
            
            # Start monitoring loop
            asyncio.create_task(self._main_system_loop())
            asyncio.create_task(self._health_monitor_loop())
            
            self.logger.info("Trading system started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            self.state = SystemState.ERROR
            raise
    
    async def stop_system(self):
        """Detener sistema"""
        try:
            self.logger.info("Stopping trading system...")
            self.state = SystemState.STOPPING
            self.running = False
            
            # Stop all components
            for component_type, component in self.components.items():
                self.logger.info(f"Stopping {component_type.value}...")
                await component.stop()
            
            self.state = SystemState.STOPPED
            self.logger.info("Trading system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
            self.state = SystemState.ERROR
    
    async def _main_system_loop(self):
        """Loop principal del sistema"""
        while self.running:
            try:
                # Check if market is open
                if self._is_market_open():
                    # Process any pending system events
                    await self._process_system_events()
                    
                    # Check component health
                    await self._check_component_health()
                
                # Sleep until next iteration
                await asyncio.sleep(self.config.update_frequency_ms / 1000)
                
            except Exception as e:
                self.logger.error(f"Error in main system loop: {e}")
                await asyncio.sleep(1)
    
    async def _health_monitor_loop(self):
        """Loop de monitoreo de salud del sistema"""
        while self.running:
            try:
                # Check system health
                health_status = await self._get_system_health()
                
                # Log health status
                if health_status['overall_status'] != 'healthy':
                    self.logger.warning(f"System health warning: {health_status}")
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(30)
    
    def _is_market_open(self) -> bool:
        """Verificar si el mercado está abierto"""
        now = datetime.now()
        market_open = now.replace(
            hour=int(self.config.market_open_time.split(':')[0]),
            minute=int(self.config.market_open_time.split(':')[1]),
            second=0, microsecond=0
        )
        market_close = now.replace(
            hour=int(self.config.market_close_time.split(':')[0]),
            minute=int(self.config.market_close_time.split(':')[1]),
            second=0, microsecond=0
        )
        
        # Check if weekday and within market hours
        return now.weekday() < 5 and market_open <= now <= market_close
    
    async def _process_system_events(self):
        """Procesar eventos del sistema"""
        try:
            while not self.message_queues['system_events'].empty():
                event = self.message_queues['system_events'].get_nowait()
                await self._handle_system_event(event)
        except queue.Empty:
            pass
    
    async def _handle_system_event(self, event: Dict):
        """Manejar evento del sistema"""
        event_type = event.get('type')
        
        if event_type == 'pause_trading':
            await self._pause_trading()
        elif event_type == 'resume_trading':
            await self._resume_trading()
        elif event_type == 'emergency_stop':
            await self._emergency_stop()
        elif event_type == 'rebalance_portfolio':
            await self._rebalance_portfolio()
    
    async def _check_component_health(self):
        """Verificar salud de componentes"""
        for component_type, component in self.components.items():
            try:
                health = await component.get_health()
                if not health['is_healthy']:
                    self.logger.warning(f"Component {component_type.value} is unhealthy: {health}")
            except Exception as e:
                self.logger.error(f"Error checking health of {component_type.value}: {e}")
    
    async def _get_system_health(self) -> Dict:
        """Obtener estado de salud del sistema"""
        health_status = {
            'timestamp': datetime.now(),
            'overall_status': 'healthy',
            'components': {},
            'issues': []
        }
        
        # Check each component
        for component_type, component in self.components.items():
            try:
                component_health = await component.get_health()
                health_status['components'][component_type.value] = component_health
                
                if not component_health['is_healthy']:
                    health_status['overall_status'] = 'warning'
                    health_status['issues'].append(f"{component_type.value}: {component_health}")
                    
            except Exception as e:
                health_status['overall_status'] = 'error'
                health_status['issues'].append(f"{component_type.value}: Error - {e}")
        
        return health_status

class BaseComponent(ABC):
    """Clase base para componentes del sistema"""
    
    def __init__(self, config: SystemConfig, name: str):
        self.config = config
        self.name = name
        self.logger = logging.getLogger(f'TradingSystem.{name}')
        self.running = False
        self.last_heartbeat = datetime.now()
    
    @abstractmethod
    async def start(self):
        """Iniciar componente"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Detener componente"""
        pass
    
    @abstractmethod
    async def process(self):
        """Procesar lógica principal del componente"""
        pass
    
    async def get_health(self) -> Dict:
        """Obtener estado de salud del componente"""
        time_since_heartbeat = datetime.now() - self.last_heartbeat
        
        return {
            'is_healthy': time_since_heartbeat < timedelta(minutes=5),
            'last_heartbeat': self.last_heartbeat,
            'running': self.running,
            'component_name': self.name
        }
    
    def update_heartbeat(self):
        """Actualizar heartbeat"""
        self.last_heartbeat = datetime.now()

class DataManager(BaseComponent):
    """Gestor de datos de mercado"""
    
    def __init__(self, config: SystemConfig, market_data_queue: queue.Queue):
        super().__init__(config, "DataManager")
        self.market_data_queue = market_data_queue
        self.data_providers = {}
        self.subscribed_symbols = set()
        
    async def start(self):
        """Iniciar data manager"""
        self.logger.info("Starting Data Manager...")
        
        # Initialize data providers
        await self._initialize_data_providers()
        
        # Start data processing loop
        self.running = True
        asyncio.create_task(self._data_processing_loop())
        
        self.logger.info("Data Manager started")
    
    async def stop(self):
        """Detener data manager"""
        self.logger.info("Stopping Data Manager...")
        self.running = False
        
        # Disconnect from data providers
        for provider in self.data_providers.values():
            await provider.disconnect()
        
        self.logger.info("Data Manager stopped")
    
    async def process(self):
        """Procesar datos de mercado"""
        # This is handled by the data processing loop
        pass
    
    async def _initialize_data_providers(self):
        """Inicializar proveedores de datos"""
        # Initialize primary data provider
        primary_provider = self._create_data_provider(self.config.primary_data_provider)
        await primary_provider.connect()
        self.data_providers['primary'] = primary_provider
        
        # Initialize backup data provider if configured
        if self.config.backup_data_provider:
            backup_provider = self._create_data_provider(self.config.backup_data_provider)
            await backup_provider.connect()
            self.data_providers['backup'] = backup_provider
    
    def _create_data_provider(self, provider_name: str):
        """Crear proveedor de datos"""
        # Factory method to create data providers
        if provider_name == "polygon":
            from src.data_acquisition.polygon_provider import PolygonDataProvider
            return PolygonDataProvider(self.config.polygon_api_key)
        elif provider_name == "yahoo":
            from src.data_acquisition.yahoo_provider import YahooDataProvider
            return YahooDataProvider()
        else:
            raise ValueError(f"Unknown data provider: {provider_name}")
    
    async def _data_processing_loop(self):
        """Loop de procesamiento de datos"""
        while self.running:
            try:
                # Get data from primary provider
                market_data = await self._get_market_data()
                
                if market_data:
                    # Put data in queue for strategy engine
                    try:
                        self.market_data_queue.put_nowait(market_data)
                    except queue.Full:
                        self.logger.warning("Market data queue is full")
                
                self.update_heartbeat()
                await asyncio.sleep(1)  # 1 second intervals
                
            except Exception as e:
                self.logger.error(f"Error in data processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _get_market_data(self) -> Optional[Dict]:
        """Obtener datos de mercado"""
        try:
            # Try primary provider first
            primary_provider = self.data_providers.get('primary')
            if primary_provider:
                data = await primary_provider.get_real_time_data(list(self.subscribed_symbols))
                if data:
                    return data
            
            # Fallback to backup provider
            backup_provider = self.data_providers.get('backup')
            if backup_provider:
                data = await backup_provider.get_real_time_data(list(self.subscribed_symbols))
                return data
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
        
        return None
    
    def subscribe_symbol(self, symbol: str):
        """Suscribirse a datos de un símbolo"""
        self.subscribed_symbols.add(symbol)
        self.logger.info(f"Subscribed to {symbol}")
    
    def unsubscribe_symbol(self, symbol: str):
        """Desuscribirse de datos de un símbolo"""
        self.subscribed_symbols.discard(symbol)
        self.logger.info(f"Unsubscribed from {symbol}")

class StrategyEngine(BaseComponent):
    """Motor de estrategias"""
    
    def __init__(self, config: SystemConfig, market_data_queue: queue.Queue, 
                 signals_queue: queue.Queue):
        super().__init__(config, "StrategyEngine")
        self.market_data_queue = market_data_queue
        self.signals_queue = signals_queue
        self.strategies = {}
        self.strategy_positions = {}
    
    async def start(self):
        """Iniciar strategy engine"""
        self.logger.info("Starting Strategy Engine...")
        
        # Load and initialize strategies
        await self._load_strategies()
        
        # Start strategy processing loop
        self.running = True
        asyncio.create_task(self._strategy_processing_loop())
        
        self.logger.info("Strategy Engine started")
    
    async def stop(self):
        """Detener strategy engine"""
        self.logger.info("Stopping Strategy Engine...")
        self.running = False
        self.logger.info("Strategy Engine stopped")
    
    async def process(self):
        """Procesar estrategias"""
        # This is handled by the strategy processing loop
        pass
    
    async def _load_strategies(self):
        """Cargar estrategias configuradas"""
        for strategy_name in self.config.active_strategies:
            try:
                strategy = self._create_strategy(strategy_name)
                self.strategies[strategy_name] = strategy
                self.strategy_positions[strategy_name] = {}
                self.logger.info(f"Loaded strategy: {strategy_name}")
            except Exception as e:
                self.logger.error(f"Error loading strategy {strategy_name}: {e}")
    
    def _create_strategy(self, strategy_name: str):
        """Crear instancia de estrategia"""
        # Factory method to create strategies
        if strategy_name == "gap_and_go":
            from src.strategies.gap_and_go import GapAndGoStrategy
            return GapAndGoStrategy()
        elif strategy_name == "vwap_reclaim":
            from src.strategies.vwap_reclaim import VWAPReclaimStrategy
            return VWAPReclaimStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    async def _strategy_processing_loop(self):
        """Loop de procesamiento de estrategias"""
        while self.running:
            try:
                # Get market data
                try:
                    market_data = self.market_data_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process data with each strategy
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signals = await strategy.process_market_data(market_data)
                        
                        # Put signals in queue for risk manager
                        for signal in signals:
                            signal['strategy'] = strategy_name
                            try:
                                self.signals_queue.put_nowait(signal)
                            except queue.Full:
                                self.logger.warning("Signals queue is full")
                    
                    except Exception as e:
                        self.logger.error(f"Error processing strategy {strategy_name}: {e}")
                
                self.update_heartbeat()
                
            except Exception as e:
                self.logger.error(f"Error in strategy processing loop: {e}")
                await asyncio.sleep(1)

class RiskManager(BaseComponent):
    """Gestor de riesgo"""
    
    def __init__(self, config: SystemConfig, signals_queue: queue.Queue,
                 orders_queue: queue.Queue, alerts_queue: queue.Queue):
        super().__init__(config, "RiskManager")
        self.signals_queue = signals_queue
        self.orders_queue = orders_queue
        self.alerts_queue = alerts_queue
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.account_equity = config.account_size
    
    async def start(self):
        """Iniciar risk manager"""
        self.logger.info("Starting Risk Manager...")
        
        # Start risk processing loop
        self.running = True
        asyncio.create_task(self._risk_processing_loop())
        
        self.logger.info("Risk Manager started")
    
    async def stop(self):
        """Detener risk manager"""
        self.logger.info("Stopping Risk Manager...")
        self.running = False
        self.logger.info("Risk Manager stopped")
    
    async def process(self):
        """Procesar gestión de riesgo"""
        # This is handled by the risk processing loop
        pass
    
    async def _risk_processing_loop(self):
        """Loop de procesamiento de riesgo"""
        while self.running:
            try:
                # Get trade signals
                try:
                    signal = self.signals_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                # Validate signal against risk rules
                if await self._validate_signal(signal):
                    # Convert signal to order
                    order = await self._create_order_from_signal(signal)
                    
                    if order:
                        try:
                            self.orders_queue.put_nowait(order)
                            self.logger.info(f"Order created: {order}")
                        except queue.Full:
                            self.logger.warning("Orders queue is full")
                else:
                    self.logger.info(f"Signal rejected by risk management: {signal}")
                
                self.update_heartbeat()
                
            except Exception as e:
                self.logger.error(f"Error in risk processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _validate_signal(self, signal: Dict) -> bool:
        """Validar señal contra reglas de riesgo"""
        
        # Check daily loss limit
        daily_loss_pct = self.daily_pnl / self.account_equity
        if daily_loss_pct <= -self.config.max_daily_loss_pct:
            await self._send_risk_alert("Daily loss limit exceeded", signal)
            return False
        
        # Check maximum positions
        if len(self.current_positions) >= self.config.max_positions:
            return False
        
        # Check position size
        position_value = signal.get('quantity', 0) * signal.get('price', 0)
        position_pct = position_value / self.account_equity
        
        if position_pct > self.config.max_position_size_pct:
            return False
        
        return True
    
    async def _create_order_from_signal(self, signal: Dict) -> Optional[Dict]:
        """Crear orden desde señal"""
        
        return {
            'symbol': signal['symbol'],
            'side': signal['side'],
            'quantity': signal['quantity'],
            'order_type': signal.get('order_type', 'market'),
            'price': signal.get('price'),
            'stop_price': signal.get('stop_price'),
            'strategy': signal['strategy'],
            'timestamp': datetime.now()
        }
    
    async def _send_risk_alert(self, message: str, data: Dict):
        """Enviar alerta de riesgo"""
        alert = {
            'type': 'risk_alert',
            'message': message,
            'data': data,
            'timestamp': datetime.now()
        }
        
        try:
            self.alerts_queue.put_nowait(alert)
        except queue.Full:
            self.logger.error("Alerts queue is full")

class ExecutionEngine(BaseComponent):
    """Motor de ejecución de órdenes"""
    
    def __init__(self, config: SystemConfig, orders_queue: queue.Queue):
        super().__init__(config, "ExecutionEngine")
        self.orders_queue = orders_queue
        self.brokers = {}
        self.order_history = []
    
    async def start(self):
        """Iniciar execution engine"""
        self.logger.info("Starting Execution Engine...")
        
        # Initialize brokers
        await self._initialize_brokers()
        
        # Start order processing loop
        self.running = True
        asyncio.create_task(self._order_processing_loop())
        
        self.logger.info("Execution Engine started")
    
    async def stop(self):
        """Detener execution engine"""
        self.logger.info("Stopping Execution Engine...")
        self.running = False
        
        # Disconnect from brokers
        for broker in self.brokers.values():
            await broker.disconnect()
        
        self.logger.info("Execution Engine stopped")
    
    async def process(self):
        """Procesar ejecución de órdenes"""
        # This is handled by the order processing loop
        pass
    
    async def _initialize_brokers(self):
        """Inicializar conexiones a brokers"""
        # Initialize primary broker
        primary_broker = self._create_broker(self.config.primary_broker)
        await primary_broker.connect()
        self.brokers['primary'] = primary_broker
        
        # Initialize backup broker if configured
        if self.config.backup_broker:
            backup_broker = self._create_broker(self.config.backup_broker)
            await backup_broker.connect()
            self.brokers['backup'] = backup_broker
    
    def _create_broker(self, broker_name: str):
        """Crear conexión a broker"""
        if broker_name == "ibkr":
            from src.execution.ibkr_broker import IBKRBroker
            return IBKRBroker()
        elif broker_name == "alpaca":
            from src.execution.alpaca_broker import AlpacaBroker
            return AlpacaBroker()
        else:
            raise ValueError(f"Unknown broker: {broker_name}")
    
    async def _order_processing_loop(self):
        """Loop de procesamiento de órdenes"""
        while self.running:
            try:
                # Get orders
                try:
                    order = self.orders_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute order
                execution_result = await self._execute_order(order)
                
                # Log execution result
                self.order_history.append({
                    'order': order,
                    'result': execution_result,
                    'timestamp': datetime.now()
                })
                
                self.logger.info(f"Order execution result: {execution_result}")
                self.update_heartbeat()
                
            except Exception as e:
                self.logger.error(f"Error in order processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _execute_order(self, order: Dict) -> Dict:
        """Ejecutar orden"""
        try:
            # Try primary broker first
            primary_broker = self.brokers.get('primary')
            if primary_broker:
                result = await primary_broker.place_order(order)
                if result.get('success'):
                    return result
            
            # Fallback to backup broker
            backup_broker = self.brokers.get('backup')
            if backup_broker:
                result = await backup_broker.place_order(order)
                return result
            
            return {'success': False, 'error': 'No available brokers'}
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return {'success': False, 'error': str(e)}

# Demo de la arquitectura
async def demo_trading_system():
    """Demo del sistema de trading"""
    
    # Configuración
    config = SystemConfig(
        account_size=100000,
        max_positions=3,
        max_daily_loss_pct=0.03,
        active_strategies=['gap_and_go', 'vwap_reclaim'],
        primary_broker='ibkr',
        primary_data_provider='polygon'
    )
    
    # Crear sistema
    trading_system = TradingSystemCore(config)
    
    try:
        # Iniciar sistema
        await trading_system.start_system()
        
        # Ejecutar por 30 segundos para demo
        await asyncio.sleep(30)
        
        # Obtener estado de salud
        health = await trading_system._get_system_health()
        print(f"System health: {health}")
        
    finally:
        # Detener sistema
        await trading_system.stop_system()

if __name__ == "__main__":
    asyncio.run(demo_trading_system())
```

## Deployment y Infraestructura

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose ports
EXPOSE 8080 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Start command
CMD ["python", "main.py"]
```

### Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  trading-system:
    build: .
    container_name: trading_system
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./config:/app/config
    ports:
      - "8080:8080"
      - "8501:8501"
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "scripts/health_check.py"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: trading_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:14-alpine
    container_name: trading_postgres
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: trading_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  streamlit:
    build: .
    container_name: trading_dashboard
    command: streamlit run dashboard/main.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      - trading-system
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: trading_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: trading_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: trading_network
```

### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-system
  labels:
    app: trading-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trading-system
  template:
    metadata:
      labels:
        app: trading-system
    spec:
      containers:
      - name: trading-system
        image: trading-system:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: database-url
        volumeMounts:
        - name: logs-volume
          mountPath: /app/logs
        - name: data-volume
          mountPath: /app/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: logs-volume
        persistentVolumeClaim:
          claimName: logs-pvc
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: trading-system-service
spec:
  selector:
    app: trading-system
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: logs-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
```

### Monitoring y Observabilidad
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Métricas de Prometheus
TRADES_TOTAL = Counter('trading_trades_total', 'Total number of trades', ['strategy', 'side', 'status'])
TRADE_DURATION = Histogram('trading_trade_duration_seconds', 'Trade duration in seconds')
ACCOUNT_EQUITY = Gauge('trading_account_equity_dollars', 'Current account equity in dollars')
POSITIONS_COUNT = Gauge('trading_positions_count', 'Number of open positions')
DAILY_PNL = Gauge('trading_daily_pnl_dollars', 'Daily P&L in dollars')
SYSTEM_ERRORS = Counter('trading_system_errors_total', 'Total system errors', ['component', 'error_type'])

class TradingMetrics:
    """Sistema de métricas para trading"""
    
    def __init__(self, port=8000):
        self.port = port
        
    def start_metrics_server(self):
        """Iniciar servidor de métricas"""
        start_http_server(self.port)
        print(f"Metrics server started on port {self.port}")
    
    def record_trade(self, strategy: str, side: str, status: str):
        """Registrar trade"""
        TRADES_TOTAL.labels(strategy=strategy, side=side, status=status).inc()
    
    def record_trade_duration(self, duration: float):
        """Registrar duración de trade"""
        TRADE_DURATION.observe(duration)
    
    def update_account_equity(self, equity: float):
        """Actualizar equity de cuenta"""
        ACCOUNT_EQUITY.set(equity)
    
    def update_positions_count(self, count: int):
        """Actualizar conteo de posiciones"""
        POSITIONS_COUNT.set(count)
    
    def update_daily_pnl(self, pnl: float):
        """Actualizar P&L diario"""
        DAILY_PNL.set(pnl)
    
    def record_system_error(self, component: str, error_type: str):
        """Registrar error de sistema"""
        SYSTEM_ERRORS.labels(component=component, error_type=error_type).inc()

def measure_time(metric_name):
    """Decorator para medir tiempo de ejecución"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                TRADE_DURATION.observe(duration)
        return wrapper
    return decorator

# Configuración de Grafana Dashboard
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "Trading System Dashboard",
        "panels": [
            {
                "title": "Account Equity",
                "type": "stat",
                "targets": [
                    {
                        "expr": "trading_account_equity_dollars",
                        "legendFormat": "Equity"
                    }
                ]
            },
            {
                "title": "Daily P&L",
                "type": "stat",
                "targets": [
                    {
                        "expr": "trading_daily_pnl_dollars",
                        "legendFormat": "Daily P&L"
                    }
                ]
            },
            {
                "title": "Trade Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(trading_trades_total[5m])",
                        "legendFormat": "Trades/sec"
                    }
                ]
            },
            {
                "title": "Open Positions",
                "type": "stat",
                "targets": [
                    {
                        "expr": "trading_positions_count",
                        "legendFormat": "Positions"
                    }
                ]
            },
            {
                "title": "System Errors",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(trading_system_errors_total[5m])",
                        "legendFormat": "{{component}} - {{error_type}}"
                    }
                ]
            }
        ]
    }
}
```

Esta arquitectura proporciona un sistema de trading robusto, escalable y observable, con capacidades de deployment tanto en contenedores como en Kubernetes, junto con monitoreo completo de métricas y salud del sistema.