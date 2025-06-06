# Integración con DAS Trader Pro

## Introducción

DAS Trader Pro es una **plataforma de trading profesional** (no un broker) muy popular entre day traders, especialmente para small caps. Se conecta a diversos brokers como Charles Schwab, Zimtra, Lightspeed, y otros.

Para integrar DAS con nuestras estrategias cuantitativas, utilizamos el **das-bridge** desarrollado específicamente para este propósito.

## DAS Trader Pro: Plataforma vs Broker

### ¿Qué es DAS Trader Pro?
- **Plataforma de software** para trading profesional
- Se **conecta a brokers** que soporten su protocolo
- Provee **interfaz avanzada** con herramientas profesionales
- **No maneja dinero** directamente (eso lo hace el broker)

### Brokers Compatibles Populares
- **Charles Schwab** (muy popular con DAS)
- **Zimtra Trading** (especialista en small caps)
- **Lightspeed Trading** (low latency)
- **Cobra Trading** (day trading enfocado)
- **TradeZero** (sin PDT rule)
- **Vision Financial Markets**

### Workflow Típico
```
Tu Estrategia Python → das-bridge → DAS Trader Pro → Broker (ej: Schwab) → Mercado
```

1. **Tu código Python** genera señales de trading
2. **das-bridge** traduce las órdenes al protocolo DAS
3. **DAS Trader Pro** procesa la orden y la optimiza (routing, etc.)
4. **El broker** (ej: Charles Schwab) ejecuta la orden en el mercado
5. **Las confirmaciones** regresan por el mismo camino

> **Nota:** El das-bridge está en desarrollo activo en: https://github.com/jefrnc/das-bridge

## ¿Por Qué Usar DAS Trader Pro?

### Ventajas para Small Caps
- **Borrows amplios** para short selling
- **Routing avanzado** para mejor ejecución
- **Level 2 premium** con data de alta calidad
- **Hotkeys configurables** para ejecución rápida
- **Comisiones competitivas** para high-frequency trading

### Casos de Uso Ideales
- Day trading en small caps volátiles
- Short selling estrategias
- Scalping con volumen alto
- Trading con margin elevado

## Instalación del DAS Bridge

### Requisitos Previos
```bash
# DAS Trader Pro debe estar instalado y configurado
# Cuenta activa con DAS
# Python 3.8+
```

### Instalación
```bash
# Clonar el bridge
git clone https://github.com/jefrnc/das-bridge.git
cd das-bridge

# Instalar dependencias
pip install -r requirements.txt

# Instalar el package
pip install -e .
```

### Configuración DAS
```ini
# En DAS Trader Pro -> Setup -> API Setup
[API Settings]
Enable CMD API: True
Port: 9910
Allow Localhost: True
Authentication: Required
```

## Configuración Básica

### Variables de Entorno
```bash
# .env
DAS_HOST=localhost
DAS_PORT=9910
DAS_USERNAME=tu_usuario
DAS_PASSWORD=tu_password
DAS_ACCOUNT=tu_cuenta
DAS_PAPER_TRADING=True  # Para testing
```

### Configuración por Broker

#### Charles Schwab + DAS
```bash
# .env para Schwab
DAS_USERNAME=tu_usuario_schwab
DAS_PASSWORD=tu_password_schwab
DAS_ACCOUNT=12345678  # Tu cuenta Schwab
DAS_BROKER=schwab
DAS_PAPER_TRADING=True
```

#### Zimtra + DAS
```bash
# .env para Zimtra
DAS_USERNAME=tu_usuario_zimtra
DAS_PASSWORD=tu_password_zimtra
DAS_ACCOUNT=ZIM12345  # Tu cuenta Zimtra
DAS_BROKER=zimtra
DAS_PAPER_TRADING=True
```

### Configuración Python
```python
# config/das_config.py
import os
from dataclasses import dataclass

@dataclass
class DASConfig:
    host: str = os.getenv('DAS_HOST', 'localhost')
    port: int = int(os.getenv('DAS_PORT', '9910'))
    username: str = os.getenv('DAS_USERNAME')
    password: str = os.getenv('DAS_PASSWORD')
    account: str = os.getenv('DAS_ACCOUNT')
    broker: str = os.getenv('DAS_BROKER', 'unknown')  # schwab, zimtra, etc.
    paper_trading: bool = os.getenv('DAS_PAPER_TRADING', 'True').lower() == 'true'
    
    # Configuraciones de trading
    max_position_size: float = 10000.0  # $10k máximo por posición
    max_daily_loss: float = 1000.0      # $1k pérdida máxima diaria
    risk_per_trade: float = 0.02        # 2% riesgo por trade
    
    # Timeouts y reconexión
    connection_timeout: int = 30
    order_timeout: int = 10
    max_reconnect_attempts: int = 5
```

## Implementación del Cliente DAS

### Cliente Unificado
```python
# src/brokers/das_client.py
import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from das_trader import DASTraderClient, OrderSide, OrderType
from dataclasses import dataclass

@dataclass
class DASPosition:
    symbol: str
    quantity: int
    avg_price: float
    market_value: float
    unrealized_pnl: float
    side: str

@dataclass
class DASOrder:
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: Optional[float]
    status: str
    filled_qty: int
    timestamp: datetime

class DASBrokerIntegration:
    """Integración completa con DAS Trader"""
    
    def __init__(self, config: DASConfig):
        self.config = config
        self.client = None
        self.connected = False
        self.positions: Dict[str, DASPosition] = {}
        self.orders: Dict[str, DASOrder] = {}
        
        # Callbacks
        self.quote_callbacks: List[Callable] = []
        self.order_callbacks: List[Callable] = []
        self.position_callbacks: List[Callable] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """Conectar a DAS Trader"""
        try:
            self.client = DASTraderClient(
                host=self.config.host,
                port=self.config.port
            )
            
            success = await self.client.connect(
                username=self.config.username,
                password=self.config.password,
                account=self.config.account
            )
            
            if success:
                self.connected = True
                self.logger.info("Conectado a DAS Trader exitosamente")
                
                # Configurar callbacks
                self.client.on_quote = self._handle_quote_update
                self.client.on_order_update = self._handle_order_update
                self.client.on_position_update = self._handle_position_update
                
                # Cargar estado inicial
                await self._load_initial_state()
                return True
            else:
                self.logger.error("Falló conexión a DAS Trader")
                return False
                
        except Exception as e:
            self.logger.error(f"Error conectando a DAS: {e}")
            return False
    
    async def disconnect(self):
        """Desconectar de DAS Trader"""
        if self.client and self.connected:
            await self.client.disconnect()
            self.connected = False
            self.logger.info("Desconectado de DAS Trader")
    
    async def get_buying_power(self) -> float:
        """Obtener poder de compra disponible"""
        if not self.connected:
            return 0.0
        
        try:
            bp_info = await self.client.get_buying_power()
            return bp_info.get('available', 0.0)
        except Exception as e:
            self.logger.error(f"Error obteniendo buying power: {e}")
            return 0.0
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Obtener cotización de un símbolo"""
        if not self.connected:
            return None
        
        try:
            quote = await self.client.get_quote(symbol)
            return {
                'symbol': symbol,
                'bid': quote.bid,
                'ask': quote.ask,
                'last': quote.last,
                'volume': quote.volume,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error obteniendo quote para {symbol}: {e}")
            return None
    
    async def subscribe_quotes(self, symbols: List[str]):
        """Suscribirse a quotes en tiempo real"""
        if not self.connected:
            return False
        
        try:
            for symbol in symbols:
                await self.client.subscribe_quote(symbol)
            return True
        except Exception as e:
            self.logger.error(f"Error suscribiendo quotes: {e}")
            return False
    
    async def send_order(self, symbol: str, side: str, quantity: int,
                        order_type: str = "MARKET", price: Optional[float] = None,
                        stop_price: Optional[float] = None) -> Optional[str]:
        """Enviar orden a DAS"""
        
        if not self.connected:
            self.logger.error("No conectado a DAS")
            return None
        
        # Validaciones previas
        if not await self._validate_order(symbol, side, quantity, price):
            return None
        
        try:
            # Convertir tipos
            das_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            das_type = getattr(OrderType, order_type.upper())
            
            # Enviar orden
            order_result = await self.client.send_order(
                symbol=symbol,
                side=das_side,
                quantity=quantity,
                order_type=das_type,
                price=price,
                stop_price=stop_price
            )
            
            if order_result.success:
                order_id = order_result.order_id
                
                # Registrar orden
                self.orders[order_id] = DASOrder(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    price=price,
                    status="PENDING",
                    filled_qty=0,
                    timestamp=datetime.now()
                )
                
                self.logger.info(f"Orden enviada: {order_id} - {side} {quantity} {symbol}")
                return order_id
            else:
                self.logger.error(f"Error enviando orden: {order_result.error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error enviando orden: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancelar orden"""
        if not self.connected:
            return False
        
        try:
            result = await self.client.cancel_order(order_id)
            if result.success:
                if order_id in self.orders:
                    self.orders[order_id].status = "CANCELLED"
                self.logger.info(f"Orden cancelada: {order_id}")
                return True
            else:
                self.logger.error(f"Error cancelando orden: {result.error}")
                return False
        except Exception as e:
            self.logger.error(f"Error cancelando orden {order_id}: {e}")
            return False
    
    async def _validate_order(self, symbol: str, side: str, quantity: int, 
                            price: Optional[float]) -> bool:
        """Validar orden antes de enviar"""
        
        # Validar buying power
        buying_power = await self.get_buying_power()
        estimated_cost = quantity * (price or 0)
        
        if side.upper() == "BUY" and estimated_cost > buying_power:
            self.logger.error(f"Insufficient buying power: ${buying_power:.2f} vs ${estimated_cost:.2f}")
            return False
        
        # Validar tamaño máximo de posición
        if estimated_cost > self.config.max_position_size:
            self.logger.error(f"Position size too large: ${estimated_cost:.2f}")
            return False
        
        # Validar si ya tenemos posición
        if symbol in self.positions:
            current_pos = self.positions[symbol]
            if (side.upper() == "BUY" and current_pos.side == "LONG") or \
               (side.upper() == "SELL" and current_pos.side == "SHORT"):
                self.logger.warning(f"Adding to existing {current_pos.side} position in {symbol}")
        
        return True
    
    async def _load_initial_state(self):
        """Cargar estado inicial de posiciones y órdenes"""
        try:
            # Cargar posiciones
            positions = await self.client.get_positions()
            for pos_data in positions:
                position = DASPosition(
                    symbol=pos_data.symbol,
                    quantity=pos_data.quantity,
                    avg_price=pos_data.avg_price,
                    market_value=pos_data.market_value,
                    unrealized_pnl=pos_data.unrealized_pnl,
                    side="LONG" if pos_data.quantity > 0 else "SHORT"
                )
                self.positions[pos_data.symbol] = position
            
            # Cargar órdenes abiertas
            open_orders = await self.client.get_open_orders()
            for order_data in open_orders:
                order = DASOrder(
                    order_id=order_data.order_id,
                    symbol=order_data.symbol,
                    side=order_data.side,
                    quantity=order_data.quantity,
                    order_type=order_data.order_type,
                    price=order_data.price,
                    status=order_data.status,
                    filled_qty=order_data.filled_qty,
                    timestamp=order_data.timestamp
                )
                self.orders[order_data.order_id] = order
                
        except Exception as e:
            self.logger.error(f"Error cargando estado inicial: {e}")
    
    def _handle_quote_update(self, quote_data):
        """Manejar actualización de quote"""
        for callback in self.quote_callbacks:
            try:
                callback(quote_data)
            except Exception as e:
                self.logger.error(f"Error en quote callback: {e}")
    
    def _handle_order_update(self, order_data):
        """Manejar actualización de orden"""
        order_id = order_data.order_id
        
        if order_id in self.orders:
            self.orders[order_id].status = order_data.status
            self.orders[order_id].filled_qty = order_data.filled_qty
        
        for callback in self.order_callbacks:
            try:
                callback(order_data)
            except Exception as e:
                self.logger.error(f"Error en order callback: {e}")
    
    def _handle_position_update(self, position_data):
        """Manejar actualización de posición"""
        symbol = position_data.symbol
        
        if position_data.quantity == 0:
            # Posición cerrada
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            # Actualizar posición
            position = DASPosition(
                symbol=symbol,
                quantity=position_data.quantity,
                avg_price=position_data.avg_price,
                market_value=position_data.market_value,
                unrealized_pnl=position_data.unrealized_pnl,
                side="LONG" if position_data.quantity > 0 else "SHORT"
            )
            self.positions[symbol] = position
        
        for callback in self.position_callbacks:
            try:
                callback(position_data)
            except Exception as e:
                self.logger.error(f"Error en position callback: {e}")
    
    def add_quote_callback(self, callback: Callable):
        """Agregar callback para quotes"""
        self.quote_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable):
        """Agregar callback para órdenes"""
        self.order_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable):
        """Agregar callback para posiciones"""
        self.position_callbacks.append(callback)
    
    def get_positions_summary(self) -> Dict:
        """Obtener resumen de posiciones"""
        total_value = sum(pos.market_value for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'positions_count': len(self.positions),
            'total_market_value': total_value,
            'total_unrealized_pnl': total_pnl,
            'positions': list(self.positions.values())
        }
```

## Integración con Estrategias

### Adaptador para el Framework Principal
```python
# src/brokers/das_adapter.py
from typing import Dict, Optional
from .das_client import DASBrokerIntegration, DASConfig
from ..core.broker_interface import BrokerInterface

class DASBrokerAdapter(BrokerInterface):
    """Adaptador para integrar DAS con el framework principal"""
    
    def __init__(self, config: DASConfig):
        self.das_client = DASBrokerIntegration(config)
        self.name = "DAS Trader"
    
    async def connect(self) -> bool:
        """Conectar al broker"""
        return await self.das_client.connect()
    
    async def disconnect(self):
        """Desconectar del broker"""
        await self.das_client.disconnect()
    
    async def get_account_info(self) -> Dict:
        """Obtener información de cuenta"""
        buying_power = await self.das_client.get_buying_power()
        positions_summary = self.das_client.get_positions_summary()
        
        return {
            'broker': self.name,
            'buying_power': buying_power,
            'positions_count': positions_summary['positions_count'],
            'total_market_value': positions_summary['total_market_value'],
            'total_unrealized_pnl': positions_summary['total_unrealized_pnl'],
            'positions': positions_summary['positions']
        }
    
    async def place_order(self, symbol: str, side: str, quantity: int,
                         order_type: str = "MARKET", price: float = None) -> Optional[str]:
        """Colocar orden"""
        return await self.das_client.send_order(symbol, side, quantity, order_type, price)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancelar orden"""
        return await self.das_client.cancel_order(order_id)
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Obtener cotización"""
        return await self.das_client.get_quote(symbol)
    
    def is_connected(self) -> bool:
        """Verificar si está conectado"""
        return self.das_client.connected
```

## Ejemplo de Uso Completo

### Script de Ejemplo
```python
# examples/das_trading_example.py
import asyncio
import logging
from datetime import datetime
from config.das_config import DASConfig
from src.brokers.das_client import DASBrokerIntegration

async def main():
    """Ejemplo completo de trading con DAS"""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuración
    config = DASConfig()
    das_client = DASBrokerIntegration(config)
    
    # Callbacks para eventos
    def on_quote_update(quote_data):
        print(f"Quote update: {quote_data.symbol} - ${quote_data.last:.2f}")
    
    def on_order_update(order_data):
        print(f"Order update: {order_data.order_id} - {order_data.status}")
    
    def on_position_update(position_data):
        print(f"Position update: {position_data.symbol} - {position_data.quantity} shares")
    
    # Agregar callbacks
    das_client.add_quote_callback(on_quote_update)
    das_client.add_order_callback(on_order_update)
    das_client.add_position_callback(on_position_update)
    
    try:
        # Conectar
        print("Conectando a DAS Trader...")
        connected = await das_client.connect()
        
        if not connected:
            print("❌ No se pudo conectar a DAS")
            return
        
        print("✅ Conectado a DAS Trader")
        
        # Obtener información de cuenta
        buying_power = await das_client.get_buying_power()
        print(f"💰 Buying Power: ${buying_power:,.2f}")
        
        # Suscribirse a quotes
        symbols = ["AAPL", "TSLA", "NVDA"]
        await das_client.subscribe_quotes(symbols)
        print(f"📊 Suscrito a quotes: {symbols}")
        
        # Obtener quotes individuales
        for symbol in symbols:
            quote = await das_client.get_quote(symbol)
            if quote:
                print(f"  {symbol}: ${quote['last']:.2f} (bid: ${quote['bid']:.2f}, ask: ${quote['ask']:.2f})")
        
        # Ejemplo de orden (comentado para seguridad)
        # print("\n📝 Enviando orden de ejemplo...")
        # order_id = await das_client.send_order("AAPL", "BUY", 10, "LIMIT", price=150.00)
        # if order_id:
        #     print(f"✅ Orden enviada: {order_id}")
        #     
        #     # Esperar un poco y cancelar
        #     await asyncio.sleep(5)
        #     cancelled = await das_client.cancel_order(order_id)
        #     if cancelled:
        #         print(f"🚫 Orden cancelada: {order_id}")
        
        # Mostrar posiciones actuales
        positions_summary = das_client.get_positions_summary()
        print(f"\n📋 Posiciones actuales: {positions_summary['positions_count']}")
        for position in positions_summary['positions']:
            print(f"  {position.symbol}: {position.quantity} @ ${position.avg_price:.2f} "
                  f"(P&L: ${position.unrealized_pnl:+.2f})")
        
        # Mantener conexión para recibir updates
        print("\n🔄 Manteniendo conexión para updates... (Ctrl+C para salir)")
        await asyncio.sleep(60)  # Esperar 1 minuto
        
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Desconectar
        await das_client.disconnect()
        print("📴 Desconectado de DAS Trader")

if __name__ == "__main__":
    asyncio.run(main())
```

## Integración con Estrategias Cuantitativas

### Ejemplo con Gap & Go Strategy
```python
# examples/das_gap_and_go.py
import asyncio
from src.strategies.gap_and_go import GapAndGoStrategy
from src.brokers.das_adapter import DASBrokerAdapter
from config.das_config import DASConfig

async def run_gap_and_go_with_das():
    """Ejecutar estrategia Gap & Go con DAS"""
    
    # Configurar DAS
    das_config = DASConfig()
    das_config.paper_trading = True  # Usar paper trading para testing
    broker = DASBrokerAdapter(das_config)
    
    # Configurar estrategia
    strategy = GapAndGoStrategy(broker)
    
    try:
        # Conectar broker
        if await broker.connect():
            print("✅ Conectado a DAS Trader")
            
            # Ejecutar estrategia
            await strategy.run()
            
        else:
            print("❌ No se pudo conectar a DAS")
    
    finally:
        await broker.disconnect()

if __name__ == "__main__":
    asyncio.run(run_gap_and_go_with_das())
```

## Consideraciones Importantes

### Ventajas del Stack DAS + Broker
- **Routing avanzado** que DAS optimiza automáticamente
- **Borrows amplios** (depende del broker subyacente)
- **Latencia baja** para estrategias rápidas
- **Data de calidad** Level 2 premium
- **Herramientas profesionales** (hotkeys, scripts, etc.)
- **Flexibilidad** para cambiar de broker manteniendo la plataforma

### Limitaciones y Costos
- **Licencia DAS** (~$150-300/mes dependiendo del plan)
- **Cuenta de broker** (Schwab, Zimtra, etc.) con sus propios requisitos
- **Windows principalmente** (DAS Pro requiere Windows)
- **Bridge en desarrollo** (puede tener bugs)
- **Curva de aprendizaje** mayor vs APIs directas

### Estructura de Costos Típica
```
Broker (ej: Schwab):        $0-25/mes + comisiones
DAS Trader Pro:             $150-300/mes
Market Data:                $50-100/mes (Level 2)
das-bridge:                 Gratis (open source)
Total mensual:              ~$200-425/mes
```

### Mejores Prácticas
1. **Siempre usar paper trading** para testing inicial
2. **Validar órdenes** antes de enviar
3. **Manejar reconexiones** automáticamente
4. **Logear todo** para debugging
5. **Implementar circuit breakers** para emergencias

### Troubleshooting Común
```python
# Problemas comunes y soluciones
COMMON_ISSUES = {
    "Connection refused": "Verificar que DAS Pro esté ejecutándose y API habilitada",
    "Authentication failed": "Revisar credenciales en .env",
    "Order rejected": "Verificar buying power y límites de cuenta",
    "Symbol not found": "Verificar que el símbolo esté disponible en DAS",
    "Position mismatch": "Sincronizar estado con get_positions()"
}
```

Este bridge para DAS Trader completa el ecosistema de trading, proporcionando acceso a una plataforma profesional con capacidades avanzadas de short selling y routing, especialmente valiosa para estrategias de small caps.