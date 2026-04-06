> 🇪🇸 [Leer en Español](das_trader_integration.es.md) | 🇺🇸 **English**

# DAS Trader Pro Integration

## Introduction

DAS Trader Pro is a **professional trading platform** (not a broker) very popular among day traders, especially for small caps. It connects to various brokers such as Charles Schwab, Zimtra, Lightspeed, and others.

To integrate DAS with our quantitative strategies, we use the **das-bridge** developed specifically for this purpose.

## DAS Trader Pro: Platform vs Broker

### What is DAS Trader Pro?
- **Software platform** for professional trading
- **Connects to brokers** that support its protocol
- Provides an **advanced interface** with professional tools
- **Does not handle money** directly (that's the broker's job)

### Popular Compatible Brokers
- **Charles Schwab** (very popular with DAS)
- **Zimtra Trading** (small caps specialist)
- **Lightspeed Trading** (low latency)
- **Cobra Trading** (day trading focused)
- **TradeZero** (no PDT rule)
- **Vision Financial Markets**

### Typical Workflow
```
Your Python Strategy -> das-bridge -> DAS Trader Pro -> Broker (e.g., Schwab) -> Market
```

1. **Your Python code** generates trading signals
2. **das-bridge** translates orders to the DAS protocol
3. **DAS Trader Pro** processes the order and optimizes it (routing, etc.)
4. **The broker** (e.g., Charles Schwab) executes the order on the market
5. **Confirmations** return through the same path

> **Note:** The das-bridge is under active development at: https://github.com/jefrnc/das-bridge

## Why Use DAS Trader Pro?

### Advantages for Small Caps
- **Extensive borrows** for short selling
- **Advanced routing** for better execution
- **Premium Level 2** with high-quality data
- **Configurable hotkeys** for fast execution
- **Competitive commissions** for high-frequency trading

### Ideal Use Cases
- Day trading volatile small caps
- Short selling strategies
- High-volume scalping
- Trading with high margin

## DAS Bridge Installation

### Prerequisites
```bash
# DAS Trader Pro must be installed and configured
# Active DAS account
# Python 3.8+
```

### Installation
```bash
# Clone the bridge
git clone https://github.com/jefrnc/das-bridge.git
cd das-bridge

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### DAS Configuration
```ini
# In DAS Trader Pro -> Setup -> API Setup
[API Settings]
Enable CMD API: True
Port: 9910
Allow Localhost: True
Authentication: Required
```

## Basic Configuration

### Environment Variables
```bash
# .env
DAS_HOST=localhost
DAS_PORT=9910
DAS_USERNAME=your_username
DAS_PASSWORD=your_password
DAS_ACCOUNT=your_account
DAS_PAPER_TRADING=True  # For testing
```

### Broker-Specific Configuration

#### Charles Schwab + DAS
```bash
# .env for Schwab
DAS_USERNAME=your_schwab_username
DAS_PASSWORD=your_schwab_password
DAS_ACCOUNT=12345678  # Your Schwab account
DAS_BROKER=schwab
DAS_PAPER_TRADING=True
```

#### Zimtra + DAS
```bash
# .env for Zimtra
DAS_USERNAME=your_zimtra_username
DAS_PASSWORD=your_zimtra_password
DAS_ACCOUNT=ZIM12345  # Your Zimtra account
DAS_BROKER=zimtra
DAS_PAPER_TRADING=True
```

### Python Configuration
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
    
    # Trading settings
    max_position_size: float = 10000.0  # $10k max per position
    max_daily_loss: float = 1000.0      # $1k max daily loss
    risk_per_trade: float = 0.02        # 2% risk per trade
    
    # Timeouts and reconnection
    connection_timeout: int = 30
    order_timeout: int = 10
    max_reconnect_attempts: int = 5
```

## DAS Client Implementation

### Unified Client
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
    """Full integration with DAS Trader"""
    
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
        """Connect to DAS Trader"""
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
                self.logger.info("Successfully connected to DAS Trader")
                
                # Set up callbacks
                self.client.on_quote = self._handle_quote_update
                self.client.on_order_update = self._handle_order_update
                self.client.on_position_update = self._handle_position_update
                
                # Load initial state
                await self._load_initial_state()
                return True
            else:
                self.logger.error("Failed to connect to DAS Trader")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to DAS: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from DAS Trader"""
        if self.client and self.connected:
            await self.client.disconnect()
            self.connected = False
            self.logger.info("Disconnected from DAS Trader")
    
    async def get_buying_power(self) -> float:
        """Get available buying power"""
        if not self.connected:
            return 0.0
        
        try:
            bp_info = await self.client.get_buying_power()
            return bp_info.get('available', 0.0)
        except Exception as e:
            self.logger.error(f"Error getting buying power: {e}")
            return 0.0
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote for a symbol"""
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
            self.logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    async def subscribe_quotes(self, symbols: List[str]):
        """Subscribe to real-time quotes"""
        if not self.connected:
            return False
        
        try:
            for symbol in symbols:
                await self.client.subscribe_quote(symbol)
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to quotes: {e}")
            return False
    
    async def send_order(self, symbol: str, side: str, quantity: int,
                        order_type: str = "MARKET", price: Optional[float] = None,
                        stop_price: Optional[float] = None) -> Optional[str]:
        """Send order to DAS"""
        
        if not self.connected:
            self.logger.error("Not connected to DAS")
            return None
        
        # Pre-validation
        if not await self._validate_order(symbol, side, quantity, price):
            return None
        
        try:
            # Convert types
            das_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            das_type = getattr(OrderType, order_type.upper())
            
            # Send order
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
                
                # Record order
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
                
                self.logger.info(f"Order sent: {order_id} - {side} {quantity} {symbol}")
                return order_id
            else:
                self.logger.error(f"Error sending order: {order_result.error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error sending order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.connected:
            return False
        
        try:
            result = await self.client.cancel_order(order_id)
            if result.success:
                if order_id in self.orders:
                    self.orders[order_id].status = "CANCELLED"
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Error cancelling order: {result.error}")
                return False
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def _validate_order(self, symbol: str, side: str, quantity: int, 
                            price: Optional[float]) -> bool:
        """Validate order before sending"""
        
        # Validate buying power
        buying_power = await self.get_buying_power()
        estimated_cost = quantity * (price or 0)
        
        if side.upper() == "BUY" and estimated_cost > buying_power:
            self.logger.error(f"Insufficient buying power: ${buying_power:.2f} vs ${estimated_cost:.2f}")
            return False
        
        # Validate max position size
        if estimated_cost > self.config.max_position_size:
            self.logger.error(f"Position size too large: ${estimated_cost:.2f}")
            return False
        
        # Check if we already have a position
        if symbol in self.positions:
            current_pos = self.positions[symbol]
            if (side.upper() == "BUY" and current_pos.side == "LONG") or \
               (side.upper() == "SELL" and current_pos.side == "SHORT"):
                self.logger.warning(f"Adding to existing {current_pos.side} position in {symbol}")
        
        return True
    
    async def _load_initial_state(self):
        """Load initial state of positions and orders"""
        try:
            # Load positions
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
            
            # Load open orders
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
            self.logger.error(f"Error loading initial state: {e}")
    
    def _handle_quote_update(self, quote_data):
        """Handle quote update"""
        for callback in self.quote_callbacks:
            try:
                callback(quote_data)
            except Exception as e:
                self.logger.error(f"Error in quote callback: {e}")
    
    def _handle_order_update(self, order_data):
        """Handle order update"""
        order_id = order_data.order_id
        
        if order_id in self.orders:
            self.orders[order_id].status = order_data.status
            self.orders[order_id].filled_qty = order_data.filled_qty
        
        for callback in self.order_callbacks:
            try:
                callback(order_data)
            except Exception as e:
                self.logger.error(f"Error in order callback: {e}")
    
    def _handle_position_update(self, position_data):
        """Handle position update"""
        symbol = position_data.symbol
        
        if position_data.quantity == 0:
            # Position closed
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            # Update position
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
                self.logger.error(f"Error in position callback: {e}")
    
    def add_quote_callback(self, callback: Callable):
        """Add callback for quotes"""
        self.quote_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable):
        """Add callback for orders"""
        self.order_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable):
        """Add callback for positions"""
        self.position_callbacks.append(callback)
    
    def get_positions_summary(self) -> Dict:
        """Get positions summary"""
        total_value = sum(pos.market_value for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'positions_count': len(self.positions),
            'total_market_value': total_value,
            'total_unrealized_pnl': total_pnl,
            'positions': list(self.positions.values())
        }
```

## Strategy Integration

### Adapter for the Main Framework
```python
# src/brokers/das_adapter.py
from typing import Dict, Optional
from .das_client import DASBrokerIntegration, DASConfig
from ..core.broker_interface import BrokerInterface

class DASBrokerAdapter(BrokerInterface):
    """Adapter to integrate DAS with the main framework"""
    
    def __init__(self, config: DASConfig):
        self.das_client = DASBrokerIntegration(config)
        self.name = "DAS Trader"
    
    async def connect(self) -> bool:
        """Connect to the broker"""
        return await self.das_client.connect()
    
    async def disconnect(self):
        """Disconnect from the broker"""
        await self.das_client.disconnect()
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
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
        """Place an order"""
        return await self.das_client.send_order(symbol, side, quantity, order_type, price)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        return await self.das_client.cancel_order(order_id)
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote"""
        return await self.das_client.get_quote(symbol)
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.das_client.connected
```

## Complete Usage Example

### Example Script
```python
# examples/das_trading_example.py
import asyncio
import logging
from datetime import datetime
from config.das_config import DASConfig
from src.brokers.das_client import DASBrokerIntegration

async def main():
    """Complete DAS trading example"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = DASConfig()
    das_client = DASBrokerIntegration(config)
    
    # Event callbacks
    def on_quote_update(quote_data):
        print(f"Quote update: {quote_data.symbol} - ${quote_data.last:.2f}")
    
    def on_order_update(order_data):
        print(f"Order update: {order_data.order_id} - {order_data.status}")
    
    def on_position_update(position_data):
        print(f"Position update: {position_data.symbol} - {position_data.quantity} shares")
    
    # Add callbacks
    das_client.add_quote_callback(on_quote_update)
    das_client.add_order_callback(on_order_update)
    das_client.add_position_callback(on_position_update)
    
    try:
        # Connect
        print("Connecting to DAS Trader...")
        connected = await das_client.connect()
        
        if not connected:
            print("Could not connect to DAS")
            return
        
        print("Connected to DAS Trader")
        
        # Get account information
        buying_power = await das_client.get_buying_power()
        print(f"Buying Power: ${buying_power:,.2f}")
        
        # Subscribe to quotes
        symbols = ["AAPL", "TSLA", "NVDA"]
        await das_client.subscribe_quotes(symbols)
        print(f"Subscribed to quotes: {symbols}")
        
        # Get individual quotes
        for symbol in symbols:
            quote = await das_client.get_quote(symbol)
            if quote:
                print(f"  {symbol}: ${quote['last']:.2f} (bid: ${quote['bid']:.2f}, ask: ${quote['ask']:.2f})")
        
        # Order example (commented out for safety)
        # print("\nSending example order...")
        # order_id = await das_client.send_order("AAPL", "BUY", 10, "LIMIT", price=150.00)
        # if order_id:
        #     print(f"Order sent: {order_id}")
        #     
        #     # Wait a bit and cancel
        #     await asyncio.sleep(5)
        #     cancelled = await das_client.cancel_order(order_id)
        #     if cancelled:
        #         print(f"Order cancelled: {order_id}")
        
        # Show current positions
        positions_summary = das_client.get_positions_summary()
        print(f"\nCurrent positions: {positions_summary['positions_count']}")
        for position in positions_summary['positions']:
            print(f"  {position.symbol}: {position.quantity} @ ${position.avg_price:.2f} "
                  f"(P&L: ${position.unrealized_pnl:+.2f})")
        
        # Keep connection alive for updates
        print("\nKeeping connection alive for updates... (Ctrl+C to exit)")
        await asyncio.sleep(60)  # Wait 1 minute
        
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Disconnect
        await das_client.disconnect()
        print("Disconnected from DAS Trader")

if __name__ == "__main__":
    asyncio.run(main())
```

## Quantitative Strategy Integration

### Example with Gap & Go Strategy
```python
# examples/das_gap_and_go.py
import asyncio
from src.strategies.gap_and_go import GapAndGoStrategy
from src.brokers.das_adapter import DASBrokerAdapter
from config.das_config import DASConfig

async def run_gap_and_go_with_das():
    """Run Gap & Go strategy with DAS"""
    
    # Configure DAS
    das_config = DASConfig()
    das_config.paper_trading = True  # Use paper trading for testing
    broker = DASBrokerAdapter(das_config)
    
    # Configure strategy
    strategy = GapAndGoStrategy(broker)
    
    try:
        # Connect broker
        if await broker.connect():
            print("Connected to DAS Trader")
            
            # Run strategy
            await strategy.run()
            
        else:
            print("Could not connect to DAS")
    
    finally:
        await broker.disconnect()

if __name__ == "__main__":
    asyncio.run(run_gap_and_go_with_das())
```

## Important Considerations

### Advantages of the DAS + Broker Stack
- **Advanced routing** that DAS optimizes automatically
- **Extensive borrows** (depends on the underlying broker)
- **Low latency** for fast strategies
- **Quality data** with premium Level 2
- **Professional tools** (hotkeys, scripts, etc.)
- **Flexibility** to switch brokers while keeping the platform

### Limitations and Costs
- **DAS license** (~$150-300/month depending on the plan)
- **Broker account** (Schwab, Zimtra, etc.) with its own requirements
- **Primarily Windows** (DAS Pro requires Windows)
- **Bridge under development** (may have bugs)
- **Steeper learning curve** vs direct APIs

### Typical Cost Structure
```
Broker (e.g., Schwab):      $0-25/month + commissions
DAS Trader Pro:              $150-300/month
Market Data:                 $50-100/month (Level 2)
das-bridge:                  Free (open source)
Monthly total:               ~$200-425/month
```

### Best Practices
1. **Always use paper trading** for initial testing
2. **Validate orders** before sending
3. **Handle reconnections** automatically
4. **Log everything** for debugging
5. **Implement circuit breakers** for emergencies

### Common Troubleshooting
```python
# Common issues and solutions
COMMON_ISSUES = {
    "Connection refused": "Verify that DAS Pro is running and API is enabled",
    "Authentication failed": "Check credentials in .env",
    "Order rejected": "Verify buying power and account limits",
    "Symbol not found": "Verify the symbol is available in DAS",
    "Position mismatch": "Sync state with get_positions()"
}
```

This DAS Trader bridge completes the trading ecosystem, providing access to a professional platform with advanced short selling and routing capabilities, especially valuable for small cap strategies.
