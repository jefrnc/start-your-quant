> 🇪🇸 [Leer en Español](broker_integration.es.md) | 🇺🇸 **English**

# Broker Integration

Comparison of recommended brokers for quantitative trading and their specific use cases.

## Broker Comparison

| Broker | Best For | API Quality | Commissions | Data Quality |
|--------|----------|-------------|-------------|--------------|
| **IBKR** | All-around, institutional | ⭐⭐⭐⭐⭐ | Very low | ⭐⭐⭐⭐⭐ |
| **Alpaca** | Development, algorithms | ⭐⭐⭐⭐⭐ | Commission-free | ⭐⭐⭐⭐ |
| **Charles Schwab** | Retail, 401k | ⭐⭐⭐ | Commission-free | ⭐⭐⭐⭐ |
| **TD Ameritrade** | Advanced retail | ⭐⭐⭐⭐ | Commission-free | ⭐⭐⭐⭐ |

## Trading Platforms

| Platform | Compatible With | Best For | Key Features |
|----------|----------------|----------|--------------|
| **DAS Trader Pro** | Charles Schwab, Zimtra, others | Day trading, small caps | Level 2, advanced routing, hotkeys |
| **TWS (IBKR)** | Interactive Brokers | Institutional trading | Robust API, global data |
| **Think or Swim** | TD Ameritrade | Technical analysis | Advanced charting, paper trading |
| **Webull Desktop** | Webull | Retail trading | Free, basic analysis |

## Interactive Brokers (IBKR) - Primary Recommendation

### IBKR Advantages
- **Low commissions**: $0.005 per share, $1 minimum per trade
- **Robust API**: TWS API with multiple language support
- **Real-time data**: Level 1 and Level 2 market data
- **Short selling**: Large inventory for locates
- **Margin requirements**: Competitive margin rates
- **Global access**: Access to multiple markets

### TWS (Trader Workstation) Setup

```python
# config/ibkr_setup.py
from ib_insync import IB, Stock, MarketOrder, LimitOrder
import asyncio
import logging

class IBKRConnection:
    def __init__(self, host='localhost', port=7497, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
    def connect(self):
        """Connect to TWS"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logging.info(f"Connected to IBKR TWS at {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"Error connecting to IBKR: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from TWS"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logging.info("Disconnected from IBKR TWS")
    
    def get_account_info(self):
        """Get account information"""
        if not self.connected:
            return None
            
        account_values = self.ib.accountValues()
        portfolio = self.ib.portfolio()
        positions = self.ib.positions()
        
        return {
            'account_values': account_values,
            'portfolio': portfolio,
            'positions': positions,
            'buying_power': self.get_buying_power(),
            'total_liquidity': self.get_total_liquidity()
        }
    
    def get_buying_power(self):
        """Get available buying power"""
        account_values = self.ib.accountValues()
        for item in account_values:
            if item.tag == 'BuyingPower':
                return float(item.value)
        return 0
    
    def get_total_liquidity(self):
        """Get total liquidity"""
        account_values = self.ib.accountValues()
        for item in account_values:
            if item.tag == 'TotalCashValue':
                return float(item.value)
        return 0
    
    def get_market_data(self, symbol, exchange='SMART'):
        """Get real-time market data"""
        contract = Stock(symbol, exchange)
        self.ib.reqMktData(contract)
        
        # Wait for data to arrive
        self.ib.sleep(2)
        
        ticker = self.ib.ticker(contract)
        
        return {
            'symbol': symbol,
            'bid': ticker.bid,
            'ask': ticker.ask,
            'last': ticker.last,
            'volume': ticker.volume,
            'bid_size': ticker.bidSize,
            'ask_size': ticker.askSize
        }
    
    def place_market_order(self, symbol, quantity, action='BUY'):
        """Place a market order"""
        contract = Stock(symbol, 'SMART')
        order = MarketOrder(action, abs(quantity))
        
        trade = self.ib.placeOrder(contract, order)
        
        return {
            'trade': trade,
            'order_id': trade.order.orderId,
            'status': trade.orderStatus.status
        }
    
    def place_limit_order(self, symbol, quantity, price, action='BUY'):
        """Place a limit order"""
        contract = Stock(symbol, 'SMART')
        order = LimitOrder(action, abs(quantity), price)
        
        trade = self.ib.placeOrder(contract, order)
        
        return {
            'trade': trade,
            'order_id': trade.order.orderId,
            'status': trade.orderStatus.status
        }
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        try:
            self.ib.cancelOrder(order_id)
            return True
        except Exception as e:
            logging.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_historical_data(self, symbol, duration='1 Y', bar_size='1 day'):
        """Get historical data"""
        contract = Stock(symbol, 'SMART')
        
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )
        
        # Convert to DataFrame
        if bars:
            df = pd.DataFrame([{
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars])
            
            df.set_index('date', inplace=True)
            return df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data

# Usage example
def test_ibkr_connection():
    """IBKR connection test"""
    ibkr = IBKRConnection()
    
    if ibkr.connect():
        # Test account info
        account_info = ibkr.get_account_info()
        print(f"Buying Power: ${account_info['buying_power']:,.2f}")
        
        # Test market data
        spy_data = ibkr.get_market_data('SPY')
        print(f"SPY: Bid={spy_data['bid']}, Ask={spy_data['ask']}")
        
        # Test historical data
        historical = ibkr.get_historical_data('SPY', '1 M', '1 hour')
        print(f"Historical data: {len(historical)} bars")
        
        ibkr.disconnect()
        return True
    
    return False
```

## DAS Trader Pro - Day Trading Platform

> **New:** Integration available through the [das-bridge](https://github.com/jefrnc/das-bridge)

**Important note:** DAS Trader Pro is a **trading platform**, not a broker. It connects to brokers like Charles Schwab, Zimtra, Lightspeed, and others that offer DAS compatibility.

### Why Use DAS as a Platform?
- **Extensive borrows** for short selling small caps
- **Advanced routing** (ARCA, EDGX, BATS) for better fills
- **Premium Level 2** with deep Times & Sales
- **Configurable hotkeys** for ultra-fast execution
- **Active community** of professional day traders

### Ideal Use Cases
- Aggressive day trading in small caps
- Short selling pump & dumps
- High-volume scalping
- Trading with 4:1 or 6:1 buying power

### Basic DAS Bridge Setup

```bash
# Install the bridge
git clone https://github.com/jefrnc/das-bridge.git
cd das-bridge
pip install -r requirements.txt
pip install -e .
```

```python
# config/das_config.py
import os

DAS_CONFIG = {
    'host': os.getenv('DAS_HOST', 'localhost'),
    'port': int(os.getenv('DAS_PORT', '9910')),
    'username': os.getenv('DAS_USERNAME'),
    'password': os.getenv('DAS_PASSWORD'),
    'account': os.getenv('DAS_ACCOUNT'),
    'paper_trading': True  # Change to False for live trading
}

# DAS-specific risk limits
DAS_RISK_LIMITS = {
    'max_position_size': 10000,      # $10k max per position
    'max_daily_loss': 1000,          # $1k max daily loss
    'max_buying_power_usage': 0.8,   # 80% max buying power usage
    'max_trades_per_minute': 5       # Rate limit
}
```

### Integration Example

```python
# examples/das_trading_basic.py
import asyncio
from das_trader import DASTraderClient

async def das_example():
    """Basic DAS example"""
    
    client = DASTraderClient(host="localhost", port=9910)
    
    try:
        # Connect
        success = await client.connect("username", "password", "account")
        if not success:
            print("Error connecting to DAS")
            return
        
        print("Connected to DAS Trader")
        
        # Get buying power
        bp = await client.get_buying_power()
        print(f"Buying Power: ${bp:,.2f}")
        
        # Subscribe to quote
        await client.subscribe_quote("AAPL")
        
        # Order example (commented out for safety)
        # order = await client.send_order("AAPL", "BUY", 100, "MARKET")
        # print(f"Order sent: {order.order_id}")
        
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(das_example())
```

> **Full documentation:** See [DAS Trader Integration](das_trader_integration.md) for advanced setup.

### TWS Configuration

```python
# scripts/setup_tws.py
"""
Script to configure TWS for API trading
"""

TWS_CONFIGURATION = """
Manual TWS Configuration:

1. Download TWS:
   - Go to https://www.interactivebrokers.com/en/trading/tws.php
   - Download Trader Workstation

2. Configure API:
   - Open TWS
   - Go to Configure -> API -> Settings
   - Enable ActiveX and Socket Clients
   - Read-Only API
   - Socket port: 7497 (paper) / 7496 (live)
   - Master API client ID: 0
   - Download open orders on connection

3. Configure Paper Trading:
   - Go to Configure -> API -> Settings
   - Enable API
   - Port: 7497
   - Allowed IPs: 127.0.0.1

4. Market Data:
   - Go to Configure -> Market Data Subscriptions
   - Enable US Securities Snapshot and Futures Value Bundle (free)
   - For Level 2: US Equity and Options Add-On Streaming Bundle

5. Configure Alerts:
   - Configure -> Alerts
   - Enable Popup alerts
   - Email alerts (optional)

IMPORTANT:
- Use Paper Trading account initially
- TWS must be open and connected to use the API
- Verify the port is correct (7497 paper / 7496 live)
"""

print(TWS_CONFIGURATION)
```

## Alpaca - Alternative Broker

### Alpaca Advantages
- **Commission-free**: No commissions on stocks
- **API-first**: Designed for algorithmic trading
- **Paper trading**: Robust sandbox environment
- **Modern REST API**: Easy to use
- **Real-time data**: WebSocket feeds

### Alpaca Setup

```python
# config/alpaca_setup.py
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pandas as pd

class AlpacaConnection:
    def __init__(self, api_key, secret_key, base_url='https://paper-api.alpaca.markets'):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.base_url = base_url
        
    def get_account(self):
        """Get account information"""
        account = self.api.get_account()
        
        return {
            'account_id': account.id,
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'day_trade_count': account.daytrade_count,
            'pattern_day_trader': account.pattern_day_trader
        }
    
    def get_positions(self):
        """Get current positions"""
        positions = self.api.list_positions()
        
        position_data = []
        for pos in positions:
            position_data.append({
                'symbol': pos.symbol,
                'qty': int(pos.qty),
                'side': 'long' if int(pos.qty) > 0 else 'short',
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'avg_entry_price': float(pos.avg_entry_price)
            })
        
        return position_data
    
    def get_orders(self, status='all', limit=100):
        """Get orders"""
        orders = self.api.list_orders(status=status, limit=limit)
        
        order_data = []
        for order in orders:
            order_data.append({
                'id': order.id,
                'symbol': order.symbol,
                'qty': int(order.qty),
                'side': order.side,
                'order_type': order.order_type,
                'time_in_force': order.time_in_force,
                'status': order.status,
                'filled_qty': int(order.filled_qty or 0),
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'submitted_at': order.submitted_at
            })
        
        return order_data
    
    def place_order(self, symbol, qty, side, order_type='market', 
                   limit_price=None, stop_price=None, time_in_force='day'):
        """Place an order"""
        
        order_params = {
            'symbol': symbol,
            'qty': abs(qty),
            'side': side,
            'type': order_type,
            'time_in_force': time_in_force
        }
        
        if order_type == 'limit' and limit_price:
            order_params['limit_price'] = limit_price
        elif order_type == 'stop' and stop_price:
            order_params['stop_price'] = stop_price
        elif order_type == 'stop_limit' and limit_price and stop_price:
            order_params['limit_price'] = limit_price
            order_params['stop_price'] = stop_price
        
        try:
            order = self.api.submit_order(**order_params)
            return {
                'success': True,
                'order_id': order.id,
                'status': order.status,
                'symbol': order.symbol,
                'qty': int(order.qty),
                'side': order.side
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False
    
    def get_historical_data(self, symbol, timeframe='1Day', start=None, end=None):
        """Get historical data"""
        if not start:
            start = datetime.now() - timedelta(days=365)
        if not end:
            end = datetime.now()
        
        # Convert to Alpaca format
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        barset = self.api.get_bars(
            symbol,
            timeframe,
            start=start_str,
            end=end_str,
            adjustment='raw'
        )
        
        # Convert to DataFrame
        data = []
        for bar in barset:
            data.append({
                'timestamp': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_quote(self, symbol):
        """Get current quote"""
        try:
            quote = self.api.get_latest_quote(symbol)
            return {
                'symbol': symbol,
                'bid': quote.bid_price,
                'ask': quote.ask_price,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': quote.timestamp
            }
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None

# Alpaca connection test
def test_alpaca_connection():
    """Alpaca connection test"""
    from config.api_keys import APIKeys
    
    if not APIKeys.ALPACA_API_KEY or not APIKeys.ALPACA_SECRET_KEY:
        print("Alpaca API Keys not configured")
        return False
    
    alpaca = AlpacaConnection(
        APIKeys.ALPACA_API_KEY,
        APIKeys.ALPACA_SECRET_KEY,
        APIKeys.ALPACA_BASE_URL
    )
    
    try:
        # Test account
        account = alpaca.get_account()
        print(f"Alpaca account connected")
        print(f"Buying Power: ${account['buying_power']:,.2f}")
        print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
        
        # Test quote
        quote = alpaca.get_quote('SPY')
        if quote:
            print(f"SPY Quote: ${quote['bid']} x ${quote['ask']}")
        
        # Test historical data
        historical = alpaca.get_historical_data('SPY', '1Day')
        print(f"Historical data: {len(historical)} days")
        
        return True
        
    except Exception as e:
        print(f"Error in Alpaca connection: {e}")
        return False
```

## TD Ameritrade - Backup Option

### TD Ameritrade Setup

```python
# config/td_setup.py
import requests
import json
from datetime import datetime, timedelta

class TDAConnection:
    def __init__(self, api_key, refresh_token=None):
        self.api_key = api_key
        self.refresh_token = refresh_token
        self.access_token = None
        self.base_url = 'https://api.tdameritrade.com/v1'
        
    def authenticate(self):
        """Authenticate with TD Ameritrade"""
        if self.refresh_token:
            return self._refresh_access_token()
        else:
            print("You need to configure the OAuth flow for TD Ameritrade")
            return False
    
    def _refresh_access_token(self):
        """Refresh the access token"""
        url = f"{self.base_url}/oauth2/token"
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.api_key
        }
        
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            return True
        else:
            print(f"Error refreshing token: {response.status_code}")
            return False
    
    def get_quote(self, symbol):
        """Get quote"""
        if not self.access_token:
            return None
        
        url = f"{self.base_url}/marketdata/quotes"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        params = {'symbol': symbol}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()[symbol]
            return {
                'symbol': symbol,
                'bid': data['bidPrice'],
                'ask': data['askPrice'],
                'last': data['lastPrice'],
                'volume': data['totalVolume']
            }
        
        return None
    
    def get_historical_data(self, symbol, period_type='year', period=1, 
                          frequency_type='daily', frequency=1):
        """Get historical data"""
        if not self.access_token:
            return None
        
        url = f"{self.base_url}/marketdata/{symbol}/pricehistory"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        params = {
            'periodType': period_type,
            'period': period,
            'frequencyType': frequency_type,
            'frequency': frequency
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            candles = data['candles']
            
            # Convert to DataFrame
            df_data = []
            for candle in candles:
                df_data.append({
                    'timestamp': pd.to_datetime(candle['datetime'], unit='ms'),
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': candle['volume']
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df
        
        return None
```

## Unified Broker Interface

### Abstraction Layer

```python
# src/execution/broker_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class OrderRequest:
    """Unified order structure"""
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'day'

@dataclass
class Position:
    """Unified position structure"""
    symbol: str
    quantity: int
    avg_price: float
    market_value: float
    unrealized_pnl: float

class BrokerInterface(ABC):
    """Abstract broker interface"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the broker"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """Get account information"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def place_order(self, order: OrderRequest) -> Dict:
        """Place an order"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str) -> Dict:
        """Get market data"""
        pass

class UnifiedBroker:
    """Unified broker that manages multiple brokers"""
    
    def __init__(self):
        self.brokers = {}
        self.primary_broker = None
        
    def add_broker(self, name: str, broker: BrokerInterface, is_primary=False):
        """Add a broker"""
        self.brokers[name] = broker
        if is_primary:
            self.primary_broker = name
    
    def execute_order(self, order: OrderRequest, broker_name=None):
        """Execute order on a specific or primary broker"""
        broker_name = broker_name or self.primary_broker
        
        if broker_name not in self.brokers:
            return {'success': False, 'error': f'Broker {broker_name} not found'}
        
        try:
            result = self.brokers[broker_name].place_order(order)
            return result
        except Exception as e:
            # Fallback to another broker if it fails
            for backup_name, backup_broker in self.brokers.items():
                if backup_name != broker_name:
                    try:
                        result = backup_broker.place_order(order)
                        result['executed_on_backup'] = backup_name
                        return result
                    except:
                        continue
            
            return {'success': False, 'error': str(e)}
    
    def get_consolidated_positions(self):
        """Get consolidated positions from all brokers"""
        all_positions = {}
        
        for broker_name, broker in self.brokers.items():
            try:
                positions = broker.get_positions()
                all_positions[broker_name] = positions
            except Exception as e:
                print(f"Error getting positions from {broker_name}: {e}")
        
        return all_positions
```

This broker infrastructure will allow you to start with paper trading on IBKR or Alpaca, and then expand to live trading with multiple brokers as your needs grow.
