> 🇪🇸 [Leer en Español](platform_integration.es.md) | 🇺🇸 **English**

# Practical Platform Integration

## Complete Workflow: Yahoo Finance -> Polygon -> IBKR -> QuantConnect

### Unified Data Pipeline
```python
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import time

@dataclass
class UnifiedQuote:
    """Unified quote across platforms"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    source: str

@dataclass
class UnifiedBar:
    """Unified OHLCV bar"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str

class DataProvider(ABC):
    """Base class for data providers"""
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[UnifiedQuote]:
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, start_date: str, 
                                end_date: str, interval: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class YahooFinanceProvider(DataProvider):
    """Yahoo Finance provider"""
    
    def __init__(self):
        self.name = "yahoo"
        self.session = requests.Session()
    
    async def get_quote(self, symbol: str) -> Optional[UnifiedQuote]:
        """Get real-time quote"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Yahoo doesn't always have real-time bid/ask
            last_price = info.get('regularMarketPrice', 0)
            bid = info.get('bid', last_price * 0.999)
            ask = info.get('ask', last_price * 1.001)
            volume = info.get('regularMarketVolume', 0)
            
            return UnifiedQuote(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=bid,
                ask=ask,
                last=last_price,
                volume=volume,
                source=self.name
            )
        except Exception as e:
            logging.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, start_date: str, 
                                end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if not data.empty:
                # Add metadata
                data['source'] = self.name
                data['symbol'] = symbol
            
            return data
        except Exception as e:
            logging.error(f"Yahoo Finance historical error for {symbol}: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check availability"""
        try:
            test_ticker = yf.Ticker("AAPL")
            info = test_ticker.info
            return 'regularMarketPrice' in info
        except:
            return False

class PolygonProvider(DataProvider):
    """Polygon.io provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "polygon"
        self.base_url = "https://api.polygon.io"
        self.session = aiohttp.ClientSession()
    
    async def get_quote(self, symbol: str) -> Optional[UnifiedQuote]:
        """Get real-time quote"""
        try:
            url = f"{self.base_url}/v2/last/nbbo/{symbol}"
            params = {"apikey": self.api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'OK' and 'results' in data:
                        result = data['results']
                        
                        return UnifiedQuote(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(result['t'] / 1000),
                            bid=result.get('P', 0),
                            ask=result.get('p', 0),
                            last=result.get('p', 0),  # Use ask as approximation
                            volume=result.get('S', 0),
                            source=self.name
                        )
        except Exception as e:
            logging.error(f"Polygon error for {symbol}: {e}")
        
        return None
    
    async def get_historical_data(self, symbol: str, start_date: str, 
                                end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical data"""
        try:
            # Convert interval to Polygon format
            timespan_map = {
                "1m": ("minute", 1),
                "5m": ("minute", 5),
                "1h": ("hour", 1),
                "1d": ("day", 1)
            }
            
            if interval not in timespan_map:
                interval = "1d"
            
            timespan, multiplier = timespan_map[interval]
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
            params = {
                "apikey": self.api_key,
                "adjusted": "true",
                "sort": "asc"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'OK' and 'results' in data:
                        results = data['results']
                        
                        df_data = []
                        for bar in results:
                            df_data.append({
                                'timestamp': pd.to_datetime(bar['t'], unit='ms'),
                                'open': bar['o'],
                                'high': bar['h'],
                                'low': bar['l'],
                                'close': bar['c'],
                                'volume': bar['v']
                            })
                        
                        df = pd.DataFrame(df_data)
                        if not df.empty:
                            df.set_index('timestamp', inplace=True)
                            df['source'] = self.name
                            df['symbol'] = symbol
                        
                        return df
        except Exception as e:
            logging.error(f"Polygon historical error for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check availability (simplified)"""
        return bool(self.api_key)
    
    async def close(self):
        """Close session"""
        await self.session.close()

class UnifiedDataManager:
    """Unified data manager with fallback"""
    
    def __init__(self):
        self.providers: Dict[str, DataProvider] = {}
        self.provider_priority = []
        self.cache = {}
        self.cache_ttl = 60  # 60 seconds
    
    def add_provider(self, provider: DataProvider, priority: int = 0):
        """Add provider with priority"""
        self.providers[provider.name] = provider
        
        # Insert in priority order
        inserted = False
        for i, (name, prio) in enumerate(self.provider_priority):
            if priority > prio:
                self.provider_priority.insert(i, (provider.name, priority))
                inserted = True
                break
        
        if not inserted:
            self.provider_priority.append((provider.name, priority))
    
    async def get_quote(self, symbol: str) -> Optional[UnifiedQuote]:
        """Get quote with automatic fallback"""
        
        # Check cache
        cache_key = f"quote_{symbol}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return cached_data
        
        # Try providers in priority order
        for provider_name, _ in self.provider_priority:
            provider = self.providers.get(provider_name)
            
            if provider and provider.is_available():
                try:
                    quote = await provider.get_quote(symbol)
                    if quote:
                        # Save to cache
                        self.cache[cache_key] = (quote, datetime.now())
                        logging.info(f"Quote for {symbol} from {provider_name}")
                        return quote
                except Exception as e:
                    logging.warning(f"Provider {provider_name} failed: {e}")
                    continue
        
        logging.error(f"No provider could fetch quote for {symbol}")
        return None
    
    async def get_historical_data(self, symbol: str, start_date: str, 
                                end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical data with fallback"""
        
        # Try providers in priority order
        for provider_name, _ in self.provider_priority:
            provider = self.providers.get(provider_name)
            
            if provider and provider.is_available():
                try:
                    data = await provider.get_historical_data(symbol, start_date, end_date, interval)
                    if not data.empty:
                        logging.info(f"Historical data for {symbol} from {provider_name}")
                        return data
                except Exception as e:
                    logging.warning(f"Provider {provider_name} failed: {e}")
                    continue
        
        logging.error(f"No provider could fetch historical data for {symbol}")
        return pd.DataFrame()
    
    def get_provider_status(self) -> Dict:
        """Get status of all providers"""
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                'available': provider.is_available(),
                'priority': next((p for pname, p in self.provider_priority if pname == name), 0)
            }
        return status

# Unified system demo
async def demo_unified_data_system():
    """Unified data system demo"""
    
    print("🔄 Initializing unified data system...")
    
    # Create manager
    data_manager = UnifiedDataManager()
    
    # Add providers (Yahoo as backup, Polygon as primary)
    yahoo_provider = YahooFinanceProvider()
    data_manager.add_provider(yahoo_provider, priority=1)
    
    # Polygon only if we have API key
    polygon_api_key = "YOUR_POLYGON_API_KEY"  # Replace with your API key
    if polygon_api_key != "YOUR_POLYGON_API_KEY":
        polygon_provider = PolygonProvider(polygon_api_key)
        data_manager.add_provider(polygon_provider, priority=2)
    
    # Check status
    status = data_manager.get_provider_status()
    print("📊 Provider status:")
    for name, info in status.items():
        print(f"  {name}: {'✅' if info['available'] else '❌'} (priority: {info['priority']})")
    
    # Get quotes
    symbols = ["AAPL", "TSLA", "NVDA"]
    
    print(f"\n💰 Fetching quotes...")
    for symbol in symbols:
        quote = await data_manager.get_quote(symbol)
        if quote:
            print(f"  {symbol}: ${quote.last:.2f} (bid: ${quote.bid:.2f}, ask: ${quote.ask:.2f}) [{quote.source}]")
        else:
            print(f"  {symbol}: ❌ Not available")
    
    # Get historical data
    print(f"\n📈 Fetching historical data...")
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    historical_data = await data_manager.get_historical_data("AAPL", start_date, end_date)
    if not historical_data.empty:
        print(f"  AAPL: {len(historical_data)} days of data [{historical_data['source'].iloc[0]}]")
        print(f"  Range: ${historical_data['Low'].min():.2f} - ${historical_data['High'].max():.2f}")
    
    # Cleanup
    for provider in data_manager.providers.values():
        if hasattr(provider, 'close'):
            await provider.close()

# Run demo
if __name__ == "__main__":
    asyncio.run(demo_unified_data_system())
```

## Interactive Brokers TWS Integration

### Connection and Order Execution
```python
from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract
import pandas as pd
from typing import Dict, List, Optional
import asyncio

class IBKRIntegration:
    """Interactive Brokers TWS integration"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
        # Configure callbacks
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.openOrderEvent += self._on_open_order
        self.ib.execDetailsEvent += self._on_execution
    
    async def connect(self) -> bool:
        """Connect to TWS/Gateway"""
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.connected = True
            print(f"✅ Connected to IBKR TWS at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Error connecting to IBKR: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from TWS"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("📴 Disconnected from IBKR TWS")
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.connected:
            return {}
        
        account_values = self.ib.accountValues()
        portfolio = self.ib.portfolio()
        positions = self.ib.positions()
        
        # Process account values
        account_info = {}
        for av in account_values:
            if av.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower']:
                account_info[av.tag] = float(av.value)
        
        # Process positions
        position_info = []
        for pos in positions:
            position_info.append({
                'symbol': pos.contract.symbol,
                'position': pos.position,
                'market_price': pos.marketPrice,
                'market_value': pos.marketValue,
                'avg_cost': pos.averageCost,
                'unrealized_pnl': pos.unrealizedPNL
            })
        
        return {
            'account_values': account_info,
            'positions': position_info,
            'portfolio_items': len(portfolio)
        }
    
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time market data"""
        if not self.connected:
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            # Wait for data
            await asyncio.sleep(2)
            
            if ticker.bid and ticker.ask:
                return {
                    'symbol': symbol,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'last': ticker.last,
                    'volume': ticker.volume,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
        
        return None
    
    async def place_order(self, symbol: str, action: str, quantity: int, 
                         order_type: str = "MKT", limit_price: float = None) -> Optional[int]:
        """Place order"""
        if not self.connected:
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Create order by type
            if order_type.upper() == "MKT":
                order = MarketOrder(action.upper(), quantity)
            elif order_type.upper() == "LMT" and limit_price:
                order = LimitOrder(action.upper(), quantity, limit_price)
            else:
                print(f"Order type not supported: {order_type}")
                return None
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            print(f"📝 Order placed: {action} {quantity} {symbol} @ {order_type}")
            if limit_price:
                print(f"   Limit price: ${limit_price:.2f}")
            
            return trade.order.orderId
            
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    def _on_order_status(self, trade):
        """Callback for order status changes"""
        order = trade.order
        status = trade.orderStatus
        
        print(f"🔄 Orden {order.orderId}: {status.status}")
        if status.status == 'Filled':
            print(f"   ✅ Ejecutada: {status.filled} @ ${status.avgFillPrice:.2f}")
    
    def _on_open_order(self, trade):
        """Callback for open orders"""
        order = trade.order
        print(f"📋 Open order: {order.orderId} - {order.action} {order.totalQuantity} {trade.contract.symbol}")
    
    def _on_execution(self, trade, fill):
        """Callback for executions"""
        print(f"⚡ Execution: {fill.shares} shares @ ${fill.price:.2f}")

class IBKRDataFeed:
    """Real-time data feed from IBKR"""
    
    def __init__(self, ibkr_integration: IBKRIntegration):
        self.ibkr = ibkr_integration
        self.subscriptions = {}
        self.data_callbacks = []
    
    def subscribe(self, symbol: str, callback=None):
        """Subscribe to symbol data"""
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        
        if callback:
            self.subscriptions[symbol].append(callback)
    
    def add_data_callback(self, callback):
        """Add global data callback"""
        self.data_callbacks.append(callback)
    
    async def start_feed(self):
        """Start data feed"""
        if not self.ibkr.connected:
            print("❌ IBKR not connected")
            return
        
        print("🔄 Starting data feed...")
        
        # Subscribe to each symbol
        for symbol in self.subscriptions.keys():
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                ticker = self.ibkr.ib.reqMktData(contract, '', False, False)
                print(f"✅ Subscribed to {symbol}")
            except Exception as e:
                print(f"❌ Error subscribing to {symbol}: {e}")
        
        # Process data in loop
        while True:
            try:
                for symbol in self.subscriptions.keys():
                    data = await self.ibkr.get_market_data(symbol)
                    if data:
                        # Call symbol-specific callbacks
                        for callback in self.subscriptions[symbol]:
                            callback(data)
                        
                        # Call global callbacks
                        for callback in self.data_callbacks:
                            callback(data)
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Error in feed: {e}")
                await asyncio.sleep(5)

# IBKR integration demo
async def demo_ibkr_integration():
    """IBKR integration demo"""
    
    print("🔌 Interactive Brokers integration demo...")
    
    # Create integration
    ibkr = IBKRIntegration()
    
    # Connect (requires TWS/Gateway running)
    connected = await ibkr.connect()
    if not connected:
        print("❌ Could not connect to TWS. Make sure it is running.")
        return
    
    try:
        # Get account info
        account_info = await ibkr.get_account_info()
        print(f"\n💰 Account information:")
        for key, value in account_info.get('account_values', {}).items():
            if isinstance(value, float):
                print(f"  {key}: ${value:,.2f}")
        
        print(f"\n📊 Current positions:")
        for pos in account_info.get('positions', []):
            if pos['position'] != 0:
                print(f"  {pos['symbol']}: {pos['position']} shares @ ${pos['avg_cost']:.2f}")
                print(f"    Unrealized P&L: ${pos['unrealized_pnl']:.2f}")
        
        # Get market data
        print(f"\n📈 Market data:")
        symbols = ["AAPL", "TSLA"]
        for symbol in symbols:
            data = await ibkr.get_market_data(symbol)
            if data:
                print(f"  {symbol}: ${data['last']:.2f} (bid: ${data['bid']:.2f}, ask: ${data['ask']:.2f})")
        
        # Order demo (commented out for safety)
        # order_id = await ibkr.place_order("AAPL", "BUY", 1, "LMT", 150.00)
        # if order_id:
        #     print(f"Order placed with ID: {order_id}")
        
    finally:
        # Disconnect
        ibkr.disconnect()

if __name__ == "__main__":
    asyncio.run(demo_ibkr_integration())
```

## QuantConnect Integration

### Hybrid Local/Cloud Strategy
```python
# Local strategy development that can be ported to QuantConnect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class QuantConnectCompatibleStrategy:
    """QuantConnect-compatible strategy"""
    
    def __init__(self):
        self.name = "HybridMomentumStrategy"
        self.symbols = ["SPY", "QQQ", "IWM"]
        self.lookback_period = 20
        self.portfolio = {}
        self.universe = {}
        
        # Configuration
        self.config = {
            'rebalance_frequency': 'daily',
            'max_positions': 3,
            'risk_per_trade': 0.02,
            'momentum_threshold': 0.1
        }
    
    def initialize(self, data_manager):
        """Initialize strategy (compatible with QC.Initialize)"""
        self.data_manager = data_manager
        
        # Configure universe
        for symbol in self.symbols:
            self.universe[symbol] = {
                'data': pd.DataFrame(),
                'indicators': {},
                'signals': []
            }
        
        print(f"Strategy {self.name} initialized")
    
    async def on_data(self, data: Dict):
        """Process new data (compatible with QC.OnData)"""
        
        # Update data for each symbol
        for symbol, price_data in data.items():
            if symbol in self.universe:
                await self._update_symbol_data(symbol, price_data)
        
        # Generate signals
        signals = await self._generate_signals()
        
        # Execute trades if there are signals
        if signals:
            await self._execute_signals(signals)
    
    async def _update_symbol_data(self, symbol: str, price_data: Dict):
        """Update symbol data"""
        
        # Create new row
        new_row = pd.DataFrame([{
            'timestamp': price_data['timestamp'],
            'open': price_data['open'],
            'high': price_data['high'],
            'low': price_data['low'],
            'close': price_data['close'],
            'volume': price_data['volume']
        }])
        
        # Add to existing data
        symbol_data = self.universe[symbol]['data']
        symbol_data = pd.concat([symbol_data, new_row], ignore_index=True)
        
        # Keep only the last N periods
        if len(symbol_data) > self.lookback_period * 2:
            symbol_data = symbol_data.tail(self.lookback_period * 2)
        
        self.universe[symbol]['data'] = symbol_data
        
        # Update indicators
        await self._update_indicators(symbol)
    
    async def _update_indicators(self, symbol: str):
        """Update technical indicators"""
        
        data = self.universe[symbol]['data']
        
        if len(data) < self.lookback_period:
            return
        
        indicators = {}
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(data['close'], 14)
        
        # Moving averages
        indicators['sma_20'] = data['close'].rolling(20).mean()
        indicators['ema_12'] = data['close'].ewm(span=12).mean()
        indicators['ema_26'] = data['close'].ewm(span=26).mean()
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
        
        # Momentum
        indicators['momentum'] = data['close'].pct_change(10)
        
        # Volatility
        indicators['volatility'] = data['close'].pct_change().rolling(20).std()
        
        self.universe[symbol]['indicators'] = indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def _generate_signals(self) -> List[Dict]:
        """Generate trading signals"""
        
        signals = []
        
        for symbol in self.symbols:
            symbol_info = self.universe[symbol]
            data = symbol_info['data']
            indicators = symbol_info['indicators']
            
            if len(data) < self.lookback_period:
                continue
            
            # Get current values
            current_price = data['close'].iloc[-1]
            current_rsi = indicators['rsi'].iloc[-1]
            current_momentum = indicators['momentum'].iloc[-1]
            current_macd = indicators['macd'].iloc[-1]
            current_signal = indicators['macd_signal'].iloc[-1]
            
            # Signal logic
            signal_strength = 0
            signal_type = None
            
            # Positive momentum
            if current_momentum > self.config['momentum_threshold']:
                signal_strength += 30
            
            # RSI not overbought
            if 30 < current_rsi < 70:
                signal_strength += 20
            
            # MACD bullish
            if current_macd > current_signal:
                signal_strength += 25
            
            # Price above moving average
            if current_price > indicators['sma_20'].iloc[-1]:
                signal_strength += 25
            
            # Determine signal type
            if signal_strength >= 70:
                signal_type = "BUY"
            elif signal_strength <= 30:
                signal_type = "SELL"
            
            if signal_type:
                signals.append({
                    'symbol': symbol,
                    'type': signal_type,
                    'strength': signal_strength,
                    'price': current_price,
                    'timestamp': data['timestamp'].iloc[-1],
                    'metadata': {
                        'rsi': current_rsi,
                        'momentum': current_momentum,
                        'macd': current_macd
                    }
                })
        
        return signals
    
    async def _execute_signals(self, signals: List[Dict]):
        """Execute signals (placeholder for real integration)"""
        
        for signal in signals:
            symbol = signal['symbol']
            signal_type = signal['type']
            strength = signal['strength']
            
            print(f"🎯 Signal {signal_type} for {symbol} (strength: {strength})")
            
            # Here you would integrate with the real broker
            # await self.broker.place_order(...)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        
        # Placeholder - in real implementation would calculate real metrics
        return {
            'strategy_name': self.name,
            'total_signals': sum(len(info['signals']) for info in self.universe.values()),
            'symbols_tracked': len(self.symbols),
            'last_update': datetime.now()
        }

# Code for QuantConnect (separate file: main.py)
QUANTCONNECT_CODE = '''
# QuantConnect Strategy Implementation
from AlgorithmImports import *

class HybridMomentumAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetCash(100000)
        
        # Add securities
        self.symbols = {
            self.AddEquity("SPY", Resolution.Daily).Symbol: "SPY",
            self.AddEquity("QQQ", Resolution.Daily).Symbol: "QQQ", 
            self.AddEquity("IWM", Resolution.Daily).Symbol: "IWM"
        }
        
        # Strategy parameters
        self.lookback_period = 20
        self.momentum_threshold = 0.1
        self.max_positions = 3
        
        # Indicators
        self.indicators = {}
        for symbol in self.symbols.keys():
            self.indicators[symbol] = {
                'rsi': self.RSI(symbol, 14, Resolution.Daily),
                'sma_20': self.SMA(symbol, 20, Resolution.Daily),
                'ema_12': self.EMA(symbol, 12, Resolution.Daily),
                'ema_26': self.EMA(symbol, 26, Resolution.Daily),
                'momentum': self.MOMP(symbol, 10, Resolution.Daily)
            }
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )
    
    def OnData(self, data):
        # Data processing happens in scheduled rebalance
        pass
    
    def Rebalance(self):
        # Generate signals
        signals = self._generate_signals()
        
        # Execute trades
        self._execute_signals(signals)
    
    def _generate_signals(self):
        signals = []
        
        for symbol in self.symbols.keys():
            if not self.indicators[symbol]['rsi'].IsReady:
                continue
            
            # Get current values
            current_price = self.Securities[symbol].Price
            current_rsi = self.indicators[symbol]['rsi'].Current.Value
            current_momentum = self.indicators[symbol]['momentum'].Current.Value
            
            # Signal logic (same as local version)
            signal_strength = 0
            
            if current_momentum > self.momentum_threshold:
                signal_strength += 30
            
            if 30 < current_rsi < 70:
                signal_strength += 20
            
            if current_price > self.indicators[symbol]['sma_20'].Current.Value:
                signal_strength += 25
            
            # MACD logic
            ema_12 = self.indicators[symbol]['ema_12'].Current.Value
            ema_26 = self.indicators[symbol]['ema_26'].Current.Value
            if ema_12 > ema_26:
                signal_strength += 25
            
            if signal_strength >= 70:
                signals.append({
                    'symbol': symbol,
                    'type': 'BUY',
                    'strength': signal_strength
                })
            elif signal_strength <= 30:
                signals.append({
                    'symbol': symbol,
                    'type': 'SELL', 
                    'strength': signal_strength
                })
        
        return signals
    
    def _execute_signals(self, signals):
        # Calculate position sizing
        target_positions = len([s for s in signals if s['type'] == 'BUY'])
        if target_positions == 0:
            self.Liquidate()
            return
        
        position_size = 1.0 / target_positions
        
        # Execute buy signals
        for signal in signals:
            symbol = signal['symbol']
            
            if signal['type'] == 'BUY':
                self.SetHoldings(symbol, position_size)
                self.Debug(f"Buying {self.symbols[symbol]} with {position_size:.2%} allocation")
            elif signal['type'] == 'SELL':
                self.Liquidate(symbol)
                self.Debug(f"Selling {self.symbols[symbol]}")
'''

# Utility for synchronizing strategies
class StrategySync:
    """Synchronize strategy between local and QuantConnect"""
    
    def __init__(self, local_strategy: QuantConnectCompatibleStrategy):
        self.local_strategy = local_strategy
        self.qc_code_template = QUANTCONNECT_CODE
    
    def export_to_quantconnect(self, filename: str = "main.py"):
        """Export strategy to QuantConnect format"""
        
        # Customize code based on local configuration
        qc_code = self.qc_code_template
        
        # Replace parameters
        qc_code = qc_code.replace(
            "self.momentum_threshold = 0.1",
            f"self.momentum_threshold = {self.local_strategy.config['momentum_threshold']}"
        )
        
        qc_code = qc_code.replace(
            "self.max_positions = 3",
            f"self.max_positions = {self.local_strategy.config['max_positions']}"
        )
        
        # Save file
        with open(filename, 'w') as f:
            f.write(qc_code)
        
        print(f"Strategy exported to {filename}")
        print("📁 Upload this file to your QuantConnect project")
    
    def backtest_locally(self, start_date: str, end_date: str):
        """Run local backtest before using in QC"""
        
        print(f"🔬 Running local backtest...")
        
        # Simulate data
        symbols = self.local_strategy.symbols
        
        # This would be the integration with your real data manager
        # data = await self.data_manager.get_historical_data(symbols, start_date, end_date)
        
        print(f"📊 Backtest completed for period {start_date} to {end_date}")
        
        # Return example metrics
        return {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'trades': 45
        }

# QuantConnect integration demo
async def demo_quantconnect_integration():
    """QuantConnect integration demo"""
    
    print("🚀 QuantConnect integration demo...")
    
    # Create local strategy
    strategy = QuantConnectCompatibleStrategy()
    
    # Simulate initialization (you would need your real data manager)
    # await strategy.initialize(data_manager)
    
    # Simulate some example data
    sample_data = {
        "SPY": {
            'timestamp': datetime.now(),
            'open': 400.0,
            'high': 402.0,
            'low': 399.0,
            'close': 401.5,
            'volume': 1000000
        }
    }
    
    # await strategy.on_data(sample_data)
    
    # Configure synchronization
    sync = StrategySync(strategy)
    
    # Run local backtest
    backtest_results = sync.backtest_locally("2023-01-01", "2023-12-31")
    print(f"📈 Backtest results:")
    for metric, value in backtest_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2%}" if 'return' in metric or 'drawdown' in metric else f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")
    
    # Export to QuantConnect
    sync.export_to_quantconnect("hybrid_momentum_strategy.py")
    
    print(f"\n📋 Next steps:")
    print("1. Review the hybrid_momentum_strategy.py file")
    print("2. Upload the file to your QuantConnect project")
    print("3. Run the backtest on the platform")
    print("4. Compare results with the local backtest")

if __name__ == "__main__":
    asyncio.run(demo_quantconnect_integration())
```

## Multi-Platform Monitoring Dashboard

### Centralized Monitoring System
```python
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List

class MultiPlatformDashboard:
    """Dashboard for monitoring multiple platforms"""
    
    def __init__(self):
        self.data_sources = {}
        self.trading_accounts = {}
        self.strategies = {}
        self.alerts = []
    
    def add_data_source(self, name: str, provider):
        """Add data source"""
        self.data_sources[name] = provider
    
    def add_trading_account(self, name: str, account_integration):
        """Add trading account"""
        self.trading_accounts[name] = account_integration
    
    def add_strategy(self, name: str, strategy):
        """Add strategy"""
        self.strategies[name] = strategy
    
    async def get_unified_portfolio_view(self) -> Dict:
        """Get unified portfolio view"""
        
        portfolio_summary = {
            'total_equity': 0,
            'total_pnl': 0,
            'positions': [],
            'account_breakdown': {}
        }
        
        # Add data from each account
        for account_name, account in self.trading_accounts.items():
            try:
                account_info = await account.get_account_info()
                
                account_equity = account_info.get('account_values', {}).get('NetLiquidation', 0)
                portfolio_summary['total_equity'] += account_equity
                
                # Add positions
                for pos in account_info.get('positions', []):
                    if pos['position'] != 0:
                        portfolio_summary['positions'].append({
                            **pos,
                            'account': account_name
                        })
                        portfolio_summary['total_pnl'] += pos.get('unrealized_pnl', 0)
                
                portfolio_summary['account_breakdown'][account_name] = {
                    'equity': account_equity,
                    'positions_count': len([p for p in account_info.get('positions', []) if p['position'] != 0])
                }
                
            except Exception as e:
                st.error(f"Error fetching data from {account_name}: {e}")
        
        return portfolio_summary
    
    async def get_data_feed_status(self) -> Dict:
        """Get data feed status"""
        
        feed_status = {}
        
        for source_name, source in self.data_sources.items():
            try:
                # Test connection
                test_symbol = "AAPL"
                start_time = datetime.now()
                
                if hasattr(source, 'get_quote'):
                    quote = await source.get_quote(test_symbol)
                    success = quote is not None
                else:
                    success = source.is_available() if hasattr(source, 'is_available') else True
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                feed_status[source_name] = {
                    'status': 'online' if success else 'offline',
                    'response_time_ms': response_time,
                    'last_check': datetime.now()
                }
                
            except Exception as e:
                feed_status[source_name] = {
                    'status': 'error',
                    'error': str(e),
                    'last_check': datetime.now()
                }
        
        return feed_status
    
    async def get_strategy_performance(self) -> Dict:
        """Get strategy performance"""
        
        strategy_performance = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'get_performance_metrics'):
                    metrics = strategy.get_performance_metrics()
                    strategy_performance[strategy_name] = metrics
                else:
                    # Mock metrics for demo
                    strategy_performance[strategy_name] = {
                        'total_trades': np.random.randint(10, 100),
                        'win_rate': np.random.uniform(0.4, 0.8),
                        'profit_factor': np.random.uniform(1.0, 2.5),
                        'total_pnl': np.random.uniform(-1000, 5000)
                    }
            except Exception as e:
                strategy_performance[strategy_name] = {'error': str(e)}
        
        return strategy_performance

def create_streamlit_dashboard():
    """Create dashboard with Streamlit"""
    
    st.set_page_config(
        page_title="Trading Multi-Platform Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Multi-Platform Trading Dashboard")
    
    # Initialize dashboard (normally this would be in session_state)
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = MultiPlatformDashboard()
        
        # Add mock sources for demo
        st.session_state.dashboard.data_sources = {
            'Yahoo Finance': {'status': 'online'},
            'Polygon.io': {'status': 'online'},
            'IBKR TWS': {'status': 'online'}
        }
        
        st.session_state.dashboard.trading_accounts = {
            'IBKR Main': {'equity': 85000, 'positions': 3},
            'TD Ameritrade': {'equity': 25000, 'positions': 1}
        }
    
    dashboard = st.session_state.dashboard
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Controls")
        
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Interval (sec)", 10, 300, 30)
        
        if st.button("🔄 Refresh Manual"):
            st.rerun()
        
        st.header("📡 Data Sources")
        for source, info in dashboard.data_sources.items():
            status_icon = "🟢" if info.get('status') == 'online' else "🔴"
            st.write(f"{status_icon} {source}")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Mock data for demo
    total_equity = sum(acc['equity'] for acc in dashboard.trading_accounts.values())
    total_positions = sum(acc['positions'] for acc in dashboard.trading_accounts.values())
    daily_pnl = np.random.uniform(-2000, 3000)
    
    with col1:
        st.metric(
            "Total Equity",
            f"${total_equity:,.2f}",
            f"{daily_pnl:+.2f}"
        )
    
    with col2:
        st.metric(
            "Active Positions",
            total_positions,
            "+2"
        )
    
    with col3:
        st.metric(
            "Daily P&L",
            f"${daily_pnl:,.2f}",
            f"{daily_pnl/total_equity:.2%}"
        )
    
    with col4:
        data_sources_online = len([s for s in dashboard.data_sources.values() if s.get('status') == 'online'])
        st.metric(
            "Data Sources",
            f"{data_sources_online}/{len(dashboard.data_sources)}",
            "All Online" if data_sources_online == len(dashboard.data_sources) else "Some Offline"
        )
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Portfolio Allocation")
        
        # Mock allocation data
        allocation_data = pd.DataFrame({
            'Account': list(dashboard.trading_accounts.keys()),
            'Value': [acc['equity'] for acc in dashboard.trading_accounts.values()]
        })
        
        fig_pie = px.pie(
            allocation_data,
            values='Value',
            names='Account',
            title="Allocation by Account"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 Data Feed Status")
        
        # Feed status table
        feed_status_df = pd.DataFrame([
            {
                'Source': source,
                'Status': info.get('status', 'unknown'),
                'Response Time': f"{np.random.randint(50, 200)}ms"
            }
            for source, info in dashboard.data_sources.items()
        ])
        
        # Color code status
        def color_status(val):
            if val == 'online':
                return 'background-color: #90EE90'
            elif val == 'offline':
                return 'background-color: #FFB6C1'
            else:
                return 'background-color: #FFFFE0'
        
        styled_df = feed_status_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
    
    # Current positions
    st.subheader("📋 Current Positions")
    
    # Mock positions data
    positions_data = []
    symbols = ["AAPL", "TSLA", "NVDA", "MSFT"]
    
    for i, symbol in enumerate(symbols):
        if i < total_positions:
            account = list(dashboard.trading_accounts.keys())[i % len(dashboard.trading_accounts)]
            positions_data.append({
                'Account': account,
                'Symbol': symbol,
                'Quantity': np.random.randint(10, 500),
                'Avg Price': np.random.uniform(100, 300),
                'Current Price': np.random.uniform(100, 300),
                'Unrealized P&L': np.random.uniform(-2000, 5000),
                'P&L %': np.random.uniform(-0.15, 0.25)
            })
    
    if positions_data:
        positions_df = pd.DataFrame(positions_data)
        
        # Color code P&L
        def color_pnl(val):
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
            else:
                return 'color: black'
        
        styled_positions = positions_df.style.applymap(
            color_pnl, 
            subset=['Unrealized P&L', 'P&L %']
        ).format({
            'Avg Price': '${:.2f}',
            'Current Price': '${:.2f}',
            'Unrealized P&L': '${:.2f}',
            'P&L %': '{:.2%}'
        })
        
        st.dataframe(styled_positions, use_container_width=True)
    else:
        st.info("No open positions")
    
    # Strategy Performance
    st.subheader("🎯 Strategy Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mock strategy data
        strategy_data = {
            'Gap & Go': {'trades': 45, 'win_rate': 0.67, 'pnl': 3200},
            'VWAP Reclaim': {'trades': 32, 'win_rate': 0.59, 'pnl': 1800},
            'Momentum': {'trades': 28, 'win_rate': 0.71, 'pnl': 2400}
        }
        
        strategy_df = pd.DataFrame([
            {
                'Strategy': strategy,
                'Trades': data['trades'],
                'Win Rate': data['win_rate'],
                'Total P&L': data['pnl']
            }
            for strategy, data in strategy_data.items()
        ])
        
        st.dataframe(
            strategy_df.style.format({
                'Win Rate': '{:.1%}',
                'Total P&L': '${:.0f}'
            }),
            use_container_width=True
        )
    
    with col2:
        # Strategy P&L chart
        fig_strategy = go.Figure()
        
        for strategy, data in strategy_data.items():
            fig_strategy.add_trace(go.Bar(
                name=strategy,
                x=[strategy],
                y=[data['pnl']],
                text=f"${data['pnl']:.0f}",
                textposition='auto'
            ))
        
        fig_strategy.update_layout(
            title="Strategy P&L Comparison",
            showlegend=False,
            yaxis_title="P&L ($)"
        )
        
        st.plotly_chart(fig_strategy, use_container_width=True)
    
    # Alerts and notifications
    st.subheader("🚨 Alerts & Notifications")
    
    # Mock alerts
    mock_alerts = [
        {"time": "10:30 AM", "type": "INFO", "message": "AAPL gap up 3.2% with high volume"},
        {"time": "11:15 AM", "type": "WARNING", "message": "TSLA position approaching stop loss"},
        {"time": "12:00 PM", "type": "SUCCESS", "message": "NVDA target reached - position closed"},
    ]
    
    for alert in mock_alerts:
        alert_type = alert['type']
        if alert_type == "WARNING":
            st.warning(f"⚠️ {alert['time']}: {alert['message']}")
        elif alert_type == "SUCCESS":
            st.success(f"✅ {alert['time']}: {alert['message']}")
        else:
            st.info(f"ℹ️ {alert['time']}: {alert['message']}")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

# Run dashboard
if __name__ == "__main__":
    create_streamlit_dashboard()
```

This multi-platform integration system provides a solid foundation for connecting and coordinating different data sources, brokers, and strategies in a unified quantitative trading workflow.