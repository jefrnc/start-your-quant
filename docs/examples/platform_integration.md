# Integración Práctica entre Plataformas

## Workflow Completo: Yahoo Finance → Polygon → IBKR → QuantConnect

### Pipeline de Datos Unificado
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
    """Quote unificado entre plataformas"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    source: str

@dataclass
class UnifiedBar:
    """Barra OHLCV unificada"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str

class DataProvider(ABC):
    """Clase base para proveedores de datos"""
    
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
    """Proveedor Yahoo Finance"""
    
    def __init__(self):
        self.name = "yahoo"
        self.session = requests.Session()
    
    async def get_quote(self, symbol: str) -> Optional[UnifiedQuote]:
        """Obtener quote en tiempo real"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Yahoo no siempre tiene bid/ask en tiempo real
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
        """Obtener datos históricos"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if not data.empty:
                # Agregar metadatos
                data['source'] = self.name
                data['symbol'] = symbol
            
            return data
        except Exception as e:
            logging.error(f"Yahoo Finance historical error for {symbol}: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Verificar disponibilidad"""
        try:
            test_ticker = yf.Ticker("AAPL")
            info = test_ticker.info
            return 'regularMarketPrice' in info
        except:
            return False

class PolygonProvider(DataProvider):
    """Proveedor Polygon.io"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "polygon"
        self.base_url = "https://api.polygon.io"
        self.session = aiohttp.ClientSession()
    
    async def get_quote(self, symbol: str) -> Optional[UnifiedQuote]:
        """Obtener quote en tiempo real"""
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
                            last=result.get('p', 0),  # Usar ask como aproximación
                            volume=result.get('S', 0),
                            source=self.name
                        )
        except Exception as e:
            logging.error(f"Polygon error for {symbol}: {e}")
        
        return None
    
    async def get_historical_data(self, symbol: str, start_date: str, 
                                end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Obtener datos históricos"""
        try:
            # Convertir intervalo a formato Polygon
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
        """Verificar disponibilidad (simplificado)"""
        return bool(self.api_key)
    
    async def close(self):
        """Cerrar sesión"""
        await self.session.close()

class UnifiedDataManager:
    """Gestor unificado de datos con fallback"""
    
    def __init__(self):
        self.providers: Dict[str, DataProvider] = {}
        self.provider_priority = []
        self.cache = {}
        self.cache_ttl = 60  # 60 segundos
    
    def add_provider(self, provider: DataProvider, priority: int = 0):
        """Agregar proveedor con prioridad"""
        self.providers[provider.name] = provider
        
        # Insertar en orden de prioridad
        inserted = False
        for i, (name, prio) in enumerate(self.provider_priority):
            if priority > prio:
                self.provider_priority.insert(i, (provider.name, priority))
                inserted = True
                break
        
        if not inserted:
            self.provider_priority.append((provider.name, priority))
    
    async def get_quote(self, symbol: str) -> Optional[UnifiedQuote]:
        """Obtener quote con fallback automático"""
        
        # Verificar cache
        cache_key = f"quote_{symbol}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return cached_data
        
        # Intentar proveedores en orden de prioridad
        for provider_name, _ in self.provider_priority:
            provider = self.providers.get(provider_name)
            
            if provider and provider.is_available():
                try:
                    quote = await provider.get_quote(symbol)
                    if quote:
                        # Guardar en cache
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
        """Obtener datos históricos con fallback"""
        
        # Intentar proveedores en orden de prioridad
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
        """Obtener estado de todos los proveedores"""
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                'available': provider.is_available(),
                'priority': next((p for pname, p in self.provider_priority if pname == name), 0)
            }
        return status

# Demo del sistema unificado
async def demo_unified_data_system():
    """Demo del sistema unificado de datos"""
    
    print("🔄 Inicializando sistema unificado de datos...")
    
    # Crear manager
    data_manager = UnifiedDataManager()
    
    # Agregar proveedores (Yahoo como backup, Polygon como primario)
    yahoo_provider = YahooFinanceProvider()
    data_manager.add_provider(yahoo_provider, priority=1)
    
    # Polygon solo si tenemos API key
    polygon_api_key = "YOUR_POLYGON_API_KEY"  # Reemplazar con tu API key
    if polygon_api_key != "YOUR_POLYGON_API_KEY":
        polygon_provider = PolygonProvider(polygon_api_key)
        data_manager.add_provider(polygon_provider, priority=2)
    
    # Verificar estado
    status = data_manager.get_provider_status()
    print("📊 Estado de proveedores:")
    for name, info in status.items():
        print(f"  {name}: {'✅' if info['available'] else '❌'} (prioridad: {info['priority']})")
    
    # Obtener quotes
    symbols = ["AAPL", "TSLA", "NVDA"]
    
    print(f"\n💰 Obteniendo quotes...")
    for symbol in symbols:
        quote = await data_manager.get_quote(symbol)
        if quote:
            print(f"  {symbol}: ${quote.last:.2f} (bid: ${quote.bid:.2f}, ask: ${quote.ask:.2f}) [{quote.source}]")
        else:
            print(f"  {symbol}: ❌ No disponible")
    
    # Obtener datos históricos
    print(f"\n📈 Obteniendo datos históricos...")
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    historical_data = await data_manager.get_historical_data("AAPL", start_date, end_date)
    if not historical_data.empty:
        print(f"  AAPL: {len(historical_data)} días de datos [{historical_data['source'].iloc[0]}]")
        print(f"  Rango: ${historical_data['Low'].min():.2f} - ${historical_data['High'].max():.2f}")
    
    # Cleanup
    for provider in data_manager.providers.values():
        if hasattr(provider, 'close'):
            await provider.close()

# Ejecutar demo
if __name__ == "__main__":
    asyncio.run(demo_unified_data_system())
```

## Integración con Interactive Brokers TWS

### Conexión y Ejecución de Órdenes
```python
from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract
import pandas as pd
from typing import Dict, List, Optional
import asyncio

class IBKRIntegration:
    """Integración con Interactive Brokers TWS"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
        # Configurar callbacks
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.openOrderEvent += self._on_open_order
        self.ib.execDetailsEvent += self._on_execution
    
    async def connect(self) -> bool:
        """Conectar a TWS/Gateway"""
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.connected = True
            print(f"✅ Conectado a IBKR TWS en {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Error conectando a IBKR: {e}")
            return False
    
    def disconnect(self):
        """Desconectar de TWS"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("📴 Desconectado de IBKR TWS")
    
    async def get_account_info(self) -> Dict:
        """Obtener información de cuenta"""
        if not self.connected:
            return {}
        
        account_values = self.ib.accountValues()
        portfolio = self.ib.portfolio()
        positions = self.ib.positions()
        
        # Procesar valores de cuenta
        account_info = {}
        for av in account_values:
            if av.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower']:
                account_info[av.tag] = float(av.value)
        
        # Procesar posiciones
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
        """Obtener datos de mercado en tiempo real"""
        if not self.connected:
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Solicitar datos de mercado
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            # Esperar datos
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
            print(f"Error obteniendo datos de {symbol}: {e}")
        
        return None
    
    async def place_order(self, symbol: str, action: str, quantity: int, 
                         order_type: str = "MKT", limit_price: float = None) -> Optional[int]:
        """Colocar orden"""
        if not self.connected:
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Crear orden según tipo
            if order_type.upper() == "MKT":
                order = MarketOrder(action.upper(), quantity)
            elif order_type.upper() == "LMT" and limit_price:
                order = LimitOrder(action.upper(), quantity, limit_price)
            else:
                print(f"Tipo de orden no soportado: {order_type}")
                return None
            
            # Colocar orden
            trade = self.ib.placeOrder(contract, order)
            
            print(f"📝 Orden colocada: {action} {quantity} {symbol} @ {order_type}")
            if limit_price:
                print(f"   Precio límite: ${limit_price:.2f}")
            
            return trade.order.orderId
            
        except Exception as e:
            print(f"Error colocando orden: {e}")
            return None
    
    def _on_order_status(self, trade):
        """Callback para cambios de estado de orden"""
        order = trade.order
        status = trade.orderStatus
        
        print(f"🔄 Orden {order.orderId}: {status.status}")
        if status.status == 'Filled':
            print(f"   ✅ Ejecutada: {status.filled} @ ${status.avgFillPrice:.2f}")
    
    def _on_open_order(self, trade):
        """Callback para órdenes abiertas"""
        order = trade.order
        print(f"📋 Orden abierta: {order.orderId} - {order.action} {order.totalQuantity} {trade.contract.symbol}")
    
    def _on_execution(self, trade, fill):
        """Callback para ejecuciones"""
        print(f"⚡ Ejecución: {fill.shares} shares @ ${fill.price:.2f}")

class IBKRDataFeed:
    """Feed de datos en tiempo real desde IBKR"""
    
    def __init__(self, ibkr_integration: IBKRIntegration):
        self.ibkr = ibkr_integration
        self.subscriptions = {}
        self.data_callbacks = []
    
    def subscribe(self, symbol: str, callback=None):
        """Suscribirse a datos de un símbolo"""
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        
        if callback:
            self.subscriptions[symbol].append(callback)
    
    def add_data_callback(self, callback):
        """Agregar callback global para datos"""
        self.data_callbacks.append(callback)
    
    async def start_feed(self):
        """Iniciar feed de datos"""
        if not self.ibkr.connected:
            print("❌ IBKR no conectado")
            return
        
        print("🔄 Iniciando feed de datos...")
        
        # Suscribirse a cada símbolo
        for symbol in self.subscriptions.keys():
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                ticker = self.ibkr.ib.reqMktData(contract, '', False, False)
                print(f"✅ Suscrito a {symbol}")
            except Exception as e:
                print(f"❌ Error suscribiendo a {symbol}: {e}")
        
        # Procesar datos en loop
        while True:
            try:
                for symbol in self.subscriptions.keys():
                    data = await self.ibkr.get_market_data(symbol)
                    if data:
                        # Llamar callbacks específicos del símbolo
                        for callback in self.subscriptions[symbol]:
                            callback(data)
                        
                        # Llamar callbacks globales
                        for callback in self.data_callbacks:
                            callback(data)
                
                await asyncio.sleep(1)  # Actualizar cada segundo
                
            except Exception as e:
                print(f"Error en feed: {e}")
                await asyncio.sleep(5)

# Demo de integración con IBKR
async def demo_ibkr_integration():
    """Demo de integración con IBKR"""
    
    print("🔌 Demo de integración con Interactive Brokers...")
    
    # Crear integración
    ibkr = IBKRIntegration()
    
    # Conectar (requiere TWS/Gateway ejecutándose)
    connected = await ibkr.connect()
    if not connected:
        print("❌ No se pudo conectar a TWS. Asegúrate de que esté ejecutándose.")
        return
    
    try:
        # Obtener info de cuenta
        account_info = await ibkr.get_account_info()
        print(f"\n💰 Información de cuenta:")
        for key, value in account_info.get('account_values', {}).items():
            if isinstance(value, float):
                print(f"  {key}: ${value:,.2f}")
        
        print(f"\n📊 Posiciones actuales:")
        for pos in account_info.get('positions', []):
            if pos['position'] != 0:
                print(f"  {pos['symbol']}: {pos['position']} shares @ ${pos['avg_cost']:.2f}")
                print(f"    P&L no realizado: ${pos['unrealized_pnl']:.2f}")
        
        # Obtener datos de mercado
        print(f"\n📈 Datos de mercado:")
        symbols = ["AAPL", "TSLA"]
        for symbol in symbols:
            data = await ibkr.get_market_data(symbol)
            if data:
                print(f"  {symbol}: ${data['last']:.2f} (bid: ${data['bid']:.2f}, ask: ${data['ask']:.2f})")
        
        # Demo de orden (comentado para seguridad)
        # order_id = await ibkr.place_order("AAPL", "BUY", 1, "LMT", 150.00)
        # if order_id:
        #     print(f"Orden colocada con ID: {order_id}")
        
    finally:
        # Desconectar
        ibkr.disconnect()

if __name__ == "__main__":
    asyncio.run(demo_ibkr_integration())
```

## Integración con QuantConnect

### Estrategia Híbrida Local/Cloud
```python
# Local strategy development que se puede portar a QuantConnect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class QuantConnectCompatibleStrategy:
    """Estrategia compatible con QuantConnect"""
    
    def __init__(self):
        self.name = "HybridMomentumStrategy"
        self.symbols = ["SPY", "QQQ", "IWM"]
        self.lookback_period = 20
        self.portfolio = {}
        self.universe = {}
        
        # Configuración
        self.config = {
            'rebalance_frequency': 'daily',
            'max_positions': 3,
            'risk_per_trade': 0.02,
            'momentum_threshold': 0.1
        }
    
    def initialize(self, data_manager):
        """Inicializar estrategia (compatible con QC.Initialize)"""
        self.data_manager = data_manager
        
        # Configurar universo
        for symbol in self.symbols:
            self.universe[symbol] = {
                'data': pd.DataFrame(),
                'indicators': {},
                'signals': []
            }
        
        print(f"✅ Estrategia {self.name} inicializada")
    
    async def on_data(self, data: Dict):
        """Procesar nuevos datos (compatible con QC.OnData)"""
        
        # Actualizar datos para cada símbolo
        for symbol, price_data in data.items():
            if symbol in self.universe:
                await self._update_symbol_data(symbol, price_data)
        
        # Generar señales
        signals = await self._generate_signals()
        
        # Ejecutar trades si hay señales
        if signals:
            await self._execute_signals(signals)
    
    async def _update_symbol_data(self, symbol: str, price_data: Dict):
        """Actualizar datos de símbolo"""
        
        # Crear nueva fila
        new_row = pd.DataFrame([{
            'timestamp': price_data['timestamp'],
            'open': price_data['open'],
            'high': price_data['high'],
            'low': price_data['low'],
            'close': price_data['close'],
            'volume': price_data['volume']
        }])
        
        # Agregar a datos existentes
        symbol_data = self.universe[symbol]['data']
        symbol_data = pd.concat([symbol_data, new_row], ignore_index=True)
        
        # Mantener solo los últimos N períodos
        if len(symbol_data) > self.lookback_period * 2:
            symbol_data = symbol_data.tail(self.lookback_period * 2)
        
        self.universe[symbol]['data'] = symbol_data
        
        # Actualizar indicadores
        await self._update_indicators(symbol)
    
    async def _update_indicators(self, symbol: str):
        """Actualizar indicadores técnicos"""
        
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
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def _generate_signals(self) -> List[Dict]:
        """Generar señales de trading"""
        
        signals = []
        
        for symbol in self.symbols:
            symbol_info = self.universe[symbol]
            data = symbol_info['data']
            indicators = symbol_info['indicators']
            
            if len(data) < self.lookback_period:
                continue
            
            # Obtener valores actuales
            current_price = data['close'].iloc[-1]
            current_rsi = indicators['rsi'].iloc[-1]
            current_momentum = indicators['momentum'].iloc[-1]
            current_macd = indicators['macd'].iloc[-1]
            current_signal = indicators['macd_signal'].iloc[-1]
            
            # Lógica de señales
            signal_strength = 0
            signal_type = None
            
            # Momentum positivo
            if current_momentum > self.config['momentum_threshold']:
                signal_strength += 30
            
            # RSI no sobrecomprado
            if 30 < current_rsi < 70:
                signal_strength += 20
            
            # MACD bullish
            if current_macd > current_signal:
                signal_strength += 25
            
            # Price above moving average
            if current_price > indicators['sma_20'].iloc[-1]:
                signal_strength += 25
            
            # Determinar tipo de señal
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
        """Ejecutar señales (placeholder para integración real)"""
        
        for signal in signals:
            symbol = signal['symbol']
            signal_type = signal['type']
            strength = signal['strength']
            
            print(f"🎯 Señal {signal_type} para {symbol} (fuerza: {strength})")
            
            # Aquí se integraría con el broker real
            # await self.broker.place_order(...)
    
    def get_performance_metrics(self) -> Dict:
        """Obtener métricas de performance"""
        
        # Placeholder - en implementación real calcularía métricas reales
        return {
            'strategy_name': self.name,
            'total_signals': sum(len(info['signals']) for info in self.universe.values()),
            'symbols_tracked': len(self.symbols),
            'last_update': datetime.now()
        }

# Código para QuantConnect (archivo separado: main.py)
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

# Utilidad para sincronizar estrategias
class StrategySync:
    """Sincronizar estrategia entre local y QuantConnect"""
    
    def __init__(self, local_strategy: QuantConnectCompatibleStrategy):
        self.local_strategy = local_strategy
        self.qc_code_template = QUANTCONNECT_CODE
    
    def export_to_quantconnect(self, filename: str = "main.py"):
        """Exportar estrategia a formato QuantConnect"""
        
        # Personalizar código basado en configuración local
        qc_code = self.qc_code_template
        
        # Reemplazar parámetros
        qc_code = qc_code.replace(
            "self.momentum_threshold = 0.1",
            f"self.momentum_threshold = {self.local_strategy.config['momentum_threshold']}"
        )
        
        qc_code = qc_code.replace(
            "self.max_positions = 3",
            f"self.max_positions = {self.local_strategy.config['max_positions']}"
        )
        
        # Guardar archivo
        with open(filename, 'w') as f:
            f.write(qc_code)
        
        print(f"✅ Estrategia exportada a {filename}")
        print("📁 Sube este archivo a tu proyecto en QuantConnect")
    
    def backtest_locally(self, start_date: str, end_date: str):
        """Ejecutar backtest local antes de usar en QC"""
        
        print(f"🔬 Ejecutando backtest local...")
        
        # Simular datos
        symbols = self.local_strategy.symbols
        
        # Esta sería la integración con tu data manager real
        # data = await self.data_manager.get_historical_data(symbols, start_date, end_date)
        
        print(f"📊 Backtest completado para período {start_date} a {end_date}")
        
        # Retornar métricas de ejemplo
        return {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'trades': 45
        }

# Demo de integración QuantConnect
async def demo_quantconnect_integration():
    """Demo de integración con QuantConnect"""
    
    print("🚀 Demo de integración con QuantConnect...")
    
    # Crear estrategia local
    strategy = QuantConnectCompatibleStrategy()
    
    # Simular inicialización (necesitarías tu data manager real)
    # await strategy.initialize(data_manager)
    
    # Simular algunos datos de ejemplo
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
    
    # Configurar sincronización
    sync = StrategySync(strategy)
    
    # Ejecutar backtest local
    backtest_results = sync.backtest_locally("2023-01-01", "2023-12-31")
    print(f"📈 Resultados del backtest:")
    for metric, value in backtest_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2%}" if 'return' in metric or 'drawdown' in metric else f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")
    
    # Exportar a QuantConnect
    sync.export_to_quantconnect("hybrid_momentum_strategy.py")
    
    print(f"\n📋 Próximos pasos:")
    print("1. Revisa el archivo hybrid_momentum_strategy.py")
    print("2. Sube el archivo a tu proyecto en QuantConnect")
    print("3. Ejecuta el backtest en la plataforma")
    print("4. Compara resultados con el backtest local")

if __name__ == "__main__":
    asyncio.run(demo_quantconnect_integration())
```

## Dashboard de Monitoreo Multi-Plataforma

### Sistema de Monitoreo Centralizado
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
    """Dashboard para monitorear múltiples plataformas"""
    
    def __init__(self):
        self.data_sources = {}
        self.trading_accounts = {}
        self.strategies = {}
        self.alerts = []
    
    def add_data_source(self, name: str, provider):
        """Agregar fuente de datos"""
        self.data_sources[name] = provider
    
    def add_trading_account(self, name: str, account_integration):
        """Agregar cuenta de trading"""
        self.trading_accounts[name] = account_integration
    
    def add_strategy(self, name: str, strategy):
        """Agregar estrategia"""
        self.strategies[name] = strategy
    
    async def get_unified_portfolio_view(self) -> Dict:
        """Obtener vista unificada del portfolio"""
        
        portfolio_summary = {
            'total_equity': 0,
            'total_pnl': 0,
            'positions': [],
            'account_breakdown': {}
        }
        
        # Agregar datos de cada cuenta
        for account_name, account in self.trading_accounts.items():
            try:
                account_info = await account.get_account_info()
                
                account_equity = account_info.get('account_values', {}).get('NetLiquidation', 0)
                portfolio_summary['total_equity'] += account_equity
                
                # Agregar posiciones
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
                st.error(f"Error obteniendo datos de {account_name}: {e}")
        
        return portfolio_summary
    
    async def get_data_feed_status(self) -> Dict:
        """Obtener estado de feeds de datos"""
        
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
        """Obtener performance de estrategias"""
        
        strategy_performance = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'get_performance_metrics'):
                    metrics = strategy.get_performance_metrics()
                    strategy_performance[strategy_name] = metrics
                else:
                    # Mock metrics para demo
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
    """Crear dashboard con Streamlit"""
    
    st.set_page_config(
        page_title="Trading Multi-Platform Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Multi-Platform Trading Dashboard")
    
    # Inicializar dashboard (normalmente esto estaría en session_state)
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = MultiPlatformDashboard()
        
        # Agregar fuentes mock para demo
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
    
    # Sidebar para controles
    with st.sidebar:
        st.header("🔧 Controles")
        
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Intervalo (seg)", 10, 300, 30)
        
        if st.button("🔄 Refresh Manual"):
            st.rerun()
        
        st.header("📡 Data Sources")
        for source, info in dashboard.data_sources.items():
            status_icon = "🟢" if info.get('status') == 'online' else "🔴"
            st.write(f"{status_icon} {source}")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    # Mock data para demo
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
    
    # Gráficos principales
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
    
    # Posiciones actuales
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
    
    # Alerts y notificaciones
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

# Ejecutar dashboard
if __name__ == "__main__":
    create_streamlit_dashboard()
```

Este sistema de integración multi-plataforma proporciona una base sólida para conectar y coordinar diferentes fuentes de datos, brokers y estrategias en un workflow unificado de trading cuantitativo.