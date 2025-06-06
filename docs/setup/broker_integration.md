# Integración con Brokers

## Interactive Brokers (IBKR) - Recomendado Principal

### Ventajas de IBKR
- **Comisiones bajas**: $0.005 por acción, mínimo $1 por trade
- **API robusta**: TWS API con múltiples lenguajes
- **Datos en tiempo real**: Level 1 y Level 2 market data
- **Short selling**: Amplio inventario para locates
- **Margin requirements**: Competitive margin rates
- **Global access**: Acceso a múltiples mercados

### Setup de TWS (Trader Workstation)

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
        """Conectar a TWS"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logging.info(f"✅ Conectado a IBKR TWS en {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"❌ Error conectando a IBKR: {e}")
            return False
    
    def disconnect(self):
        """Desconectar de TWS"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logging.info("🔌 Desconectado de IBKR TWS")
    
    def get_account_info(self):
        """Obtener información de cuenta"""
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
        """Obtener buying power disponible"""
        account_values = self.ib.accountValues()
        for item in account_values:
            if item.tag == 'BuyingPower':
                return float(item.value)
        return 0
    
    def get_total_liquidity(self):
        """Obtener liquidez total"""
        account_values = self.ib.accountValues()
        for item in account_values:
            if item.tag == 'TotalCashValue':
                return float(item.value)
        return 0
    
    def get_market_data(self, symbol, exchange='SMART'):
        """Obtener market data en tiempo real"""
        contract = Stock(symbol, exchange)
        self.ib.reqMktData(contract)
        
        # Esperar a que lleguen los datos
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
        """Colocar orden de mercado"""
        contract = Stock(symbol, 'SMART')
        order = MarketOrder(action, abs(quantity))
        
        trade = self.ib.placeOrder(contract, order)
        
        return {
            'trade': trade,
            'order_id': trade.order.orderId,
            'status': trade.orderStatus.status
        }
    
    def place_limit_order(self, symbol, quantity, price, action='BUY'):
        """Colocar orden límite"""
        contract = Stock(symbol, 'SMART')
        order = LimitOrder(action, abs(quantity), price)
        
        trade = self.ib.placeOrder(contract, order)
        
        return {
            'trade': trade,
            'order_id': trade.order.orderId,
            'status': trade.orderStatus.status
        }
    
    def cancel_order(self, order_id):
        """Cancelar orden"""
        try:
            self.ib.cancelOrder(order_id)
            return True
        except Exception as e:
            logging.error(f"Error cancelando orden {order_id}: {e}")
            return False
    
    def get_historical_data(self, symbol, duration='1 Y', bar_size='1 day'):
        """Obtener datos históricos"""
        contract = Stock(symbol, 'SMART')
        
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )
        
        # Convertir a DataFrame
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

# Ejemplo de uso
def test_ibkr_connection():
    """Test de conexión IBKR"""
    ibkr = IBKRConnection()
    
    if ibkr.connect():
        # Test account info
        account_info = ibkr.get_account_info()
        print(f"💰 Buying Power: ${account_info['buying_power']:,.2f}")
        
        # Test market data
        spy_data = ibkr.get_market_data('SPY')
        print(f"📊 SPY: Bid={spy_data['bid']}, Ask={spy_data['ask']}")
        
        # Test historical data
        historical = ibkr.get_historical_data('SPY', '1 M', '1 hour')
        print(f"📈 Historical data: {len(historical)} bars")
        
        ibkr.disconnect()
        return True
    
    return False
```

### Configuración TWS

```python
# scripts/setup_tws.py
"""
Script para configurar TWS para API trading
"""

TWS_CONFIGURATION = """
⚙️  Configuración Manual de TWS:

1. 📥 Descargar TWS:
   - Ir a https://www.interactivebrokers.com/en/trading/tws.php
   - Descargar Trader Workstation

2. 🔧 Configurar API:
   - Abrir TWS
   - Ir a Configure → API → Settings
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Read-Only API
   - Socket port: 7497 (paper) / 7496 (live)
   - Master API client ID: 0
   - ✅ Download open orders on connection

3. 🔐 Configurar Paper Trading:
   - Ir a Configure → API → Settings
   - ✅ Enable API
   - Puerto: 7497
   - IP permitidas: 127.0.0.1

4. ⚡ Market Data:
   - Ir a Configure → Market Data Subscriptions
   - Activar US Securities Snapshot and Futures Value Bundle (gratis)
   - Para Level 2: US Equity and Options Add-On Streaming Bundle

5. 🚨 Configurar Alertas:
   - Configure → Alerts
   - ✅ Enable Popup alerts
   - ✅ Email alerts (opcional)

⚠️  IMPORTANTE:
- Usar Paper Trading account inicialmente
- TWS debe estar abierto y conectado para usar API
- Verificar que el puerto esté correcto (7497 paper / 7496 live)
"""

print(TWS_CONFIGURATION)
```

## Alpaca - Alternative Broker

### Ventajas de Alpaca
- **Commission-free**: Sin comisiones en stocks
- **API-first**: Diseñado para trading algorítmico
- **Paper trading**: Sandbox environment robusto
- **Modern REST API**: Fácil de usar
- **Real-time data**: WebSocket feeds

### Setup Alpaca

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
        """Obtener información de cuenta"""
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
        """Obtener posiciones actuales"""
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
        """Obtener órdenes"""
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
        """Colocar orden"""
        
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
        """Cancelar orden"""
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"Error cancelando orden: {e}")
            return False
    
    def get_historical_data(self, symbol, timeframe='1Day', start=None, end=None):
        """Obtener datos históricos"""
        if not start:
            start = datetime.now() - timedelta(days=365)
        if not end:
            end = datetime.now()
        
        # Convertir a formato Alpaca
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        barset = self.api.get_bars(
            symbol,
            timeframe,
            start=start_str,
            end=end_str,
            adjustment='raw'
        )
        
        # Convertir a DataFrame
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
        """Obtener quote actual"""
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
            print(f"Error obteniendo quote para {symbol}: {e}")
            return None

# Test de conexión Alpaca
def test_alpaca_connection():
    """Test de conexión Alpaca"""
    from config.api_keys import APIKeys
    
    if not APIKeys.ALPACA_API_KEY or not APIKeys.ALPACA_SECRET_KEY:
        print("❌ API Keys de Alpaca no configuradas")
        return False
    
    alpaca = AlpacaConnection(
        APIKeys.ALPACA_API_KEY,
        APIKeys.ALPACA_SECRET_KEY,
        APIKeys.ALPACA_BASE_URL
    )
    
    try:
        # Test account
        account = alpaca.get_account()
        print(f"✅ Cuenta Alpaca conectada")
        print(f"💰 Buying Power: ${account['buying_power']:,.2f}")
        print(f"📊 Portfolio Value: ${account['portfolio_value']:,.2f}")
        
        # Test quote
        quote = alpaca.get_quote('SPY')
        if quote:
            print(f"📈 SPY Quote: ${quote['bid']} x ${quote['ask']}")
        
        # Test historical data
        historical = alpaca.get_historical_data('SPY', '1Day')
        print(f"📊 Historical data: {len(historical)} days")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en conexión Alpaca: {e}")
        return False
```

## TD Ameritrade - Backup Option

### Setup TD Ameritrade

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
        """Autenticar con TD Ameritrade"""
        if self.refresh_token:
            return self._refresh_access_token()
        else:
            print("⚠️  Necesitas configurar OAuth flow para TD Ameritrade")
            return False
    
    def _refresh_access_token(self):
        """Renovar access token"""
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
            print(f"❌ Error renovando token: {response.status_code}")
            return False
    
    def get_quote(self, symbol):
        """Obtener quote"""
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
        """Obtener datos históricos"""
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
            
            # Convertir a DataFrame
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
    """Estructura de orden unificada"""
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'day'

@dataclass
class Position:
    """Estructura de posición unificada"""
    symbol: str
    quantity: int
    avg_price: float
    market_value: float
    unrealized_pnl: float

class BrokerInterface(ABC):
    """Interface abstracta para brokers"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Conectar al broker"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Desconectar del broker"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """Obtener información de cuenta"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Obtener posiciones actuales"""
        pass
    
    @abstractmethod
    def place_order(self, order: OrderRequest) -> Dict:
        """Colocar orden"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancelar orden"""
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str) -> Dict:
        """Obtener market data"""
        pass

class UnifiedBroker:
    """Broker unificado que maneja múltiples brokers"""
    
    def __init__(self):
        self.brokers = {}
        self.primary_broker = None
        
    def add_broker(self, name: str, broker: BrokerInterface, is_primary=False):
        """Agregar broker"""
        self.brokers[name] = broker
        if is_primary:
            self.primary_broker = name
    
    def execute_order(self, order: OrderRequest, broker_name=None):
        """Ejecutar orden en broker específico o primario"""
        broker_name = broker_name or self.primary_broker
        
        if broker_name not in self.brokers:
            return {'success': False, 'error': f'Broker {broker_name} no encontrado'}
        
        try:
            result = self.brokers[broker_name].place_order(order)
            return result
        except Exception as e:
            # Fallback a otro broker si falla
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
        """Obtener posiciones consolidadas de todos los brokers"""
        all_positions = {}
        
        for broker_name, broker in self.brokers.items():
            try:
                positions = broker.get_positions()
                all_positions[broker_name] = positions
            except Exception as e:
                print(f"Error obteniendo posiciones de {broker_name}: {e}")
        
        return all_positions
```

Esta infraestructura de brokers te permitirá comenzar con paper trading en IBKR o Alpaca, y luego expandir a trading real con múltiples brokers según tus necesidades.