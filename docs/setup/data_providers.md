# Configuración de Data Providers

## Yahoo Finance - Free Tier

### Ventajas y Limitaciones
- ✅ **Gratuito** para uso personal
- ✅ **Históricos** disponibles por años
- ✅ **Fácil implementación** con yfinance
- ❌ **Rate limits** no documentados
- ❌ **No real-time** verdadero (15-20 min delay)
- ❌ **No level 2** data

### Implementación Yahoo Finance

```python
# src/data_acquisition/yahoo_provider.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional

class YahooDataProvider:
    def __init__(self, rate_limit_delay=1.0):
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Control de rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def get_historical_data(self, symbol: str, period: str = "1y", 
                           interval: str = "1d") -> pd.DataFrame:
        """Obtener datos históricos"""
        self._rate_limit()
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logging.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Limpiar nombres de columnas
            data.columns = [col.lower() for col in data.columns]
            
            # Agregar símbolo
            data['symbol'] = symbol
            
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_symbols(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Obtener datos para múltiples símbolos"""
        results = {}
        
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            data = self.get_historical_data(symbol, period)
            if not data.empty:
                results[symbol] = data
            
            # Rate limiting entre requests
            time.sleep(self.rate_limit_delay)
        
        return results
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtener precio actual (delayed)"""
        self._rate_limit()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice') or info.get('previousClose')
        except Exception as e:
            logging.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """Obtener datos fundamentales"""
        self._rate_limit()
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'market_cap': info.get('marketCap'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'revenue': info.get('totalRevenue'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'exchange': info.get('exchange'),
                'currency': info.get('currency')
            }
        except Exception as e:
            logging.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}
    
    def scan_by_criteria(self, criteria: Dict) -> List[str]:
        """Screening básico usando Yahoo Finance"""
        # Yahoo Finance no tiene screening API público
        # Esta es una implementación básica usando símbolos conocidos
        
        popular_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
            'NFLX', 'CRM', 'PYPL', 'SHOP', 'SQ', 'ROKU', 'ZOOM', 'DOCU'
        ]
        
        filtered_symbols = []
        
        for symbol in popular_symbols:
            try:
                fundamentals = self.get_fundamentals(symbol)
                
                # Aplicar criterios básicos
                if criteria.get('min_market_cap'):
                    if not fundamentals.get('market_cap') or fundamentals['market_cap'] < criteria['min_market_cap']:
                        continue
                
                if criteria.get('max_market_cap'):
                    if not fundamentals.get('market_cap') or fundamentals['market_cap'] > criteria['max_market_cap']:
                        continue
                
                filtered_symbols.append(symbol)
                
            except Exception:
                continue
        
        return filtered_symbols

# Ejemplo de uso
def demo_yahoo_provider():
    """Demo del provider de Yahoo Finance"""
    provider = YahooDataProvider()
    
    # Obtener datos históricos
    print("📊 Obteniendo datos históricos de AAPL...")
    aapl_data = provider.get_historical_data('AAPL', period='3mo', interval='1d')
    print(f"Datos obtenidos: {len(aapl_data)} días")
    print(aapl_data.tail())
    
    # Obtener precio actual
    print(f"\n💰 Precio actual AAPL: ${provider.get_current_price('AAPL')}")
    
    # Obtener fundamentales
    print(f"\n📈 Fundamentales AAPL:")
    fundamentals = provider.get_fundamentals('AAPL')
    for key, value in fundamentals.items():
        if value:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    demo_yahoo_provider()
```

## Polygon.io - Professional Tier

### Ventajas
- ✅ **Real-time data** sub-second
- ✅ **Level 2** market data
- ✅ **Extensive API** con múltiples endpoints
- ✅ **WebSocket** feeds para streaming
- ✅ **Historical** data de alta calidad
- 💰 **Paid service** ($99-249/month)

### Implementación Polygon.io

```python
# src/data_acquisition/polygon_provider.py
from polygon import RESTClient
import pandas as pd
import asyncio
import websocket
import json
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Callable

class PolygonDataProvider:
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)
        self.api_key = api_key
        self.ws = None
        self.ws_callbacks = {}
        
    def get_historical_bars(self, symbol: str, timespan: str = "day", 
                           from_date: str = None, to_date: str = None,
                           limit: int = 5000) -> pd.DataFrame:
        """Obtener barras históricas"""
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            bars = self.client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                limit=limit
            )
            
            data = []
            for bar in bars:
                data.append({
                    'timestamp': pd.to_datetime(bar.timestamp, unit='ms'),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap,
                    'transactions': bar.transactions
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching bars for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Obtener quote en tiempo real"""
        try:
            quote = self.client.get_last_quote(ticker=symbol)
            
            return {
                'symbol': symbol,
                'bid': quote.bid,
                'ask': quote.ask,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': pd.to_datetime(quote.timestamp, unit='ns'),
                'exchange': quote.exchange
            }
        except Exception as e:
            logging.error(f"Error fetching quote for {symbol}: {e}")
            return {}
    
    def get_real_time_trade(self, symbol: str) -> Dict:
        """Obtener último trade"""
        try:
            trade = self.client.get_last_trade(ticker=symbol)
            
            return {
                'symbol': symbol,
                'price': trade.price,
                'size': trade.size,
                'timestamp': pd.to_datetime(trade.timestamp, unit='ns'),
                'exchange': trade.exchange,
                'conditions': trade.conditions
            }
        except Exception as e:
            logging.error(f"Error fetching trade for {symbol}: {e}")
            return {}
    
    def get_stock_details(self, symbol: str) -> Dict:
        """Obtener detalles del stock"""
        try:
            details = self.client.get_ticker_details(symbol)
            
            return {
                'symbol': symbol,
                'name': details.name,
                'market_cap': details.market_cap,
                'shares_outstanding': details.share_class_shares_outstanding,
                'weighted_shares_outstanding': details.weighted_shares_outstanding,
                'description': details.description,
                'homepage_url': details.homepage_url,
                'logo_url': details.branding.logo_url if details.branding else None,
                'primary_exchange': details.primary_exchange,
                'type': details.type,
                'currency_name': details.currency_name,
                'cik': details.cik,
                'composite_figi': details.composite_figi,
                'phone_number': details.phone_number
            }
        except Exception as e:
            logging.error(f"Error fetching details for {symbol}: {e}")
            return {}
    
    def get_market_holidays(self, year: int = None) -> List[Dict]:
        """Obtener días feriados del mercado"""
        if not year:
            year = datetime.now().year
            
        try:
            holidays = self.client.get_market_holidays()
            return [
                {
                    'date': holiday.date,
                    'name': holiday.name,
                    'status': holiday.status
                }
                for holiday in holidays
                if holiday.date.startswith(str(year))
            ]
        except Exception as e:
            logging.error(f"Error fetching market holidays: {e}")
            return []
    
    def get_market_status(self) -> Dict:
        """Obtener status del mercado"""
        try:
            status = self.client.get_market_status()
            
            return {
                'market': status.market,
                'server_time': status.serverTime,
                'exchanges': {
                    'nasdaq': status.exchanges.nasdaq,
                    'nyse': status.exchanges.nyse,
                    'otc': status.exchanges.otc
                },
                'currencies': {
                    'fx': status.currencies.fx,
                    'crypto': status.currencies.crypto
                }
            }
        except Exception as e:
            logging.error(f"Error fetching market status: {e}")
            return {}
    
    def search_tickers(self, search_term: str, market: str = "stocks", 
                      active: bool = True, limit: int = 100) -> List[Dict]:
        """Buscar tickers"""
        try:
            tickers = self.client.list_tickers(
                market=market,
                search=search_term,
                active=active,
                limit=limit
            )
            
            results = []
            for ticker in tickers:
                results.append({
                    'ticker': ticker.ticker,
                    'name': ticker.name,
                    'market': ticker.market,
                    'locale': ticker.locale,
                    'primary_exchange': ticker.primary_exchange,
                    'type': ticker.type,
                    'active': ticker.active,
                    'currency_name': ticker.currency_name,
                    'cik': ticker.cik,
                    'composite_figi': ticker.composite_figi
                })
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching tickers: {e}")
            return []
    
    def setup_websocket(self, symbols: List[str], callback: Callable):
        """Configurar WebSocket para datos en tiempo real"""
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                callback(data)
            except Exception as e:
                logging.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logging.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logging.info("WebSocket connection closed")
        
        def on_open(ws):
            logging.info("WebSocket connection opened")
            
            # Autenticarse
            auth_msg = {
                "action": "auth",
                "params": self.api_key
            }
            ws.send(json.dumps(auth_msg))
            
            # Suscribirse a símbolos
            subscribe_msg = {
                "action": "subscribe",
                "params": ",".join([f"T.{symbol}" for symbol in symbols])  # Trades
            }
            ws.send(json.dumps(subscribe_msg))
        
        # Crear conexión WebSocket
        ws_url = "wss://socket.polygon.io/stocks"
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        return self.ws
    
    def start_real_time_feed(self, symbols: List[str], callback: Callable):
        """Iniciar feed de datos en tiempo real"""
        ws = self.setup_websocket(symbols, callback)
        ws.run_forever()

# Ejemplo de uso
def demo_polygon_provider():
    """Demo del provider de Polygon"""
    from config.api_keys import APIKeys
    
    if not APIKeys.POLYGON_API_KEY:
        print("❌ Polygon API key no configurada")
        return
    
    provider = PolygonDataProvider(APIKeys.POLYGON_API_KEY)
    
    # Test market status
    print("📊 Market Status:")
    status = provider.get_market_status()
    print(f"Server time: {status.get('server_time')}")
    print(f"NASDAQ: {status.get('exchanges', {}).get('nasdaq')}")
    
    # Test historical data
    print("\n📈 Historical Data (AAPL):")
    historical = provider.get_historical_bars('AAPL', timespan='hour', 
                                             from_date='2024-01-01', limit=10)
    print(historical.head())
    
    # Test real-time quote
    print(f"\n💰 Real-time Quote (AAPL):")
    quote = provider.get_real_time_quote('AAPL')
    print(f"Bid: ${quote.get('bid')}, Ask: ${quote.get('ask')}")
    
    # Test stock details
    print(f"\n📋 Stock Details (AAPL):")
    details = provider.get_stock_details('AAPL')
    print(f"Name: {details.get('name')}")
    print(f"Market Cap: ${details.get('market_cap'):,}" if details.get('market_cap') else "Market Cap: N/A")

if __name__ == "__main__":
    demo_polygon_provider()
```

## IEX Cloud - Balanced Option

### Implementación IEX Cloud

```python
# src/data_acquisition/iex_provider.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional

class IEXCloudProvider:
    def __init__(self, api_key: str, base_url: str = "https://cloud.iexapis.com/stable"):
        self.api_key = api_key
        self.base_url = base_url
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Hacer request a IEX API"""
        if params is None:
            params = {}
        
        params['token'] = self.api_key
        
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error in IEX request to {endpoint}: {e}")
            return {}
    
    def get_quote(self, symbol: str) -> Dict:
        """Obtener quote actual"""
        data = self._make_request(f"stock/{symbol}/quote")
        
        if data:
            return {
                'symbol': symbol,
                'latest_price': data.get('latestPrice'),
                'change': data.get('change'),
                'change_percent': data.get('changePercent'),
                'volume': data.get('volume'),
                'avg_volume': data.get('avgTotalVolume'),
                'market_cap': data.get('marketCap'),
                'pe_ratio': data.get('peRatio'),
                'week_52_high': data.get('week52High'),
                'week_52_low': data.get('week52Low'),
                'latest_update': data.get('latestUpdate')
            }
        
        return {}
    
    def get_historical_prices(self, symbol: str, range_period: str = "1y") -> pd.DataFrame:
        """Obtener precios históricos"""
        data = self._make_request(f"stock/{symbol}/chart/{range_period}")
        
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['symbol'] = symbol
            return df
        
        return pd.DataFrame()
    
    def get_intraday_prices(self, symbol: str) -> pd.DataFrame:
        """Obtener precios intraday"""
        data = self._make_request(f"stock/{symbol}/intraday-prices")
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['minute'])
                df.set_index('datetime', inplace=True)
                df['symbol'] = symbol
            return df
        
        return pd.DataFrame()
    
    def get_company_info(self, symbol: str) -> Dict:
        """Obtener información de la empresa"""
        data = self._make_request(f"stock/{symbol}/company")
        
        return {
            'symbol': symbol,
            'company_name': data.get('companyName'),
            'exchange': data.get('exchange'),
            'industry': data.get('industry'),
            'sector': data.get('sector'),
            'website': data.get('website'),
            'description': data.get('description'),
            'ceo': data.get('CEO'),
            'employees': data.get('employees'),
            'address': {
                'address': data.get('address'),
                'city': data.get('city'),
                'state': data.get('state'),
                'zip': data.get('zip'),
                'country': data.get('country')
            }
        }
    
    def get_key_stats(self, symbol: str) -> Dict:
        """Obtener estadísticas clave"""
        data = self._make_request(f"stock/{symbol}/advanced-stats")
        
        return {
            'symbol': symbol,
            'market_cap': data.get('marketcap'),
            'enterprise_value': data.get('enterpriseValue'),
            'pe_ratio': data.get('peRatio'),
            'peg_ratio': data.get('pegRatio'),
            'price_to_book': data.get('priceToBook'),
            'price_to_sales': data.get('priceToSales'),
            'ev_to_revenue': data.get('enterpriseValueToRevenue'),
            'ev_to_ebitda': data.get('enterpriseValueToEBITDA'),
            'profit_margin': data.get('profitMargin'),
            'operating_margin': data.get('operatingMargin'),
            'return_on_assets': data.get('returnOnAssets'),
            'return_on_equity': data.get('returnOnEquity'),
            'revenue': data.get('revenue'),
            'gross_profit': data.get('grossProfit'),
            'ebitda': data.get('EBITDA'),
            'revenue_per_share': data.get('revenuePerShare'),
            'debt_to_equity': data.get('debtToEquity'),
            'current_ratio': data.get('currentRatio'),
            'shares_outstanding': data.get('sharesOutstanding'),
            'float': data.get('float'),
            'avg_10_day_volume': data.get('avg10Volume'),
            'avg_30_day_volume': data.get('avg30Volume'),
            'day_200_moving_avg': data.get('day200MovingAvg'),
            'day_50_moving_avg': data.get('day50MovingAvg'),
            'max_change_percent': data.get('maxChangePercent'),
            'year_5_change_percent': data.get('year5ChangePercent'),
            'year_2_change_percent': data.get('year2ChangePercent'),
            'year_1_change_percent': data.get('year1ChangePercent'),
            'ytd_change_percent': data.get('ytdChangePercent'),
            'month_6_change_percent': data.get('month6ChangePercent'),
            'month_3_change_percent': data.get('month3ChangePercent'),
            'month_1_change_percent': data.get('month1ChangePercent'),
            'day_30_change_percent': data.get('day30ChangePercent'),
            'day_5_change_percent': data.get('day5ChangePercent')
        }
    
    def get_news(self, symbol: str, count: int = 10) -> List[Dict]:
        """Obtener noticias"""
        data = self._make_request(f"stock/{symbol}/news/last/{count}")
        
        news = []
        for item in data:
            news.append({
                'datetime': pd.to_datetime(item.get('datetime'), unit='ms'),
                'headline': item.get('headline'),
                'source': item.get('source'),
                'url': item.get('url'),
                'summary': item.get('summary'),
                'related': item.get('related'),
                'image': item.get('image')
            })
        
        return news
    
    def search_symbols(self, query: str) -> List[Dict]:
        """Buscar símbolos"""
        data = self._make_request(f"search/{query}")
        
        results = []
        for item in data:
            results.append({
                'symbol': item.get('symbol'),
                'security_name': item.get('securityName'),
                'security_type': item.get('securityType'),
                'region': item.get('region'),
                'exchange': item.get('exchange')
            })
        
        return results
    
    def get_gainers_losers(self, list_type: str = "gainers") -> List[Dict]:
        """Obtener gainers/losers"""
        data = self._make_request(f"stock/market/list/{list_type}")
        
        results = []
        for item in data:
            results.append({
                'symbol': item.get('symbol'),
                'company_name': item.get('companyName'),
                'primary_exchange': item.get('primaryExchange'),
                'latest_price': item.get('latestPrice'),
                'change': item.get('change'),
                'change_percent': item.get('changePercent'),
                'volume': item.get('volume')
            })
        
        return results

# Unified Data Provider
class UnifiedDataProvider:
    """Provider unificado que combina múltiples fuentes"""
    
    def __init__(self):
        self.providers = {}
        self.primary_provider = None
        
    def add_provider(self, name: str, provider, is_primary: bool = False):
        """Agregar proveedor de datos"""
        self.providers[name] = provider
        if is_primary:
            self.primary_provider = name
    
    def get_quote(self, symbol: str, provider_name: str = None) -> Dict:
        """Obtener quote con fallback automático"""
        provider_name = provider_name or self.primary_provider
        
        # Intentar con provider primario
        if provider_name in self.providers:
            try:
                return self.providers[provider_name].get_quote(symbol)
            except Exception as e:
                logging.warning(f"Error with primary provider {provider_name}: {e}")
        
        # Fallback a otros providers
        for name, provider in self.providers.items():
            if name != provider_name:
                try:
                    result = provider.get_quote(symbol)
                    if result:
                        result['data_source'] = name
                        return result
                except Exception:
                    continue
        
        return {}
    
    def get_historical_data(self, symbol: str, period: str = "1y", 
                           provider_name: str = None) -> pd.DataFrame:
        """Obtener datos históricos con fallback"""
        provider_name = provider_name or self.primary_provider
        
        # Mapear períodos entre providers
        period_mapping = {
            'yahoo': {'1y': '1y', '6mo': '6mo', '3mo': '3mo'},
            'polygon': {'1y': 'day', '6mo': 'day', '3mo': 'day'},
            'iex': {'1y': '1y', '6mo': '6m', '3mo': '3m'}
        }
        
        for name, provider in self.providers.items():
            if provider_name and name != provider_name:
                continue
                
            try:
                if hasattr(provider, 'get_historical_data'):
                    return provider.get_historical_data(symbol, period)
                elif hasattr(provider, 'get_historical_prices'):
                    mapped_period = period_mapping.get(name, {}).get(period, period)
                    return provider.get_historical_prices(symbol, mapped_period)
                elif hasattr(provider, 'get_historical_bars'):
                    return provider.get_historical_bars(symbol)
            except Exception as e:
                logging.warning(f"Error with provider {name}: {e}")
                continue
        
        return pd.DataFrame()
```

Esta configuración te permite usar múltiples data providers con fallback automático, optimizando costos y reliability.