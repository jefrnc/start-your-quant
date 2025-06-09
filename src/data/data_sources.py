"""
Data Sources Implementation
Complementa: docs/data/data_sources.md
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod


class DataSource(ABC):
    """Clase base para fuentes de datos"""
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Obtiene datos históricos para un símbolo"""
        pass
    
    @abstractmethod
    def get_real_time_data(self, symbol: str) -> Dict:
        """Obtiene datos en tiempo real para un símbolo"""
        pass


class YahooFinanceAPI(DataSource):
    """Implementación para Yahoo Finance (simulada)"""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
    
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Obtiene datos históricos de Yahoo Finance
        
        Args:
            symbol: Símbolo del activo (ej: 'AAPL')
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            
        Returns:
            DataFrame con datos OHLCV
        """
        # NOTA: Esta es una implementación simulada para demostración
        # En un entorno real, usarías yfinance o la API oficial
        
        print(f"Simulando obtención de datos para {symbol} desde {start_date} hasta {end_date}")
        
        # Generar datos simulados
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='D')
        
        # Filtrar solo días de semana (trading days)
        dates = dates[dates.dayofweek < 5]
        
        np.random.seed(hash(symbol) % 2**32)  # Semilla basada en el símbolo
        
        # Generar precios simulados con tendencia
        n_days = len(dates)
        returns = np.random.normal(0.001, 0.02, n_days)  # Retornos diarios
        
        prices = [100.0]  # Precio inicial
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Crear datos OHLCV
        data = []
        for i, date in enumerate(dates):
            close = prices[i + 1]
            open_price = prices[i] * (1 + np.random.normal(0, 0.005))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(100000, 10000000)
            
            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """Obtiene datos en tiempo real simulados"""
        # Generar datos en tiempo real simulados
        base_price = 100 + np.random.normal(0, 10)
        
        return {
            'symbol': symbol,
            'price': round(base_price, 2),
            'bid': round(base_price - 0.01, 2),
            'ask': round(base_price + 0.01, 2),
            'volume': np.random.randint(1000, 100000),
            'timestamp': datetime.now(),
            'change': round(np.random.normal(0, 1), 2),
            'change_percent': round(np.random.normal(0, 0.02) * 100, 2)
        }


class AlphaVantageAPI(DataSource):
    """Implementación para Alpha Vantage (simulada)"""
    
    def __init__(self, api_key: str = "demo"):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Obtiene datos históricos de Alpha Vantage (simulado)"""
        print(f"Simulando Alpha Vantage para {symbol}")
        
        # Similar a Yahoo Finance pero con diferentes características
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='D')
        dates = dates[dates.dayofweek < 5]
        
        # Usar semilla diferente para Alpha Vantage
        np.random.seed((hash(symbol) + 12345) % 2**32)
        
        n_days = len(dates)
        returns = np.random.normal(0.0005, 0.018, n_days)
        
        prices = [105.0]  # Precio inicial diferente
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, date in enumerate(dates):
            close = prices[i + 1]
            open_price = prices[i] * (1 + np.random.normal(0, 0.003))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.008)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.008)))
            volume = np.random.randint(50000, 5000000)
            
            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """Obtiene datos en tiempo real de Alpha Vantage (simulado)"""
        base_price = 105 + np.random.normal(0, 8)
        
        return {
            'symbol': symbol,
            'price': round(base_price, 2),
            'bid': round(base_price - 0.02, 2),
            'ask': round(base_price + 0.02, 2),
            'volume': np.random.randint(1000, 80000),
            'timestamp': datetime.now(),
            'change': round(np.random.normal(0, 0.8), 2),
            'change_percent': round(np.random.normal(0, 0.015) * 100, 2)
        }


class DataManager:
    """Gestor centralizado de datos que puede usar múltiples fuentes"""
    
    def __init__(self):
        self.sources = {}
        self.cache = {}
        self.cache_timeout = 300  # 5 minutos
    
    def add_source(self, name: str, source: DataSource):
        """Añade una fuente de datos"""
        self.sources[name] = source
    
    def get_data(self, symbol: str, start_date: str, end_date: str, 
                source_name: str = None) -> pd.DataFrame:
        """
        Obtiene datos usando una fuente específica o la primera disponible
        
        Args:
            symbol: Símbolo del activo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            source_name: Nombre de la fuente (opcional)
            
        Returns:
            DataFrame con datos históricos
        """
        if source_name and source_name in self.sources:
            return self.sources[source_name].get_data(symbol, start_date, end_date)
        elif self.sources:
            # Usar la primera fuente disponible
            first_source = next(iter(self.sources.values()))
            return first_source.get_data(symbol, start_date, end_date)
        else:
            raise ValueError("No hay fuentes de datos configuradas")
    
    def get_real_time_data(self, symbol: str, source_name: str = None) -> Dict:
        """Obtiene datos en tiempo real"""
        cache_key = f"{symbol}_realtime"
        current_time = datetime.now()
        
        # Verificar cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (current_time - cached_time).seconds < self.cache_timeout:
                return cached_data
        
        # Obtener datos frescos
        if source_name and source_name in self.sources:
            data = self.sources[source_name].get_real_time_data(symbol)
        elif self.sources:
            first_source = next(iter(self.sources.values()))
            data = first_source.get_real_time_data(symbol)
        else:
            raise ValueError("No hay fuentes de datos configuradas")
        
        # Guardar en cache
        self.cache[cache_key] = (data, current_time)
        return data
    
    def compare_sources(self, symbol: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Compara datos de múltiples fuentes"""
        results = {}
        for source_name, source in self.sources.items():
            try:
                data = source.get_data(symbol, start_date, end_date)
                results[source_name] = data
            except Exception as e:
                print(f"Error obteniendo datos de {source_name}: {e}")
                results[source_name] = None
        
        return results
    
    def get_multiple_symbols(self, symbols: List[str], start_date: str, 
                           end_date: str, source_name: str = None) -> Dict[str, pd.DataFrame]:
        """Obtiene datos para múltiples símbolos"""
        results = {}
        for symbol in symbols:
            try:
                data = self.get_data(symbol, start_date, end_date, source_name)
                results[symbol] = data
            except Exception as e:
                print(f"Error obteniendo datos para {symbol}: {e}")
                results[symbol] = None
        
        return results


def example_usage():
    """Ejemplo de uso del gestor de datos"""
    # Crear gestor de datos
    data_manager = DataManager()
    
    # Añadir fuentes
    data_manager.add_source("yahoo", YahooFinanceAPI())
    data_manager.add_source("alphavantage", AlphaVantageAPI("demo_key"))
    
    print("=== Ejemplo de Uso del Gestor de Datos ===")
    
    # 1. Obtener datos históricos
    print("\n1. Datos Históricos:")
    data = data_manager.get_data("AAPL", "2023-01-01", "2023-01-31", "yahoo")
    print(f"Datos obtenidos: {len(data)} registros")
    print(data.head())
    
    # 2. Datos en tiempo real
    print("\n2. Datos en Tiempo Real:")
    real_time = data_manager.get_real_time_data("AAPL", "yahoo")
    print(f"Precio actual: ${real_time['price']}")
    print(f"Cambio: {real_time['change_percent']:.2f}%")
    
    # 3. Comparar fuentes
    print("\n3. Comparación de Fuentes:")
    comparison = data_manager.compare_sources("AAPL", "2023-01-01", "2023-01-10")
    for source_name, data in comparison.items():
        if data is not None:
            avg_price = data['Close'].mean()
            print(f"{source_name}: Precio promedio = ${avg_price:.2f}")
    
    # 4. Múltiples símbolos
    print("\n4. Múltiples Símbolos:")
    symbols = ["AAPL", "GOOGL", "MSFT"]
    multi_data = data_manager.get_multiple_symbols(symbols, "2023-01-01", "2023-01-05")
    for symbol, data in multi_data.items():
        if data is not None:
            print(f"{symbol}: {len(data)} registros obtenidos")


if __name__ == "__main__":
    example_usage()