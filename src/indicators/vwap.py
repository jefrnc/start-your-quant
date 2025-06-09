"""
VWAP (Volume Weighted Average Price) Implementation
Complementa: docs/indicators/vwap.md
"""

import pandas as pd
import numpy as np
from typing import Optional


class VWAP:
    """Implementación del VWAP (Volume Weighted Average Price)"""
    
    @staticmethod
    def calculate(prices: pd.Series, volumes: pd.Series, 
                 period: Optional[int] = None) -> pd.Series:
        """
        Calcula el VWAP
        
        Args:
            prices: Serie de precios (típicamente precio típico: (H+L+C)/3)
            volumes: Serie de volúmenes
            period: Período para VWAP rodante (None para VWAP acumulativo)
            
        Returns:
            Serie con valores VWAP
        """
        typical_price = prices
        pv = typical_price * volumes
        
        if period is None:
            # VWAP acumulativo
            cumulative_pv = pv.cumsum()
            cumulative_volume = volumes.cumsum()
            return cumulative_pv / cumulative_volume
        else:
            # VWAP rodante
            rolling_pv = pv.rolling(window=period).sum()
            rolling_volume = volumes.rolling(window=period).sum()
            return rolling_pv / rolling_volume
    
    @staticmethod
    def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calcula el precio típico (H+L+C)/3
        
        Args:
            high: Precios máximos
            low: Precios mínimos
            close: Precios de cierre
            
        Returns:
            Serie con precios típicos
        """
        return (high + low + close) / 3
    
    @staticmethod
    def vwap_bands(vwap: pd.Series, prices: pd.Series, volumes: pd.Series,
                   period: int = 20, std_dev: float = 1.0) -> tuple:
        """
        Calcula bandas de VWAP basadas en desviación estándar
        
        Args:
            vwap: Serie VWAP
            prices: Serie de precios
            volumes: Serie de volúmenes
            period: Período para cálculo de desviación
            std_dev: Multiplicador de desviación estándar
            
        Returns:
            Tupla (banda_superior, banda_inferior)
        """
        # Calcular desviación estándar ponderada por volumen
        price_diff_sq = (prices - vwap) ** 2
        weighted_variance = (price_diff_sq * volumes).rolling(period).sum() / volumes.rolling(period).sum()
        weighted_std = np.sqrt(weighted_variance)
        
        upper_band = vwap + (weighted_std * std_dev)
        lower_band = vwap - (weighted_std * std_dev)
        
        return upper_band, lower_band
    
    @staticmethod
    def vwap_signals(price: pd.Series, vwap: pd.Series) -> pd.Series:
        """
        Genera señales de trading basadas en VWAP
        
        Args:
            price: Serie de precios de cierre
            vwap: Serie VWAP
            
        Returns:
            Serie con señales: 1 (compra), -1 (venta), 0 (mantener)
        """
        signals = pd.Series(0, index=price.index)
        signals[price > vwap] = 1  # Precio por encima de VWAP - señal alcista
        signals[price < vwap] = -1  # Precio por debajo de VWAP - señal bajista
        return signals


def example_usage():
    """Ejemplo de uso del VWAP"""
    # Generar datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Datos OHLCV simulados
    high = pd.Series(np.random.uniform(100, 110, 100), index=dates)
    low = pd.Series(np.random.uniform(90, 100, 100), index=dates)
    close = pd.Series(np.random.uniform(95, 105, 100), index=dates)
    volume = pd.Series(np.random.randint(1000, 10000, 100), index=dates)
    
    vwap_calc = VWAP()
    
    # Calcular precio típico y VWAP
    typical_price = vwap_calc.typical_price(high, low, close)
    vwap = vwap_calc.calculate(typical_price, volume, period=20)
    
    # Generar señales
    signals = vwap_calc.vwap_signals(close, vwap)
    
    print("VWAP y señales calculados:")
    print(f"VWAP promedio: {vwap.mean():.2f}")
    print(f"Señales de compra: {(signals == 1).sum()}")
    print(f"Señales de venta: {(signals == -1).sum()}")


if __name__ == "__main__":
    example_usage()