"""
Moving Averages Implementation
Complementa: docs/indicators/moving_averages.md
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


class MovingAverages:
    """Implementación de diferentes tipos de medias móviles para análisis técnico"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """
        Simple Moving Average (SMA)
        
        Args:
            data: Serie de precios
            window: Período de la media móvil
            
        Returns:
            Serie con la media móvil simple
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """
        Exponential Moving Average (EMA)
        
        Args:
            data: Serie de precios
            window: Período de la media móvil
            alpha: Factor de suavizado (opcional)
            
        Returns:
            Serie con la media móvil exponencial
        """
        if alpha is None:
            alpha = 2 / (window + 1)
        return data.ewm(alpha=alpha).mean()
    
    @staticmethod
    def wma(data: pd.Series, window: int) -> pd.Series:
        """
        Weighted Moving Average (WMA)
        
        Args:
            data: Serie de precios
            window: Período de la media móvil
            
        Returns:
            Serie con la media móvil ponderada
        """
        weights = np.arange(1, window + 1)
        return data.rolling(window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def crossover_signal(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """
        Genera señales de cruce entre dos medias móviles
        
        Args:
            fast_ma: Media móvil rápida
            slow_ma: Media móvil lenta
            
        Returns:
            Serie con señales: 1 (compra), -1 (venta), 0 (mantener)
        """
        signals = pd.Series(0, index=fast_ma.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        return signals.diff().fillna(0)


def example_usage():
    """Ejemplo de uso de las medias móviles"""
    # Generar datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5), 
        index=dates
    )
    
    # Calcular medias móviles
    ma = MovingAverages()
    sma_20 = ma.sma(prices, 20)
    ema_12 = ma.ema(prices, 12)
    
    # Generar señales de trading
    signals = ma.crossover_signal(ema_12, sma_20)
    
    print("Señales de trading generadas:")
    print(signals[signals != 0].head())


if __name__ == "__main__":
    example_usage()