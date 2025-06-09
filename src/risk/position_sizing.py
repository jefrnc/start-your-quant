"""
Position Sizing Models
Complementa: docs/risk/position_sizing.md
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from abc import ABC, abstractmethod


class PositionSizingModel(ABC):
    """Clase base para modelos de dimensionamiento de posiciones"""
    
    @abstractmethod
    def calculate_position_size(self, **kwargs) -> float:
        """Calcula el tamaño de la posición"""
        pass


class FixedDollarAmount(PositionSizingModel):
    """Modelo de monto fijo en dólares"""
    
    def __init__(self, amount: float):
        self.amount = amount
    
    def calculate_position_size(self, price: float, **kwargs) -> int:
        """
        Calcula el número de acciones basado en un monto fijo
        
        Args:
            price: Precio actual del activo
            
        Returns:
            Número de acciones a comprar
        """
        return int(self.amount / price)


class FixedPercentage(PositionSizingModel):
    """Modelo de porcentaje fijo del capital"""
    
    def __init__(self, percentage: float):
        self.percentage = percentage  # Como decimal (ej: 0.1 para 10%)
    
    def calculate_position_size(self, portfolio_value: float, price: float, **kwargs) -> int:
        """
        Calcula el número de acciones basado en porcentaje del portafolio
        
        Args:
            portfolio_value: Valor total del portafolio
            price: Precio actual del activo
            
        Returns:
            Número de acciones a comprar
        """
        amount = portfolio_value * self.percentage
        return int(amount / price)


class KellyCriterion(PositionSizingModel):
    """Modelo basado en el Criterio de Kelly"""
    
    def __init__(self, max_position_size: float = 0.25):
        self.max_position_size = max_position_size  # Límite máximo de posición
    
    def calculate_position_size(self, win_rate: float, avg_win: float, 
                              avg_loss: float, portfolio_value: float, 
                              price: float, **kwargs) -> int:
        """
        Calcula el tamaño de posición usando el Criterio de Kelly
        
        Args:
            win_rate: Tasa de acierto (como decimal)
            avg_win: Ganancia promedio
            avg_loss: Pérdida promedio (valor positivo)
            portfolio_value: Valor del portafolio
            price: Precio del activo
            
        Returns:
            Número de acciones a comprar
        """
        # Fórmula de Kelly: f = (bp - q) / b
        # donde b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        
        if avg_loss <= 0:
            return 0
        
        b = avg_win / avg_loss  # Ratio ganancia/pérdida
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Aplicar límites de seguridad
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        amount = portfolio_value * kelly_fraction
        return int(amount / price)


class RiskParity(PositionSizingModel):
    """Modelo de paridad de riesgo"""
    
    def __init__(self, target_volatility: float = 0.15):
        self.target_volatility = target_volatility  # Volatilidad objetivo anual
    
    def calculate_position_size(self, asset_volatility: float, 
                              portfolio_value: float, price: float, 
                              **kwargs) -> int:
        """
        Calcula el tamaño de posición basado en volatilidad objetivo
        
        Args:
            asset_volatility: Volatilidad del activo (anualizada)
            portfolio_value: Valor del portafolio
            price: Precio del activo
            
        Returns:
            Número de acciones a comprar
        """
        if asset_volatility <= 0:
            return 0
        
        # Calcular el factor de escala basado en volatilidad
        volatility_factor = self.target_volatility / asset_volatility
        
        # Limitar el factor para evitar posiciones excesivas
        volatility_factor = min(volatility_factor, 1.0)
        
        amount = portfolio_value * volatility_factor
        return int(amount / price)


class ATRPositionSizing(PositionSizingModel):
    """Dimensionamiento basado en Average True Range (ATR)"""
    
    def __init__(self, risk_per_trade: float = 0.02, atr_multiplier: float = 2.0):
        self.risk_per_trade = risk_per_trade  # Riesgo por operación (% del capital)
        self.atr_multiplier = atr_multiplier  # Multiplicador del ATR para stop loss
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 14) -> float:
        """
        Calcula el Average True Range
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            period: Período para el cálculo
            
        Returns:
            Valor ATR actual
        """
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
    
    def calculate_position_size(self, portfolio_value: float, price: float,
                              high: pd.Series, low: pd.Series, 
                              close: pd.Series, **kwargs) -> int:
        """
        Calcula el tamaño de posición basado en ATR
        
        Args:
            portfolio_value: Valor del portafolio
            price: Precio actual
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            
        Returns:
            Número de acciones a comprar
        """
        atr = self.calculate_atr(high, low, close)
        
        if atr <= 0:
            return 0
        
        # Calcular stop loss basado en ATR
        stop_distance = atr * self.atr_multiplier
        
        # Cantidad de riesgo en dólares
        risk_amount = portfolio_value * self.risk_per_trade
        
        # Calcular tamaño de posición
        position_size = int(risk_amount / stop_distance)
        
        # Verificar que no exceda el valor del portafolio
        max_shares = int(portfolio_value / price)
        return min(position_size, max_shares)


class PortfolioManager:
    """Gestor de portafolio que utiliza diferentes modelos de posicionamiento"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.positions = {}
        self.position_sizing_model = None
    
    def set_position_sizing_model(self, model: PositionSizingModel):
        """Establece el modelo de dimensionamiento de posiciones"""
        self.position_sizing_model = model
    
    def calculate_position_size(self, symbol: str, price: float, **kwargs) -> int:
        """
        Calcula el tamaño de posición para un símbolo dado
        
        Args:
            symbol: Símbolo del activo
            price: Precio actual
            **kwargs: Parámetros adicionales para el modelo
            
        Returns:
            Número de acciones a comprar
        """
        if self.position_sizing_model is None:
            raise ValueError("No se ha establecido un modelo de posicionamiento")
        
        kwargs.update({
            'portfolio_value': self.current_value,
            'price': price
        })
        
        return self.position_sizing_model.calculate_position_size(**kwargs)


def example_usage():
    """Ejemplo de uso de los modelos de posicionamiento"""
    # Crear gestor de portafolio
    portfolio = PortfolioManager(initial_capital=100000)
    
    # Datos de ejemplo
    price = 50.0
    portfolio_value = 100000
    
    print("Ejemplos de Dimensionamiento de Posiciones:")
    print("=" * 50)
    
    # 1. Monto fijo
    fixed_amount = FixedDollarAmount(10000)
    shares = fixed_amount.calculate_position_size(price=price)
    print(f"Monto Fijo ($10,000): {shares} acciones")
    
    # 2. Porcentaje fijo
    fixed_pct = FixedPercentage(0.1)  # 10%
    shares = fixed_pct.calculate_position_size(portfolio_value=portfolio_value, price=price)
    print(f"Porcentaje Fijo (10%): {shares} acciones")
    
    # 3. Criterio de Kelly
    kelly = KellyCriterion()
    shares = kelly.calculate_position_size(
        win_rate=0.6, avg_win=100, avg_loss=50,
        portfolio_value=portfolio_value, price=price
    )
    print(f"Criterio de Kelly: {shares} acciones")
    
    # 4. Paridad de riesgo
    risk_parity = RiskParity(target_volatility=0.15)
    shares = risk_parity.calculate_position_size(
        asset_volatility=0.25, portfolio_value=portfolio_value, price=price
    )
    print(f"Paridad de Riesgo: {shares} acciones")
    
    # 5. ATR Position Sizing
    atr_sizing = ATRPositionSizing(risk_per_trade=0.02)
    
    # Generar datos de ejemplo para ATR
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    high = pd.Series(np.random.uniform(48, 52, 20), index=dates)
    low = pd.Series(np.random.uniform(47, 51, 20), index=dates)
    close = pd.Series(np.random.uniform(47.5, 51.5, 20), index=dates)
    
    shares = atr_sizing.calculate_position_size(
        portfolio_value=portfolio_value, price=price,
        high=high, low=low, close=close
    )
    print(f"ATR Position Sizing: {shares} acciones")


if __name__ == "__main__":
    example_usage()