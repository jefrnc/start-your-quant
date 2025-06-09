"""
Simple Backtesting Engine
Complementa: docs/backtesting/simple_engine.md
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Representa una operación individual"""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str  # 'long' o 'short'
    pnl: Optional[float] = None
    commission: float = 0.0
    
    def close_trade(self, exit_date: datetime, exit_price: float, commission: float = 0.0):
        """Cierra la operación"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.commission += commission
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.commission
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.commission


class Portfolio:
    """Gestiona el portafolio y las posiciones"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: quantity}
        self.equity_curve = []
        self.trades = []
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calcula el valor total del portafolio"""
        positions_value = sum(
            qty * prices.get(symbol, 0) 
            for symbol, qty in self.positions.items()
        )
        return self.cash + positions_value
    
    def can_buy(self, symbol: str, quantity: int, price: float, commission: float = 0.0) -> bool:
        """Verifica si se puede realizar una compra"""
        total_cost = quantity * price + commission
        return self.cash >= total_cost
    
    def buy(self, symbol: str, quantity: int, price: float, date: datetime, commission: float = 0.0) -> bool:
        """Ejecuta una compra"""
        if not self.can_buy(symbol, quantity, price, commission):
            return False
        
        total_cost = quantity * price + commission
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        # Registrar trade
        trade = Trade(
            entry_date=date,
            exit_date=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            side='long',
            commission=commission
        )
        self.trades.append(trade)
        return True
    
    def sell(self, symbol: str, quantity: int, price: float, date: datetime, commission: float = 0.0) -> bool:
        """Ejecuta una venta"""
        if self.positions.get(symbol, 0) < quantity:
            return False
        
        total_proceeds = quantity * price - commission
        self.cash += total_proceeds
        self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Cerrar trades correspondientes (FIFO)
        remaining_qty = quantity
        for trade in reversed(self.trades):
            if (trade.exit_date is None and 
                trade.side == 'long' and 
                remaining_qty > 0):
                
                qty_to_close = min(remaining_qty, trade.quantity)
                if qty_to_close == trade.quantity:
                    trade.close_trade(date, price, commission / quantity * qty_to_close)
                remaining_qty -= qty_to_close
                
                if remaining_qty == 0:
                    break
        
        return True


class SimpleBacktester:
    """Motor de backtesting simple"""
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio = Portfolio(initial_capital)
        
    def run_backtest(self, data: pd.DataFrame, strategy_func: Callable, 
                    **strategy_params) -> Dict:
        """
        Ejecuta el backtest
        
        Args:
            data: DataFrame con datos OHLCV
            strategy_func: Función que genera señales de trading
            **strategy_params: Parámetros para la estrategia
            
        Returns:
            Diccionario con resultados del backtest
        """
        # Generar señales
        signals = strategy_func(data, **strategy_params)
        
        # Ejecutar trades
        for date, row in data.iterrows():
            current_price = row['Close']
            signal = signals.get(date, 0) if isinstance(signals, dict) else signals.loc[date] if date in signals.index else 0
            
            # Calcular valor del portafolio
            portfolio_value = self.portfolio.get_portfolio_value({'ASSET': current_price})
            self.portfolio.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.portfolio.cash
            })
            
            # Ejecutar operaciones basadas en señales
            if signal == 1:  # Señal de compra
                # Invertir el 10% del portafolio
                investment_amount = portfolio_value * 0.1
                quantity = int(investment_amount / current_price)
                if quantity > 0:
                    commission = investment_amount * self.commission
                    self.portfolio.buy('ASSET', quantity, current_price, date, commission)
            
            elif signal == -1:  # Señal de venta
                # Vender toda la posición
                current_position = self.portfolio.positions.get('ASSET', 0)
                if current_position > 0:
                    commission = current_position * current_price * self.commission
                    self.portfolio.sell('ASSET', current_position, current_price, date, commission)
        
        return self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calcula métricas de rendimiento"""
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calcular retornos
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        
        # Métricas básicas
        total_return = (equity_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Volatilidad anualizada
        volatility = equity_df['returns'].std() * np.sqrt(252) * 100
        
        # Sharpe ratio (asumiendo tasa libre de riesgo = 0)
        sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
        
        # Maximum drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Estadísticas de trades
        completed_trades = [t for t in self.portfolio.trades if t.pnl is not None]
        winning_trades = len([t for t in completed_trades if t.pnl > 0])
        total_trades = len(completed_trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in completed_trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in completed_trades if t.pnl < 0]) if (total_trades - winning_trades) > 0 else 0
        
        return {
            'total_return_percent': total_return,
            'volatility_percent': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_percent': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate_percent': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'equity_curve': equity_df,
            'trades': completed_trades
        }


def simple_ma_crossover_strategy(data: pd.DataFrame, fast_period: int = 10, 
                                slow_period: int = 20) -> pd.Series:
    """
    Estrategia simple de cruce de medias móviles
    
    Args:
        data: DataFrame con datos OHLCV
        fast_period: Período de la media móvil rápida
        slow_period: Período de la media móvil lenta
        
    Returns:
        Serie con señales de trading
    """
    fast_ma = data['Close'].rolling(fast_period).mean()
    slow_ma = data['Close'].rolling(slow_period).mean()
    
    signals = pd.Series(0, index=data.index)
    signals[fast_ma > slow_ma] = 1
    signals[fast_ma < slow_ma] = -1
    
    # Solo generar señal en cruces
    signal_changes = signals.diff()
    final_signals = pd.Series(0, index=data.index)
    final_signals[signal_changes == 2] = 1  # Cruce alcista
    final_signals[signal_changes == -2] = -1  # Cruce bajista
    
    return final_signals


def example_usage():
    """Ejemplo de uso del backtester"""
    # Generar datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')  # Un año de datos
    
    # Simular precios con tendencia
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Retornos diarios
    prices = [100]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Open': prices[:-1],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'Close': prices[1:],
        'Volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # Ejecutar backtest
    backtester = SimpleBacktester(initial_capital=100000, commission=0.001)
    results = backtester.run_backtest(data, simple_ma_crossover_strategy, 
                                     fast_period=10, slow_period=20)
    
    print("Resultados del Backtest:")
    print(f"Retorno Total: {results['total_return_percent']:.2f}%")
    print(f"Volatilidad: {results['volatility_percent']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Máximo Drawdown: {results['max_drawdown_percent']:.2f}%")
    print(f"Total de Trades: {results['total_trades']}")
    print(f"Tasa de Acierto: {results['win_rate_percent']:.2f}%")
    print(f"Ganancia Promedio: ${results['avg_win']:.2f}")
    print(f"Pérdida Promedio: ${results['avg_loss']:.2f}")


if __name__ == "__main__":
    example_usage()