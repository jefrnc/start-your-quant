"""
Gap and Go Strategy Implementation
Complementa: docs/strategies/gap_and_go.md
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class GapAndGoStrategy:
    """
    Estrategia Gap and Go para trading de acciones
    
    Esta estrategia busca gaps de apertura significativos y opera
    en la dirección del gap si se confirma con volumen
    """
    
    def __init__(self, min_gap_percent: float = 2.0, 
                 min_volume_ratio: float = 1.5,
                 profit_target: float = 0.05,
                 stop_loss: float = 0.03):
        """
        Args:
            min_gap_percent: Gap mínimo requerido (%)
            min_volume_ratio: Ratio mínimo de volumen vs promedio
            profit_target: Objetivo de ganancia (%)
            stop_loss: Stop loss (%)
        """
        self.min_gap_percent = min_gap_percent
        self.min_volume_ratio = min_volume_ratio
        self.profit_target = profit_target
        self.stop_loss = stop_loss
    
    def calculate_gap(self, current_open: float, previous_close: float) -> float:
        """
        Calcula el gap porcentual
        
        Args:
            current_open: Precio de apertura actual
            previous_close: Precio de cierre anterior
            
        Returns:
            Gap porcentual
        """
        return ((current_open - previous_close) / previous_close) * 100
    
    def identify_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifica gaps en los datos
        
        Args:
            data: DataFrame con columnas ['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
            DataFrame con información de gaps
        """
        result = data.copy()
        result['Previous_Close'] = data['Close'].shift(1)
        result['Gap_Percent'] = result.apply(
            lambda row: self.calculate_gap(row['Open'], row['Previous_Close'])
            if pd.notna(row['Previous_Close']) else 0, axis=1
        )
        
        # Calcular volumen promedio (20 días)
        result['Avg_Volume'] = data['Volume'].rolling(20).mean()
        result['Volume_Ratio'] = result['Volume'] / result['Avg_Volume']
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de trading basadas en gaps
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con señales de trading
        """
        signals_df = self.identify_gaps(data).copy()
        signals_df['Signal'] = 0
        signals_df['Entry_Price'] = np.nan
        signals_df['Target_Price'] = np.nan
        signals_df['Stop_Price'] = np.nan
        
        # Condiciones para gap up
        gap_up_condition = (
            (signals_df['Gap_Percent'] >= self.min_gap_percent) &
            (signals_df['Volume_Ratio'] >= self.min_volume_ratio)
        )
        
        # Condiciones para gap down
        gap_down_condition = (
            (signals_df['Gap_Percent'] <= -self.min_gap_percent) &
            (signals_df['Volume_Ratio'] >= self.min_volume_ratio)
        )
        
        # Asignar señales
        signals_df.loc[gap_up_condition, 'Signal'] = 1  # Compra
        signals_df.loc[gap_down_condition, 'Signal'] = -1  # Venta
        
        # Calcular precios de entrada y objetivos
        entry_mask = signals_df['Signal'] != 0
        signals_df.loc[entry_mask, 'Entry_Price'] = signals_df.loc[entry_mask, 'Open']
        
        # Para posiciones largas
        long_mask = signals_df['Signal'] == 1
        signals_df.loc[long_mask, 'Target_Price'] = (
            signals_df.loc[long_mask, 'Entry_Price'] * (1 + self.profit_target)
        )
        signals_df.loc[long_mask, 'Stop_Price'] = (
            signals_df.loc[long_mask, 'Entry_Price'] * (1 - self.stop_loss)
        )
        
        # Para posiciones cortas
        short_mask = signals_df['Signal'] == -1
        signals_df.loc[short_mask, 'Target_Price'] = (
            signals_df.loc[short_mask, 'Entry_Price'] * (1 - self.profit_target)
        )
        signals_df.loc[short_mask, 'Stop_Price'] = (
            signals_df.loc[short_mask, 'Entry_Price'] * (1 + self.stop_loss)
        )
        
        return signals_df
    
    def backtest_strategy(self, data: pd.DataFrame) -> Dict:
        """
        Ejecuta backtest de la estrategia
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con resultados del backtest
        """
        signals = self.generate_signals(data)
        trades = []
        
        for idx, row in signals.iterrows():
            if row['Signal'] != 0:
                trade = {
                    'date': idx,
                    'signal': row['Signal'],
                    'entry_price': row['Entry_Price'],
                    'target_price': row['Target_Price'],
                    'stop_price': row['Stop_Price'],
                    'gap_percent': row['Gap_Percent'],
                    'volume_ratio': row['Volume_Ratio']
                }
                trades.append(trade)
        
        # Calcular estadísticas
        total_trades = len(trades)
        long_trades = sum(1 for t in trades if t['signal'] == 1)
        short_trades = sum(1 for t in trades if t['signal'] == -1)
        
        results = {
            'total_trades': total_trades,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'trades': trades,
            'signals_df': signals
        }
        
        return results


def example_usage():
    """Ejemplo de uso de la estrategia Gap and Go"""
    # Generar datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Simular datos OHLCV con algunos gaps
    base_price = 100
    data = []
    
    for i in range(100):
        if i == 0:
            open_price = base_price
        else:
            # Simular gap ocasional
            if np.random.random() < 0.1:  # 10% probabilidad de gap
                gap_direction = np.random.choice([-1, 1])
                gap_size = np.random.uniform(0.02, 0.05)  # 2-5% gap
                open_price = data[i-1]['Close'] * (1 + gap_direction * gap_size)
            else:
                open_price = data[i-1]['Close'] * (1 + np.random.uniform(-0.01, 0.01))
        
        high = open_price * (1 + abs(np.random.normal(0, 0.02)))
        low = open_price * (1 - abs(np.random.normal(0, 0.02)))
        close = np.random.uniform(low, high)
        volume = np.random.randint(10000, 100000)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Ejecutar estrategia
    strategy = GapAndGoStrategy()
    results = strategy.backtest_strategy(df)
    
    print("Resultados del backtest Gap and Go:")
    print(f"Total de trades: {results['total_trades']}")
    print(f"Trades largos: {results['long_trades']}")
    print(f"Trades cortos: {results['short_trades']}")
    
    if results['trades']:
        print("\nPrimeros 5 trades:")
        for trade in results['trades'][:5]:
            print(f"Fecha: {trade['date'].date()}, "
                  f"Señal: {trade['signal']}, "
                  f"Gap: {trade['gap_percent']:.2f}%, "
                  f"Entrada: ${trade['entry_price']:.2f}")


if __name__ == "__main__":
    example_usage()