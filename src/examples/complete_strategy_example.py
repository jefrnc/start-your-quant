"""
Complete Strategy Example
Ejemplo completo que integra múltiples componentes del framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Importar componentes del framework
from indicators.moving_averages import MovingAverages
from indicators.vwap import VWAP
from strategies.gap_and_go import GapAndGoStrategy
from backtesting.simple_engine import SimpleBacktester
from risk.position_sizing import FixedPercentage, ATRPositionSizing
from data.data_sources import DataManager, YahooFinanceAPI


class AdvancedTradingStrategy:
    """
    Estrategia avanzada que combina múltiples indicadores y técnicas
    """
    
    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        
        # Inicializar componentes
        self.ma = MovingAverages()
        self.vwap = VWAP()
        self.gap_strategy = GapAndGoStrategy()
        self.data_manager = DataManager()
        
        # Configurar fuentes de datos
        self.data_manager.add_source("yahoo", YahooFinanceAPI())
        
        # Inicializar position sizing
        if self.config['position_sizing']['method'] == 'fixed_percentage':
            self.position_sizer = FixedPercentage(
                self.config['position_sizing']['percentage']
            )
        elif self.config['position_sizing']['method'] == 'atr':
            self.position_sizer = ATRPositionSizing(
                risk_per_trade=self.config['position_sizing']['risk_per_trade'],
                atr_multiplier=self.config['position_sizing']['atr_multiplier']
            )
    
    def _default_config(self) -> dict:
        """Configuración por defecto de la estrategia"""
        return {
            'indicators': {
                'ma_fast': 10,
                'ma_slow': 20,
                'vwap_period': 20
            },
            'entry_conditions': {
                'ma_crossover': True,
                'price_above_vwap': True,
                'volume_confirmation': True,
                'min_volume_ratio': 1.2
            },
            'exit_conditions': {
                'profit_target': 0.08,  # 8%
                'stop_loss': 0.04,      # 4%
                'trailing_stop': True,
                'trailing_percent': 0.02  # 2%
            },
            'position_sizing': {
                'method': 'fixed_percentage',  # 'fixed_percentage' o 'atr'
                'percentage': 0.1,  # 10% del portafolio
                'risk_per_trade': 0.02,  # 2% riesgo por trade (para ATR)
                'atr_multiplier': 2.0
            },
            'filters': {
                'min_price': 5.0,
                'max_price': 500.0,
                'min_avg_volume': 100000
            }
        }
    
    def analyze_market_conditions(self, data: pd.DataFrame) -> dict:
        """
        Analiza las condiciones del mercado
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con análisis de mercado
        """
        # Calcular indicadores técnicos
        ma_fast = self.ma.sma(data['Close'], self.config['indicators']['ma_fast'])
        ma_slow = self.ma.sma(data['Close'], self.config['indicators']['ma_slow'])
        
        # VWAP
        typical_price = self.vwap.typical_price(data['High'], data['Low'], data['Close'])
        vwap = self.vwap.calculate(typical_price, data['Volume'], 
                                  self.config['indicators']['vwap_period'])
        
        # Identificar gaps
        gaps_data = self.gap_strategy.identify_gaps(data)
        
        # Análisis de volumen
        avg_volume = data['Volume'].rolling(20).mean()
        volume_ratio = data['Volume'] / avg_volume
        
        # Condiciones de mercado
        current_price = data['Close'].iloc[-1]
        current_vwap = vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else 0
        
        market_analysis = {
            'trend_direction': 'bullish' if ma_fast.iloc[-1] > ma_slow.iloc[-1] else 'bearish',
            'price_vs_vwap': 'above' if current_price > current_vwap else 'below',
            'volume_strength': 'high' if volume_ratio.iloc[-1] > 1.2 else 'normal',
            'recent_gap': abs(gaps_data['Gap_Percent'].iloc[-1]) > 1.0,
            'gap_direction': 'up' if gaps_data['Gap_Percent'].iloc[-1] > 0 else 'down',
            'volatility': data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252),
            'indicators': {
                'ma_fast': ma_fast.iloc[-1],
                'ma_slow': ma_slow.iloc[-1],
                'vwap': current_vwap,
                'volume_ratio': volume_ratio.iloc[-1]
            }
        }
        
        return market_analysis
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de trading basadas en la estrategia completa
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con señales y análisis
        """
        signals_df = data.copy()
        
        # Calcular indicadores
        ma_fast = self.ma.sma(data['Close'], self.config['indicators']['ma_fast'])
        ma_slow = self.ma.sma(data['Close'], self.config['indicators']['ma_slow'])
        
        typical_price = self.vwap.typical_price(data['High'], data['Low'], data['Close'])
        vwap = self.vwap.calculate(typical_price, data['Volume'], 
                                  self.config['indicators']['vwap_period'])
        
        # Análisis de volumen
        avg_volume = data['Volume'].rolling(20).mean()
        volume_ratio = data['Volume'] / avg_volume
        
        # Añadir indicadores al DataFrame
        signals_df['MA_Fast'] = ma_fast
        signals_df['MA_Slow'] = ma_slow
        signals_df['VWAP'] = vwap
        signals_df['Volume_Ratio'] = volume_ratio
        
        # Inicializar señales
        signals_df['Signal'] = 0
        signals_df['Entry_Reason'] = ''
        signals_df['Position_Size'] = 0
        
        # Condiciones de entrada
        for i in range(len(signals_df)):
            current_row = signals_df.iloc[i]
            
            # Filtros básicos
            if (current_row['Close'] < self.config['filters']['min_price'] or
                current_row['Close'] > self.config['filters']['max_price'] or
                current_row['Volume'] < self.config['filters']['min_avg_volume']):
                continue
            
            entry_conditions = []
            
            # Condición 1: Cruce de medias móviles
            if (self.config['entry_conditions']['ma_crossover'] and 
                i > 0 and
                not pd.isna(current_row['MA_Fast']) and 
                not pd.isna(current_row['MA_Slow'])):
                
                prev_row = signals_df.iloc[i-1]
                if (current_row['MA_Fast'] > current_row['MA_Slow'] and 
                    prev_row['MA_Fast'] <= prev_row['MA_Slow']):
                    entry_conditions.append('MA_Crossover_Bullish')
            
            # Condición 2: Precio por encima de VWAP
            if (self.config['entry_conditions']['price_above_vwap'] and 
                not pd.isna(current_row['VWAP'])):
                if current_row['Close'] > current_row['VWAP']:
                    entry_conditions.append('Price_Above_VWAP')
            
            # Condición 3: Confirmación de volumen
            if (self.config['entry_conditions']['volume_confirmation'] and
                not pd.isna(current_row['Volume_Ratio'])):
                if current_row['Volume_Ratio'] > self.config['entry_conditions']['min_volume_ratio']:
                    entry_conditions.append('Volume_Confirmation')
            
            # Generar señal si se cumplen las condiciones
            required_conditions = sum([
                self.config['entry_conditions']['ma_crossover'],
                self.config['entry_conditions']['price_above_vwap'],
                self.config['entry_conditions']['volume_confirmation']
            ])
            
            if len(entry_conditions) >= required_conditions:
                signals_df.at[signals_df.index[i], 'Signal'] = 1
                signals_df.at[signals_df.index[i], 'Entry_Reason'] = ', '.join(entry_conditions)
                
                # Calcular tamaño de posición
                if hasattr(self, 'position_sizer'):
                    if isinstance(self.position_sizer, FixedPercentage):
                        position_size = self.position_sizer.calculate_position_size(
                            portfolio_value=100000,  # Valor por defecto
                            price=current_row['Close']
                        )
                    elif isinstance(self.position_sizer, ATRPositionSizing):
                        position_size = self.position_sizer.calculate_position_size(
                            portfolio_value=100000,
                            price=current_row['Close'],
                            high=data['High'][:i+1],
                            low=data['Low'][:i+1],
                            close=data['Close'][:i+1]
                        )
                    
                    signals_df.at[signals_df.index[i], 'Position_Size'] = position_size
        
        return signals_df
    
    def run_complete_analysis(self, symbol: str, start_date: str, end_date: str):
        """
        Ejecuta análisis completo de la estrategia
        
        Args:
            symbol: Símbolo del activo
            start_date: Fecha de inicio
            end_date: Fecha de fin
        """
        print(f"=== Análisis Completo para {symbol} ===")
        print(f"Período: {start_date} a {end_date}")
        print()
        
        # 1. Obtener datos
        print("1. Obteniendo datos...")
        data = self.data_manager.get_data(symbol, start_date, end_date, "yahoo")
        print(f"   Datos obtenidos: {len(data)} registros")
        
        # 2. Análisis de mercado
        print("\n2. Analizando condiciones de mercado...")
        market_analysis = self.analyze_market_conditions(data)
        print(f"   Tendencia: {market_analysis['trend_direction']}")
        print(f"   Precio vs VWAP: {market_analysis['price_vs_vwap']}")
        print(f"   Fuerza del volumen: {market_analysis['volume_strength']}")
        print(f"   Volatilidad anual: {market_analysis['volatility']:.2%}")
        
        # 3. Generar señales
        print("\n3. Generando señales de trading...")
        signals = self.generate_signals(data)
        trade_signals = signals[signals['Signal'] != 0]
        print(f"   Señales generadas: {len(trade_signals)}")
        
        if len(trade_signals) > 0:
            print("\n   Últimas 3 señales:")
            for idx, row in trade_signals.tail(3).iterrows():
                print(f"   {idx.date()}: {row['Entry_Reason']} - ${row['Close']:.2f}")
        
        # 4. Backtest básico
        print("\n4. Ejecutando backtest...")
        def strategy_func(data, **kwargs):
            strategy_signals = self.generate_signals(data)
            return strategy_signals['Signal']
        
        backtester = SimpleBacktester(initial_capital=100000, commission=0.001)
        results = backtester.run_backtest(data, strategy_func)
        
        print(f"   Retorno Total: {results['total_return_percent']:.2f}%")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Máximo Drawdown: {results['max_drawdown_percent']:.2f}%")
        print(f"   Trades ejecutados: {results['total_trades']}")
        print(f"   Tasa de acierto: {results['win_rate_percent']:.2f}%")
        
        return {
            'data': data,
            'signals': signals,
            'market_analysis': market_analysis,
            'backtest_results': results
        }


def main():
    """Función principal que ejecuta el ejemplo completo"""
    # Configuración personalizada
    custom_config = {
        'indicators': {
            'ma_fast': 12,
            'ma_slow': 26,
            'vwap_period': 20
        },
        'entry_conditions': {
            'ma_crossover': True,
            'price_above_vwap': True,
            'volume_confirmation': True,
            'min_volume_ratio': 1.5
        },
        'exit_conditions': {
            'profit_target': 0.10,
            'stop_loss': 0.05,
            'trailing_stop': True,
            'trailing_percent': 0.03
        },
        'position_sizing': {
            'method': 'atr',
            'risk_per_trade': 0.02,
            'atr_multiplier': 2.5
        },
        'filters': {
            'min_price': 10.0,
            'max_price': 300.0,
            'min_avg_volume': 500000
        }
    }
    
    # Crear estrategia
    strategy = AdvancedTradingStrategy(custom_config)
    
    # Ejecutar análisis
    results = strategy.run_complete_analysis(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print("\n=== Análisis Completado ===")
    print("Los resultados incluyen:")
    print("- Datos históricos")
    print("- Señales de trading generadas")
    print("- Análisis de condiciones de mercado")
    print("- Resultados del backtest")


if __name__ == "__main__":
    main()