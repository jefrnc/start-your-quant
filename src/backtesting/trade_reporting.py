"""
Trade Reporting y Exportación para Análisis
Genera reportes CSV compatibles con TraderVue, TradesViz, y otras plataformas
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import csv
import json


class TradeReporter:
    """Generador de reportes de trading para análisis post-trade"""
    
    def __init__(self, trades: List):
        """
        Args:
            trades: Lista de objetos Trade del backtesting
        """
        self.trades = trades
        
    def to_tradervue_csv(self, filename: str, account_name: str = "Backtest"):
        """
        Exporta trades en formato CSV compatible con TraderVue
        
        TraderVue Format:
        - Date/Time: Entry time
        - Symbol: Trading symbol
        - Side: Buy/Sell
        - Quantity: Number of shares
        - Price: Entry price
        - Commission: Trading commission
        - Exit Date/Time: Exit time
        - Exit Price: Exit price
        - P&L: Profit/Loss
        """
        
        rows = []
        for trade in self.trades:
            if trade.exit_date is not None:  # Solo trades cerrados
                # Entry row
                entry_row = {
                    'Date/Time': trade.entry_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Symbol': 'BACKTEST',  # Usar símbolo real si está disponible
                    'Side': 'BUY' if trade.side == 'long' else 'SELL SHORT',
                    'Quantity': trade.quantity,
                    'Price': trade.entry_price,
                    'Commission': trade.commission / 2,  # Dividir comisión entre entry/exit
                    'Account': account_name
                }
                
                # Exit row
                exit_row = {
                    'Date/Time': trade.exit_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Symbol': 'BACKTEST',
                    'Side': 'SELL' if trade.side == 'long' else 'BUY TO COVER',
                    'Quantity': trade.quantity,
                    'Price': trade.exit_price,
                    'Commission': trade.commission / 2,
                    'Account': account_name
                }
                
                rows.append(entry_row)
                rows.append(exit_row)
        
        # Escribir CSV
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Exportado {len(self.trades)} trades a {filename} para TraderVue")
        
    def to_generic_csv(self, filename: str):
        """
        Exporta trades en formato CSV genérico detallado
        """
        
        rows = []
        for i, trade in enumerate(self.trades):
            if trade.exit_date is not None:
                row = {
                    'Trade_ID': i + 1,
                    'Entry_Date': trade.entry_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit_Date': trade.exit_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Duration_Minutes': (trade.exit_date - trade.entry_date).total_seconds() / 60,
                    'Side': trade.side,
                    'Entry_Price': trade.entry_price,
                    'Exit_Price': trade.exit_price,
                    'Quantity': trade.quantity,
                    'Gross_PnL': trade.pnl + trade.commission,
                    'Commission': trade.commission,
                    'Net_PnL': trade.pnl,
                    'Return_Percent': (trade.pnl / (trade.entry_price * trade.quantity)) * 100,
                    'MAE': 0,  # Maximum Adverse Excursion (requiere datos intraday)
                    'MFE': 0   # Maximum Favorable Excursion (requiere datos intraday)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Exportado reporte detallado de {len(rows)} trades a {filename}")
        
    def to_journal_format(self, filename: str):
        """
        Exporta en formato para journal de trading con métricas adicionales
        """
        
        # Agrupar trades por día
        daily_stats = {}
        
        for trade in self.trades:
            if trade.exit_date is not None:
                trade_date = trade.entry_date.date()
                
                if trade_date not in daily_stats:
                    daily_stats[trade_date] = {
                        'date': trade_date,
                        'trades': 0,
                        'gross_pnl': 0,
                        'net_pnl': 0,
                        'winners': 0,
                        'losers': 0,
                        'commissions': 0,
                        'largest_win': 0,
                        'largest_loss': 0
                    }
                
                daily_stats[trade_date]['trades'] += 1
                daily_stats[trade_date]['gross_pnl'] += trade.pnl + trade.commission
                daily_stats[trade_date]['net_pnl'] += trade.pnl
                daily_stats[trade_date]['commissions'] += trade.commission
                
                if trade.pnl > 0:
                    daily_stats[trade_date]['winners'] += 1
                    daily_stats[trade_date]['largest_win'] = max(
                        daily_stats[trade_date]['largest_win'], trade.pnl
                    )
                else:
                    daily_stats[trade_date]['losers'] += 1
                    daily_stats[trade_date]['largest_loss'] = min(
                        daily_stats[trade_date]['largest_loss'], trade.pnl
                    )
        
        # Calcular métricas adicionales
        for date, stats in daily_stats.items():
            stats['win_rate'] = (stats['winners'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            stats['avg_trade'] = stats['net_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
        
        # Convertir a DataFrame y exportar
        df = pd.DataFrame(list(daily_stats.values()))
        df = df.sort_values('date')
        df.to_csv(filename, index=False)
        print(f"Exportado journal con estadísticas diarias a {filename}")
        
    def generate_performance_report(self, filename: str, initial_capital: float = 100000):
        """
        Genera reporte de rendimiento detallado
        """
        
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        
        if not completed_trades:
            print("No hay trades completados para generar reporte")
            return
        
        # Calcular estadísticas
        total_trades = len(completed_trades)
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        report = {
            'Performance Summary': {
                'Total Trades': total_trades,
                'Winning Trades': len(winning_trades),
                'Losing Trades': len(losing_trades),
                'Win Rate %': len(winning_trades) / total_trades * 100 if total_trades > 0 else 0,
                'Total P&L': sum(t.pnl for t in completed_trades),
                'Total Commissions': sum(t.commission for t in completed_trades),
                'Net P&L': sum(t.pnl for t in completed_trades),
                'Return on Capital %': sum(t.pnl for t in completed_trades) / initial_capital * 100
            },
            
            'Trade Statistics': {
                'Average Win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
                'Average Loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
                'Largest Win': max([t.pnl for t in winning_trades]) if winning_trades else 0,
                'Largest Loss': min([t.pnl for t in losing_trades]) if losing_trades else 0,
                'Win/Loss Ratio': abs(np.mean([t.pnl for t in winning_trades]) / np.mean([t.pnl for t in losing_trades])) if winning_trades and losing_trades else 0,
                'Profit Factor': abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf'),
                'Average Trade Duration (hours)': np.mean([(t.exit_date - t.entry_date).total_seconds() / 3600 for t in completed_trades])
            },
            
            'Risk Metrics': {
                'Max Consecutive Wins': self._max_consecutive(completed_trades, True),
                'Max Consecutive Losses': self._max_consecutive(completed_trades, False),
                'Recovery Factor': sum(t.pnl for t in completed_trades) / abs(min([t.pnl for t in losing_trades])) if losing_trades else float('inf'),
                'Expectancy': sum(t.pnl for t in completed_trades) / total_trades if total_trades > 0 else 0
            }
        }
        
        # Guardar como JSON
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Reporte de rendimiento guardado en {filename}")
        
        # También crear versión CSV
        csv_filename = filename.replace('.json', '.csv')
        self._flatten_dict_to_csv(report, csv_filename)
        
    def _max_consecutive(self, trades: List, wins: bool) -> int:
        """Calcula máximo de trades consecutivos ganadores/perdedores"""
        max_count = 0
        current_count = 0
        
        for trade in trades:
            if (wins and trade.pnl > 0) or (not wins and trade.pnl < 0):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
                
        return max_count
    
    def _flatten_dict_to_csv(self, data: Dict, filename: str):
        """Convierte diccionario anidado a CSV plano"""
        rows = []
        for category, metrics in data.items():
            for metric, value in metrics.items():
                rows.append({
                    'Category': category,
                    'Metric': metric,
                    'Value': value
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Reporte CSV guardado en {filename}")
        
    def create_equity_curve_csv(self, equity_curve: pd.DataFrame, filename: str):
        """
        Exporta la curva de equity para análisis
        """
        equity_curve.to_csv(filename)
        print(f"Curva de equity exportada a {filename}")


def export_backtest_results(backtest_results: Dict, output_dir: str = "./backtest_reports/"):
    """
    Función de conveniencia para exportar todos los reportes de un backtest
    
    Args:
        backtest_results: Diccionario retornado por SimpleBacktester
        output_dir: Directorio donde guardar los reportes
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp para nombres únicos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear reporter
    reporter = TradeReporter(backtest_results['trades'])
    
    # Generar todos los reportes
    reporter.to_tradervue_csv(f"{output_dir}/tradervue_{timestamp}.csv")
    reporter.to_generic_csv(f"{output_dir}/trades_detail_{timestamp}.csv")
    reporter.to_journal_format(f"{output_dir}/daily_journal_{timestamp}.csv")
    reporter.generate_performance_report(f"{output_dir}/performance_{timestamp}.json")
    reporter.create_equity_curve_csv(
        backtest_results['equity_curve'], 
        f"{output_dir}/equity_curve_{timestamp}.csv"
    )
    
    print(f"\n✅ Todos los reportes exportados en: {output_dir}")
    print("   - tradervue_*.csv: Para importar en TraderVue")
    print("   - trades_detail_*.csv: Detalle completo de trades")
    print("   - daily_journal_*.csv: Resumen diario para journal")
    print("   - performance_*.json/csv: Métricas de rendimiento")
    print("   - equity_curve_*.csv: Evolución del capital")


if __name__ == "__main__":
    # Ejemplo de uso
    from simple_engine import SimpleBacktester, simple_ma_crossover_strategy
    import pandas as pd
    import numpy as np
    
    # Generar datos de ejemplo
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = [100]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Open': prices[:-1],
        'High': [p * 1.01 for p in prices[:-1]],
        'Low': [p * 0.99 for p in prices[:-1]],
        'Close': prices[1:],
        'Volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # Ejecutar backtest
    backtester = SimpleBacktester()
    results = backtester.run_backtest(data, simple_ma_crossover_strategy)
    
    # Exportar resultados
    export_backtest_results(results)