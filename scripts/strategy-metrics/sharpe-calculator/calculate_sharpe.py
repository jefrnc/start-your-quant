"""
Calculadora de Sharpe Ratio
===========================

Script para calcular Sharpe ratio de estrategias de trading con múltiples
métodos y validaciones. Incluye Sharpe anualizado, ajustado por riesgo
y comparaciones con benchmarks.

Uso:
    python calculate_sharpe.py --returns returns.csv --period daily
    python calculate_sharpe.py --trades trades.csv --benchmark SPY
"""

import pandas as pd
import numpy as np
import argparse
from typing import Union, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SharpeCalculator:
    """
    Calculadora completa de Sharpe Ratio para trading cuantitativo

    Soporta:
    - Múltiples períodos (daily, weekly, monthly)
    - Risk-free rates variables
    - Benchmarks personalizados
    - Análisis de drawdowns
    """

    ANNUALIZATION_FACTORS = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'hourly': 252 * 24,
        'minute': 252 * 24 * 60
    }

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Tasa libre de riesgo anual (default: 2%)
        """
        self.risk_free_rate = risk_free_rate

    def from_returns(self,
                    returns: Union[pd.Series, np.array, list],
                    period: str = 'daily') -> float:
        """
        Calcula Sharpe ratio desde serie de returns

        Args:
            returns: Serie de returns (decimal, ej: 0.05 = 5%)
            period: Frecuencia de los datos ('daily', 'weekly', 'monthly')

        Returns:
            Sharpe ratio anualizado
        """
        if isinstance(returns, (list, np.ndarray)):
            returns = pd.Series(returns)

        # Remover NaN y valores infinitos
        returns = returns.dropna()
        returns = returns[np.isfinite(returns)]

        if len(returns) < 2:
            raise ValueError("Se necesitan al menos 2 observaciones válidas")

        # Calcular estadísticas
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return np.inf if mean_return > 0 else np.nan

        # Risk-free rate ajustado al período
        annualization_factor = self.ANNUALIZATION_FACTORS.get(period, 252)
        rf_adjusted = self.risk_free_rate / annualization_factor

        # Sharpe ratio
        sharpe = (mean_return - rf_adjusted) / std_return

        # Anualizar
        sharpe_annualized = sharpe * np.sqrt(annualization_factor)

        return sharpe_annualized

    def from_trades(self,
                   trades_df: pd.DataFrame,
                   capital: float = 10000) -> Tuple[float, pd.DataFrame]:
        """
        Calcula Sharpe ratio desde DataFrame de trades

        Args:
            trades_df: DataFrame con columnas ['date', 'pnl'] como mínimo
            capital: Capital inicial para calcular returns

        Returns:
            Tuple de (sharpe_ratio, equity_curve_df)
        """
        required_cols = ['date', 'pnl']
        if not all(col in trades_df.columns for col in required_cols):
            raise ValueError(f"DataFrame debe contener columnas: {required_cols}")

        # Preparar datos
        df = trades_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Calcular equity curve
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['equity'] = capital + df['cumulative_pnl']

        # Calcular returns diarios
        df['returns'] = df['equity'].pct_change()

        # Calcular Sharpe
        returns = df['returns'].dropna()
        sharpe = self.from_returns(returns, period='daily')

        return sharpe, df[['date', 'equity', 'returns', 'cumulative_pnl']]

    def rolling_sharpe(self,
                      returns: pd.Series,
                      window: int = 60,
                      period: str = 'daily') -> pd.Series:
        """
        Calcula Sharpe ratio rodante

        Args:
            returns: Serie de returns
            window: Ventana para cálculo rodante
            period: Frecuencia de datos

        Returns:
            Serie con Sharpe ratio rodante
        """
        def sharpe_window(x):
            try:
                return self.from_returns(x, period)
            except:
                return np.nan

        return returns.rolling(window=window).apply(sharpe_window, raw=False)

    def sharpe_with_benchmark(self,
                             strategy_returns: pd.Series,
                             benchmark_returns: pd.Series,
                             period: str = 'daily') -> dict:
        """
        Compara Sharpe de estrategia vs benchmark

        Args:
            strategy_returns: Returns de la estrategia
            benchmark_returns: Returns del benchmark
            period: Frecuencia de datos

        Returns:
            Dict con métricas comparativas
        """
        # Alinear fechas
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) < 2:
            raise ValueError("No hay suficientes datos alineados")

        # Calcular Sharpe ratios
        strategy_sharpe = self.from_returns(aligned['strategy'], period)
        benchmark_sharpe = self.from_returns(aligned['benchmark'], period)

        # Excess returns
        excess_returns = aligned['strategy'] - aligned['benchmark']
        information_ratio = self.from_returns(excess_returns, period)

        return {
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'sharpe_difference': strategy_sharpe - benchmark_sharpe,
            'information_ratio': information_ratio,
            'tracking_error': excess_returns.std() * np.sqrt(self.ANNUALIZATION_FACTORS[period])
        }

    def detailed_analysis(self,
                         returns: pd.Series,
                         period: str = 'daily') -> dict:
        """
        Análisis detallado incluyendo métricas adicionales

        Returns:
            Dict con múltiples métricas de risk-adjusted performance
        """
        returns = pd.Series(returns).dropna()

        if len(returns) < 2:
            raise ValueError("Datos insuficientes para análisis")

        # Métricas básicas
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** self.ANNUALIZATION_FACTORS[period] - 1
        volatility = returns.std() * np.sqrt(self.ANNUALIZATION_FACTORS[period])

        # Sharpe y variantes
        sharpe = self.from_returns(returns, period)

        # Sortino ratio (solo downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(self.ANNUALIZATION_FACTORS[period])
            sortino = (annualized_return - self.risk_free_rate) / downside_deviation
        else:
            sortino = np.inf

        # Calmar ratio (annual return / max drawdown)
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

        # Win rate y profit factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(returns),
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }


def load_data(file_path: str) -> pd.DataFrame:
    """Carga datos desde CSV con validaciones"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error cargando archivo: {e}")


def print_analysis(analysis: dict, title: str = "Análisis de Performance"):
    """Imprime análisis formateado"""
    print(f"\n{title}")
    print("=" * len(title))

    for metric, value in analysis.items():
        if isinstance(value, float):
            if abs(value) > 1000 or value == np.inf:
                print(f"{metric:<20}: {'∞' if value == np.inf else f'{value:.2e}'}")
            else:
                print(f"{metric:<20}: {value:.4f}")
        else:
            print(f"{metric:<20}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Calculadora de Sharpe Ratio")

    # Argumentos principales
    parser.add_argument('--returns', type=str, help="Archivo CSV con returns")
    parser.add_argument('--trades', type=str, help="Archivo CSV con trades")
    parser.add_argument('--period', type=str, default='daily',
                       choices=['daily', 'weekly', 'monthly'],
                       help="Período de los datos")
    parser.add_argument('--benchmark', type=str, help="Archivo CSV con benchmark")
    parser.add_argument('--rf-rate', type=float, default=0.02,
                       help="Tasa libre de riesgo (anual)")
    parser.add_argument('--capital', type=float, default=10000,
                       help="Capital inicial para análisis de trades")
    parser.add_argument('--rolling-window', type=int, default=60,
                       help="Ventana para Sharpe ratio rodante")

    args = parser.parse_args()

    # Inicializar calculadora
    calc = SharpeCalculator(risk_free_rate=args.rf_rate)

    try:
        if args.returns:
            # Análisis desde returns
            print(f"Analizando returns desde: {args.returns}")
            data = load_data(args.returns)

            if 'returns' not in data.columns:
                raise ValueError("CSV debe contener columna 'returns'")

            returns = data['returns'].dropna()

            # Análisis completo
            analysis = calc.detailed_analysis(returns, args.period)
            print_analysis(analysis)

            # Sharpe rolling si hay suficientes datos
            if len(returns) >= args.rolling_window:
                rolling_sharpe = calc.rolling_sharpe(returns, args.rolling_window, args.period)
                print(f"\nSharpe Rolling (últimos 10 valores):")
                print(rolling_sharpe.tail(10).to_string())

        elif args.trades:
            # Análisis desde trades
            print(f"Analizando trades desde: {args.trades}")
            trades_df = load_data(args.trades)

            sharpe, equity_df = calc.from_trades(trades_df, args.capital)

            print(f"\nSharpe Ratio: {sharpe:.4f}")
            print(f"Capital Final: ${equity_df['equity'].iloc[-1]:,.2f}")
            print(f"Total P&L: ${equity_df['cumulative_pnl'].iloc[-1]:,.2f}")

            # Análisis detallado de los returns
            analysis = calc.detailed_analysis(equity_df['returns'].dropna(), args.period)
            print_analysis(analysis)

        # Comparación con benchmark si se proporciona
        if args.benchmark and (args.returns or args.trades):
            print(f"\nComparando con benchmark: {args.benchmark}")
            benchmark_data = load_data(args.benchmark)

            if 'returns' not in benchmark_data.columns:
                raise ValueError("Benchmark CSV debe contener columna 'returns'")

            if args.returns:
                strategy_returns = load_data(args.returns)['returns']
            else:
                # Usar returns de trades
                strategy_returns = equity_df['returns']

            benchmark_returns = benchmark_data['returns']

            comparison = calc.sharpe_with_benchmark(
                strategy_returns,
                benchmark_returns,
                args.period
            )

            print_analysis(comparison, "Comparación vs Benchmark")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


"""
EJEMPLOS DE USO:
================

1. Análisis básico desde archivo de returns:
   python calculate_sharpe.py --returns daily_returns.csv

2. Análisis desde trades con capital personalizado:
   python calculate_sharpe.py --trades my_trades.csv --capital 50000

3. Comparación con benchmark:
   python calculate_sharpe.py --returns strategy_returns.csv --benchmark spy_returns.csv

4. Análisis con configuración personalizada:
   python calculate_sharpe.py --trades trades.csv --rf-rate 0.03 --rolling-window 90

FORMATO DE ARCHIVOS CSV:
========================

returns.csv:
date,returns
2024-01-01,0.02
2024-01-02,-0.01
2024-01-03,0.015

trades.csv:
date,pnl
2024-01-01,150.50
2024-01-02,-75.25
2024-01-03,200.00

INTERPRETACIÓN DE SHARPE RATIO:
===============================

> 2.0  : Excelente
1.0-2.0: Muy bueno
0.5-1.0: Bueno
0.0-0.5: Pobre
< 0.0  : Muy pobre (destruye valor)

NOTAS IMPORTANTES:
==================

1. Sharpe ratio es sensible a outliers
2. Requiere distribución normal de returns (considerar Sortino para asimetría)
3. Período de medición afecta significativamente el resultado
4. Compara siempre con benchmarks relevantes
5. Considera otros ratios (Calmar, Information) para análisis completo
"""