"""
Calculadora de Maximum Drawdown
===============================

Script completo para calcular maximum drawdown y métricas relacionadas
para estrategias de trading cuantitativo. Incluye análisis temporal,
recovery time, y underwater curves.

Maximum Drawdown es critical para small cap trading porque:
- Small caps pueden tener drawdowns brutales (30%+ possible)
- Psychology impact puede ser severe
- Capital preservation es paramount
- Risk management debe basarse en worst-case scenarios

Uso:
    python calculate_drawdown.py --trades trades.csv --capital 10000
    python calculate_drawdown.py --equity equity_curve.csv --plot
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DrawdownAnalyzer:
    """
    Analizador completo de Drawdown para trading cuantitativo

    Calcula:
    - Maximum Drawdown (absoluto y porcentual)
    - Drawdown duration y recovery time
    - Underwater curves
    - Drawdown distribution
    - Pain index y otras métricas avanzadas
    """

    def __init__(self):
        self.drawdown_periods = []

    def from_equity_curve(self, equity_curve: Union[pd.Series, np.array, list]) -> Dict:
        """
        Calcula métricas de drawdown desde equity curve

        Args:
            equity_curve: Serie de valores de cuenta/equity

        Returns:
            Dict con métricas completas de drawdown
        """
        if isinstance(equity_curve, (list, np.ndarray)):
            equity_curve = pd.Series(equity_curve)

        equity_curve = equity_curve.dropna()

        if len(equity_curve) < 2:
            raise ValueError("Se necesitan al menos 2 observaciones válidas")

        # Calcular running maximum (peak values)
        running_max = equity_curve.expanding().max()

        # Calcular drawdown series
        drawdown_dollars = equity_curve - running_max
        drawdown_percent = drawdown_dollars / running_max

        # Maximum drawdown
        max_dd_dollars = drawdown_dollars.min()
        max_dd_percent = drawdown_percent.min()
        max_dd_date = drawdown_percent.idxmin()

        # Encontrar inicio del maximum drawdown
        peak_before_max_dd = running_max.loc[max_dd_date]
        peak_date = equity_curve[equity_curve == peak_before_max_dd].index[0]

        # Recovery analysis
        recovery_info = self._analyze_recovery(
            equity_curve, max_dd_date, peak_before_max_dd
        )

        # Drawdown periods analysis
        dd_periods = self._identify_drawdown_periods(equity_curve, running_max)

        # Advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(
            equity_curve, drawdown_percent, dd_periods
        )

        return {
            'max_drawdown_dollars': max_dd_dollars,
            'max_drawdown_percent': max_dd_percent,
            'max_dd_date': max_dd_date,
            'peak_date': peak_date,
            'peak_value': peak_before_max_dd,
            'trough_value': equity_curve.loc[max_dd_date],
            'recovery_info': recovery_info,
            'drawdown_periods': dd_periods,
            'advanced_metrics': advanced_metrics,
            'underwater_curve': drawdown_percent,
            'running_max': running_max
        }

    def from_trades(self, trades_df: pd.DataFrame, initial_capital: float = 10000) -> Tuple[Dict, pd.DataFrame]:
        """
        Calcula drawdown desde DataFrame de trades

        Args:
            trades_df: DataFrame con columnas ['date', 'pnl']
            initial_capital: Capital inicial

        Returns:
            Tuple de (drawdown_metrics, equity_curve_df)
        """
        required_cols = ['date', 'pnl']
        if not all(col in trades_df.columns for col in required_cols):
            raise ValueError(f"DataFrame debe contener columnas: {required_cols}")

        # Preparar datos
        df = trades_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Crear equity curve
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['equity'] = initial_capital + df['cumulative_pnl']

        # Calcular drawdown metrics
        equity_series = df.set_index('date')['equity']
        drawdown_metrics = self.from_equity_curve(equity_series)

        return drawdown_metrics, df

    def _analyze_recovery(self, equity_curve: pd.Series,
                         max_dd_date, peak_value: float) -> Dict:
        """
        Analiza recovery time y patrones de recuperación
        """
        # Buscar recovery (cuando equity alcanza o supera peak anterior)
        post_dd_equity = equity_curve[max_dd_date:]
        recovery_mask = post_dd_equity >= peak_value

        if recovery_mask.any():
            recovery_date = recovery_mask[recovery_mask].index[0]
            peak_date = equity_curve[equity_curve == peak_value].index[0]

            recovery_days = (recovery_date - peak_date).days
            dd_duration_days = (max_dd_date - peak_date).days

            # Análisis de recovery path
            recovery_equity = equity_curve[max_dd_date:recovery_date]
            recovery_slope = (recovery_equity.iloc[-1] - recovery_equity.iloc[0]) / len(recovery_equity)

            return {
                'recovered': True,
                'recovery_date': recovery_date,
                'recovery_days': recovery_days,
                'drawdown_duration_days': dd_duration_days,
                'total_days_to_recovery': recovery_days,
                'recovery_slope': recovery_slope,
                'recovery_path_volatility': recovery_equity.pct_change().std()
            }
        else:
            # Still in drawdown
            days_since_peak = (equity_curve.index[-1] -
                              equity_curve[equity_curve == peak_value].index[0]).days

            return {
                'recovered': False,
                'recovery_date': None,
                'recovery_days': None,
                'drawdown_duration_days': days_since_peak,
                'total_days_to_recovery': None,
                'current_dd_duration': days_since_peak
            }

    def _identify_drawdown_periods(self, equity_curve: pd.Series,
                                  running_max: pd.Series) -> List[Dict]:
        """
        Identifica todos los períodos de drawdown

        Returns:
            Lista de dicts con información de cada período de drawdown
        """
        drawdown_periods = []
        in_drawdown = False
        current_period = {}

        for i, (date, equity) in enumerate(equity_curve.items()):
            peak = running_max.loc[date]
            dd_percent = (equity - peak) / peak

            if not in_drawdown and dd_percent < -0.001:  # Start drawdown (>0.1% loss)
                in_drawdown = True
                current_period = {
                    'start_date': date,
                    'peak_value': peak,
                    'start_equity': equity
                }

            elif in_drawdown and dd_percent >= 0:  # End drawdown (recovery)
                current_period.update({
                    'end_date': date,
                    'trough_value': min(equity_curve[current_period['start_date']:date]),
                    'trough_date': equity_curve[current_period['start_date']:date].idxmin(),
                    'recovery_value': equity
                })

                # Calculate period metrics
                trough_value = current_period['trough_value']
                peak_value = current_period['peak_value']

                current_period.update({
                    'max_dd_dollars': trough_value - peak_value,
                    'max_dd_percent': (trough_value - peak_value) / peak_value,
                    'duration_days': (current_period['end_date'] - current_period['start_date']).days,
                    'recovery_days': (date - current_period['trough_date']).days
                })

                drawdown_periods.append(current_period)
                in_drawdown = False
                current_period = {}

        # Handle ongoing drawdown
        if in_drawdown:
            last_date = equity_curve.index[-1]
            trough_value = equity_curve[current_period['start_date']:].min()
            trough_date = equity_curve[current_period['start_date']:].idxmin()

            current_period.update({
                'end_date': None,
                'trough_value': trough_value,
                'trough_date': trough_date,
                'recovery_value': None,
                'max_dd_dollars': trough_value - current_period['peak_value'],
                'max_dd_percent': (trough_value - current_period['peak_value']) / current_period['peak_value'],
                'duration_days': (last_date - current_period['start_date']).days,
                'recovery_days': None,
                'ongoing': True
            })

            drawdown_periods.append(current_period)

        return drawdown_periods

    def _calculate_advanced_metrics(self, equity_curve: pd.Series,
                                   drawdown_percent: pd.Series,
                                   dd_periods: List[Dict]) -> Dict:
        """
        Calcula métricas avanzadas de drawdown
        """
        # Pain Index (average drawdown)
        pain_index = drawdown_percent.mean()

        # Ulcer Index (RMS of drawdowns)
        ulcer_index = np.sqrt((drawdown_percent ** 2).mean())

        # Calmar Ratio (anual return / max drawdown)
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        periods = len(equity_curve)
        days_per_year = 252  # Trading days

        if periods > 0:
            annual_return = ((1 + total_return) ** (days_per_year / periods)) - 1
        else:
            annual_return = 0

        max_dd = abs(drawdown_percent.min())
        calmar_ratio = annual_return / max_dd if max_dd != 0 else np.inf

        # Sterling Ratio (similar to Calmar but uses average of worst N drawdowns)
        completed_periods = [p for p in dd_periods if not p.get('ongoing', False)]
        if len(completed_periods) >= 3:
            worst_3_dd = sorted([abs(p['max_dd_percent']) for p in completed_periods], reverse=True)[:3]
            avg_worst_dd = np.mean(worst_3_dd)
            sterling_ratio = annual_return / avg_worst_dd if avg_worst_dd != 0 else np.inf
        else:
            sterling_ratio = calmar_ratio

        # Burke Ratio (return / sqrt(sum of squared drawdowns))
        if len(completed_periods) > 0:
            sum_squared_dd = sum([p['max_dd_percent'] ** 2 for p in completed_periods])
            burke_ratio = annual_return / np.sqrt(sum_squared_dd) if sum_squared_dd > 0 else np.inf
        else:
            burke_ratio = np.inf

        # Drawdown distribution
        dd_durations = [p['duration_days'] for p in completed_periods]
        recovery_times = [p['recovery_days'] for p in completed_periods if p['recovery_days'] is not None]

        return {
            'pain_index': pain_index,
            'ulcer_index': ulcer_index,
            'calmar_ratio': calmar_ratio,
            'sterling_ratio': sterling_ratio,
            'burke_ratio': burke_ratio,
            'total_drawdown_periods': len(dd_periods),
            'completed_drawdown_periods': len(completed_periods),
            'avg_drawdown_duration': np.mean(dd_durations) if dd_durations else 0,
            'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0,
            'max_drawdown_duration': max(dd_durations) if dd_durations else 0,
            'drawdown_frequency': len(completed_periods) / (periods / days_per_year) if periods > 0 else 0
        }

    def plot_drawdown_analysis(self, equity_curve: pd.Series,
                              drawdown_metrics: Dict,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Crea visualización completa del análisis de drawdown
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Análisis Completo de Drawdown', fontsize=16, fontweight='bold')

        # 1. Equity Curve con Drawdown Shading
        ax1 = axes[0, 0]
        equity_curve.plot(ax=ax1, color='blue', linewidth=2, label='Equity Curve')
        drawdown_metrics['running_max'].plot(ax=ax1, color='green', alpha=0.7, label='Running Maximum')

        # Shade drawdown areas
        underwater = drawdown_metrics['underwater_curve']
        ax1.fill_between(underwater.index, 0, underwater, alpha=0.3, color='red', label='Drawdown')

        ax1.set_title('Equity Curve y Running Maximum')
        ax1.set_ylabel('Account Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Underwater Curve
        ax2 = axes[0, 1]
        underwater_percent = drawdown_metrics['underwater_curve'] * 100
        underwater_percent.plot(ax=ax2, color='red', linewidth=2)
        ax2.fill_between(underwater_percent.index, 0, underwater_percent, alpha=0.3, color='red')
        ax2.axhline(y=drawdown_metrics['max_drawdown_percent'] * 100,
                   color='darkred', linestyle='--', linewidth=2,
                   label=f'Max DD: {drawdown_metrics["max_drawdown_percent"]*100:.1f}%')

        ax2.set_title('Underwater Curve')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Drawdown Duration Distribution
        ax3 = axes[1, 0]
        dd_periods = drawdown_metrics['drawdown_periods']
        completed_periods = [p for p in dd_periods if not p.get('ongoing', False)]

        if completed_periods:
            durations = [p['duration_days'] for p in completed_periods]
            ax3.hist(durations, bins=min(len(durations), 20), alpha=0.7, color='orange', edgecolor='black')
            ax3.axvline(np.mean(durations), color='red', linestyle='--',
                       label=f'Avg: {np.mean(durations):.1f} days')

        ax3.set_title('Distribución de Duración de Drawdowns')
        ax3.set_xlabel('Duración (días)')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Drawdown Metrics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        metrics_text = f"""
        MÉTRICAS DE DRAWDOWN

        Maximum Drawdown: {drawdown_metrics['max_drawdown_percent']*100:.2f}%
        Max DD ($): ${drawdown_metrics['max_drawdown_dollars']:,.2f}

        Pain Index: {drawdown_metrics['advanced_metrics']['pain_index']*100:.2f}%
        Ulcer Index: {drawdown_metrics['advanced_metrics']['ulcer_index']*100:.2f}%

        Calmar Ratio: {drawdown_metrics['advanced_metrics']['calmar_ratio']:.2f}
        Sterling Ratio: {drawdown_metrics['advanced_metrics']['sterling_ratio']:.2f}

        Total DD Periods: {drawdown_metrics['advanced_metrics']['total_drawdown_periods']}
        Avg Duration: {drawdown_metrics['advanced_metrics']['avg_drawdown_duration']:.1f} days
        Avg Recovery: {drawdown_metrics['advanced_metrics']['avg_recovery_time']:.1f} days

        Recovery Status: {'✅ Recovered' if drawdown_metrics['recovery_info']['recovered'] else '❌ In Drawdown'}
        """

        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        return fig

    def generate_drawdown_report(self, equity_curve: pd.Series) -> str:
        """
        Genera reporte textual completo de drawdown
        """
        metrics = self.from_equity_curve(equity_curve)

        report = f"""
REPORTE DE ANÁLISIS DE DRAWDOWN
{'='*50}

MÁXIMO DRAWDOWN
- Drawdown máximo: {metrics['max_drawdown_percent']*100:.2f}% (${metrics['max_drawdown_dollars']:,.2f})
- Fecha del mínimo: {metrics['max_dd_date']}
- Valor en el peak: ${metrics['peak_value']:,.2f}
- Valor en el trough: ${metrics['trough_value']:,.2f}

RECUPERACIÓN
- Estado: {'Recuperado' if metrics['recovery_info']['recovered'] else 'En drawdown'}
- Duración del DD: {metrics['recovery_info']['drawdown_duration_days']} días
"""

        if metrics['recovery_info']['recovered']:
            report += f"""- Tiempo de recuperación: {metrics['recovery_info']['recovery_days']} días
- Fecha de recuperación: {metrics['recovery_info']['recovery_date']}
"""

        report += f"""
MÉTRICAS AVANZADAS
- Pain Index: {metrics['advanced_metrics']['pain_index']*100:.2f}%
- Ulcer Index: {metrics['advanced_metrics']['ulcer_index']*100:.2f}%
- Calmar Ratio: {metrics['advanced_metrics']['calmar_ratio']:.2f}
- Sterling Ratio: {metrics['advanced_metrics']['sterling_ratio']:.2f}

ESTADÍSTICAS DE DRAWDOWN
- Total períodos de drawdown: {metrics['advanced_metrics']['total_drawdown_periods']}
- Duración promedio: {metrics['advanced_metrics']['avg_drawdown_duration']:.1f} días
- Tiempo de recuperación promedio: {metrics['advanced_metrics']['avg_recovery_time']:.1f} días
- Duración máxima: {metrics['advanced_metrics']['max_drawdown_duration']} días
- Frecuencia anual: {metrics['advanced_metrics']['drawdown_frequency']:.1f} drawdowns/año

PERÍODOS DE DRAWDOWN PRINCIPALES
"""

        # Top 5 worst drawdowns
        completed_periods = [p for p in metrics['drawdown_periods'] if not p.get('ongoing', False)]
        worst_periods = sorted(completed_periods, key=lambda x: x['max_dd_percent'])[:5]

        for i, period in enumerate(worst_periods, 1):
            report += f"""
{i}. {period['start_date'].strftime('%Y-%m-%d')} - {period['end_date'].strftime('%Y-%m-%d')}
   Drawdown: {period['max_dd_percent']*100:.2f}% (${period['max_dd_dollars']:,.2f})
   Duración: {period['duration_days']} días, Recuperación: {period['recovery_days']} días
"""

        return report


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


def main():
    parser = argparse.ArgumentParser(description="Calculadora de Maximum Drawdown")

    # Input options
    parser.add_argument('--trades', type=str, help="Archivo CSV con trades")
    parser.add_argument('--equity', type=str, help="Archivo CSV con equity curve")
    parser.add_argument('--capital', type=float, default=10000,
                       help="Capital inicial para análisis de trades")

    # Output options
    parser.add_argument('--plot', action='store_true',
                       help="Generar gráficos de análisis")
    parser.add_argument('--report', action='store_true',
                       help="Generar reporte textual detallado")
    parser.add_argument('--save-plot', type=str,
                       help="Guardar gráfico en archivo")

    args = parser.parse_args()

    analyzer = DrawdownAnalyzer()

    try:
        if args.equity:
            # Análisis desde equity curve
            print(f"Analizando equity curve desde: {args.equity}")
            data = load_data(args.equity)

            if 'equity' not in data.columns:
                raise ValueError("CSV debe contener columna 'equity'")

            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                equity_series = data.set_index('date')['equity']
            else:
                equity_series = data['equity']

            metrics = analyzer.from_equity_curve(equity_series)

        elif args.trades:
            # Análisis desde trades
            print(f"Analizando trades desde: {args.trades}")
            trades_df = load_data(args.trades)

            metrics, equity_df = analyzer.from_trades(trades_df, args.capital)
            equity_series = equity_df.set_index('date')['equity']

        else:
            raise ValueError("Debe proporcionar --trades o --equity")

        # Mostrar métricas principales
        print(f"\n{'='*50}")
        print("MÉTRICAS PRINCIPALES DE DRAWDOWN")
        print(f"{'='*50}")
        print(f"Maximum Drawdown: {metrics['max_drawdown_percent']*100:.2f}%")
        print(f"Maximum Drawdown ($): ${metrics['max_drawdown_dollars']:,.2f}")
        print(f"Pain Index: {metrics['advanced_metrics']['pain_index']*100:.2f}%")
        print(f"Calmar Ratio: {metrics['advanced_metrics']['calmar_ratio']:.2f}")
        print(f"Recovery Status: {'✅ Recovered' if metrics['recovery_info']['recovered'] else '❌ In Drawdown'}")

        # Generar reporte detallado
        if args.report:
            report = analyzer.generate_drawdown_report(equity_series)
            print(report)

        # Generar gráficos
        if args.plot or args.save_plot:
            fig = analyzer.plot_drawdown_analysis(equity_series, metrics)

            if args.save_plot:
                fig.savefig(args.save_plot, dpi=300, bbox_inches='tight')
                print(f"Gráfico guardado en: {args.save_plot}")

            if args.plot:
                plt.show()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


"""
EJEMPLOS DE USO:
================

1. Análisis básico desde trades:
   python calculate_drawdown.py --trades my_trades.csv --capital 10000

2. Análisis con gráficos:
   python calculate_drawdown.py --equity equity_curve.csv --plot

3. Reporte completo:
   python calculate_drawdown.py --trades trades.csv --report --plot

4. Guardar análisis:
   python calculate_drawdown.py --equity equity.csv --save-plot drawdown_analysis.png

FORMATO DE ARCHIVOS CSV:
========================

trades.csv:
date,pnl
2024-01-01,150.50
2024-01-02,-75.25
2024-01-03,200.00

equity.csv:
date,equity
2024-01-01,10150.50
2024-01-02,10075.25
2024-01-03,10275.25

INTERPRETACIÓN DE MÉTRICAS:
===========================

Maximum Drawdown:
- < 5%    : Excelente control de riesgo
- 5-10%   : Bueno para small caps
- 10-20%  : Aceptable pero preocupante
- > 20%   : Señal de alerta - revisar strategy

Pain Index (Average Drawdown):
- < 2%    : Muy consistente
- 2-5%    : Buena consistencia
- > 5%    : Puede indicar problemas

Calmar Ratio (Return/Max DD):
- > 3.0   : Excelente
- 1.0-3.0 : Bueno
- < 1.0   : Pobre risk-adjusted performance

NOTAS IMPORTANTES:
==================

1. Small caps típicamente tienen drawdowns más altos
2. Recovery time es crucial - long recovery indica problemas structurales
3. Frequency of drawdowns puede indicar strategy instability
4. Pain Index es mejor que Max DD para evaluar consistency
5. Siempre compare con benchmarks relevantes (IWM, IJR)

INTEGRACIÓN CON RISK MANAGEMENT:
===============================

1. Set position sizing based on historical Max DD
2. Use real-time drawdown monitoring for circuit breakers
3. Adjust strategy parameters if drawdown exceeds historical patterns
4. Consider strategy correlation during portfolio-level DD analysis
"""