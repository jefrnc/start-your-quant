"""
Calculadora de Profit Factor
============================

Script completo para calcular Profit Factor y métricas relacionadas
para estrategias de trading cuantitativo. Incluye análisis detallado
de winning vs losing trades, distribuciones, y optimización.

Profit Factor = Gross Profit / Gross Loss = Total $ ganado / Total $ perdido

Para small cap trading:
- Profit Factor > 1.5 = Good
- Profit Factor > 2.0 = Excellent
- Profit Factor > 3.0 = Elite

¿Por qué es importante?:
- Mide eficiencia real de la estrategia
- Independent de win rate (puede tener low win rate pero high PF)
- Critical para small caps por volatilidad extrema
- Más realistic que otras métricas

Uso:
    python calculate_profit_factor.py --trades trades.csv
    python calculate_profit_factor.py --trades trades.csv --analysis detailed
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


class ProfitFactorAnalyzer:
    """
    Analizador completo de Profit Factor para trading cuantitativo

    Calcula:
    - Basic Profit Factor (gross profit / gross loss)
    - Modified Profit Factor (considera frequency)
    - Conditional Profit Factor por subgrupos
    - Distribution analysis de wins vs losses
    - Trade efficiency metrics
    """

    def __init__(self):
        self.trades_df = None

    def calculate_basic_profit_factor(self, trades: Union[pd.Series, list, np.array]) -> Dict:
        """
        Calcula Profit Factor básico y métricas relacionadas

        Args:
            trades: Serie de P&L de trades individuales

        Returns:
            Dict con métricas básicas de profit factor
        """
        if isinstance(trades, (list, np.ndarray)):
            trades = pd.Series(trades)

        trades = trades.dropna()

        if len(trades) == 0:
            raise ValueError("No hay trades válidos para analizar")

        # Separar winning y losing trades
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]
        breakeven_trades = trades[trades == 0]

        # Calcular gross profit y gross loss
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0

        # Profit Factor
        if gross_loss == 0:
            profit_factor = np.inf if gross_profit > 0 else np.nan
        else:
            profit_factor = gross_profit / gross_loss

        # Win rate y métricas relacionadas
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        loss_rate = len(losing_trades) / total_trades if total_trades > 0 else 0

        # Average wins y losses
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0

        # Win/Loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        # Expectancy
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        # Largest win/loss
        largest_win = winning_trades.max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades.min() if len(losing_trades) > 0 else 0

        return {
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': gross_profit - gross_loss,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'breakeven_trades': len(breakeven_trades),
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'expectancy': expectancy,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }

    def analyze_from_dataframe(self, trades_df: pd.DataFrame) -> Dict:
        """
        Análisis completo desde DataFrame de trades

        Args:
            trades_df: DataFrame con al menos columna 'pnl'

        Returns:
            Dict con análisis completo
        """
        if 'pnl' not in trades_df.columns:
            raise ValueError("DataFrame debe contener columna 'pnl'")

        self.trades_df = trades_df.copy()

        # Análisis básico
        basic_metrics = self.calculate_basic_profit_factor(trades_df['pnl'])

        # Análisis por categorías
        categorical_analysis = {}

        # Por símbolo (si existe)
        if 'symbol' in trades_df.columns:
            categorical_analysis['by_symbol'] = self._analyze_by_category(trades_df, 'symbol')

        # Por día de la semana (si existe fecha)
        if 'date' in trades_df.columns or 'entry_time' in trades_df.columns:
            date_col = 'date' if 'date' in trades_df.columns else 'entry_time'
            trades_df[date_col] = pd.to_datetime(trades_df[date_col])
            trades_df['day_of_week'] = trades_df[date_col].dt.day_name()
            categorical_analysis['by_day_of_week'] = self._analyze_by_category(trades_df, 'day_of_week')

        # Por hora del día (si existe timestamp)
        if 'entry_time' in trades_df.columns:
            trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
            categorical_analysis['by_hour'] = self._analyze_by_category(trades_df, 'entry_hour')

        # Por estrategia (si existe)
        if 'strategy' in trades_df.columns:
            categorical_analysis['by_strategy'] = self._analyze_by_category(trades_df, 'strategy')

        # Análisis de distribución
        distribution_analysis = self._analyze_distribution(trades_df['pnl'])

        # Análisis temporal
        temporal_analysis = {}
        if 'date' in trades_df.columns or 'entry_time' in trades_df.columns:
            temporal_analysis = self._analyze_temporal_trends(trades_df)

        # Análisis de outliers
        outlier_analysis = self._analyze_outliers(trades_df['pnl'])

        return {
            'basic_metrics': basic_metrics,
            'categorical_analysis': categorical_analysis,
            'distribution_analysis': distribution_analysis,
            'temporal_analysis': temporal_analysis,
            'outlier_analysis': outlier_analysis
        }

    def _analyze_by_category(self, df: pd.DataFrame, category_col: str) -> Dict:
        """
        Analiza profit factor por categoría específica
        """
        category_stats = {}

        for category, group in df.groupby(category_col):
            if len(group) >= 5:  # Mínimo 5 trades para analysis significativo
                stats = self.calculate_basic_profit_factor(group['pnl'])
                category_stats[category] = stats

        # Encontrar mejores y peores categorías
        if category_stats:
            best_pf = max(category_stats.items(), key=lambda x: x[1]['profit_factor'])
            worst_pf = min(category_stats.items(), key=lambda x: x[1]['profit_factor'])

            return {
                'category_stats': category_stats,
                'best_category': best_pf[0],
                'best_profit_factor': best_pf[1]['profit_factor'],
                'worst_category': worst_pf[0],
                'worst_profit_factor': worst_pf[1]['profit_factor'],
                'total_categories': len(category_stats)
            }

        return {'category_stats': {}}

    def _analyze_distribution(self, pnl_series: pd.Series) -> Dict:
        """
        Analiza distribución de wins vs losses
        """
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]

        distribution_stats = {}

        # Estadísticas de wins
        if len(wins) > 0:
            distribution_stats['wins'] = {
                'count': len(wins),
                'mean': wins.mean(),
                'median': wins.median(),
                'std': wins.std(),
                'min': wins.min(),
                'max': wins.max(),
                'percentiles': {
                    '25th': wins.quantile(0.25),
                    '75th': wins.quantile(0.75),
                    '90th': wins.quantile(0.90),
                    '95th': wins.quantile(0.95)
                }
            }

        # Estadísticas de losses
        if len(losses) > 0:
            distribution_stats['losses'] = {
                'count': len(losses),
                'mean': losses.mean(),
                'median': losses.median(),
                'std': losses.std(),
                'min': losses.min(),
                'max': losses.max(),
                'percentiles': {
                    '25th': losses.quantile(0.25),
                    '75th': losses.quantile(0.75),
                    '10th': losses.quantile(0.10),
                    '5th': losses.quantile(0.05)
                }
            }

        # Comparación de distribuciones
        if len(wins) > 0 and len(losses) > 0:
            distribution_stats['comparison'] = {
                'win_std_vs_loss_std': wins.std() / abs(losses.std()),
                'largest_win_vs_largest_loss': wins.max() / abs(losses.min()),
                'median_win_vs_median_loss': wins.median() / abs(losses.median()),
                'win_skewness': wins.skew(),
                'loss_skewness': losses.skew()
            }

        return distribution_stats

    def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analiza trends temporales en profit factor
        """
        date_col = 'date' if 'date' in df.columns else 'entry_time'
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])

        # Profit factor por mes
        df_temp['month'] = df_temp[date_col].dt.to_period('M')
        monthly_pf = {}

        for month, group in df_temp.groupby('month'):
            if len(group) >= 5:
                pf_stats = self.calculate_basic_profit_factor(group['pnl'])
                monthly_pf[str(month)] = pf_stats['profit_factor']

        # Rolling profit factor (30 trades)
        rolling_pf = []
        window_size = min(30, len(df) // 3)  # Adaptive window size

        for i in range(window_size, len(df)):
            window_trades = df.iloc[i-window_size:i]['pnl']
            pf_stats = self.calculate_basic_profit_factor(window_trades)
            rolling_pf.append(pf_stats['profit_factor'])

        # Trend analysis
        if len(rolling_pf) > 10:
            # Simple linear trend
            x = np.arange(len(rolling_pf))
            slope = np.polyfit(x, rolling_pf, 1)[0]
            trend_direction = "Improving" if slope > 0.01 else "Deteriorating" if slope < -0.01 else "Stable"
        else:
            slope = 0
            trend_direction = "Insufficient data"

        return {
            'monthly_profit_factors': monthly_pf,
            'rolling_profit_factor': rolling_pf,
            'trend_slope': slope,
            'trend_direction': trend_direction,
            'recent_pf': rolling_pf[-5:] if len(rolling_pf) >= 5 else [],
            'pf_volatility': np.std(rolling_pf) if len(rolling_pf) > 1 else 0
        }

    def _analyze_outliers(self, pnl_series: pd.Series) -> Dict:
        """
        Analiza outliers y su impacto en profit factor
        """
        # Calcular profit factor sin outliers
        q75, q25 = np.percentile(pnl_series, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - (iqr * 1.5)
        upper_bound = q75 + (iqr * 1.5)

        # Identificar outliers
        outliers = pnl_series[(pnl_series < lower_bound) | (pnl_series > upper_bound)]
        clean_trades = pnl_series[(pnl_series >= lower_bound) & (pnl_series <= upper_bound)]

        # Calcular métricas
        original_pf = self.calculate_basic_profit_factor(pnl_series)
        clean_pf = self.calculate_basic_profit_factor(clean_trades) if len(clean_trades) > 0 else {}

        # Outliers impact
        positive_outliers = outliers[outliers > upper_bound]
        negative_outliers = outliers[outliers < lower_bound]

        return {
            'total_outliers': len(outliers),
            'positive_outliers': len(positive_outliers),
            'negative_outliers': len(negative_outliers),
            'outlier_percentage': len(outliers) / len(pnl_series) * 100,
            'original_profit_factor': original_pf['profit_factor'],
            'clean_profit_factor': clean_pf.get('profit_factor', np.nan),
            'outlier_impact': original_pf['profit_factor'] - clean_pf.get('profit_factor', 0),
            'largest_positive_outlier': positive_outliers.max() if len(positive_outliers) > 0 else 0,
            'largest_negative_outlier': negative_outliers.min() if len(negative_outliers) > 0 else 0
        }

    def calculate_modified_profit_factor(self, trades: pd.Series,
                                       frequency_weight: float = 0.1) -> float:
        """
        Calcula Modified Profit Factor que considera frequency

        Modified PF = (Gross Profit * Win Rate) / (Gross Loss * Loss Rate)
        """
        basic_metrics = self.calculate_basic_profit_factor(trades)

        if basic_metrics['gross_loss'] == 0:
            return np.inf

        modified_pf = (
            (basic_metrics['gross_profit'] * basic_metrics['win_rate']) /
            (basic_metrics['gross_loss'] * basic_metrics['loss_rate'])
        )

        return modified_pf

    def plot_profit_factor_analysis(self, analysis_results: Dict,
                                   figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Crea visualización completa del análisis de profit factor
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Análisis Completo de Profit Factor', fontsize=16, fontweight='bold')

        # 1. Win vs Loss Distribution
        ax1 = axes[0, 0]
        dist_data = analysis_results['distribution_analysis']

        if 'wins' in dist_data and 'losses' in dist_data:
            wins_data = [v for v in self.trades_df['pnl'] if v > 0]
            losses_data = [v for v in self.trades_df['pnl'] if v < 0]

            ax1.hist(wins_data, bins=20, alpha=0.7, color='green', label='Wins', density=True)
            ax1.hist(losses_data, bins=20, alpha=0.7, color='red', label='Losses', density=True)

        ax1.set_title('Distribución de Wins vs Losses')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Profit Factor por Categoría (ejemplo: por símbolo)
        ax2 = axes[0, 1]
        if 'by_symbol' in analysis_results['categorical_analysis']:
            symbol_data = analysis_results['categorical_analysis']['by_symbol']['category_stats']
            if symbol_data:
                symbols = list(symbol_data.keys())[:10]  # Top 10
                pf_values = [symbol_data[s]['profit_factor'] for s in symbols]

                bars = ax2.bar(range(len(symbols)), pf_values, color=['green' if pf > 1 else 'red' for pf in pf_values])
                ax2.set_xticks(range(len(symbols)))
                ax2.set_xticklabels(symbols, rotation=45)
                ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7)

        ax2.set_title('Profit Factor por Símbolo (Top 10)')
        ax2.set_ylabel('Profit Factor')
        ax2.grid(True, alpha=0.3)

        # 3. Rolling Profit Factor
        ax3 = axes[0, 2]
        if 'rolling_profit_factor' in analysis_results['temporal_analysis']:
            rolling_pf = analysis_results['temporal_analysis']['rolling_profit_factor']
            if rolling_pf:
                ax3.plot(rolling_pf, linewidth=2, color='blue')
                ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even')
                ax3.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='Good (1.5)')
                ax3.axhline(y=2.0, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent (2.0)')

        ax3.set_title('Rolling Profit Factor (30 trades)')
        ax3.set_xlabel('Trade Window')
        ax3.set_ylabel('Profit Factor')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Win Rate vs Avg Win/Loss
        ax4 = axes[1, 0]
        basic = analysis_results['basic_metrics']

        # Scatter plot conceptual
        ax4.scatter([basic['win_rate']], [basic['win_loss_ratio']],
                   s=basic['profit_factor']*50, alpha=0.7, color='blue')

        ax4.set_xlabel('Win Rate')
        ax4.set_ylabel('Win/Loss Ratio')
        ax4.set_title('Win Rate vs Win/Loss Ratio\n(Size = Profit Factor)')
        ax4.grid(True, alpha=0.3)

        # 5. Monthly Profit Factor
        ax5 = axes[1, 1]
        if 'monthly_profit_factors' in analysis_results['temporal_analysis']:
            monthly_data = analysis_results['temporal_analysis']['monthly_profit_factors']
            if monthly_data:
                months = list(monthly_data.keys())
                pf_values = list(monthly_data.values())

                bars = ax5.bar(range(len(months)), pf_values,
                              color=['green' if pf > 1.5 else 'orange' if pf > 1 else 'red' for pf in pf_values])
                ax5.set_xticks(range(len(months)))
                ax5.set_xticklabels(months, rotation=45)
                ax5.axhline(y=1, color='black', linestyle='-', alpha=0.7)
                ax5.axhline(y=1.5, color='green', linestyle='--', alpha=0.7)

        ax5.set_title('Profit Factor por Mes')
        ax5.set_ylabel('Profit Factor')
        ax5.grid(True, alpha=0.3)

        # 6. Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')

        basic = analysis_results['basic_metrics']
        outlier = analysis_results['outlier_analysis']

        summary_text = f"""
        RESUMEN DE PROFIT FACTOR

        Profit Factor: {basic['profit_factor']:.2f}
        Gross Profit: ${basic['gross_profit']:,.2f}
        Gross Loss: ${basic['gross_loss']:,.2f}

        Total Trades: {basic['total_trades']}
        Win Rate: {basic['win_rate']*100:.1f}%
        Avg Win: ${basic['avg_win']:.2f}
        Avg Loss: ${basic['avg_loss']:.2f}

        Win/Loss Ratio: {basic['win_loss_ratio']:.2f}
        Expectancy: ${basic['expectancy']:.2f}

        Largest Win: ${basic['largest_win']:.2f}
        Largest Loss: ${basic['largest_loss']:.2f}

        Outliers: {outlier['total_outliers']} ({outlier['outlier_percentage']:.1f}%)
        Clean PF: {outlier['clean_profit_factor']:.2f}
        """

        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()
        return fig

    def generate_profit_factor_report(self, analysis_results: Dict) -> str:
        """
        Genera reporte textual completo de profit factor
        """
        basic = analysis_results['basic_metrics']
        dist = analysis_results['distribution_analysis']
        outlier = analysis_results['outlier_analysis']

        # Classification
        pf = basic['profit_factor']
        if pf >= 3.0:
            classification = "🏆 ELITE"
        elif pf >= 2.0:
            classification = "🚀 EXCELLENT"
        elif pf >= 1.5:
            classification = "✅ GOOD"
        elif pf >= 1.2:
            classification = "⚠️ FAIR"
        elif pf >= 1.0:
            classification = "❌ POOR"
        else:
            classification = "💀 LOSING"

        report = f"""
REPORTE DE ANÁLISIS DE PROFIT FACTOR
{'='*50}

CLASIFICACIÓN: {classification}
Profit Factor: {pf:.3f}

MÉTRICAS BÁSICAS
- Gross Profit: ${basic['gross_profit']:,.2f}
- Gross Loss: ${basic['gross_loss']:,.2f}
- Net Profit: ${basic['net_profit']:,.2f}
- Total Trades: {basic['total_trades']}

DISTRIBUCIÓN DE TRADES
- Winning Trades: {basic['winning_trades']} ({basic['win_rate']*100:.1f}%)
- Losing Trades: {basic['losing_trades']} ({basic['loss_rate']*100:.1f}%)
- Breakeven Trades: {basic['breakeven_trades']}

EFICIENCIA
- Average Win: ${basic['avg_win']:.2f}
- Average Loss: ${basic['avg_loss']:.2f}
- Win/Loss Ratio: {basic['win_loss_ratio']:.2f}
- Expectancy per Trade: ${basic['expectancy']:.2f}

EXTREMOS
- Largest Win: ${basic['largest_win']:.2f}
- Largest Loss: ${basic['largest_loss']:.2f}
"""

        # Distribución analysis
        if 'wins' in dist and 'losses' in dist:
            report += f"""
ANÁLISIS DE DISTRIBUCIÓN
Wins:
- Median Win: ${dist['wins']['median']:.2f}
- Win Std Dev: ${dist['wins']['std']:.2f}
- 90th Percentile: ${dist['wins']['percentiles']['90th']:.2f}

Losses:
- Median Loss: ${dist['losses']['median']:.2f}
- Loss Std Dev: ${dist['losses']['std']:.2f}
- 10th Percentile: ${dist['losses']['percentiles']['10th']:.2f}
"""

        # Outliers analysis
        report += f"""
ANÁLISIS DE OUTLIERS
- Total Outliers: {outlier['total_outliers']} ({outlier['outlier_percentage']:.1f}%)
- Profit Factor sin Outliers: {outlier['clean_profit_factor']:.2f}
- Impacto de Outliers: {outlier['outlier_impact']:+.2f}
- Mayor Outlier Positivo: ${outlier['largest_positive_outlier']:.2f}
- Mayor Outlier Negativo: ${outlier['largest_negative_outlier']:.2f}
"""

        # Recommendations
        report += f"""
RECOMENDACIONES
"""
        if pf < 1.2:
            report += "❌ Profit Factor muy bajo - revisar completamente la estrategia\n"
        elif pf < 1.5:
            report += "⚠️ Profit Factor marginal - optimizar exits y risk management\n"
        elif pf < 2.0:
            report += "✅ Profit Factor sólido - buscar oportunidades de scaling\n"
        else:
            report += "🚀 Profit Factor excelente - mantener disciplina y consistency\n"

        if basic['win_rate'] < 0.4:
            report += "• Win rate bajo - revisar entry criteria\n"
        elif basic['win_rate'] > 0.8:
            report += "• Win rate muy alto - verificar que no sea overfitting\n"

        if basic['win_loss_ratio'] < 1.0:
            report += "• Avg loss > avg win - mejorar profit taking\n"
        elif basic['win_loss_ratio'] > 3.0:
            report += "• Excelente win/loss ratio - mantener discipline\n"

        if outlier['outlier_percentage'] > 20:
            report += "• Muchos outliers - considerar filtros adicionales\n"

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
    parser = argparse.ArgumentParser(description="Calculadora de Profit Factor")

    parser.add_argument('--trades', type=str, required=True,
                       help="Archivo CSV con trades")
    parser.add_argument('--analysis', type=str, default='basic',
                       choices=['basic', 'detailed'],
                       help="Nivel de análisis")
    parser.add_argument('--plot', action='store_true',
                       help="Generar gráficos de análisis")
    parser.add_argument('--report', action='store_true',
                       help="Generar reporte textual detallado")
    parser.add_argument('--save-plot', type=str,
                       help="Guardar gráfico en archivo")

    args = parser.parse_args()

    analyzer = ProfitFactorAnalyzer()

    try:
        print(f"Analizando trades desde: {args.trades}")
        trades_df = load_data(args.trades)

        if args.analysis == 'basic':
            # Análisis básico
            basic_metrics = analyzer.calculate_basic_profit_factor(trades_df['pnl'])

            print(f"\n{'='*40}")
            print("PROFIT FACTOR ANALYSIS")
            print(f"{'='*40}")
            print(f"Profit Factor: {basic_metrics['profit_factor']:.3f}")
            print(f"Win Rate: {basic_metrics['win_rate']*100:.1f}%")
            print(f"Win/Loss Ratio: {basic_metrics['win_loss_ratio']:.2f}")
            print(f"Expectancy: ${basic_metrics['expectancy']:.2f}")

        else:
            # Análisis detallado
            analysis_results = analyzer.analyze_from_dataframe(trades_df)

            # Mostrar métricas principales
            basic = analysis_results['basic_metrics']
            print(f"\n{'='*50}")
            print("ANÁLISIS DETALLADO DE PROFIT FACTOR")
            print(f"{'='*50}")
            print(f"Profit Factor: {basic['profit_factor']:.3f}")
            print(f"Gross Profit: ${basic['gross_profit']:,.2f}")
            print(f"Gross Loss: ${basic['gross_loss']:,.2f}")
            print(f"Total Trades: {basic['total_trades']}")

            # Generar reporte detallado
            if args.report:
                report = analyzer.generate_profit_factor_report(analysis_results)
                print(report)

            # Generar gráficos
            if args.plot or args.save_plot:
                fig = analyzer.plot_profit_factor_analysis(analysis_results)

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

1. Análisis básico:
   python calculate_profit_factor.py --trades trades.csv

2. Análisis detallado con reporte:
   python calculate_profit_factor.py --trades trades.csv --analysis detailed --report

3. Análisis con gráficos:
   python calculate_profit_factor.py --trades trades.csv --analysis detailed --plot

4. Guardar análisis completo:
   python calculate_profit_factor.py --trades trades.csv --analysis detailed --report --save-plot pf_analysis.png

FORMATO CSV REQUERIDO:
======================

trades.csv (mínimo):
date,pnl
2024-01-01,150.50
2024-01-02,-75.25
2024-01-03,200.00

trades.csv (completo):
date,symbol,pnl,strategy,entry_time
2024-01-01,AAPL,150.50,gap_go,09:30:00
2024-01-02,TSLA,-75.25,vwap_reclaim,10:15:00

INTERPRETACIÓN:
===============

Profit Factor Ranges:
- > 3.0   : Elite performance 🏆
- 2.0-3.0 : Excellent 🚀
- 1.5-2.0 : Good ✅
- 1.2-1.5 : Fair ⚠️
- 1.0-1.2 : Poor ❌
- < 1.0   : Losing strategy 💀

Para Small Caps:
- PF > 2.0 es especialmente impressive debido a volatilidad
- Consider frequency: high PF con low frequency puede no ser scalable
- Watch for outlier dependency: PF no debe depender de pocos big wins

FACTORES QUE AFECTAN PROFIT FACTOR:
===================================

1. Entry Timing: Better entries → higher avg wins
2. Exit Strategy: Good exits → lower avg losses
3. Position Sizing: Proper sizing → more consistent results
4. Market Conditions: Bull markets inflate PF temporarily
5. Outliers: Few big wins/losses can skew results significantly

LIMITACIONES:
=============

1. No considera frequency of trades
2. Puede ser dominado por outliers
3. No refleja risk-adjusted performance
4. High PF con low sample size puede ser misleading

MEJORES PRÁCTICAS:
==================

1. Combinar con otras métricas (Sharpe, Max DD)
2. Analizar distribution de wins vs losses
3. Monitor rolling PF para detect degradation
4. Consider profit factor por different market conditions
5. Validate con out-of-sample data
"""