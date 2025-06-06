# Análisis Fundamental para Trading Cuantitativo

## Introducción

El análisis fundamental evalúa el valor intrínseco de una empresa basándose en sus estados financieros, posición competitiva y perspectivas futuras. Para trading cuantitativo, automatizamos estos análisis para identificar oportunidades de value investing y integrarlas con estrategias técnicas.

## Conceptos Fundamentales

### ¿Por Qué Funciona el Análisis Fundamental?

**Teoría del Valor Intrínseco:**
- Toda empresa tiene un valor real basado en sus fundamentos
- Los precios de mercado fluctúan alrededor del valor intrínseco
- Las discrepancias crean oportunidades de inversión
- A largo plazo, el precio converge hacia el valor intrínseco

**Ventajas para Small Caps:**
- Menos cobertura de analistas = más ineficiencias
- Mayor volatilidad = mayores discrepancias precio/valor
- Información menos procesada por el mercado
- Oportunidades antes de que se descubran

### Métricas Clave

**Rentabilidad:**
- ROE (Return on Equity)
- ROA (Return on Assets) 
- ROC (Return on Capital)
- Profit Margins

**Valoración:**
- P/E Ratio (Price to Earnings)
- P/B Ratio (Price to Book)
- P/S Ratio (Price to Sales)
- EV/EBITDA

**Eficiencia:**
- Asset Turnover
- Inventory Turnover
- Working Capital Management

## Valor Intrínseco - Modelo DCF

El Discounted Cash Flow (DCF) es el método más riguroso para calcular el valor intrínseco de una empresa.

### Implementación Completa

```python
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DCFValuator:
    """
    Calculadora de Valor Intrínseco usando modelo DCF
    """
    
    def __init__(self, ticker):
        """
        Parámetros
        ----------
        ticker : str
            Símbolo de la acción a evaluar
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.financial_data = {}
        self.dcf_assumptions = {}
        
    def get_financial_data(self):
        """
        Obtener datos financieros necesarios para DCF
        """
        try:
            # Estados financieros
            self.cashflow = self.stock.cashflow
            self.financials = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.info = self.stock.info
            
            # Datos clave
            self.financial_data = {
                'free_cash_flow': self.cashflow.loc["Free Cash Flow"].iloc[0],
                'total_debt': self.balance_sheet.loc["Long Term Debt"].iloc[0] if "Long Term Debt" in self.balance_sheet.index else 0,
                'cash_and_equivalents': self.balance_sheet.loc["Cash And Cash Equivalents"].iloc[0] if "Cash And Cash Equivalents" in self.balance_sheet.index else 0,
                'shares_outstanding': self.info.get("sharesOutstanding", self.info.get("impliedSharesOutstanding", 0)),
                'current_price': self.info.get("currentPrice", 0),
                'market_cap': self.info.get("marketCap", 0),
                'revenue': self.financials.loc["Total Revenue"].iloc[0] if "Total Revenue" in self.financials.index else 0,
                'net_income': self.financials.loc["Net Income"].iloc[0] if "Net Income" in self.financials.index else 0
            }
            
            return True
            
        except Exception as e:
            print(f"Error obteniendo datos financieros para {self.ticker}: {e}")
            return False
    
    def calculate_growth_rates(self, years=5):
        """
        Calcular tasas de crecimiento históricas
        """
        try:
            # Crecimiento de FCF histórico
            fcf_historical = self.cashflow.loc["Free Cash Flow"]
            if len(fcf_historical) >= 2:
                fcf_growth = (fcf_historical.iloc[0] / fcf_historical.iloc[-1]) ** (1/len(fcf_historical)) - 1
            else:
                fcf_growth = 0.05  # Default 5%
            
            # Crecimiento de revenue histórico
            revenue_historical = self.financials.loc["Total Revenue"]
            if len(revenue_historical) >= 2:
                revenue_growth = (revenue_historical.iloc[0] / revenue_historical.iloc[-1]) ** (1/len(revenue_historical)) - 1
            else:
                revenue_growth = 0.03  # Default 3%
            
            return {
                'fcf_growth': max(min(fcf_growth, 0.15), -0.05),  # Cap entre -5% y 15%
                'revenue_growth': max(min(revenue_growth, 0.12), -0.02)  # Cap entre -2% y 12%
            }
            
        except Exception as e:
            print(f"Error calculando tasas de crecimiento: {e}")
            return {'fcf_growth': 0.05, 'revenue_growth': 0.03}
    
    def set_dcf_assumptions(self, growth_rate=None, discount_rate=0.10, 
                           terminal_growth_rate=0.025, projection_years=5):
        """
        Establecer supuestos para el modelo DCF
        
        Parámetros
        ----------
        growth_rate : float
            Tasa de crecimiento anual de FCF (si None, se calcula automáticamente)
        discount_rate : float
            Tasa de descuento (WACC estimado)
        terminal_growth_rate : float
            Tasa de crecimiento perpetuo
        projection_years : int
            Años de proyección explícita
        """
        if growth_rate is None:
            growth_rates = self.calculate_growth_rates()
            growth_rate = growth_rates['fcf_growth']
        
        self.dcf_assumptions = {
            'growth_rate': growth_rate,
            'discount_rate': discount_rate,
            'terminal_growth_rate': terminal_growth_rate,
            'projection_years': projection_years
        }
        
        return self.dcf_assumptions
    
    def calculate_dcf_valuation(self):
        """
        Calcular valoración DCF completa
        
        Returns
        -------
        dict
            Resultados de valoración DCF
        """
        if not self.financial_data:
            if not self.get_financial_data():
                return {'error': 'No se pudieron obtener datos financieros'}
        
        if not self.dcf_assumptions:
            self.set_dcf_assumptions()
        
        # Extraer datos
        fcf_base = self.financial_data['free_cash_flow']
        if fcf_base <= 0:
            return {'error': 'Free Cash Flow negativo o cero'}
        
        growth_rate = self.dcf_assumptions['growth_rate']
        discount_rate = self.dcf_assumptions['discount_rate']
        terminal_growth = self.dcf_assumptions['terminal_growth_rate']
        years = self.dcf_assumptions['projection_years']
        
        # Proyectar FCF futuro
        projected_fcf = []
        for year in range(1, years + 1):
            fcf_year = fcf_base * ((1 + growth_rate) ** year)
            projected_fcf.append(fcf_year)
        
        # Calcular valor terminal
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        
        # Descontar flujos al presente
        present_value_fcf = []
        for year, fcf in enumerate(projected_fcf, 1):
            pv = fcf / ((1 + discount_rate) ** year)
            present_value_fcf.append(pv)
        
        # Descontar valor terminal
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)
        
        # Valor total de la empresa
        enterprise_value = sum(present_value_fcf) + pv_terminal
        
        # Ajustar por deuda neta
        net_debt = self.financial_data['total_debt'] - self.financial_data['cash_and_equivalents']
        equity_value = enterprise_value - net_debt
        
        # Valor por acción
        shares_outstanding = self.financial_data['shares_outstanding']
        if shares_outstanding <= 0:
            return {'error': 'Acciones en circulación inválidas'}
        
        intrinsic_value_per_share = equity_value / shares_outstanding
        current_price = self.financial_data['current_price']
        
        # Margen de seguridad
        margin_of_safety = (intrinsic_value_per_share - current_price) / intrinsic_value_per_share if intrinsic_value_per_share > 0 else -1
        
        return {
            'ticker': self.ticker,
            'intrinsic_value_per_share': intrinsic_value_per_share,
            'current_price': current_price,
            'margin_of_safety': margin_of_safety,
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'projected_fcf': projected_fcf,
            'terminal_value': terminal_value,
            'assumptions': self.dcf_assumptions,
            'recommendation': self.get_recommendation(margin_of_safety),
            'upside_potential': (intrinsic_value_per_share / current_price - 1) if current_price > 0 else 0
        }
    
    def get_recommendation(self, margin_of_safety):
        """
        Generar recomendación basada en margen de seguridad
        """
        if margin_of_safety > 0.3:
            return "STRONG BUY - Margen de seguridad excelente"
        elif margin_of_safety > 0.15:
            return "BUY - Buen margen de seguridad"
        elif margin_of_safety > 0:
            return "HOLD - Margen de seguridad marginal"
        elif margin_of_safety > -0.15:
            return "HOLD - Ligeramente sobrevaluada"
        else:
            return "SELL - Significativamente sobrevaluada"
    
    def sensitivity_analysis(self, growth_range=(-0.02, 0.02), discount_range=(-0.02, 0.02)):
        """
        Análisis de sensibilidad para supuestos clave
        """
        base_valuation = self.calculate_dcf_valuation()
        if 'error' in base_valuation:
            return base_valuation
        
        base_growth = self.dcf_assumptions['growth_rate']
        base_discount = self.dcf_assumptions['discount_rate']
        
        sensitivity_results = []
        
        # Variaciones en growth rate
        for delta_growth in np.linspace(growth_range[0], growth_range[1], 5):
            self.dcf_assumptions['growth_rate'] = base_growth + delta_growth
            valuation = self.calculate_dcf_valuation()
            if 'error' not in valuation:
                sensitivity_results.append({
                    'parameter': 'growth_rate',
                    'value': base_growth + delta_growth,
                    'intrinsic_value': valuation['intrinsic_value_per_share'],
                    'margin_of_safety': valuation['margin_of_safety']
                })
        
        # Variaciones en discount rate
        for delta_discount in np.linspace(discount_range[0], discount_range[1], 5):
            self.dcf_assumptions['growth_rate'] = base_growth  # Reset
            self.dcf_assumptions['discount_rate'] = base_discount + delta_discount
            valuation = self.calculate_dcf_valuation()
            if 'error' not in valuation:
                sensitivity_results.append({
                    'parameter': 'discount_rate',
                    'value': base_discount + delta_discount,
                    'intrinsic_value': valuation['intrinsic_value_per_share'],
                    'margin_of_safety': valuation['margin_of_safety']
                })
        
        # Restaurar supuestos base
        self.dcf_assumptions['growth_rate'] = base_growth
        self.dcf_assumptions['discount_rate'] = base_discount
        
        return {
            'base_valuation': base_valuation,
            'sensitivity_results': sensitivity_results
        }

def dcf_screening(tickers, min_margin_of_safety=0.15):
    """
    Screening de múltiples acciones usando DCF
    """
    results = []
    
    for ticker in tickers:
        print(f"Analizando {ticker}...")
        
        try:
            valuator = DCFValuator(ticker)
            valuation = valuator.calculate_dcf_valuation()
            
            if 'error' not in valuation:
                # Solo incluir si cumple criterios mínimos
                if (valuation['margin_of_safety'] > min_margin_of_safety and 
                    valuation['intrinsic_value_per_share'] > 0):
                    results.append(valuation)
                    
        except Exception as e:
            print(f"Error analizando {ticker}: {e}")
            continue
    
    # Ordenar por margen de seguridad
    results.sort(key=lambda x: x['margin_of_safety'], reverse=True)
    
    return results

# Ejemplo de uso
def dcf_example_analysis():
    """
    Ejemplo completo de análisis DCF
    """
    ticker = "AAPL"
    print(f"=== ANÁLISIS DCF: {ticker} ===\\n")
    
    # Crear valuador
    valuator = DCFValuator(ticker)
    
    # Obtener datos
    if not valuator.get_financial_data():
        print("❌ Error obteniendo datos financieros")
        return
    
    # Mostrar datos clave
    print("📊 DATOS FINANCIEROS CLAVE:")
    for key, value in valuator.financial_data.items():
        if isinstance(value, (int, float)):
            if abs(value) > 1e9:
                print(f"   {key}: ${value/1e9:.2f}B")
            elif abs(value) > 1e6:
                print(f"   {key}: ${value/1e6:.2f}M")
            else:
                print(f"   {key}: ${value:,.2f}")
    
    # Calcular tasas de crecimiento
    growth_rates = valuator.calculate_growth_rates()
    print(f"\\n📈 TASAS DE CRECIMIENTO HISTÓRICAS:")
    print(f"   FCF Growth: {growth_rates['fcf_growth']:.1%}")
    print(f"   Revenue Growth: {growth_rates['revenue_growth']:.1%}")
    
    # Establecer supuestos (usando crecimiento histórico)
    assumptions = valuator.set_dcf_assumptions(
        growth_rate=growth_rates['fcf_growth']
    )
    print(f"\\n⚙️ SUPUESTOS DCF:")
    for key, value in assumptions.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.1%}")
        else:
            print(f"   {key}: {value}")
    
    # Calcular valoración
    valuation = valuator.calculate_dcf_valuation()
    
    if 'error' in valuation:
        print(f"❌ Error en valoración: {valuation['error']}")
        return
    
    print(f"\\n💰 RESULTADOS DCF:")
    print(f"   Valor Intrínseco: ${valuation['intrinsic_value_per_share']:.2f}")
    print(f"   Precio Actual: ${valuation['current_price']:.2f}")
    print(f"   Margen de Seguridad: {valuation['margin_of_safety']:.1%}")
    print(f"   Potencial Upside: {valuation['upside_potential']:.1%}")
    print(f"   Recomendación: {valuation['recommendation']}")
    
    # Análisis de sensibilidad
    print(f"\\n🔍 ANÁLISIS DE SENSIBILIDAD:")
    sensitivity = valuator.sensitivity_analysis()
    
    if 'error' not in sensitivity:
        print(f"   Rango de valores intrínsecos:")
        values = [result['intrinsic_value'] for result in sensitivity['sensitivity_results']]
        print(f"   Mínimo: ${min(values):.2f}")
        print(f"   Máximo: ${max(values):.2f}")
        print(f"   Rango: ±{(max(values) - min(values))/2:.2f}")
    
    return valuation

if __name__ == "__main__":
    dcf_example_analysis()
```

## Fórmula Mágica de Joel Greenblatt

La Fórmula Mágica combina rentabilidad (ROC) y valoración (Earnings Yield) para identificar acciones infravaloradas de calidad.

### Implementación Completa

```python
class MagicFormulaScreener:
    """
    Implementación de la Fórmula Mágica de Joel Greenblatt
    """
    
    def __init__(self, min_market_cap=50e6):
        """
        Parámetros
        ----------
        min_market_cap : float
            Capitalización de mercado mínima en USD
        """
        self.min_market_cap = min_market_cap
        self.results = []
    
    def calculate_metrics(self, ticker):
        """
        Calcular métricas de la Fórmula Mágica para una acción
        
        Returns
        -------
        dict
            Métricas calculadas o None si hay error
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Obtener datos necesarios
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            
            # Verificar capitalización mínima
            market_cap = info.get('marketCap', 0)
            if market_cap < self.min_market_cap:
                return None
            
            # Datos básicos
            current_price = info.get('currentPrice', 0)
            shares_outstanding = info.get('sharesOutstanding', 0)
            
            if current_price <= 0 or shares_outstanding <= 0:
                return None
            
            # EBIT (Earnings Before Interest and Taxes)
            if "EBIT" in financials.index:
                ebit = financials.loc["EBIT"].iloc[0]
            else:
                # Calcular EBIT aproximado
                operating_income = financials.loc["Operating Income"].iloc[0] if "Operating Income" in financials.index else 0
                ebit = operating_income
            
            if ebit <= 0:
                return None
            
            # Net Income
            net_income = financials.loc["Net Income"].iloc[0] if "Net Income" in financials.index else 0
            
            # Balance Sheet items
            total_assets = balance_sheet.loc["Total Assets"].iloc[0] if "Total Assets" in balance_sheet.index else 0
            current_liabilities = balance_sheet.loc["Current Liabilities"].iloc[0] if "Current Liabilities" in balance_sheet.index else 0
            
            # Working Capital
            working_capital = total_assets - current_liabilities
            
            if working_capital <= 0:
                return None
            
            # 1. Return on Capital (ROC)
            # ROC = EBIT / Working Capital
            roc = (ebit / working_capital) * 100
            
            # 2. Earnings Yield (EY)
            # EY = EBIT / Market Cap
            earnings_yield = (ebit / market_cap) * 100
            
            # P/E Ratio para referencia
            pe_ratio = market_cap / net_income if net_income > 0 else float('inf')
            
            return {
                'ticker': ticker,
                'market_cap': market_cap,
                'current_price': current_price,
                'roc': roc,
                'earnings_yield': earnings_yield,
                'pe_ratio': pe_ratio,
                'ebit': ebit,
                'working_capital': working_capital,
                'net_income': net_income
            }
            
        except Exception as e:
            print(f"Error calculando métricas para {ticker}: {e}")
            return None
    
    def screen_stocks(self, tickers):
        """
        Aplicar screening de Fórmula Mágica a lista de tickers
        
        Parámetros
        ----------
        tickers : list
            Lista de símbolos de acciones
            
        Returns
        -------
        pd.DataFrame
            Resultados ordenados por ranking
        """
        results = []
        
        print(f"Screening {len(tickers)} acciones con Fórmula Mágica...")
        
        for ticker in tickers:
            print(f"Analizando {ticker}...")
            metrics = self.calculate_metrics(ticker)
            
            if metrics is not None:
                results.append(metrics)
        
        if not results:
            print("❌ No se encontraron acciones válidas")
            return pd.DataFrame()
        
        # Convertir a DataFrame
        df = pd.DataFrame(results)
        
        # Calcular rankings
        # Ranking ROC (1 = mejor ROC)
        df['roc_rank'] = df['roc'].rank(ascending=False, na_option='bottom')
        
        # Ranking Earnings Yield (1 = mejor EY)
        df['ey_rank'] = df['earnings_yield'].rank(ascending=False, na_option='bottom')
        
        # Ranking combinado (menor es mejor)
        df['magic_formula_rank'] = df['roc_rank'] + df['ey_rank']
        
        # Ordenar por ranking
        df = df.sort_values('magic_formula_rank')
        
        # Agregar percentiles
        df['roc_percentile'] = df['roc'].rank(pct=True) * 100
        df['ey_percentile'] = df['earnings_yield'].rank(pct=True) * 100
        
        self.results = df
        return df
    
    def get_top_stocks(self, n=10):
        """
        Obtener las mejores n acciones según Fórmula Mágica
        """
        if self.results.empty:
            return pd.DataFrame()
        
        return self.results.head(n)
    
    def analyze_results(self):
        """
        Análisis estadístico de los resultados
        """
        if self.results.empty:
            return {}
        
        stats = {
            'total_stocks_analyzed': len(self.results),
            'average_roc': self.results['roc'].mean(),
            'average_earnings_yield': self.results['earnings_yield'].mean(),
            'average_pe_ratio': self.results['pe_ratio'].mean(),
            'median_market_cap': self.results['market_cap'].median(),
            'roc_stats': self.results['roc'].describe(),
            'ey_stats': self.results['earnings_yield'].describe()
        }
        
        return stats
    
    def backtest_strategy(self, start_date="2023-01-01", end_date="2024-01-01", top_n=10):
        """
        Backtest simple de la estrategia Fórmula Mágica
        """
        if self.results.empty:
            return {'error': 'No hay resultados para backtest'}
        
        # Seleccionar top stocks
        top_stocks = self.get_top_stocks(top_n)
        tickers = top_stocks['ticker'].tolist()
        
        try:
            # Obtener precios históricos
            price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
            
            if price_data.empty:
                return {'error': 'No se pudieron obtener datos de precios'}
            
            # Calcular retornos
            returns = price_data.pct_change().dropna()
            
            # Retorno igual ponderado del portfolio
            portfolio_returns = returns.mean(axis=1)
            
            # Benchmark (SPY)
            spy_data = yf.download('SPY', start=start_date, end=end_date)['Close']
            spy_returns = spy_data.pct_change().dropna()
            
            # Métricas de performance
            portfolio_cumret = (1 + portfolio_returns).cumprod().iloc[-1] - 1
            spy_cumret = (1 + spy_returns).cumprod().iloc[-1] - 1
            
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            spy_volatility = spy_returns.std() * np.sqrt(252)
            
            portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            spy_sharpe = spy_returns.mean() / spy_returns.std() * np.sqrt(252)
            
            return {
                'portfolio_return': portfolio_cumret,
                'spy_return': spy_cumret,
                'outperformance': portfolio_cumret - spy_cumret,
                'portfolio_volatility': portfolio_volatility,
                'spy_volatility': spy_volatility,
                'portfolio_sharpe': portfolio_sharpe,
                'spy_sharpe': spy_sharpe,
                'stocks_used': tickers,
                'period': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            return {'error': f'Error en backtest: {e}'}

def magic_formula_example():
    """
    Ejemplo completo de Fórmula Mágica
    """
    print("=== FÓRMULA MÁGICA DE JOEL GREENBLATT ===\\n")
    
    # Lista de tickers para analizar (expandir según necesidad)
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA",
        "JPM", "JNJ", "PG", "KO", "DIS", "INTC", "CSCO", "VZ",
        "PFE", "WMT", "HD", "MRK", "ABBV", "CRM", "ADBE", "TMO"
    ]
    
    # Crear screener
    screener = MagicFormulaScreener(min_market_cap=1e9)  # Min $1B market cap
    
    # Ejecutar screening
    results = screener.screen_stocks(tickers)
    
    if results.empty:
        print("❌ No se encontraron resultados")
        return
    
    print(f"✅ Analizadas {len(results)} acciones\\n")
    
    # Mostrar top 10
    top_10 = screener.get_top_stocks(10)
    print("🏆 TOP 10 SEGÚN FÓRMULA MÁGICA:")
    print(top_10[['ticker', 'roc', 'earnings_yield', 'pe_ratio', 'magic_formula_rank']].round(2))
    
    # Estadísticas
    stats = screener.analyze_results()
    print(f"\\n📊 ESTADÍSTICAS:")
    print(f"   ROC Promedio: {stats['average_roc']:.1f}%")
    print(f"   Earnings Yield Promedio: {stats['average_earnings_yield']:.1f}%")
    print(f"   P/E Promedio: {stats['average_pe_ratio']:.1f}")
    print(f"   Market Cap Mediana: ${stats['median_market_cap']/1e9:.1f}B")
    
    # Backtest
    backtest = screener.backtest_strategy(
        start_date="2023-01-01", 
        end_date="2024-01-01",
        top_n=5
    )
    
    if 'error' not in backtest:
        print(f"\\n📈 BACKTEST (Top 5 acciones):")
        print(f"   Portfolio Return: {backtest['portfolio_return']:.1%}")
        print(f"   SPY Return: {backtest['spy_return']:.1%}")
        print(f"   Outperformance: {backtest['outperformance']:.1%}")
        print(f"   Portfolio Sharpe: {backtest['portfolio_sharpe']:.2f}")
        print(f"   SPY Sharpe: {backtest['spy_sharpe']:.2f}")
    
    return screener

if __name__ == "__main__":
    magic_formula_example()
```

## Métricas Financieras Avanzadas

### 1. Quality Score
```python
def calculate_quality_score(ticker):
    """
    Calcular score de calidad de una empresa
    """
    stock = yf.Ticker(ticker)
    
    try:
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        
        quality_metrics = {}
        
        # 1. Profitability
        net_margin = info.get('profitMargins', 0) * 100
        roa = info.get('returnOnAssets', 0) * 100
        roe = info.get('returnOnEquity', 0) * 100
        
        # 2. Financial Strength
        debt_to_equity = info.get('debtToEquity', 0)
        current_ratio = info.get('currentRatio', 0)
        quick_ratio = info.get('quickRatio', 0)
        
        # 3. Growth
        revenue_growth = info.get('revenueGrowth', 0) * 100
        earnings_growth = info.get('earningsGrowth', 0) * 100
        
        # 4. Efficiency
        asset_turnover = info.get('assetTurnover', 0)
        inventory_turnover = info.get('inventoryTurnover', 0)
        
        # Calculate composite score
        profitability_score = min((net_margin + roa + roe) / 3, 100)
        strength_score = min((10 - debt_to_equity + current_ratio + quick_ratio) * 10, 100)
        growth_score = min((revenue_growth + earnings_growth) / 2, 100)
        efficiency_score = min((asset_turnover + inventory_turnover) * 50, 100)
        
        quality_score = (profitability_score + strength_score + growth_score + efficiency_score) / 4
        
        return {
            'ticker': ticker,
            'quality_score': quality_score,
            'profitability_score': profitability_score,
            'strength_score': strength_score,
            'growth_score': growth_score,
            'efficiency_score': efficiency_score,
            'individual_metrics': {
                'net_margin': net_margin,
                'roa': roa,
                'roe': roe,
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth
            }
        }
        
    except Exception as e:
        return {'error': f'Error calculando quality score para {ticker}: {e}'}
```

### 2. Value Investing Screener
```python
class ValueInvestingScreener:
    """
    Screener completo para value investing
    """
    
    def __init__(self):
        self.criteria = {
            'max_pe': 15,           # P/E < 15
            'max_pb': 1.5,          # P/B < 1.5
            'min_dividend_yield': 2, # Dividend Yield > 2%
            'max_debt_equity': 0.5,  # Debt/Equity < 0.5
            'min_roe': 15,          # ROE > 15%
            'min_roa': 5,           # ROA > 5%
            'min_current_ratio': 1.5 # Current Ratio > 1.5
        }
    
    def screen_stock(self, ticker):
        """
        Evaluar una acción contra criterios value
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            metrics = {
                'ticker': ticker,
                'pe_ratio': info.get('trailingPE', float('inf')),
                'pb_ratio': info.get('priceToBook', float('inf')),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'debt_to_equity': info.get('debtToEquity', float('inf')),
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
                'current_ratio': info.get('currentRatio', 0)
            }
            
            # Evaluar criterios
            passed_criteria = 0
            total_criteria = len(self.criteria)
            
            if metrics['pe_ratio'] < self.criteria['max_pe']:
                passed_criteria += 1
            if metrics['pb_ratio'] < self.criteria['max_pb']:
                passed_criteria += 1
            if metrics['dividend_yield'] > self.criteria['min_dividend_yield']:
                passed_criteria += 1
            if metrics['debt_to_equity'] < self.criteria['max_debt_equity']:
                passed_criteria += 1
            if metrics['roe'] > self.criteria['min_roe']:
                passed_criteria += 1
            if metrics['roa'] > self.criteria['min_roa']:
                passed_criteria += 1
            if metrics['current_ratio'] > self.criteria['min_current_ratio']:
                passed_criteria += 1
            
            metrics['value_score'] = (passed_criteria / total_criteria) * 100
            metrics['passed_criteria'] = passed_criteria
            metrics['total_criteria'] = total_criteria
            
            return metrics
            
        except Exception as e:
            return {'error': f'Error screening {ticker}: {e}'}
    
    def batch_screen(self, tickers, min_score=70):
        """
        Screen múltiples acciones
        """
        results = []
        
        for ticker in tickers:
            result = self.screen_stock(ticker)
            if 'error' not in result and result['value_score'] >= min_score:
                results.append(result)
        
        # Ordenar por score
        results.sort(key=lambda x: x['value_score'], reverse=True)
        
        return results
```

## Integración con Trading Cuantitativo

### 1. Fundamental + Technical Strategy
```python
def fundamental_technical_strategy(ticker, dcf_margin_threshold=0.15):
    """
    Combinar análisis fundamental con señales técnicas
    """
    # Análisis fundamental
    valuator = DCFValuator(ticker)
    dcf_result = valuator.calculate_dcf_valuation()
    
    if 'error' in dcf_result:
        return {'error': 'No se pudo completar análisis fundamental'}
    
    # Solo considerar si hay margen de seguridad
    if dcf_result['margin_of_safety'] < dcf_margin_threshold:
        return {'signal': 'HOLD', 'reason': 'Insufficient margin of safety'}
    
    # Obtener datos técnicos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=252)
    price_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Indicadores técnicos simples
    current_price = price_data['Close'].iloc[-1]
    sma_50 = price_data['Close'].rolling(50).mean().iloc[-1]
    sma_200 = price_data['Close'].rolling(200).mean().iloc[-1]
    rsi = calculate_rsi(price_data['Close']).iloc[-1]
    
    # Lógica de señales combinada
    fundamental_bullish = dcf_result['margin_of_safety'] > dcf_margin_threshold
    technical_bullish = (current_price > sma_50 and 
                        sma_50 > sma_200 and 
                        rsi < 70)
    
    if fundamental_bullish and technical_bullish:
        signal = 'STRONG BUY'
    elif fundamental_bullish and not technical_bullish:
        signal = 'BUY (Wait for better entry)'
    elif not fundamental_bullish and technical_bullish:
        signal = 'HOLD (Technical only)'
    else:
        signal = 'SELL/AVOID'
    
    return {
        'ticker': ticker,
        'signal': signal,
        'dcf_analysis': dcf_result,
        'technical_indicators': {
            'current_price': current_price,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'rsi': rsi
        },
        'fundamental_score': dcf_result['margin_of_safety'],
        'technical_score': int(technical_bullish)
    }

def calculate_rsi(series, period=14):
    """Helper function para RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### 2. Portfolio Construction
```python
def build_fundamental_portfolio(tickers, max_positions=10, min_margin_safety=0.2):
    """
    Construir portfolio basado en análisis fundamental
    """
    candidates = []
    
    # Screening fundamental
    for ticker in tickers:
        try:
            # DCF Analysis
            valuator = DCFValuator(ticker)
            dcf = valuator.calculate_dcf_valuation()
            
            if ('error' not in dcf and 
                dcf['margin_of_safety'] > min_margin_safety):
                
                # Quality metrics
                quality = calculate_quality_score(ticker)
                
                if 'error' not in quality:
                    candidates.append({
                        'ticker': ticker,
                        'margin_of_safety': dcf['margin_of_safety'],
                        'upside_potential': dcf['upside_potential'],
                        'quality_score': quality['quality_score'],
                        'composite_score': (dcf['margin_of_safety'] * 0.6 + 
                                          quality['quality_score']/100 * 0.4)
                    })
                    
        except Exception as e:
            continue
    
    # Seleccionar mejores
    candidates.sort(key=lambda x: x['composite_score'], reverse=True)
    selected = candidates[:max_positions]
    
    # Calcular pesos (basado en composite score)
    total_score = sum(stock['composite_score'] for stock in selected)
    for stock in selected:
        stock['weight'] = stock['composite_score'] / total_score
    
    return selected
```

## Métricas de Evaluación

### Performance Tracking
```python
def track_fundamental_performance(portfolio, start_date, end_date):
    """
    Hacer seguimiento del performance de portfolio fundamental
    """
    tickers = [stock['ticker'] for stock in portfolio]
    weights = [stock['weight'] for stock in portfolio]
    
    # Obtener precios
    price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = price_data.pct_change().dropna()
    
    # Portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Benchmark
    spy_data = yf.download('SPY', start=start_date, end=end_date)['Close']
    spy_returns = spy_data.pct_change().dropna()
    
    # Métricas
    portfolio_cumret = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    spy_cumret = (1 + spy_returns).cumprod().iloc[-1] - 1
    
    portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    spy_sharpe = spy_returns.mean() / spy_returns.std() * np.sqrt(252)
    
    # Calcular max drawdown
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_drawdown = (portfolio_cumulative / portfolio_cumulative.cummax() - 1).min()
    
    return {
        'portfolio_return': portfolio_cumret,
        'benchmark_return': spy_cumret,
        'outperformance': portfolio_cumret - spy_cumret,
        'portfolio_sharpe': portfolio_sharpe,
        'benchmark_sharpe': spy_sharpe,
        'max_drawdown': portfolio_drawdown,
        'win_rate': (portfolio_returns > spy_returns).mean()
    }
```

## Limitaciones y Mejores Prácticas

### Limitaciones del Análisis Fundamental
1. **Datos históricos**: Los estados financieros reflejan el pasado
2. **Accounting practices**: Diferentes métodos contables pueden distorsionar métricas
3. **Market timing**: El valor puede tardar años en realizarse
4. **Small caps data**: Datos menos confiables o disponibles

### Mejores Prácticas
```python
FUNDAMENTAL_BEST_PRACTICES = {
    'data_quality': {
        'min_market_cap': 50e6,     # Mínimo $50M para datos confiables
        'max_pe_outlier': 100,      # Evitar P/E extremos
        'min_trading_volume': 100000 # Mínimo volumen diario
    },
    'valuation': {
        'dcf_sensitivity': True,     # Siempre hacer análisis de sensibilidad
        'multiple_methods': True,    # Usar DCF + múltiples comparables
        'margin_of_safety': 0.20,   # Mínimo 20% margen de seguridad
        'growth_cap': 0.15          # Cap de crecimiento al 15% anual
    },
    'portfolio': {
        'max_position_size': 0.10,   # Máximo 10% por posición
        'diversification': True,     # Diversificar por sectores
        'rebalancing': 'quarterly',  # Rebalancear cada trimestre
        'fundamental_weight': 0.70   # 70% peso fundamental, 30% técnico
    }
}
```

## Siguiente Paso

Con Análisis Fundamental completado, ahora tienes un arsenal completo de herramientas cuantitativas profesionales. La documentación cubre desde indicadores técnicos hasta machine learning y análisis fundamental, proporcionando una base sólida para trading cuantitativo institucional.