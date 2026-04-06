> 🇪🇸 [Leer en Español](fundamental_analysis.es.md) | 🇺🇸 **English**

# Fundamental Analysis for Quantitative Trading

## Introduction

Fundamental analysis evaluates a company's intrinsic value based on its financial statements, competitive position, and future prospects. For quantitative trading, we automate these analyses to identify value investing opportunities and integrate them with technical strategies.

## Core Concepts

### Why Does Fundamental Analysis Work?

**Intrinsic Value Theory:**
- Every company has a real value based on its fundamentals
- Market prices fluctuate around intrinsic value
- Discrepancies create investment opportunities
- Over the long term, price converges toward intrinsic value

**Advantages for Small Caps:**
- Less analyst coverage = more inefficiencies
- Higher volatility = larger price/value discrepancies
- Information less processed by the market
- Opportunities before they are discovered

### Key Metrics

**Profitability:**
- ROE (Return on Equity)
- ROA (Return on Assets) 
- ROC (Return on Capital)
- Profit Margins

**Valuation:**
- P/E Ratio (Price to Earnings)
- P/B Ratio (Price to Book)
- P/S Ratio (Price to Sales)
- EV/EBITDA

**Efficiency:**
- Asset Turnover
- Inventory Turnover
- Working Capital Management

## Intrinsic Value - DCF Model

The Discounted Cash Flow (DCF) is the most rigorous method for calculating a company's intrinsic value.

### Complete Implementation

```python
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DCFValuator:
    """
    Intrinsic Value Calculator using DCF model
    """
    
    def __init__(self, ticker):
        """
        Parameters
        ----------
        ticker : str
            Stock symbol to evaluate
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.financial_data = {}
        self.dcf_assumptions = {}
        
    def get_financial_data(self):
        """
        Get financial data needed for DCF
        """
        try:
            # Financial statements
            self.cashflow = self.stock.cashflow
            self.financials = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.info = self.stock.info
            
            # Key data
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
            print(f"Error getting financial data for {self.ticker}: {e}")
            return False
    
    def calculate_growth_rates(self, years=5):
        """
        Calculate historical growth rates
        """
        try:
            # Historical FCF growth
            fcf_historical = self.cashflow.loc["Free Cash Flow"]
            if len(fcf_historical) >= 2:
                fcf_growth = (fcf_historical.iloc[0] / fcf_historical.iloc[-1]) ** (1/len(fcf_historical)) - 1
            else:
                fcf_growth = 0.05  # Default 5%
            
            # Historical revenue growth
            revenue_historical = self.financials.loc["Total Revenue"]
            if len(revenue_historical) >= 2:
                revenue_growth = (revenue_historical.iloc[0] / revenue_historical.iloc[-1]) ** (1/len(revenue_historical)) - 1
            else:
                revenue_growth = 0.03  # Default 3%
            
            return {
                'fcf_growth': max(min(fcf_growth, 0.15), -0.05),  # Cap between -5% and 15%
                'revenue_growth': max(min(revenue_growth, 0.12), -0.02)  # Cap between -2% and 12%
            }
            
        except Exception as e:
            print(f"Error calculating growth rates: {e}")
            return {'fcf_growth': 0.05, 'revenue_growth': 0.03}
    
    def set_dcf_assumptions(self, growth_rate=None, discount_rate=0.10, 
                           terminal_growth_rate=0.025, projection_years=5):
        """
        Set assumptions for the DCF model
        
        Parameters
        ----------
        growth_rate : float
            Annual FCF growth rate (if None, calculated automatically)
        discount_rate : float
            Discount rate (estimated WACC)
        terminal_growth_rate : float
            Perpetual growth rate
        projection_years : int
            Explicit projection years
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
        Calculate complete DCF valuation
        
        Returns
        -------
        dict
            DCF valuation results
        """
        if not self.financial_data:
            if not self.get_financial_data():
                return {'error': 'Could not get financial data'}
        
        if not self.dcf_assumptions:
            self.set_dcf_assumptions()
        
        # Extract data
        fcf_base = self.financial_data['free_cash_flow']
        if fcf_base <= 0:
            return {'error': 'Free Cash Flow negativo o cero'}
        
        growth_rate = self.dcf_assumptions['growth_rate']
        discount_rate = self.dcf_assumptions['discount_rate']
        terminal_growth = self.dcf_assumptions['terminal_growth_rate']
        years = self.dcf_assumptions['projection_years']
        
        # Project future FCF
        projected_fcf = []
        for year in range(1, years + 1):
            fcf_year = fcf_base * ((1 + growth_rate) ** year)
            projected_fcf.append(fcf_year)
        
        # Calculate terminal value
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        
        # Discount cash flows to present
        present_value_fcf = []
        for year, fcf in enumerate(projected_fcf, 1):
            pv = fcf / ((1 + discount_rate) ** year)
            present_value_fcf.append(pv)
        
        # Discount terminal value
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)
        
        # Total enterprise value
        enterprise_value = sum(present_value_fcf) + pv_terminal
        
        # Adjust for net debt
        net_debt = self.financial_data['total_debt'] - self.financial_data['cash_and_equivalents']
        equity_value = enterprise_value - net_debt
        
        # Value per share
        shares_outstanding = self.financial_data['shares_outstanding']
        if shares_outstanding <= 0:
            return {'error': 'Invalid shares outstanding'}
        
        intrinsic_value_per_share = equity_value / shares_outstanding
        current_price = self.financial_data['current_price']
        
        # Margin of safety
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
        Generate recommendation based on margin of safety
        """
        if margin_of_safety > 0.3:
            return "STRONG BUY - Margin of safety excelente"
        elif margin_of_safety > 0.15:
            return "BUY - Good margin of safety"
        elif margin_of_safety > 0:
            return "HOLD - Margin of safety marginal"
        elif margin_of_safety > -0.15:
            return "HOLD - Slightly overvalued"
        else:
            return "SELL - Significantly overvalued"
    
    def sensitivity_analysis(self, growth_range=(-0.02, 0.02), discount_range=(-0.02, 0.02)):
        """
        Sensitivity analysis for key assumptions
        """
        base_valuation = self.calculate_dcf_valuation()
        if 'error' in base_valuation:
            return base_valuation
        
        base_growth = self.dcf_assumptions['growth_rate']
        base_discount = self.dcf_assumptions['discount_rate']
        
        sensitivity_results = []
        
        # Growth rate variations
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
        
        # Discount rate variations
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
        
        # Restore base assumptions
        self.dcf_assumptions['growth_rate'] = base_growth
        self.dcf_assumptions['discount_rate'] = base_discount
        
        return {
            'base_valuation': base_valuation,
            'sensitivity_results': sensitivity_results
        }

def dcf_screening(tickers, min_margin_of_safety=0.15):
    """
    Screening multiple stocks using DCF
    """
    results = []
    
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        
        try:
            valuator = DCFValuator(ticker)
            valuation = valuator.calculate_dcf_valuation()
            
            if 'error' not in valuation:
                # Only include if minimum criteria are met
                if (valuation['margin_of_safety'] > min_margin_of_safety and 
                    valuation['intrinsic_value_per_share'] > 0):
                    results.append(valuation)
                    
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue
    
    # Sort by margin of safety
    results.sort(key=lambda x: x['margin_of_safety'], reverse=True)
    
    return results

# Usage example
def dcf_example_analysis():
    """
    Complete DCF analysis example
    """
    ticker = "AAPL"
    print(f"=== DCF ANALYSIS: {ticker} ===\\n")
    
    # Create valuator
    valuator = DCFValuator(ticker)
    
    # Get data
    if not valuator.get_financial_data():
        print("❌ Error getting financial data")
        return
    
    # Show key data
    print("📊 KEY FINANCIAL DATA:")
    for key, value in valuator.financial_data.items():
        if isinstance(value, (int, float)):
            if abs(value) > 1e9:
                print(f"   {key}: ${value/1e9:.2f}B")
            elif abs(value) > 1e6:
                print(f"   {key}: ${value/1e6:.2f}M")
            else:
                print(f"   {key}: ${value:,.2f}")
    
    # Calculate growth rates
    growth_rates = valuator.calculate_growth_rates()
    print(f"\\n📈 HISTORICAL GROWTH RATES:")
    print(f"   FCF Growth: {growth_rates['fcf_growth']:.1%}")
    print(f"   Revenue Growth: {growth_rates['revenue_growth']:.1%}")
    
    # Set assumptions (using historical growth)
    assumptions = valuator.set_dcf_assumptions(
        growth_rate=growth_rates['fcf_growth']
    )
    print(f"\\n⚙️ DCF ASSUMPTIONS:")
    for key, value in assumptions.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.1%}")
        else:
            print(f"   {key}: {value}")
    
    # Calculate valuation
    valuation = valuator.calculate_dcf_valuation()
    
    if 'error' in valuation:
        print(f"❌ Error in valuation: {valuation['error']}")
        return
    
    print(f"\\n💰 DCF RESULTS:")
    print(f"   Intrinsic Value: ${valuation['intrinsic_value_per_share']:.2f}")
    print(f"   Current Price: ${valuation['current_price']:.2f}")
    print(f"   Margin of Safety: {valuation['margin_of_safety']:.1%}")
    print(f"   Upside Potential: {valuation['upside_potential']:.1%}")
    print(f"   Recommendation: {valuation['recommendation']}")
    
    # Sensitivity analysis
    print(f"\\n🔍 SENSITIVITY ANALYSIS:")
    sensitivity = valuator.sensitivity_analysis()
    
    if 'error' not in sensitivity:
        print(f"   Range of intrinsic values:")
        values = [result['intrinsic_value'] for result in sensitivity['sensitivity_results']]
        print(f"   Minimum: ${min(values):.2f}")
        print(f"   Maximum: ${max(values):.2f}")
        print(f"   Range: ±{(max(values) - min(values))/2:.2f}")
    
    return valuation

if __name__ == "__main__":
    dcf_example_analysis()
```

## Joel Greenblatt's Magic Formula

The Magic Formula combines profitability (ROC) and valuation (Earnings Yield) to identify high-quality undervalued stocks.

### Complete Implementation

```python
class MagicFormulaScreener:
    """
    Joel Greenblatt's Magic Formula Implementation
    """
    
    def __init__(self, min_market_cap=50e6):
        """
        Parameters
        ----------
        min_market_cap : float
            Minimum market capitalization in USD
        """
        self.min_market_cap = min_market_cap
        self.results = []
    
    def calculate_metrics(self, ticker):
        """
        Calculate Magic Formula metrics for a stock
        
        Returns
        -------
        dict
            Calculated metrics or None on error
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get data necesarios
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            
            # Verify minimum market capitalization
            market_cap = info.get('marketCap', 0)
            if market_cap < self.min_market_cap:
                return None
            
            # Basic data
            current_price = info.get('currentPrice', 0)
            shares_outstanding = info.get('sharesOutstanding', 0)
            
            if current_price <= 0 or shares_outstanding <= 0:
                return None
            
            # EBIT (Earnings Before Interest and Taxes)
            if "EBIT" in financials.index:
                ebit = financials.loc["EBIT"].iloc[0]
            else:
                # Calculate approximate EBIT
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
            
            # P/E Ratio for reference
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
            print(f"Error calculating metrics for {ticker}: {e}")
            return None
    
    def screen_stocks(self, tickers):
        """
        Apply Magic Formula screening to ticker list
        
        Parameters
        ----------
        tickers : list
            List of stock symbols
            
        Returns
        -------
        pd.DataFrame
            Results sorted by ranking
        """
        results = []
        
        print(f"Screening {len(tickers)} stocks with Magic Formula...")
        
        for ticker in tickers:
            print(f"Analyzing {ticker}...")
            metrics = self.calculate_metrics(ticker)
            
            if metrics is not None:
                results.append(metrics)
        
        if not results:
            print("❌ No valid stocks found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate rankings
        # Ranking ROC (1 = mejor ROC)
        df['roc_rank'] = df['roc'].rank(ascending=False, na_option='bottom')
        
        # Ranking Earnings Yield (1 = mejor EY)
        df['ey_rank'] = df['earnings_yield'].rank(ascending=False, na_option='bottom')
        
        # Combined ranking (lower is better)
        df['magic_formula_rank'] = df['roc_rank'] + df['ey_rank']
        
        # Sort by ranking
        df = df.sort_values('magic_formula_rank')
        
        # Add percentiles
        df['roc_percentile'] = df['roc'].rank(pct=True) * 100
        df['ey_percentile'] = df['earnings_yield'].rank(pct=True) * 100
        
        self.results = df
        return df
    
    def get_top_stocks(self, n=10):
        """
        Get the top n stocks according to Magic Formula
        """
        if self.results.empty:
            return pd.DataFrame()
        
        return self.results.head(n)
    
    def analyze_results(self):
        """
        Statistical analysis of results
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
        Simple backtest of Magic Formula strategy
        """
        if self.results.empty:
            return {'error': 'No results for backtest'}
        
        # Select top stocks
        top_stocks = self.get_top_stocks(top_n)
        tickers = top_stocks['ticker'].tolist()
        
        try:
            # Get historical prices
            price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
            
            if price_data.empty:
                return {'error': 'Could not get price data'}
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Equal-weighted portfolio return
            portfolio_returns = returns.mean(axis=1)
            
            # Benchmark (SPY)
            spy_data = yf.download('SPY', start=start_date, end=end_date)['Close']
            spy_returns = spy_data.pct_change().dropna()
            
            # Performance metrics
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
            return {'error': f'Error in backtest: {e}'}

def magic_formula_example():
    """
    Complete Magic Formula example
    """
    print("=== JOEL GREENBLATT'S MAGIC FORMULA ===\\n")
    
    # List of tickers to analyze (expand as needed)
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA",
        "JPM", "JNJ", "PG", "KO", "DIS", "INTC", "CSCO", "VZ",
        "PFE", "WMT", "HD", "MRK", "ABBV", "CRM", "ADBE", "TMO"
    ]
    
    # Create screener
    screener = MagicFormulaScreener(min_market_cap=1e9)  # Min $1B market cap
    
    # Run screening
    results = screener.screen_stocks(tickers)
    
    if results.empty:
        print("❌ No results found")
        return
    
    print(f"✅ Analyzed {len(results)} stocks\\n")
    
    # Show top 10
    top_10 = screener.get_top_stocks(10)
    print("🏆 TOP 10 ACCORDING TO MAGIC FORMULA:")
    print(top_10[['ticker', 'roc', 'earnings_yield', 'pe_ratio', 'magic_formula_rank']].round(2))
    
    # Statistics
    stats = screener.analyze_results()
    print(f"\\n📊 STATISTICS:")
    print(f"   Average ROC: {stats['average_roc']:.1f}%")
    print(f"   Average Earnings Yield: {stats['average_earnings_yield']:.1f}%")
    print(f"   Average P/E: {stats['average_pe_ratio']:.1f}")
    print(f"   Median Market Cap: ${stats['median_market_cap']/1e9:.1f}B")
    
    # Backtest
    backtest = screener.backtest_strategy(
        start_date="2023-01-01", 
        end_date="2024-01-01",
        top_n=5
    )
    
    if 'error' not in backtest:
        print(f"\\n📈 BACKTEST (Top 5 stocks):")
        print(f"   Portfolio Return: {backtest['portfolio_return']:.1%}")
        print(f"   SPY Return: {backtest['spy_return']:.1%}")
        print(f"   Outperformance: {backtest['outperformance']:.1%}")
        print(f"   Portfolio Sharpe: {backtest['portfolio_sharpe']:.2f}")
        print(f"   SPY Sharpe: {backtest['spy_sharpe']:.2f}")
    
    return screener

if __name__ == "__main__":
    magic_formula_example()
```

## Advanced Financial Metrics

### 1. Quality Score
```python
def calculate_quality_score(ticker):
    """
    Calculate quality score for a company
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
    Complete screener for value investing
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
        Evaluate a stock against value criteria
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
            
            # Evaluate criteria
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
        Screen multiple stocks
        """
        results = []
        
        for ticker in tickers:
            result = self.screen_stock(ticker)
            if 'error' not in result and result['value_score'] >= min_score:
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x['value_score'], reverse=True)
        
        return results
```

## Integration with Quantitative Trading

### 1. Fundamental + Technical Strategy
```python
def fundamental_technical_strategy(ticker, dcf_margin_threshold=0.15):
    """
    Combine fundamental analysis with technical signals
    """
    # Fundamental analysis
    valuator = DCFValuator(ticker)
    dcf_result = valuator.calculate_dcf_valuation()
    
    if 'error' in dcf_result:
        return {'error': 'Could not complete fundamental analysis'}
    
    # Only consider if there is margin of safety
    if dcf_result['margin_of_safety'] < dcf_margin_threshold:
        return {'signal': 'HOLD', 'reason': 'Insufficient margin of safety'}
    
    # Get data técnicos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=252)
    price_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Simple technical indicators
    current_price = price_data['Close'].iloc[-1]
    sma_50 = price_data['Close'].rolling(50).mean().iloc[-1]
    sma_200 = price_data['Close'].rolling(200).mean().iloc[-1]
    rsi = calculate_rsi(price_data['Close']).iloc[-1]
    
    # Combined signal logic
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
    """Helper function for RSI"""
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
    Build portfolio based on fundamental analysis
    """
    candidates = []
    
    # Fundamental screening
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
    
    # Select best candidates
    candidates.sort(key=lambda x: x['composite_score'], reverse=True)
    selected = candidates[:max_positions]
    
    # Calculate weights (based on composite score)
    total_score = sum(stock['composite_score'] for stock in selected)
    for stock in selected:
        stock['weight'] = stock['composite_score'] / total_score
    
    return selected
```

## Evaluation Metrics

### Performance Tracking
```python
def track_fundamental_performance(portfolio, start_date, end_date):
    """
    Track fundamental portfolio performance
    """
    tickers = [stock['ticker'] for stock in portfolio]
    weights = [stock['weight'] for stock in portfolio]
    
    # Get prices
    price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = price_data.pct_change().dropna()
    
    # Portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Benchmark
    spy_data = yf.download('SPY', start=start_date, end=end_date)['Close']
    spy_returns = spy_data.pct_change().dropna()
    
    # Metrics
    portfolio_cumret = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    spy_cumret = (1 + spy_returns).cumprod().iloc[-1] - 1
    
    portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    spy_sharpe = spy_returns.mean() / spy_returns.std() * np.sqrt(252)
    
    # Calculate max drawdown
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

## Limitations and Best Practices

### Limitations of Fundamental Analysis
1. **Historical data**: Financial statements reflect the past
2. **Accounting practices**: Different accounting methods can distort metrics
3. **Market timing**: Value may take years to be realized
4. **Small caps data**: Less reliable or available data

### Best Practices
```python
FUNDAMENTAL_BEST_PRACTICES = {
    'data_quality': {
        'min_market_cap': 50e6,     # Minimum $50M for reliable data
        'max_pe_outlier': 100,      # Avoid extreme P/E
        'min_trading_volume': 100000 # Minimum daily volume
    },
    'valuation': {
        'dcf_sensitivity': True,     # Always perform sensitivity analysis
        'multiple_methods': True,    # Use DCF + comparable multiples
        'margin_of_safety': 0.20,   # Minimum 20% margin of safety
        'growth_cap': 0.15          # Growth cap at 15% annual
    },
    'portfolio': {
        'max_position_size': 0.10,   # Maximum 10% per position
        'diversification': True,     # Diversify by sectors
        'rebalancing': 'quarterly',  # Rebalance quarterly
        'fundamental_weight': 0.70   # 70% fundamental weight, 30% technical
    }
}
```

## Next Step

With Fundamental Analysis completed, you now have a full arsenal of professional quantitative tools. The documentation covers everything from technical indicators to machine learning and fundamental analysis, providing a solid foundation for institutional-grade quantitative trading.