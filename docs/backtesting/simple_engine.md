> 🇪🇸 [Leer en Español](simple_engine.es.md) | 🇺🇸 **English**

# Simple Backtest Engine

## Building Your Own Engine

Before using complex frameworks, you need to understand how a backtest engine works under the hood. Here we build one from scratch that actually works.

## Basic Architecture

```python
class SimpleBacktestEngine:
    def __init__(self, initial_capital=10000, commission=5, slippage=0.0001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # {ticker: shares}
        self.portfolio_value = initial_capital
        
        # Tracking
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def reset(self):
        """Reset for new backtest"""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
```

## Exporting Results for Analysis

### TraderVue Integration

One of the best ways to analyze your results is using specialized platforms like TraderVue. Our system includes automatic export:

```python
from backtesting.trade_reporting import TradeReporter, export_backtest_results

# After running your backtest
results = backtester.run_backtest(data, strategy)

# Export all reports automatically
export_backtest_results(results, output_dir="./my_reports/")

# Or export specifically for TraderVue
reporter = TradeReporter(results['trades'])
reporter.to_tradervue_csv("trades_for_tradervue.csv", account_name="My_Backtest")
```

### Available Export Formats

1. **TraderVue CSV**: Directly compatible with TraderVue import
2. **Trade Detail**: CSV with all metrics for each trade
3. **Daily Journal**: Summary by day for pattern analysis
4. **Performance Report**: JSON/CSV with complete metrics
5. **Equity Curve**: For plotting capital evolution

### Post-Backtest Analysis

```python
# Generate detailed performance report
reporter.generate_performance_report("performance_analysis.json")

# Export for personal journal
reporter.to_journal_format("my_trading_journal.csv")

# Generic CSV with all metrics
reporter.to_generic_csv("complete_trades.csv")
```

### Benefits of External Analysis

- **Advanced visualizations**: TraderVue generates professional charts
- **Tag-based analysis**: Categorize trades by strategy, time, etc.
- **Comparison**: Compare different backtests side by side
- **Additional metrics**: MAE/MFE, time-of-day analysis, etc.
- **Share results**: Export reports for investors

## Order Execution

```python
def execute_order(self, ticker, shares, price, timestamp, order_type='market'):
    """Execute an order with realistic costs"""
    
    # Basic validations
    if shares == 0:
        return False
        
    # Calculate costs
    gross_value = abs(shares * price)
    commission_cost = max(self.commission, gross_value * 0.0001)  # Min $5 or 1bp
    
    # Apply slippage
    if shares > 0:  # Buy
        execution_price = price * (1 + self.slippage)
    else:  # Sell
        execution_price = price * (1 - self.slippage)
    
    net_cost = shares * execution_price + commission_cost
    
    # Verify we have enough cash/shares
    if shares > 0 and net_cost > self.cash:
        # Not enough cash
        return False
        
    if shares < 0 and ticker in self.positions:
        if abs(shares) > self.positions[ticker]:
            # Not enough shares to sell
            return False
    elif shares < 0 and ticker not in self.positions:
        # Trying to sell what we don't own
        return False
    
    # Execute the order
    self.cash -= net_cost
    
    if ticker in self.positions:
        self.positions[ticker] += shares
        if self.positions[ticker] == 0:
            del self.positions[ticker]
    else:
        self.positions[ticker] = shares
    
    # Record trade
    trade_record = {
        'timestamp': timestamp,
        'ticker': ticker,
        'shares': shares,
        'price': execution_price,
        'commission': commission_cost,
        'type': 'buy' if shares > 0 else 'sell'
    }
    self.trades.append(trade_record)
    
    return True

def calculate_portfolio_value(self, current_prices):
    """Calculate current portfolio value"""
    positions_value = 0
    
    for ticker, shares in self.positions.items():
        if ticker in current_prices:
            positions_value += shares * current_prices[ticker]
    
    self.portfolio_value = self.cash + positions_value
    return self.portfolio_value
```

## Strategy Framework

```python
class Strategy:
    """Base class for strategies"""
    
    def __init__(self, name):
        self.name = name
        self.parameters = {}
        
    def initialize(self, engine):
        """Initial setup"""
        self.engine = engine
        
    def on_data(self, data, timestamp):
        """Called on each data bar"""
        raise NotImplementedError
        
    def should_enter(self, data, ticker):
        """Entry logic"""
        return False
        
    def should_exit(self, data, ticker):
        """Exit logic"""
        return False
        
    def calculate_position_size(self, ticker, price):
        """Calculate position size"""
        return 0

class VWAPStrategy(Strategy):
    """Example: Simple VWAP strategy"""
    
    def __init__(self, risk_per_trade=0.02, stop_loss_pct=0.03):
        super().__init__("VWAP Reclaim")
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.entry_prices = {}
        
    def on_data(self, data, timestamp):
        """Process each bar"""
        for ticker in data.columns.get_level_values(0).unique():
            if ticker not in ['SPY', 'QQQ']:  # Skip ETFs for this example
                self.process_ticker(data, ticker, timestamp)
    
    def process_ticker(self, data, ticker, timestamp):
        """Process a specific ticker"""
        try:
            # Get ticker data
            ticker_data = data[ticker].loc[timestamp]
            
            # Calculate VWAP if it doesn't exist
            if 'vwap' not in ticker_data:
                return
            
            current_price = ticker_data['close']
            vwap = ticker_data['vwap']
            volume = ticker_data['volume']
            avg_volume = ticker_data.get('avg_volume', volume)
            
            # Signals
            above_vwap = current_price > vwap
            high_volume = volume > avg_volume * 1.5
            
            # Entry logic
            if (ticker not in self.engine.positions and 
                above_vwap and high_volume and
                ticker not in self.entry_prices):
                
                shares = self.calculate_position_size(ticker, current_price)
                if self.engine.execute_order(ticker, shares, current_price, timestamp):
                    self.entry_prices[ticker] = current_price
            
            # Exit logic
            elif ticker in self.engine.positions:
                should_exit = False
                exit_reason = ""
                
                # Stop loss
                if ticker in self.entry_prices:
                    if current_price < self.entry_prices[ticker] * (1 - self.stop_loss_pct):
                        should_exit = True
                        exit_reason = "stop_loss"
                
                # Take profit (2:1 R/R)
                if ticker in self.entry_prices:
                    if current_price > self.entry_prices[ticker] * (1 + self.stop_loss_pct * 2):
                        should_exit = True
                        exit_reason = "take_profit"
                
                # VWAP loss
                if not above_vwap:
                    should_exit = True
                    exit_reason = "vwap_loss"
                
                if should_exit:
                    shares = -self.engine.positions[ticker]  # Sell all
                    self.engine.execute_order(ticker, shares, current_price, timestamp)
                    if ticker in self.entry_prices:
                        del self.entry_prices[ticker]
        
        except KeyError as e:
            # Data not available for this timestamp
            print(f"Warning: No data available for {ticker} at {timestamp}: {e}")
            pass
        except Exception as e:
            # Unexpected error processing ticker
            print(f"Error processing {ticker} at {timestamp}: {e}")
            pass
    
    def calculate_position_size(self, ticker, price):
        """Calculate shares based on risk management"""
        risk_amount = self.engine.portfolio_value * self.risk_per_trade
        stop_distance = price * self.stop_loss_pct
        shares = int(risk_amount / stop_distance)
        
        # Don't use more than 20% of portfolio on one position
        max_position_value = self.engine.portfolio_value * 0.2
        max_shares = int(max_position_value / price)
        
        return min(shares, max_shares)
```

## Main Backtest Loop

```python
def run_backtest(engine, strategy, data, start_date=None, end_date=None):
    """Run complete backtest"""
    
    # Filter data by dates
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    
    # Initialize
    engine.reset()
    strategy.initialize(engine)
    
    print(f"Starting backtest: {strategy.name}")
    print(f"Period: {data.index[0]} to {data.index[-1]}")
    print(f"Initial capital: ${engine.initial_capital:,}")
    
    # Main loop
    for timestamp in data.index:
        current_bar = data.loc[timestamp]
        
        # Update portfolio value
        current_prices = {}
        for ticker in engine.positions.keys():
            if ticker in current_bar:
                current_prices[ticker] = current_bar[ticker]['close']
        
        portfolio_value = engine.calculate_portfolio_value(current_prices)
        engine.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': engine.cash,
            'positions_value': portfolio_value - engine.cash
        })
        
        # Strategy decision
        strategy.on_data(data.loc[:timestamp], timestamp)
        
        # Daily return calculation
        if len(engine.equity_curve) > 1:
            prev_value = engine.equity_curve[-2]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            engine.daily_returns.append(daily_return)
    
    print(f"Backtest completed. Final value: ${portfolio_value:,.2f}")
    return engine

# Usage example
def example_backtest():
    """Complete backtest example"""
    
    # 1. Prepare data
    tickers = ['AAPL', 'MSFT', 'TSLA']
    data = prepare_backtest_data(tickers, '2023-01-01', '2023-12-31')
    
    # 2. Setup engine and strategy
    engine = SimpleBacktestEngine(initial_capital=50000)
    strategy = VWAPStrategy(risk_per_trade=0.02)
    
    # 3. Run backtest
    results = run_backtest(engine, strategy, data)
    
    # 4. Analyze results
    performance = analyze_performance(results)
    print(performance)
    
    return results, performance
```

## Data Preparation

```python
def prepare_backtest_data(tickers, start_date, end_date):
    """Prepare multi-ticker data for backtest"""
    import yfinance as yf
    import pandas as pd
    
    all_data = {}
    
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        
        # Download intraday data
        stock_data = yf.download(ticker, 
                                start=start_date, 
                                end=end_date, 
                                interval='5m')
        
        if stock_data.empty:
            continue
            
        # Calculate indicators
        stock_data = calculate_indicators(stock_data)
        
        # Store with ticker as column level
        all_data[ticker] = stock_data
    
    # Combine into multi-level DataFrame
    combined = pd.concat(all_data, axis=1)
    
    # Forward fill missing data
    combined = combined.fillna(method='ffill')
    
    return combined

def calculate_indicators(df):
    """Add technical indicators"""
    # VWAP
    df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Volume average
    df['avg_volume'] = df['Volume'].rolling(20).mean()
    
    # Price metrics
    df['high_low_pct'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['close_open_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Rename columns to lowercase for consistency
    df.columns = df.columns.str.lower()
    
    return df
```

## Performance Analysis

```python
def analyze_performance(engine):
    """Complete performance analysis"""
    
    if not engine.equity_curve:
        return {"error": "No equity curve data"}
    
    # Convert to DataFrame
    equity_df = pd.DataFrame(engine.equity_curve)
    equity_df.set_index('timestamp', inplace=True)
    
    # Basic metrics
    total_return = (engine.portfolio_value - engine.initial_capital) / engine.initial_capital
    
    # Trade analysis
    trades_df = pd.DataFrame(engine.trades)
    if not trades_df.empty:
        # Group buys and sells
        buy_trades = trades_df[trades_df['type'] == 'buy']
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        # Calculate P&L per trade
        trade_pnl = []
        for ticker in buy_trades['ticker'].unique():
            ticker_buys = buy_trades[buy_trades['ticker'] == ticker].copy()
            ticker_sells = sell_trades[sell_trades['ticker'] == ticker].copy()
            
            # Match buys and sells (simplified FIFO)
            for _, sell in ticker_sells.iterrows():
                matching_buy = ticker_buys[ticker_buys['timestamp'] <= sell['timestamp']]
                if not matching_buy.empty:
                    buy = matching_buy.iloc[-1]  # Last buy before this sell
                    pnl = (sell['price'] - buy['price']) * abs(sell['shares'])
                    trade_pnl.append({
                        'ticker': ticker,
                        'entry_date': buy['timestamp'],
                        'exit_date': sell['timestamp'],
                        'entry_price': buy['price'],
                        'exit_price': sell['price'],
                        'shares': abs(sell['shares']),
                        'pnl': pnl,
                        'return_pct': (sell['price'] - buy['price']) / buy['price']
                    })
        
        trade_pnl_df = pd.DataFrame(trade_pnl)
    else:
        trade_pnl_df = pd.DataFrame()
    
    # Risk metrics
    if len(engine.daily_returns) > 0:
        daily_returns = pd.Series(engine.daily_returns)
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (total_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Drawdown calculation
        equity_df['peak'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Win rate
        if not trade_pnl_df.empty:
            winning_trades = (trade_pnl_df['pnl'] > 0).sum()
            total_trades = len(trade_pnl_df)
            win_rate = winning_trades / total_trades
            
            avg_win = trade_pnl_df[trade_pnl_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trade_pnl_df[trade_pnl_df['pnl'] < 0]['pnl'].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_trades = 0
    else:
        volatility = 0
        sharpe_ratio = 0
        max_drawdown = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        total_trades = 0
    
    performance = {
        'total_return': total_return,
        'annual_return': total_return,  # Simplified - assumes 1 year
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'final_value': engine.portfolio_value,
        'total_commission_paid': sum([t['commission'] for t in engine.trades])
    }
    
    return performance
```

## Results Visualization

```python
def plot_backtest_results(engine):
    """Create charts of the results"""
    import matplotlib.pyplot as plt
    
    # Equity curve
    equity_df = pd.DataFrame(engine.equity_curve)
    equity_df.set_index('timestamp', inplace=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Portfolio value
    ax1.plot(equity_df.index, equity_df['portfolio_value'])
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Value ($)')
    ax1.grid(True)
    
    # Drawdown
    equity_df['peak'] = equity_df['portfolio_value'].cummax()
    equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak']
    ax2.fill_between(equity_df.index, equity_df['drawdown'], 0, alpha=0.3, color='red')
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True)
    
    # Trade distribution
    if engine.trades:
        trades_df = pd.DataFrame(engine.trades)
        trade_values = trades_df['shares'] * trades_df['price']
        ax3.hist(trade_values, bins=20, alpha=0.7)
        ax3.set_title('Trade Size Distribution')
        ax3.set_xlabel('Trade Value ($)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Engine Testing

```python
def test_engine():
    """Unit tests to validate the engine"""
    
    # Test 1: Basic order execution
    engine = SimpleBacktestEngine(initial_capital=10000)
    
    # Buy 100 shares at $50
    success = engine.execute_order('TEST', 100, 50, pd.Timestamp('2023-01-01'))
    assert success == True
    assert engine.positions['TEST'] == 100
    assert engine.cash < 10000  # Reduced by purchase + commission
    
    # Sell 50 shares
    success = engine.execute_order('TEST', -50, 55, pd.Timestamp('2023-01-02'))
    assert success == True
    assert engine.positions['TEST'] == 50
    
    print("Engine tests passed")

if __name__ == "__main__":
    test_engine()
    example_backtest()
```

## Advanced Extensions

```python
# To add later:
class AdvancedFeatures:
    """More advanced features for the engine"""
    
    def add_multiple_timeframes(self):
        """Support for multiple timeframes"""
        pass
    
    def add_options_support(self):
        """Options trading"""
        pass
    
    def add_portfolio_rebalancing(self):
        """Automatic rebalancing"""
        pass
    
    def add_risk_management(self):
        """Advanced risk management"""
        pass
```

## Next Step

With our basic engine working, let's move on to [Key Metrics](metrics.md) to understand which numbers truly matter.
