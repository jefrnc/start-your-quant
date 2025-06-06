# Motor de Backtest Simple

## Construyendo Tu Propio Engine

Antes de usar frameworks complejos, necesitas entender cómo funciona un motor de backtest por dentro. Aquí construimos uno desde cero que realmente funciona.

## Arquitectura Básica

```python
class SimpleBacktestEngine:
    def __init__(self, initial_capital=10000, commission=5, slippage=0.0001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Estado del portfolio
        self.cash = initial_capital
        self.positions = {}  # {ticker: shares}
        self.portfolio_value = initial_capital
        
        # Tracking
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def reset(self):
        """Reset para nuevo backtest"""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
```

## Ejecución de Órdenes

```python
def execute_order(self, ticker, shares, price, timestamp, order_type='market'):
    """Ejecutar una orden con costos realistas"""
    
    # Validaciones básicas
    if shares == 0:
        return False
        
    # Calcular costos
    gross_value = abs(shares * price)
    commission_cost = max(self.commission, gross_value * 0.0001)  # Min $5 o 1bp
    
    # Aplicar slippage
    if shares > 0:  # Compra
        execution_price = price * (1 + self.slippage)
    else:  # Venta
        execution_price = price * (1 - self.slippage)
    
    net_cost = shares * execution_price + commission_cost
    
    # Verificar si tenemos cash/shares suficientes
    if shares > 0 and net_cost > self.cash:
        # No hay cash suficiente
        return False
        
    if shares < 0 and ticker in self.positions:
        if abs(shares) > self.positions[ticker]:
            # No hay shares suficientes para vender
            return False
    elif shares < 0 and ticker not in self.positions:
        # Trying to sell what we don't own
        return False
    
    # Ejecutar la orden
    self.cash -= net_cost
    
    if ticker in self.positions:
        self.positions[ticker] += shares
        if self.positions[ticker] == 0:
            del self.positions[ticker]
    else:
        self.positions[ticker] = shares
    
    # Registrar trade
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
    """Calcular valor actual del portfolio"""
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
    """Base class para estrategias"""
    
    def __init__(self, name):
        self.name = name
        self.parameters = {}
        
    def initialize(self, engine):
        """Setup inicial"""
        self.engine = engine
        
    def on_data(self, data, timestamp):
        """Llamada en cada barra de datos"""
        raise NotImplementedError
        
    def should_enter(self, data, ticker):
        """Lógica de entrada"""
        return False
        
    def should_exit(self, data, ticker):
        """Lógica de salida"""
        return False
        
    def calculate_position_size(self, ticker, price):
        """Calcular tamaño de posición"""
        return 0

class VWAPStrategy(Strategy):
    """Ejemplo: Estrategia VWAP simple"""
    
    def __init__(self, risk_per_trade=0.02, stop_loss_pct=0.03):
        super().__init__("VWAP Reclaim")
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.entry_prices = {}
        
    def on_data(self, data, timestamp):
        """Procesar cada barra"""
        for ticker in data.columns.get_level_values(0).unique():
            if ticker not in ['SPY', 'QQQ']:  # Skip ETFs for this example
                self.process_ticker(data, ticker, timestamp)
    
    def process_ticker(self, data, ticker, timestamp):
        """Procesar un ticker específico"""
        try:
            # Obtener datos del ticker
            ticker_data = data[ticker].loc[timestamp]
            
            # Calcular VWAP si no existe
            if 'vwap' not in ticker_data:
                return
            
            current_price = ticker_data['close']
            vwap = ticker_data['vwap']
            volume = ticker_data['volume']
            avg_volume = ticker_data.get('avg_volume', volume)
            
            # Señales
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
            # Datos no disponibles para este timestamp
            print(f"Warning: No data available for {ticker} at {timestamp}: {e}")
            pass
        except Exception as e:
            # Error inesperado al procesar ticker
            print(f"Error processing {ticker} at {timestamp}: {e}")
            pass
    
    def calculate_position_size(self, ticker, price):
        """Calcular shares basado en risk management"""
        risk_amount = self.engine.portfolio_value * self.risk_per_trade
        stop_distance = price * self.stop_loss_pct
        shares = int(risk_amount / stop_distance)
        
        # No usar más del 20% del portfolio en una posición
        max_position_value = self.engine.portfolio_value * 0.2
        max_shares = int(max_position_value / price)
        
        return min(shares, max_shares)
```

## Main Backtest Loop

```python
def run_backtest(engine, strategy, data, start_date=None, end_date=None):
    """Ejecutar backtest completo"""
    
    # Filter data por fechas
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

# Ejemplo de uso
def example_backtest():
    """Ejemplo completo de backtest"""
    
    # 1. Preparar datos
    tickers = ['AAPL', 'MSFT', 'TSLA']
    data = prepare_backtest_data(tickers, '2023-01-01', '2023-12-31')
    
    # 2. Setup engine y strategy
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
    """Preparar datos multi-ticker para backtest"""
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
    """Agregar indicadores técnicos"""
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
    """Análisis completo de performance"""
    
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

## Visualización de Resultados

```python
def plot_backtest_results(engine):
    """Crear gráficos de los resultados"""
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

## Testing del Engine

```python
def test_engine():
    """Tests unitarios para validar el engine"""
    
    # Test 1: Ejecución básica de órdenes
    engine = SimpleBacktestEngine(initial_capital=10000)
    
    # Comprar 100 shares a $50
    success = engine.execute_order('TEST', 100, 50, pd.Timestamp('2023-01-01'))
    assert success == True
    assert engine.positions['TEST'] == 100
    assert engine.cash < 10000  # Reduced by purchase + commission
    
    # Vender 50 shares
    success = engine.execute_order('TEST', -50, 55, pd.Timestamp('2023-01-02'))
    assert success == True
    assert engine.positions['TEST'] == 50
    
    print("✅ Engine tests passed")

if __name__ == "__main__":
    test_engine()
    example_backtest()
```

## Extensiones Avanzadas

```python
# Para agregar después:
class AdvancedFeatures:
    """Features más avanzadas para el engine"""
    
    def add_multiple_timeframes(self):
        """Support para múltiples timeframes"""
        pass
    
    def add_options_support(self):
        """Trading de options"""
        pass
    
    def add_portfolio_rebalancing(self):
        """Rebalanceo automático"""
        pass
    
    def add_risk_management(self):
        """Risk management avanzado"""
        pass
```

## Siguiente Paso

Con nuestro motor básico funcionando, vamos a [Métricas Clave](metrics.md) para entender qué números realmente importan.