> 🇪🇸 [Leer en Español](README.es.md) | 🇺🇸 **English**

# F4: Your First Complete Strategy

**Fundamental Module 4 - Duration: 2-3 hours**

## Module Objectives

After completing this module you will have:
- A complete and functional trading strategy
- Clear entry, exit, and risk management rules
- A backtest that proves your strategy with real data
- Metrics to evaluate whether your strategy is profitable

## The Strategy: "Golden Cross with Filters"

We will build a **real and proven** strategy step by step.

### Why This Strategy?
- **Simple but effective**: Easy to understand and program
- **Historically proven**: Used by institutional funds
- **Adaptable**: Works in different markets
- **Great for learning**: Includes all essential components

### Strategy Components

| Component | Description |
|-----------|-------------|
| **Main Signal** | Golden Cross (SMA 50 crosses SMA 200) |
| **Filter 1** | RSI not overbought (< 70) |
| **Filter 2** | Volume > 20-day average |
| **Stop Loss** | 5% from entry |
| **Take Profit** | 15% from entry |
| **Risk Management** | Maximum 2% of capital per trade |

## Step 1: Build the Strategy

### Strategy Base Code

```python
# For Google Colab
!pip install yfinance pandas numpy matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class GoldenCrossStrategy:
    """
    Golden Cross Strategy with Quality Filters
    """

    def __init__(self, symbol, start_date, end_date, initial_capital=10000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Strategy parameters
        self.short_sma = 50
        self.long_sma = 200
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.volume_filter = 1.2  # 20% above average
        self.stop_loss_pct = 0.05   # 5%
        self.take_profit_pct = 0.15 # 15%
        self.risk_per_trade = 0.02 # 2% of capital

        # Tracking data
        self.trades = []
        self.current_position = None

    def download_data(self):
        """Downloads and prepares the data"""
        print(f"Downloading data for {self.symbol}...")

        # Download with extra margin to calculate indicators
        extended_start = pd.to_datetime(self.start_date) - timedelta(days=300)
        self.data = yf.download(self.symbol, start=extended_start, end=self.end_date)

        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")

        print(f"Downloaded {len(self.data)} days of data")

    def calculate_indicators(self):
        """Calculates all required indicators"""
        print("Calculating indicators...")

        # Moving averages
        self.data['SMA_50'] = self.data['Close'].rolling(window=self.short_sma).mean()
        self.data['SMA_200'] = self.data['Close'].rolling(window=self.long_sma).mean()

        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # Average volume
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']

        # Crossover signals
        self.data['Golden_Cross'] = (
            (self.data['SMA_50'] > self.data['SMA_200']) &
            (self.data['SMA_50'].shift(1) <= self.data['SMA_200'].shift(1))
        )

        self.data['Death_Cross'] = (
            (self.data['SMA_50'] < self.data['SMA_200']) &
            (self.data['SMA_50'].shift(1) >= self.data['SMA_200'].shift(1))
        )

        # Clean NaN from beginning
        self.data = self.data[self.data.index >= self.start_date].dropna()

        print("Indicators calculated")

    def generate_signals(self):
        """Generates buy/sell signals based on the strategy"""
        print("Generating trading signals...")

        self.data['Signal'] = 0  # 0: No position, 1: Buy, -1: Sell

        for i in range(len(self.data)):
            date = self.data.index[i]
            row = self.data.iloc[i]

            # BUY SIGNAL
            if (row['Golden_Cross'] and
                row['RSI'] < self.rsi_overbought and
                row['Volume_Ratio'] > self.volume_filter and
                self.current_position is None):

                self.data.loc[date, 'Signal'] = 1
                self.open_position(date, row['Close'], 'LONG')

            # CHECK STOPS IF POSITION EXISTS
            elif self.current_position is not None:
                current_price = row['Close']
                entry_price = self.current_position['entry_price']

                # Stop Loss
                if current_price <= entry_price * (1 - self.stop_loss_pct):
                    self.data.loc[date, 'Signal'] = -1
                    self.close_position(date, current_price, 'STOP_LOSS')

                # Take Profit
                elif current_price >= entry_price * (1 + self.take_profit_pct):
                    self.data.loc[date, 'Signal'] = -1
                    self.close_position(date, current_price, 'TAKE_PROFIT')

                # Death Cross (signal exit)
                elif row['Death_Cross']:
                    self.data.loc[date, 'Signal'] = -1
                    self.close_position(date, current_price, 'DEATH_CROSS')

        # Close position at end if still open
        if self.current_position is not None:
            last_price = self.data['Close'].iloc[-1]
            self.close_position(self.data.index[-1], last_price, 'END_OF_PERIOD')

        print(f"Generated {len(self.trades)} trades")

    def open_position(self, date, price, type):
        """Opens a new position"""

        # Calculate position size based on risk
        capital_at_risk = self.capital * self.risk_per_trade
        stop_loss_price = price * (1 - self.stop_loss_pct)
        risk_per_share = price - stop_loss_price
        num_shares = int(capital_at_risk / risk_per_share)

        # Limit to available capital
        max_shares = int(self.capital * 0.95 / price)  # Use max 95% of capital
        num_shares = min(num_shares, max_shares)

        self.current_position = {
            'entry_date': date,
            'entry_price': price,
            'num_shares': num_shares,
            'type': type,
            'stop_loss': stop_loss_price,
            'take_profit': price * (1 + self.take_profit_pct)
        }

    def close_position(self, date, price, reason):
        """Closes the current position and records the trade"""

        if self.current_position is None:
            return

        # Calculate result
        entry_price = self.current_position['entry_price']
        num_shares = self.current_position['num_shares']
        profit_loss = (price - entry_price) * num_shares
        return_pct = ((price - entry_price) / entry_price) * 100

        # Update capital
        self.capital += profit_loss

        # Record trade
        trade = {
            'entry_date': self.current_position['entry_date'],
            'exit_date': date,
            'entry_price': entry_price,
            'exit_price': price,
            'num_shares': num_shares,
            'profit_loss': profit_loss,
            'return_pct': return_pct,
            'exit_reason': reason,
            'capital_after': self.capital
        }
        self.trades.append(trade)

        # Clear position
        self.current_position = None

    def calculate_metrics(self):
        """Calculates strategy performance metrics"""

        if not self.trades:
            print("No trades to analyze")
            return None

        df_trades = pd.DataFrame(self.trades)

        # Basic metrics
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['profit_loss'] > 0])
        losing_trades = len(df_trades[df_trades['profit_loss'] < 0])
        win_rate = (winning_trades / total_trades) * 100

        # Gains and losses
        total_gains = df_trades[df_trades['profit_loss'] > 0]['profit_loss'].sum()
        total_losses = abs(df_trades[df_trades['profit_loss'] < 0]['profit_loss'].sum())
        profit_factor = total_gains / total_losses if total_losses > 0 else np.inf

        # Returns
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        avg_win = df_trades[df_trades['profit_loss'] > 0]['return_pct'].mean()
        avg_loss = df_trades[df_trades['profit_loss'] < 0]['return_pct'].mean()

        # Maximum drawdown
        peak_capital = self.initial_capital
        max_drawdown = 0
        for trade in self.trades:
            peak_capital = max(peak_capital, trade['capital_after'])
            drawdown = ((peak_capital - trade['capital_after']) / peak_capital) * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Buy & Hold comparison
        initial_price = self.data['Close'].iloc[0]
        final_price = self.data['Close'].iloc[-1]
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100

        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'final_capital': self.capital,
            'buy_hold_return': buy_hold_return
        }

        return metrics

    def visualize_results(self):
        """Creates result visualizations"""

        fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

        # 1. Price and Signals
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Price', linewidth=1, alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_50'], label='SMA 50', linewidth=1, alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_200'], label='SMA 200', linewidth=1, alpha=0.7)

        # Mark entries and exits
        entries = self.data[self.data['Signal'] == 1]
        exits = self.data[self.data['Signal'] == -1]

        ax1.scatter(entries.index, entries['Close'], color='green', marker='^',
                   s=100, label=f'Buys ({len(entries)})', zorder=5)
        ax1.scatter(exits.index, exits['Close'], color='red', marker='v',
                   s=100, label=f'Sells ({len(exits)})', zorder=5)

        ax1.set_title(f'{self.symbol} - Golden Cross Strategy with Filters', fontsize=14)
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. RSI
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['RSI'], color='purple', linewidth=1)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        # 3. Volume
        ax3 = axes[2]
        colors = ['green' if s == 1 else 'red' if s == -1 else 'gray' for s in self.data['Signal']]
        ax3.bar(self.data.index, self.data['Volume'], color=colors, alpha=0.5)
        ax3.plot(self.data.index, self.data['Volume_MA'], color='blue', linewidth=1)
        ax3.set_ylabel('Volume')
        ax3.grid(True, alpha=0.3)

        # 4. Equity Curve
        ax4 = axes[3]
        if self.trades:
            equity_data = [self.initial_capital]
            equity_dates = [self.data.index[0]]

            for trade in self.trades:
                equity_dates.append(trade['exit_date'])
                equity_data.append(trade['capital_after'])

            ax4.plot(equity_dates, equity_data, color='blue', linewidth=2, label='Strategy')

            # Compare with Buy & Hold
            buy_hold_values = self.initial_capital * (self.data['Close'] / self.data['Close'].iloc[0])
            ax4.plot(self.data.index, buy_hold_values, color='gray', linewidth=1,
                    alpha=0.7, label='Buy & Hold')

            ax4.set_ylabel('Capital ($)')
            ax4.set_xlabel('Date')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_backtest(self):
        """Runs the complete backtest"""
        print("\n" + "="*60)
        print(f"BACKTESTING: {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print("="*60)

        # Run steps
        self.download_data()
        self.calculate_indicators()
        self.generate_signals()

        # Calculate metrics
        metrics = self.calculate_metrics()

        if metrics:
            print("\nBACKTEST RESULTS:")
            print("="*60)
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Winning Trades: {metrics['winning_trades']}")
            print(f"Losing Trades: {metrics['losing_trades']}")
            print(f"\nWin Rate: {metrics['win_rate']:.2f}%")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"\nFinal Capital: ${metrics['final_capital']:,.2f}")
            print(f"Total Return: {metrics['total_return']:.2f}%")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"\nBuy & Hold Return: {metrics['buy_hold_return']:.2f}%")

            if metrics['total_return'] > metrics['buy_hold_return']:
                print("\nThe strategy BEATS Buy & Hold!")
            else:
                print("\nThe strategy DOES NOT beat Buy & Hold")

        # Visualize
        self.visualize_results()

        return metrics

# RUN THE STRATEGY
if __name__ == "__main__":
    # Configure parameters
    SYMBOL = 'AAPL'  # You can change the symbol
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    INITIAL_CAPITAL = 10000

    # Create and run strategy
    strategy = GoldenCrossStrategy(SYMBOL, START_DATE, END_DATE, INITIAL_CAPITAL)
    results = strategy.run_backtest()
```

## Step 2: Optimize the Strategy

### Testing Different Parameters

```python
def optimize_parameters(symbol='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """Tests different parameter combinations"""

    print("PARAMETER OPTIMIZATION")
    print("="*60)

    optimization_results = []

    # Parameters to test
    short_smas = [20, 30, 50]
    long_smas = [100, 150, 200]
    stop_losses = [0.03, 0.05, 0.07]
    take_profits = [0.10, 0.15, 0.20]

    best_result = None
    best_return = -np.inf

    for sma_s in short_smas:
        for sma_l in long_smas:
            if sma_s >= sma_l:
                continue

            for sl in stop_losses:
                for tp in take_profits:
                    # Create strategy with these parameters
                    strategy = GoldenCrossStrategy(symbol, start_date, end_date)
                    strategy.short_sma = sma_s
                    strategy.long_sma = sma_l
                    strategy.stop_loss_pct = sl
                    strategy.take_profit_pct = tp

                    # Run backtest silently
                    try:
                        strategy.download_data()
                        strategy.calculate_indicators()
                        strategy.generate_signals()
                        metrics = strategy.calculate_metrics()

                        if metrics and metrics['total_trades'] > 0:
                            result = {
                                'short_sma': sma_s,
                                'long_sma': sma_l,
                                'stop_loss': sl,
                                'take_profit': tp,
                                'return': metrics['total_return'],
                                'win_rate': metrics['win_rate'],
                                'profit_factor': metrics['profit_factor'],
                                'max_drawdown': metrics['max_drawdown'],
                                'num_trades': metrics['total_trades']
                            }

                            optimization_results.append(result)

                            if metrics['total_return'] > best_return:
                                best_return = metrics['total_return']
                                best_result = result

                    except:
                        continue

    # Show results
    if best_result:
        print("\nBEST PARAMETERS FOUND:")
        print("="*60)
        print(f"Short SMA: {best_result['short_sma']}")
        print(f"Long SMA: {best_result['long_sma']}")
        print(f"Stop Loss: {best_result['stop_loss']*100:.1f}%")
        print(f"Take Profit: {best_result['take_profit']*100:.1f}%")
        print(f"\nReturn: {best_result['return']:.2f}%")
        print(f"Win Rate: {best_result['win_rate']:.2f}%")
        print(f"Profit Factor: {best_result['profit_factor']:.2f}")
        print(f"Max Drawdown: {best_result['max_drawdown']:.2f}%")

    return pd.DataFrame(optimization_results)

# Run optimization
df_optimization = optimize_parameters('AAPL')

# View top 5 combinations
print("\nTOP 5 COMBINATIONS:")
print(df_optimization.nlargest(5, 'return')[['short_sma', 'long_sma', 'stop_loss', 'take_profit', 'return', 'win_rate']])
```

## Step 3: Validation and Improvements

### Analyzing Weaknesses

```python
def detailed_trade_analysis(strategy):
    """Detailed analysis of each trade"""

    if not strategy.trades:
        print("No trades to analyze")
        return

    df_trades = pd.DataFrame(strategy.trades)

    print("\nDETAILED TRADE ANALYSIS:")
    print("="*60)

    # Analysis by exit reason
    print("\nExit Distribution:")
    for reason in df_trades['exit_reason'].unique():
        reason_trades = df_trades[df_trades['exit_reason'] == reason]
        avg_return = reason_trades['return_pct'].mean()
        count = len(reason_trades)
        print(f"{reason}: {count} trades, Average return: {avg_return:.2f}%")

    # Trade duration
    df_trades['duration'] = (pd.to_datetime(df_trades['exit_date']) -
                            pd.to_datetime(df_trades['entry_date'])).dt.days

    print(f"\nAverage duration: {df_trades['duration'].mean():.1f} days")
    print(f"Maximum duration: {df_trades['duration'].max()} days")
    print(f"Minimum duration: {df_trades['duration'].min()} days")

    # Best and worst trade
    best_trade = df_trades.loc[df_trades['profit_loss'].idxmax()]
    worst_trade = df_trades.loc[df_trades['profit_loss'].idxmin()]

    print(f"\nBest Trade:")
    print(f"  Entry: {best_trade['entry_date']}")
    print(f"  Gain: ${best_trade['profit_loss']:.2f} ({best_trade['return_pct']:.2f}%)")

    print(f"\nWorst Trade:")
    print(f"  Entry: {worst_trade['entry_date']}")
    print(f"  Loss: ${worst_trade['profit_loss']:.2f} ({worst_trade['return_pct']:.2f}%)")

    return df_trades

# Analyze trades from the last strategy
df_trades = detailed_trade_analysis(strategy)
```

## Final Project: Multi-Strategy

### Comparing Multiple Strategies

```python
def compare_strategies():
    """Compares different strategies over the same period"""

    strategy_configs = [
        {'name': 'Golden Cross Original', 'sma_s': 50, 'sma_l': 200, 'sl': 0.05, 'tp': 0.15},
        {'name': 'Golden Cross Aggressive', 'sma_s': 20, 'sma_l': 50, 'sl': 0.03, 'tp': 0.10},
        {'name': 'Golden Cross Conservative', 'sma_s': 50, 'sma_l': 200, 'sl': 0.07, 'tp': 0.20},
        {'name': 'Golden Cross Fast', 'sma_s': 10, 'sma_l': 30, 'sl': 0.02, 'tp': 0.05},
    ]

    comparison_results = []

    for config in strategy_configs:
        print(f"\nTesting: {config['name']}...")

        strategy = GoldenCrossStrategy('SPY', '2020-01-01', '2024-01-01', 10000)
        strategy.short_sma = config['sma_s']
        strategy.long_sma = config['sma_l']
        strategy.stop_loss_pct = config['sl']
        strategy.take_profit_pct = config['tp']

        strategy.download_data()
        strategy.calculate_indicators()
        strategy.generate_signals()
        metrics = strategy.calculate_metrics()

        if metrics:
            comparison_results.append({
                'Strategy': config['name'],
                'Return (%)': metrics['total_return'],
                'Win Rate (%)': metrics['win_rate'],
                'Profit Factor': metrics['profit_factor'],
                'Max DD (%)': metrics['max_drawdown'],
                'Trades': metrics['total_trades']
            })

    # Create comparison table
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison = df_comparison.round(2)

    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(df_comparison.to_string(index=False))

    # Identify winner
    best_strategy = df_comparison.loc[df_comparison['Return (%)'].idxmax()]
    print(f"\nBEST STRATEGY: {best_strategy['Strategy']}")
    print(f"   Return: {best_strategy['Return (%)']}%")
    print(f"   Profit Factor: {best_strategy['Profit Factor']}")

    return df_comparison

# Run comparison
df_comparison = compare_strategies()
```

## Module Checkpoint

### Complete Strategy
- [ ] My strategy has clear entry rules
- [ ] I implemented stop loss and take profit
- [ ] I use risk management (2% per trade)
- [ ] I combine multiple indicators

### Working Backtest
- [ ] I can test with real historical data
- [ ] I calculate performance metrics
- [ ] I compare with Buy & Hold
- [ ] I visualize results clearly

### Optimization
- [ ] I tested different parameters
- [ ] I identified best configurations
- [ ] I analyzed individual trades
- [ ] I compared multiple variations

### Learnings
- [ ] I understand why some strategies fail
- [ ] I know the importance of stop loss
- [ ] I see the impact of parameters
- [ ] I can improve the strategy

## CONGRATULATIONS!

**You have completed the FUNDAMENTALS of quantitative trading.**

You now have:
- Knowledge of what being a quant means
- Python trading skills
- Mastery of technical indicators
- Your first complete and tested strategy

## Next Steps

### Continue with STRATEGIES (Level 2)

**E1: Momentum Trading**
- Gap & Go for small caps
- Breakout strategies
- Momentum scanning

**E2: Mean Reversion**
- VWAP reclaim
- Oversold bounces
- Pairs trading

**E3: Advanced Backtesting**
- Walk-forward analysis
- Monte Carlo simulation
- Stress testing

---

### Final Reflection

> **"A mediocre strategy well executed is better than a perfect strategy poorly executed."**

Your first strategy won't be perfect, but you now have the tools to continuously improve it. Every day of trading generates new data to learn from.

**You are officially a Quant Trader in training!**

---

**Ready for more advanced strategies?** -> [E1: Momentum Trading](../../estrategias/e1-momentum-trading/)
