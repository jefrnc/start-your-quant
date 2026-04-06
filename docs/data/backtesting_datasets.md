> 🇪🇸 [Leer en Español](backtesting_datasets.es.md) | 🇺🇸 **English**

# Creating Datasets for Backtesting

## The Mental Framework

A good backtesting dataset is not just historical data. It's a realistic representation of the trading environment, including all the limitations and frictions you'll face in live trading.

## Base Structure

```python
class BacktestDataset:
    def __init__(self):
        self.price_data = {}      # OHLCV data
        self.fundamental_data = {} # Float, sector, etc
        self.universe = []        # Tradeable tickers each day
        self.metadata = {}        # Additional info
        
    def add_ticker(self, ticker, data, metadata=None):
        # Validate data
        if not self._validate_data(data):
            raise ValueError(f"Invalid data for {ticker}")
            
        self.price_data[ticker] = data
        if metadata:
            self.metadata[ticker] = metadata
            
    def get_universe(self, date):
        """Get tradeable tickers on a date"""
        available = []
        for ticker, data in self.price_data.items():
            if date in data.index:
                # Check minimum liquidity
                volume = data.loc[date, 'volume']
                price = data.loc[date, 'close']
                dollar_volume = volume * price
                
                if dollar_volume > 1_000_000:  # $1M minimum
                    available.append(ticker)
        
        return available
```

## Datasets by Strategy Type

### 1. Gap Trading Dataset
```python
def create_gap_trading_dataset(start_date, end_date):
    dataset = BacktestDataset()
    
    # Universe: Small caps with volume
    universe_criteria = {
        'market_cap': (10_000_000, 500_000_000),
        'avg_volume': 500_000,
        'price': (1, 50)
    }
    
    # Get tickers that meet criteria
    tickers = screen_universe(universe_criteria)
    
    for ticker in tickers:
        # We need pre-market data
        data = fetch_extended_hours_data(ticker, start_date, end_date)
        
        # Calculate gaps
        data['gap_pct'] = (data['open'] / data['close'].shift(1) - 1) * 100
        data['gap_type'] = data['gap_pct'].apply(classify_gap)
        
        # Add relevant indicators
        data['premarket_volume'] = calculate_premarket_volume(data)
        data['rvol'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Important metadata
        metadata = {
            'float': get_float(ticker),
            'sector': get_sector(ticker),
            'avg_spread': calculate_avg_spread(data)
        }
        
        dataset.add_ticker(ticker, data, metadata)
    
    return dataset

def classify_gap(gap_pct):
    if gap_pct > 20:
        return 'large_gap_up'
    elif gap_pct > 10:
        return 'medium_gap_up'
    elif gap_pct < -10:
        return 'gap_down'
    else:
        return 'small_gap'
```

### 2. Mean Reversion Dataset
```python
def create_mean_reversion_dataset(start_date, end_date):
    dataset = BacktestDataset()
    
    # For mean reversion we want liquid and stable stocks
    universe_criteria = {
        'market_cap': (1_000_000_000, None),  # Large caps
        'avg_volume': 5_000_000,
        'volatility': (0.01, 0.05)  # Moderate volatility
    }
    
    tickers = screen_universe(universe_criteria)
    
    for ticker in tickers:
        data = fetch_data(ticker, start_date, end_date)
        
        # Mean reversion indicators
        data['sma_20'] = data['close'].rolling(20).mean()
        data['distance_from_mean'] = (data['close'] - data['sma_20']) / data['sma_20']
        
        # Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = calculate_bollinger_bands(data)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # RSI
        data['rsi'] = calculate_rsi(data['close'])
        
        # Z-Score
        data['zscore'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        
        dataset.add_ticker(ticker, data)
    
    return dataset
```

### 3. Pairs Trading Dataset
```python
def create_pairs_trading_dataset(start_date, end_date):
    dataset = BacktestDataset()
    
    # Get correlated pairs
    sectors = ['XLK', 'XLF', 'XLE', 'XLV']  # Tech, Financial, Energy, Healthcare
    pairs = []
    
    for sector in sectors:
        stocks = get_sector_stocks(sector)
        correlation_matrix = calculate_correlation_matrix(stocks, start_date, end_date)
        
        # Find highly correlated pairs
        high_corr_pairs = find_high_correlation_pairs(correlation_matrix, threshold=0.8)
        pairs.extend(high_corr_pairs)
    
    # Create spreads for each pair
    for stock1, stock2 in pairs:
        data1 = fetch_data(stock1, start_date, end_date)
        data2 = fetch_data(stock2, start_date, end_date)
        
        # Align data
        spread_data = pd.DataFrame(index=data1.index)
        spread_data['price1'] = data1['close']
        spread_data['price2'] = data2['close']
        
        # Calculate ratio and z-score
        spread_data['ratio'] = spread_data['price1'] / spread_data['price2']
        spread_data['ratio_mean'] = spread_data['ratio'].rolling(20).mean()
        spread_data['ratio_std'] = spread_data['ratio'].rolling(20).std()
        spread_data['zscore'] = (spread_data['ratio'] - spread_data['ratio_mean']) / spread_data['ratio_std']
        
        # Cointegration test
        spread_data['is_cointegrated'] = test_cointegration(data1['close'], data2['close'])
        
        dataset.add_ticker(f"{stock1}_{stock2}_pair", spread_data)
    
    return dataset
```

## Adding Fundamental Data

```python
def enrich_with_fundamentals(dataset):
    """Add fundamental data to the dataset"""
    
    for ticker in dataset.price_data.keys():
        # Float data
        float_data = get_historical_float(ticker)
        
        # Short interest
        short_data = get_short_interest(ticker)
        
        # Earnings dates
        earnings_dates = get_earnings_calendar(ticker)
        
        # News sentiment
        news_sentiment = get_news_sentiment(ticker)
        
        # Add to dataset
        dataset.metadata[ticker].update({
            'float_history': float_data,
            'short_interest': short_data,
            'earnings_dates': earnings_dates,
            'news_sentiment': news_sentiment
        })
    
    return dataset
```

## Considering Costs and Frictions

```python
class RealisticBacktestData:
    def __init__(self, base_dataset):
        self.data = base_dataset
        self.cost_model = CostModel()
        
    def add_market_impact(self, ticker, date, size):
        """Estimate price impact by order size"""
        daily_volume = self.data.price_data[ticker].loc[date, 'volume']
        participation_rate = size / daily_volume
        
        # Simple impact model
        if participation_rate < 0.01:
            impact = 0.0001  # 1 basis point
        elif participation_rate < 0.05:
            impact = 0.0005  # 5 basis points
        else:
            impact = 0.001 + (participation_rate - 0.05) * 0.01
            
        return impact
    
    def calculate_realistic_fill(self, ticker, date, side, size):
        """Calculate realistic fill price"""
        bar = self.data.price_data[ticker].loc[date]
        
        if side == 'buy':
            # Buy near the ask
            base_price = bar['close'] * 1.0001  # Slight premium
            impact = self.add_market_impact(ticker, date, size)
            fill_price = base_price * (1 + impact)
        else:
            # Sell near the bid
            base_price = bar['close'] * 0.9999  # Slight discount
            impact = self.add_market_impact(ticker, date, size)
            fill_price = base_price * (1 - impact)
            
        # Add random slippage
        slippage = np.random.normal(0, 0.0001)  # 1bp std dev
        fill_price *= (1 + slippage)
        
        return fill_price
```

## Splits and Corporate Actions

```python
def handle_corporate_actions(dataset):
    """Handle splits, dividends, etc."""
    
    for ticker in dataset.price_data.keys():
        # Get corporate actions
        actions = get_corporate_actions(ticker)
        
        for action in actions:
            if action['type'] == 'split':
                ratio = action['ratio']
                date = action['date']
                
                # Adjust historical prices
                mask = dataset.price_data[ticker].index < date
                dataset.price_data[ticker].loc[mask, ['open', 'high', 'low', 'close']] /= ratio
                dataset.price_data[ticker].loc[mask, 'volume'] *= ratio
                
            elif action['type'] == 'dividend':
                # Adjust for dividends if needed
                div_amount = action['amount']
                date = action['date']
                
                # Some backtests adjust, others don't
                # Depends on your strategy
```

## Survivorship Bias

```python
def create_survivorship_bias_free_dataset(start_date, end_date):
    """Include companies that went bankrupt or were delisted"""
    
    dataset = BacktestDataset()
    
    # Get historical list of tickers
    historical_universe = get_historical_constituents('Russell3000', start_date)
    
    for ticker in historical_universe:
        try:
            data = fetch_data(ticker, start_date, end_date)
            
            # Mark if delisted
            if ticker in get_delisted_tickers():
                data['is_delisted'] = True
                data['delisting_date'] = get_delisting_date(ticker)
                
            dataset.add_ticker(ticker, data)
            
        except DataNotAvailable:
            # Important: include even if data is incomplete
            print(f"Warning: Limited data for {ticker}")
```

## Optimization for Speed

```python
class OptimizedBacktestData:
    def __init__(self, dataset):
        self.dataset = dataset
        self._create_indexes()
        
    def _create_indexes(self):
        """Pre-calculate indexes for fast lookups"""
        # Create MultiIndex for fast access
        all_data = []
        
        for ticker, df in self.dataset.price_data.items():
            df['ticker'] = ticker
            all_data.append(df)
            
        self.combined_df = pd.concat(all_data)
        self.combined_df.set_index(['ticker', self.combined_df.index], inplace=True)
        
        # Pre-calculate universes by date
        self.daily_universes = {}
        for date in self.combined_df.index.get_level_values(1).unique():
            self.daily_universes[date] = self.dataset.get_universe(date)
    
    def get_bar(self, ticker, date):
        """Ultra-fast access to a bar"""
        return self.combined_df.loc[(ticker, date)]
    
    def get_multiple_bars(self, tickers, date):
        """Get multiple tickers efficiently"""
        return self.combined_df.loc[(tickers, date)]
```

## Final Dataset Validation

```python
def validate_backtest_dataset(dataset):
    """Crucial validations before backtest"""
    
    issues = []
    
    # 1. Check for look-ahead bias
    for ticker, data in dataset.price_data.items():
        # Look for indicators that use future data
        for col in data.columns:
            if 'shift(-' in str(data[col]):
                issues.append(f"Possible look-ahead bias in {ticker}:{col}")
    
    # 2. Check temporal consistency
    dates = set()
    for ticker, data in dataset.price_data.items():
        dates.update(data.index)
    
    # All tickers should have similar dates
    date_coverage = {}
    for ticker, data in dataset.price_data.items():
        coverage = len(data) / len(dates)
        if coverage < 0.8:  # Less than 80% coverage
            issues.append(f"{ticker} has only {coverage:.1%} date coverage")
    
    # 3. Check for extreme data
    for ticker, data in dataset.price_data.items():
        returns = data['close'].pct_change()
        if (returns > 5).any():  # +500% in a single day
            issues.append(f"{ticker} has suspicious returns > 500%")
    
    return issues
```

## My Personal Setup

```python
# create_my_dataset.py
def create_my_trading_dataset():
    # Base configuration
    config = {
        'start_date': '2022-01-01',
        'end_date': '2024-01-01',
        'universe': 'small_caps',
        'min_volume': 1_000_000,
        'min_price': 1,
        'max_price': 50
    }
    
    # Create datasets for each strategy
    datasets = {
        'gap_trading': create_gap_trading_dataset(**config),
        'vwap_bounce': create_mean_reversion_dataset(**config),
        'momentum': create_momentum_dataset(**config)
    }
    
    # Enrich with fundamentals
    for name, dataset in datasets.items():
        enrich_with_fundamentals(dataset)
        handle_corporate_actions(dataset)
    
    # Optimize for speed
    optimized = {
        name: OptimizedBacktestData(dataset) 
        for name, dataset in datasets.items()
    }
    
    return optimized
```

## Next Step

With robust datasets ready, let's move on to the [Technical Indicators](../indicators/vwap.md) specific to small caps.