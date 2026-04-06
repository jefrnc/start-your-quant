> 🇪🇸 [Leer en Español](data_cleaning.es.md) | 🇺🇸 **English**

# Data Cleaning and Structuring

## Why It Matters

Garbage in, garbage out. You can have the best strategy in the world, but if your data is bad, your results will be fiction. I've lost days debugging "broken strategies" that actually had corrupt data.

## Common Problems in Trading Data

### 1. Missing Data (Gaps)
```python
# Problem: Missing bars during market hours
datetime              close   volume
2024-01-15 09:30:00  175.00  500000
2024-01-15 09:31:00  175.20  450000
2024-01-15 09:32:00  NaN     NaN      # <-- Missing!
2024-01-15 09:33:00  175.45  380000
```

**Solución:**
```python
def fill_missing_bars(df, market_hours_only=True):
    # Create complete index
    if market_hours_only:
        full_index = pd.date_range(
            start=df.index[0].replace(hour=9, minute=30),
            end=df.index[-1].replace(hour=16, minute=0),
            freq='1min'
        )
        # Filter only market hours
        full_index = full_index[(full_index.time >= pd.Timestamp('09:30').time()) & 
                               (full_index.time <= pd.Timestamp('16:00').time())]
    
    # Reindex and forward fill
    df = df.reindex(full_index)
    df['close'] = df['close'].fillna(method='ffill')
    df['volume'] = df['volume'].fillna(0)  # Volume 0 si no hubo trades
    
    # OHLC: forward fill from previous close
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df['close'])
    df['low'] = df['low'].fillna(df['close'])
    
    return df
```

### 2. Outliers and Fat Fingers
```python
# Problem: Erroneous trades that distort analysis
def detect_outliers(df, method='iqr', threshold=3):
    if method == 'iqr':
        # IQR Method (Interquartile Range)
        Q1 = df['close'].quantile(0.25)
        Q3 = df['close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
    elif method == 'zscore':
        # Z-score Method
        mean = df['close'].mean()
        std = df['close'].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    
    # Mark outliers
    df['is_outlier'] = (df['close'] < lower_bound) | (df['close'] > upper_bound)
    
    # Option 1: Remove
    # df = df[~df['is_outlier']]
    
    # Option 2: Cap (better for trading)
    df.loc[df['close'] > upper_bound, 'close'] = upper_bound
    df.loc[df['close'] < lower_bound, 'close'] = lower_bound
    
    return df
```

### 3. Adjustments for Splits and Dividends
```python
# Problem: AAPL did a 4:1 split, incorrect historical data
def adjust_for_splits(df, splits_df):
    """
    splits_df debe tener columnas: date, ratio
    """
    df = df.copy()
    
    for _, split in splits_df.iterrows():
        split_date = split['date']
        ratio = split['ratio']
        
        # Adjust prices before the split
        mask = df.index < split_date
        df.loc[mask, ['open', 'high', 'low', 'close']] /= ratio
        df.loc[mask, 'volume'] *= ratio
    
    return df

# Using yfinance (already adjusted)
ticker = yf.Ticker('AAPL')
# auto_adjust=True automatically adjusts for splits and dividends
df = ticker.history(period='2y', auto_adjust=True)
```

### 4. Timezone Hell
```python
# Problem: Mixing timezones = disaster
def standardize_timezone(df, target_tz='America/New_York'):
    """
    Convierte todo a Eastern Time (NYSE)
    """
    if df.index.tz is None:
        # Assume UTC if no timezone
        df.index = df.index.tz_localize('UTC')
    
    # Convert to Eastern
    df.index = df.index.tz_convert(target_tz)
    
    # Optional: remove timezone for faster calculations
    # df.index = df.index.tz_localize(None)
    
    return df
```

### 5. Duplicates
```python
# Problem: Same trade reported multiple times
def remove_duplicates(df):
    # Method 1: Exact duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Method 2: Duplicates in time window (more aggressive)
    df = df.sort_index()
    time_diff = df.index.to_series().diff()
    
    # Remove if there's another trade within < 1 second
    mask = time_diff > pd.Timedelta('1s')
    mask.iloc[0] = True  # Keep the first trade
    
    return df[mask]
```

## Complete Cleaning Pipeline

```python
class DataCleaner:
    def __init__(self, config=None):
        self.config = config or self.default_config()
    
    def default_config(self):
        return {
            'remove_outliers': True,
            'outlier_method': 'iqr',
            'outlier_threshold': 3,
            'fill_missing': True,
            'adjust_splits': True,
            'timezone': 'America/New_York',
            'validate_prices': True
        }
    
    def clean(self, df, ticker=None):
        """Complete cleaning pipeline"""
        original_len = len(df)
        
        # 1. Basic validation
        df = self.validate_basic(df)
        
        # 2. Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # 3. Timezone
        if self.config['timezone']:
            df = standardize_timezone(df, self.config['timezone'])
        
        # 4. Fill missing
        if self.config['fill_missing']:
            df = fill_missing_bars(df)
        
        # 5. Outliers
        if self.config['remove_outliers']:
            df = detect_outliers(
                df, 
                method=self.config['outlier_method'],
                threshold=self.config['outlier_threshold']
            )
        
        # 6. Validate OHLC relationships
        if self.config['validate_prices']:
            df = self.validate_ohlc(df)
        
        # 7. Adjust for splits if we have the info
        if self.config['adjust_splits'] and ticker:
            df = self.auto_adjust_splits(df, ticker)
        
        print(f"Cleaned {ticker}: {original_len} -> {len(df)} rows")
        return df
    
    def validate_basic(self, df):
        """Basic sanity validations"""
        # Remove rows with all NaN
        df = df.dropna(how='all')
        
        # Remove negative volume
        df = df[df['volume'] >= 0]
        
        # Remove negative or zero prices
        price_cols = ['open', 'high', 'low', 'close']
        df = df[(df[price_cols] > 0).all(axis=1)]
        
        return df
    
    def validate_ohlc(self, df):
        """Ensure high >= low, etc."""
        # High must be >= open, close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        # Low must be <= open, close
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
    
    def auto_adjust_splits(self, df, ticker):
        """Auto-detect and adjust large splits"""
        # Detect price changes > 40% overnight
        df['prev_close'] = df['close'].shift(1)
        df['overnight_change'] = df['open'] / df['prev_close']
        
        potential_splits = df[
            (df['overnight_change'] < 0.6) |  # Posible split
            (df['overnight_change'] > 1.4)     # Posible reverse split
        ]
        
        if len(potential_splits) > 0:
            print(f"Warning: Potential splits detected for {ticker}")
            print(potential_splits[['close', 'prev_close', 'overnight_change']])
        
        df = df.drop(['prev_close', 'overnight_change'], axis=1)
        return df
```

## Structuring for Analysis

### Adding Common Features
```python
def add_technical_features(df):
    """Add common indicators for analysis"""
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Range
    df['range'] = (df['high'] - df['low']) / df['low'] * 100
    df['range_avg'] = df['range'].rolling(20).mean()
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Gaps
    df['gap'] = df['open'] / df['close'].shift(1) - 1
    
    return df
```

### Structure for Backtesting
```python
class BacktestData:
    def __init__(self, data, universe=None):
        self.data = data
        self.universe = universe or list(data.keys())
        self.current_index = 0
        
    def prepare_for_backtest(self):
        """Prepare data for efficient backtesting"""
        for ticker in self.universe:
            df = self.data[ticker]
            
            # Pre-calculate indicators
            df = add_technical_features(df)
            
            # Create columns for signals
            df['signal'] = 0
            df['position'] = 0
            df['pnl'] = 0
            
            # Index by date for fast lookups
            df = df.sort_index()
            
            self.data[ticker] = df
    
    def get_snapshot(self, date, lookback=20):
        """Get data view up to a certain date"""
        snapshot = {}
        for ticker in self.universe:
            df = self.data[ticker]
            # Data up to the date, including lookback
            mask = df.index <= date
            historical = df[mask].tail(lookback)
            snapshot[ticker] = historical
        
        return snapshot
```

## Final Validation

```python
def validate_dataset(data_dict, start_date, end_date):
    """Complete dataset validation before backtest"""
    report = {
        'tickers': len(data_dict),
        'issues': [],
        'stats': {}
    }
    
    for ticker, df in data_dict.items():
        # Check temporal coverage
        actual_start = df.index[0]
        actual_end = df.index[-1]
        
        if actual_start > pd.Timestamp(start_date):
            report['issues'].append(f"{ticker}: Data starts late ({actual_start})")
        
        if actual_end < pd.Timestamp(end_date):
            report['issues'].append(f"{ticker}: Data ends early ({actual_end})")
        
        # Stats per ticker
        report['stats'][ticker] = {
            'rows': len(df),
            'missing_closes': df['close'].isna().sum(),
            'zero_volume_bars': (df['volume'] == 0).sum(),
            'date_range': f"{actual_start} to {actual_end}"
        }
    
    return report

# Uso
report = validate_dataset(cleaned_data, '2023-01-01', '2024-01-01')
if report['issues']:
    print("WARNING: Data issues found:")
    for issue in report['issues']:
        print(f"  - {issue}")
```

## My Personal Checklist

```python
# clean_all.py
def my_cleaning_pipeline(raw_data):
    steps = [
        ('Remove duplicates', lambda df: df[~df.index.duplicated()]),
        ('Fix timezone', lambda df: standardize_timezone(df)),
        ('Validate OHLC', lambda df: validate_ohlc(df)),
        ('Fill missing', lambda df: fill_missing_bars(df)),
        ('Remove outliers', lambda df: detect_outliers(df, method='iqr')),
        ('Add features', lambda df: add_technical_features(df)),
        ('Final validation', lambda df: validate_basic(df))
    ]
    
    for step_name, step_func in steps:
        print(f"Running: {step_name}")
        raw_data = step_func(raw_data)
    
    return raw_data
```

## Storage Best Practices

```python
# Save clean data
def save_clean_data(df, ticker, format='parquet'):
    date_str = pd.Timestamp.now().strftime('%Y%m%d')
    
    if format == 'parquet':
        # More efficient for reading
        df.to_parquet(f'data/clean/{ticker}_{date_str}.parquet')
    elif format == 'hdf':
        # Better for very large datasets
        df.to_hdf(f'data/clean/{ticker}_{date_str}.h5', key='data')
    elif format == 'csv':
        # Compatible but less efficient
        df.to_csv(f'data/clean/{ticker}_{date_str}.csv')
```

## Red Flags in Your Data

1. **Too many gaps**: Unreliable provider
2. **Erratic volume**: Possible incorrect adjustments
3. **Perfectly round prices**: Synthetic data
4. **Repetitive patterns**: Possibly duplicated data
5. **Unrealistic volatility**: Undetected outliers

## Next Step

With clean data, we can now create optimized [Backtesting Datasets](backtesting_datasets.md).