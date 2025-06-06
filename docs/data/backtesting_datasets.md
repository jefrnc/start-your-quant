# Creación de Datasets para Backtesting

## El Framework Mental

Un buen dataset de backtesting no es solo data histórica. Es una representación realista del ambiente de trading, incluyendo todas las limitaciones y fricciones que enfrentarás en vivo.

## Estructura Base

```python
class BacktestDataset:
    def __init__(self):
        self.price_data = {}      # OHLCV data
        self.fundamental_data = {} # Float, sector, etc
        self.universe = []        # Tickers tradeable cada día
        self.metadata = {}        # Info adicional
        
    def add_ticker(self, ticker, data, metadata=None):
        # Validar data
        if not self._validate_data(data):
            raise ValueError(f"Invalid data for {ticker}")
            
        self.price_data[ticker] = data
        if metadata:
            self.metadata[ticker] = metadata
            
    def get_universe(self, date):
        """Obtener tickers tradeables en una fecha"""
        available = []
        for ticker, data in self.price_data.items():
            if date in data.index:
                # Verificar liquidez mínima
                volume = data.loc[date, 'volume']
                price = data.loc[date, 'close']
                dollar_volume = volume * price
                
                if dollar_volume > 1_000_000:  # $1M mínimo
                    available.append(ticker)
        
        return available
```

## Datasets por Tipo de Estrategia

### 1. Dataset para Gap Trading
```python
def create_gap_trading_dataset(start_date, end_date):
    dataset = BacktestDataset()
    
    # Universo: Small caps con volumen
    universe_criteria = {
        'market_cap': (10_000_000, 500_000_000),
        'avg_volume': 500_000,
        'price': (1, 50)
    }
    
    # Obtener tickers que cumplan criterios
    tickers = screen_universe(universe_criteria)
    
    for ticker in tickers:
        # Necesitamos pre-market data
        data = fetch_extended_hours_data(ticker, start_date, end_date)
        
        # Calcular gaps
        data['gap_pct'] = (data['open'] / data['close'].shift(1) - 1) * 100
        data['gap_type'] = data['gap_pct'].apply(classify_gap)
        
        # Agregar indicadores relevantes
        data['premarket_volume'] = calculate_premarket_volume(data)
        data['rvol'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Metadata importante
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

### 2. Dataset para Mean Reversion
```python
def create_mean_reversion_dataset(start_date, end_date):
    dataset = BacktestDataset()
    
    # Para mean reversion queremos stocks líquidos y estables
    universe_criteria = {
        'market_cap': (1_000_000_000, None),  # Large caps
        'avg_volume': 5_000_000,
        'volatility': (0.01, 0.05)  # Volatilidad moderada
    }
    
    tickers = screen_universe(universe_criteria)
    
    for ticker in tickers:
        data = fetch_data(ticker, start_date, end_date)
        
        # Indicadores de mean reversion
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

### 3. Dataset para Pairs Trading
```python
def create_pairs_trading_dataset(start_date, end_date):
    dataset = BacktestDataset()
    
    # Obtener pares correlacionados
    sectors = ['XLK', 'XLF', 'XLE', 'XLV']  # Tech, Financial, Energy, Healthcare
    pairs = []
    
    for sector in sectors:
        stocks = get_sector_stocks(sector)
        correlation_matrix = calculate_correlation_matrix(stocks, start_date, end_date)
        
        # Encontrar pares con alta correlación
        high_corr_pairs = find_high_correlation_pairs(correlation_matrix, threshold=0.8)
        pairs.extend(high_corr_pairs)
    
    # Crear spreads para cada par
    for stock1, stock2 in pairs:
        data1 = fetch_data(stock1, start_date, end_date)
        data2 = fetch_data(stock2, start_date, end_date)
        
        # Alinear datos
        spread_data = pd.DataFrame(index=data1.index)
        spread_data['price1'] = data1['close']
        spread_data['price2'] = data2['close']
        
        # Calcular ratio y z-score
        spread_data['ratio'] = spread_data['price1'] / spread_data['price2']
        spread_data['ratio_mean'] = spread_data['ratio'].rolling(20).mean()
        spread_data['ratio_std'] = spread_data['ratio'].rolling(20).std()
        spread_data['zscore'] = (spread_data['ratio'] - spread_data['ratio_mean']) / spread_data['ratio_std']
        
        # Cointegration test
        spread_data['is_cointegrated'] = test_cointegration(data1['close'], data2['close'])
        
        dataset.add_ticker(f"{stock1}_{stock2}_pair", spread_data)
    
    return dataset
```

## Agregar Datos Fundamentales

```python
def enrich_with_fundamentals(dataset):
    """Agregar datos fundamentales al dataset"""
    
    for ticker in dataset.price_data.keys():
        # Float data
        float_data = get_historical_float(ticker)
        
        # Short interest
        short_data = get_short_interest(ticker)
        
        # Earnings dates
        earnings_dates = get_earnings_calendar(ticker)
        
        # News sentiment
        news_sentiment = get_news_sentiment(ticker)
        
        # Agregar al dataset
        dataset.metadata[ticker].update({
            'float_history': float_data,
            'short_interest': short_data,
            'earnings_dates': earnings_dates,
            'news_sentiment': news_sentiment
        })
    
    return dataset
```

## Considerar Costos y Fricciones

```python
class RealisticBacktestData:
    def __init__(self, base_dataset):
        self.data = base_dataset
        self.cost_model = CostModel()
        
    def add_market_impact(self, ticker, date, size):
        """Estimar impacto en precio por tamaño de orden"""
        daily_volume = self.data.price_data[ticker].loc[date, 'volume']
        participation_rate = size / daily_volume
        
        # Modelo simple de impacto
        if participation_rate < 0.01:
            impact = 0.0001  # 1 basis point
        elif participation_rate < 0.05:
            impact = 0.0005  # 5 basis points
        else:
            impact = 0.001 + (participation_rate - 0.05) * 0.01
            
        return impact
    
    def calculate_realistic_fill(self, ticker, date, side, size):
        """Calcular precio de fill realista"""
        bar = self.data.price_data[ticker].loc[date]
        
        if side == 'buy':
            # Comprar cerca del ask
            base_price = bar['close'] * 1.0001  # Slight premium
            impact = self.add_market_impact(ticker, date, size)
            fill_price = base_price * (1 + impact)
        else:
            # Vender cerca del bid
            base_price = bar['close'] * 0.9999  # Slight discount
            impact = self.add_market_impact(ticker, date, size)
            fill_price = base_price * (1 - impact)
            
        # Agregar slippage aleatorio
        slippage = np.random.normal(0, 0.0001)  # 1bp std dev
        fill_price *= (1 + slippage)
        
        return fill_price
```

## Splits y Corporate Actions

```python
def handle_corporate_actions(dataset):
    """Manejar splits, dividendos, etc."""
    
    for ticker in dataset.price_data.keys():
        # Obtener corporate actions
        actions = get_corporate_actions(ticker)
        
        for action in actions:
            if action['type'] == 'split':
                ratio = action['ratio']
                date = action['date']
                
                # Ajustar precios históricos
                mask = dataset.price_data[ticker].index < date
                dataset.price_data[ticker].loc[mask, ['open', 'high', 'low', 'close']] /= ratio
                dataset.price_data[ticker].loc[mask, 'volume'] *= ratio
                
            elif action['type'] == 'dividend':
                # Ajustar por dividendos si es necesario
                div_amount = action['amount']
                date = action['date']
                
                # Algunos backtests ajustan, otros no
                # Depende de tu estrategia
```

## Survivorship Bias

```python
def create_survivorship_bias_free_dataset(start_date, end_date):
    """Incluir empresas que quebraron o fueron delistadas"""
    
    dataset = BacktestDataset()
    
    # Obtener lista histórica de tickers
    historical_universe = get_historical_constituents('Russell3000', start_date)
    
    for ticker in historical_universe:
        try:
            data = fetch_data(ticker, start_date, end_date)
            
            # Marcar si fue delistado
            if ticker in get_delisted_tickers():
                data['is_delisted'] = True
                data['delisting_date'] = get_delisting_date(ticker)
                
            dataset.add_ticker(ticker, data)
            
        except DataNotAvailable:
            # Importante: incluir incluso si no hay data completa
            print(f"Warning: Limited data for {ticker}")
```

## Optimización para Velocidad

```python
class OptimizedBacktestData:
    def __init__(self, dataset):
        self.dataset = dataset
        self._create_indexes()
        
    def _create_indexes(self):
        """Pre-calcular índices para lookups rápidos"""
        # Crear MultiIndex para acceso rápido
        all_data = []
        
        for ticker, df in self.dataset.price_data.items():
            df['ticker'] = ticker
            all_data.append(df)
            
        self.combined_df = pd.concat(all_data)
        self.combined_df.set_index(['ticker', self.combined_df.index], inplace=True)
        
        # Pre-calcular universos por fecha
        self.daily_universes = {}
        for date in self.combined_df.index.get_level_values(1).unique():
            self.daily_universes[date] = self.dataset.get_universe(date)
    
    def get_bar(self, ticker, date):
        """Acceso ultra-rápido a una barra"""
        return self.combined_df.loc[(ticker, date)]
    
    def get_multiple_bars(self, tickers, date):
        """Obtener múltiples tickers eficientemente"""
        return self.combined_df.loc[(tickers, date)]
```

## Validación Final del Dataset

```python
def validate_backtest_dataset(dataset):
    """Validaciones cruciales antes de backtest"""
    
    issues = []
    
    # 1. Verificar look-ahead bias
    for ticker, data in dataset.price_data.items():
        # Buscar indicadores que usan datos futuros
        for col in data.columns:
            if 'shift(-' in str(data[col]):
                issues.append(f"Possible look-ahead bias in {ticker}:{col}")
    
    # 2. Verificar consistencia temporal
    dates = set()
    for ticker, data in dataset.price_data.items():
        dates.update(data.index)
    
    # Todos los tickers deben tener fechas similares
    date_coverage = {}
    for ticker, data in dataset.price_data.items():
        coverage = len(data) / len(dates)
        if coverage < 0.8:  # Menos del 80% de cobertura
            issues.append(f"{ticker} has only {coverage:.1%} date coverage")
    
    # 3. Verificar datos extremos
    for ticker, data in dataset.price_data.items():
        returns = data['close'].pct_change()
        if (returns > 5).any():  # +500% en un día
            issues.append(f"{ticker} has suspicious returns > 500%")
    
    return issues
```

## Mi Setup Personal

```python
# create_my_dataset.py
def create_my_trading_dataset():
    # Configuración base
    config = {
        'start_date': '2022-01-01',
        'end_date': '2024-01-01',
        'universe': 'small_caps',
        'min_volume': 1_000_000,
        'min_price': 1,
        'max_price': 50
    }
    
    # Crear datasets para cada estrategia
    datasets = {
        'gap_trading': create_gap_trading_dataset(**config),
        'vwap_bounce': create_mean_reversion_dataset(**config),
        'momentum': create_momentum_dataset(**config)
    }
    
    # Enriquecer con fundamentales
    for name, dataset in datasets.items():
        enrich_with_fundamentals(dataset)
        handle_corporate_actions(dataset)
    
    # Optimizar para velocidad
    optimized = {
        name: OptimizedBacktestData(dataset) 
        for name, dataset in datasets.items()
    }
    
    return optimized
```

## Siguiente Paso

Con datasets robustos listos, pasemos a los [Indicadores Técnicos](../indicators/vwap.md) específicos para small caps.