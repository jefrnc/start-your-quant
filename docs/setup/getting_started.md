> 🇪🇸 [Leer en Español](getting_started.es.md) | 🇺🇸 **English**

# Initial Setup Guide

## Prerequisites

### Required Knowledge
- **Intermediate Python**: Control flow, OOP, library management
- **Trading fundamentals**: Order types, bid/ask, volume, basic charts
- **Basic statistics**: Mean, standard deviation, correlation
- **Recommended experience**: 6+ months of paper trading or live trading

### Recommended Minimum Capital
```python
CAPITAL_REQUIREMENTS = {
    'absolute_minimum': 5_000,      # To learn without real risk
    'recommended_minimum': 25_000,   # PDT rule compliance + diversification
    'comfortable_start': 50_000,    # Multiple simultaneous positions
    'professional_level': 100_000   # Full strategy implementation
}

# Recommended capital distribution
CAPITAL_ALLOCATION = {
    'active_trading': 0.60,    # 60% for active trading
    'cash_buffer': 0.25,       # 25% cash for opportunities
    'emergency_fund': 0.15     # 15% never touch
}
```

## Environment Installation

### 1. Python Environment Setup
```bash
# Create virtual environment
python -m venv quant_trading
source quant_trading/bin/activate  # Linux/Mac
# quant_trading\Scripts\activate    # Windows

# Upgrade pip
python -m pip install --upgrade pip

# Install base dependencies
pip install pandas numpy matplotlib seaborn
pip install scipy scikit-learn statsmodels
pip install yfinance polygon-api-client
pip install jupyter jupyterlab
pip install ta-lib  # Technical Analysis Library
```

### 2. Specific Dependencies
```bash
# Trading APIs
pip install ib_insync              # Interactive Brokers
pip install alpaca-trade-api       # Alpaca
pip install python-telegram-bot    # Telegram alerts

# Data handling
pip install influxdb-client        # Time series database
pip install redis                  # Caching
pip install sqlalchemy            # Database ORM

# Backtesting
pip install backtrader             # Backtesting framework
pip install zipline-reloaded       # Alternative backtesting

# Machine Learning
pip install xgboost lightgbm       # Gradient boosting
pip install tensorflow             # Deep learning (optional)

# Visualization
pip install plotly dash            # Interactive charts
pip install mplfinance            # Financial charts
```

### 3. requirements.txt
```python
# Create requirements.txt file
REQUIREMENTS_CONTENT = """
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.8.0
scikit-learn>=1.1.0
statsmodels>=0.13.0
yfinance>=0.1.70
polygon-api-client>=1.9.0
jupyter>=1.0.0
jupyterlab>=3.4.0
TA-Lib>=0.4.24
ib_insync>=0.9.70
alpaca-trade-api>=2.3.0
python-telegram-bot>=13.0
influxdb-client>=1.30.0
redis>=4.3.0
sqlalchemy>=1.4.0
backtrader>=1.9.76.123
zipline-reloaded>=2.2.0
xgboost>=1.6.0
plotly>=5.9.0
dash>=2.6.0
mplfinance>=0.12.0
requests>=2.28.0
python-dotenv>=0.20.0
"""

# Save requirements.txt
with open('requirements.txt', 'w') as f:
    f.write(REQUIREMENTS_CONTENT)
```

## Project Structure

### Recommended Directory Layout
```python
PROJECT_STRUCTURE = """
start-your-quant/
│
├── config/
│   ├── __init__.py
│   ├── settings.py          # General settings
│   ├── api_keys.py          # API keys (don't commit to git)
│   └── broker_config.py     # Broker configurations
│
├── data/
│   ├── raw/                 # Raw downloaded data
│   ├── processed/           # Processed data
│   ├── historical/          # Historical data
│   └── live/               # Real-time data
│
├── src/
│   ├── __init__.py
│   ├── data_acquisition/    # Data fetching modules
│   ├── indicators/          # Technical indicators
│   ├── strategies/          # Trading strategies
│   ├── backtesting/         # Backtesting framework
│   ├── risk_management/     # Risk management
│   ├── execution/           # Trade execution
│   └── utils/              # General utilities
│
├── notebooks/
│   ├── exploration/         # Exploratory analysis
│   ├── strategy_development/# Strategy development
│   └── backtesting/        # Backtesting notebooks
│
├── tests/
│   ├── unit_tests/         # Unit tests
│   ├── integration_tests/  # Integration tests
│   └── strategy_tests/     # Strategy tests
│
├── logs/
│   ├── trading/            # Trading logs
│   ├── errors/             # Error logs
│   └── performance/        # Performance logs
│
├── docs/                   # Documentation (already created)
├── scripts/                # Automation scripts
├── .env                    # Environment variables
├── .gitignore             # Git ignore file
├── requirements.txt       # Dependencies
└── README.md              # Main documentation
"""

print(PROJECT_STRUCTURE)
```

### Create Structure Automatically
```python
import os

def create_project_structure(base_path="start-your-quant"):
    """Create project directory structure"""
    
    directories = [
        "config",
        "data/raw",
        "data/processed", 
        "data/historical",
        "data/live",
        "src/data_acquisition",
        "src/indicators",
        "src/strategies",
        "src/backtesting",
        "src/risk_management",
        "src/execution",
        "src/utils",
        "notebooks/exploration",
        "notebooks/strategy_development",
        "notebooks/backtesting",
        "tests/unit_tests",
        "tests/integration_tests",
        "tests/strategy_tests",
        "logs/trading",
        "logs/errors",
        "logs/performance",
        "scripts"
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        
        # Create __init__.py in Python directories
        if directory.startswith("src/") or directory == "config":
            init_file = os.path.join(full_path, "__init__.py")
            with open(init_file, 'w') as f:
                f.write(f'"""Module {directory.split("/")[-1]}"""\n')
    
    print("Project structure created successfully!")

# Run function
create_project_structure()
```

## Initial Configuration

### 1. Main Configuration File
```python
# config/settings.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TradingConfig:
    """Main trading configuration"""
    
    # Account settings
    ACCOUNT_SIZE: float = 50_000.0
    MAX_POSITION_SIZE: float = 0.20        # 20% max in one position
    MAX_DAILY_LOSS: float = 0.02           # 2% max daily loss
    MAX_POSITIONS: int = 5                 # Maximum 5 simultaneous positions
    
    # Risk settings
    DEFAULT_STOP_LOSS: float = 0.08        # 8% default stop loss
    DEFAULT_POSITION_RISK: float = 0.02    # 2% risk per position
    
    # Trading hours (EST)
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"
    PREMARKET_START: str = "04:00"
    AFTERHOURS_END: str = "20:00"
    
    # Data settings
    DEFAULT_LOOKBACK_DAYS: int = 252       # 1 year of data
    UPDATE_FREQUENCY_SECONDS: int = 60     # Update every minute
    
    # Paths
    DATA_PATH: str = "data/"
    LOG_PATH: str = "logs/"
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Load configuration from environment variables"""
        return cls(
            ACCOUNT_SIZE=float(os.getenv('ACCOUNT_SIZE', 50000)),
            MAX_POSITION_SIZE=float(os.getenv('MAX_POSITION_SIZE', 0.20)),
            MAX_DAILY_LOSS=float(os.getenv('MAX_DAILY_LOSS', 0.02))
        )

# Global instance
CONFIG = TradingConfig()
```

### 2. API Configuration
```python
# config/api_keys.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APIKeys:
    """Centralized API key management"""
    
    # Data providers
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    IEX_API_KEY = os.getenv('IEX_API_KEY')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Brokers
    IBKR_HOST = os.getenv('IBKR_HOST', 'localhost')
    IBKR_PORT = int(os.getenv('IBKR_PORT', 7497))
    IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', 1))
    
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Notifications
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
    
    @classmethod
    def validate_keys(cls) -> Dict[str, bool]:
        """Validate that API keys are configured"""
        validation = {}
        
        # Data providers
        validation['polygon'] = bool(cls.POLYGON_API_KEY)
        validation['iex'] = bool(cls.IEX_API_KEY)
        
        # Brokers
        validation['ibkr'] = bool(cls.IBKR_HOST and cls.IBKR_PORT)
        validation['alpaca'] = bool(cls.ALPACA_API_KEY and cls.ALPACA_SECRET_KEY)
        
        # Notifications
        validation['telegram'] = bool(cls.TELEGRAM_BOT_TOKEN and cls.TELEGRAM_CHAT_ID)
        
        return validation

# Example .env file
ENV_TEMPLATE = """
# Data APIs
POLYGON_API_KEY=your_polygon_key_here
IEX_API_KEY=your_iex_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Interactive Brokers
IBKR_HOST=localhost
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Alpaca (Paper Trading)
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Trading Configuration
ACCOUNT_SIZE=50000
MAX_POSITION_SIZE=0.20
MAX_DAILY_LOSS=0.02
"""

# Save .env template
with open('.env.example', 'w') as f:
    f.write(ENV_TEMPLATE)
```

### 3. Logging Configuration
```python
# config/logging_config.py
import logging
import os
from datetime import datetime

def setup_logging():
    """Configure logging system"""
    
    # Create log directories if they don't exist
    log_dirs = ['logs/trading', 'logs/errors', 'logs/performance']
    for log_dir in log_dirs:
        os.makedirs(log_dir, exist_ok=True)
    
    # Format configuration
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Main logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format
    )
    
    # Trading logger
    trading_logger = logging.getLogger('trading')
    trading_handler = logging.FileHandler(
        f'logs/trading/trading_{datetime.now().strftime("%Y%m%d")}.log'
    )
    trading_handler.setFormatter(logging.Formatter(log_format, date_format))
    trading_logger.addHandler(trading_handler)
    
    # Error logger
    error_logger = logging.getLogger('errors')
    error_handler = logging.FileHandler(
        f'logs/errors/errors_{datetime.now().strftime("%Y%m%d")}.log'
    )
    error_handler.setFormatter(logging.Formatter(log_format, date_format))
    error_logger.addHandler(error_handler)
    
    # Performance logger
    performance_logger = logging.getLogger('performance')
    performance_handler = logging.FileHandler(
        f'logs/performance/performance_{datetime.now().strftime("%Y%m%d")}.log'
    )
    performance_handler.setFormatter(logging.Formatter(log_format, date_format))
    performance_logger.addHandler(performance_handler)
    
    return {
        'trading': trading_logger,
        'errors': error_logger,
        'performance': performance_logger
    }

# Configure logging on import
LOGGERS = setup_logging()
```

## Setup Verification

### Verification Script
```python
# scripts/verify_setup.py
import sys
import importlib
from config.api_keys import APIKeys
from config.settings import CONFIG

def verify_python_packages():
    """Verify all libraries are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'scipy', 'sklearn', 'statsmodels',
        'yfinance', 'polygon', 'ib_insync',
        'jupyter', 'talib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  {package}")
        except ImportError:
            print(f"  {package} - NOT FOUND")
            missing_packages.append(package)
    
    return missing_packages

def verify_api_connections():
    """Verify API connections"""
    validation = APIKeys.validate_keys()
    
    print("\nAPI Keys Status:")
    for service, is_valid in validation.items():
        status = "OK" if is_valid else "MISSING"
        print(f"  {status} {service.upper()}")
    
    return validation

def test_data_connection():
    """Data connection test"""
    try:
        import yfinance as yf
        
        # Basic test with Yahoo Finance
        ticker = yf.Ticker("SPY")
        data = ticker.history(period="1d")
        
        if len(data) > 0:
            print("Data connection working")
            return True
        else:
            print("Could not fetch data")
            return False
            
    except Exception as e:
        print(f"Data connection error: {e}")
        return False

def main():
    """Main verification function"""
    print("Verifying Quantitative Trading System Setup\n")
    
    # Verify packages
    print("Verifying Python Packages:")
    missing = verify_python_packages()
    
    if missing:
        print(f"\nInstall missing packages: pip install {' '.join(missing)}")
    
    # Verify API keys
    api_status = verify_api_connections()
    
    # Connection test
    print("\nTesting Data Connection:")
    data_ok = test_data_connection()
    
    # Verify configuration
    print(f"\nConfiguration:")
    print(f"Account Size: ${CONFIG.ACCOUNT_SIZE:,.2f}")
    print(f"Max Position Size: {CONFIG.MAX_POSITION_SIZE:.1%}")
    print(f"Max Daily Loss: {CONFIG.MAX_DAILY_LOSS:.1%}")
    
    # Final summary
    print("\nSetup Summary:")
    packages_ok = len(missing) == 0
    apis_configured = any(api_status.values())
    
    if packages_ok and apis_configured and data_ok:
        print("Setup complete! Ready for trading.")
    else:
        print("Setup incomplete. Review items marked above.")
        
        if not packages_ok:
            print("   - Install missing packages")
        if not apis_configured:
            print("   - Configure at least one API key in .env")
        if not data_ok:
            print("   - Verify internet connection and APIs")

if __name__ == "__main__":
    main()
```

## First Steps

### 1. Initial Exploration Notebook
```python
# notebooks/exploration/first_exploration.ipynb
"""
Notebook for first steps and setup verification
"""

# Import main libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Verify everything works
print("Starting Initial Exploration")

# 1. Get sample data
ticker = "AAPL"
data = yf.download(ticker, period="1y")

print(f"Data retrieved for {ticker}: {len(data)} days")
print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")

# 2. Basic analysis
print(f"\nBasic Statistics:")
print(f"Average price: ${data['Close'].mean():.2f}")
print(f"Volatility (std): {data['Close'].pct_change().std():.4f}")
print(f"Total return: {(data['Close'].iloc[-1] / data['Close'].iloc[0] - 1):.2%}")

# 3. Basic chart
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'])
plt.title(f'{ticker} - Closing Price (Last Year)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
plt.show()

print("\nSetup verified! Ready to develop strategies.")
```

### 2. First Strategy Test
```python
# scripts/first_strategy_test.py
"""
Basic test of a simple strategy to verify everything works
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def simple_moving_average_strategy(symbol="SPY", short_window=20, long_window=50):
    """Simple moving average crossover strategy"""
    
    # Get data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate moving averages
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    data['Signal'] = 0
    data['Signal'][short_window:] = np.where(
        data['SMA_short'][short_window:] > data['SMA_long'][short_window:], 1, 0
    )
    
    # Calculate positions
    data['Position'] = data['Signal'].diff()
    
    # Calculate returns
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    
    # Basic metrics
    total_return = (1 + data['Strategy_Returns']).prod() - 1
    buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
    
    print(f"Results for {symbol}:")
    print(f"Strategy Return: {total_return:.2%}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"Excess Return: {(total_return - buy_hold_return):.2%}")
    
    # Number of trades
    trades = len(data[data['Position'] != 0])
    print(f"Number of trades: {trades}")
    
    return data

if __name__ == "__main__":
    print("Testing Simple Strategy\n")
    results = simple_moving_average_strategy()
    print("\nTest completed successfully!")
```

With this initial setup, you'll have a solid foundation to start developing and testing quantitative trading strategies. The next step will be configuring the specific APIs and beginning development of the documented strategies.
