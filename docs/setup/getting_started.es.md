> 🇺🇸 [Read in English](getting_started.md) | 🇪🇸 **Español**

# Guía de Setup Inicial

## Prerrequisitos

### Conocimientos Requeridos
- **Python intermedio**: Control de flujo, OOP, manejo de librerías
- **Fundamentos de trading**: Order types, bid/ask, volumen, charts básicos
- **Estadística básica**: Media, desviación estándar, correlación
- **Experiencia recomendada**: 6+ meses de paper trading o trading real

### Capital Mínimo Recomendado
```python
CAPITAL_REQUIREMENTS = {
    'absolute_minimum': 5_000,      # Para aprender sin riesgo real
    'recommended_minimum': 25_000,   # PDT rule compliance + diversification
    'comfortable_start': 50_000,    # Múltiples posiciones simultáneas
    'professional_level': 100_000   # Full strategy implementation
}

# Distribución recomendada del capital
CAPITAL_ALLOCATION = {
    'active_trading': 0.60,    # 60% para trading activo
    'cash_buffer': 0.25,       # 25% cash para oportunidades
    'emergency_fund': 0.15     # 15% nunca tocar
}
```

## Instalación del Entorno

### 1. Python Environment Setup
```bash
# Crear entorno virtual
python -m venv quant_trading
source quant_trading/bin/activate  # Linux/Mac
# quant_trading\Scripts\activate    # Windows

# Upgrade pip
python -m pip install --upgrade pip

# Instalar dependencias base
pip install pandas numpy matplotlib seaborn
pip install scipy scikit-learn statsmodels
pip install yfinance polygon-api-client
pip install jupyter jupyterlab
pip install ta-lib  # Technical Analysis Library
```

### 2. Dependencias Específicas
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
# Crear archivo requirements.txt
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

# Guardar requirements.txt
with open('requirements.txt', 'w') as f:
    f.write(REQUIREMENTS_CONTENT)
```

## Estructura de Proyecto

### Directorio Recomendado
```python
PROJECT_STRUCTURE = """
start-your-quant/
│
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuraciones generales
│   ├── api_keys.py          # API keys (no subir a git)
│   └── broker_config.py     # Configuraciones de brokers
│
├── data/
│   ├── raw/                 # Datos crudos descargados
│   ├── processed/           # Datos procesados
│   ├── historical/          # Datos históricos
│   └── live/               # Datos en tiempo real
│
├── src/
│   ├── __init__.py
│   ├── data_acquisition/    # Módulos de obtención de datos
│   ├── indicators/          # Indicadores técnicos
│   ├── strategies/          # Estrategias de trading
│   ├── backtesting/         # Framework de backtesting
│   ├── risk_management/     # Gestión de riesgo
│   ├── execution/           # Ejecución de trades
│   └── utils/              # Utilidades generales
│
├── notebooks/
│   ├── exploration/         # Análisis exploratorio
│   ├── strategy_development/# Desarrollo de estrategias
│   └── backtesting/        # Notebooks de backtesting
│
├── tests/
│   ├── unit_tests/         # Tests unitarios
│   ├── integration_tests/  # Tests de integración
│   └── strategy_tests/     # Tests de estrategias
│
├── logs/
│   ├── trading/            # Logs de trading
│   ├── errors/             # Logs de errores
│   └── performance/        # Logs de performance
│
├── docs/                   # Documentación (ya creada)
├── scripts/                # Scripts de automatización
├── .env                    # Variables de entorno
├── .gitignore             # Git ignore file
├── requirements.txt       # Dependencias
└── README.md              # Documentación principal
"""

print(PROJECT_STRUCTURE)
```

### Crear Estructura Automáticamente
```python
import os

def create_project_structure(base_path="start-your-quant"):
    """Crear estructura de directorios del proyecto"""
    
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
        
        # Crear __init__.py en directorios de Python
        if directory.startswith("src/") or directory == "config":
            init_file = os.path.join(full_path, "__init__.py")
            with open(init_file, 'w') as f:
                f.write(f'"""Módulo {directory.split("/")[-1]}"""\n')
    
    print("Estructura de proyecto creada exitosamente!")

# Ejecutar función
create_project_structure()
```

## Configuración Inicial

### 1. Archivo de Configuración Principal
```python
# config/settings.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TradingConfig:
    """Configuración principal de trading"""
    
    # Account settings
    ACCOUNT_SIZE: float = 50_000.0
    MAX_POSITION_SIZE: float = 0.20        # 20% max en una posición
    MAX_DAILY_LOSS: float = 0.02           # 2% max pérdida diaria
    MAX_POSITIONS: int = 5                 # Máximo 5 posiciones simultáneas
    
    # Risk settings
    DEFAULT_STOP_LOSS: float = 0.08        # 8% stop loss default
    DEFAULT_POSITION_RISK: float = 0.02    # 2% riesgo por posición
    
    # Trading hours (EST)
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"
    PREMARKET_START: str = "04:00"
    AFTERHOURS_END: str = "20:00"
    
    # Data settings
    DEFAULT_LOOKBACK_DAYS: int = 252       # 1 año de datos
    UPDATE_FREQUENCY_SECONDS: int = 60     # Update cada minuto
    
    # Paths
    DATA_PATH: str = "data/"
    LOG_PATH: str = "logs/"
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Cargar configuración desde variables de entorno"""
        return cls(
            ACCOUNT_SIZE=float(os.getenv('ACCOUNT_SIZE', 50000)),
            MAX_POSITION_SIZE=float(os.getenv('MAX_POSITION_SIZE', 0.20)),
            MAX_DAILY_LOSS=float(os.getenv('MAX_DAILY_LOSS', 0.02))
        )

# Instancia global
CONFIG = TradingConfig()
```

### 2. Configuración de APIs
```python
# config/api_keys.py
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class APIKeys:
    """Gestión centralizada de API keys"""
    
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
        """Validar que las API keys estén configuradas"""
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

# Ejemplo de archivo .env
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

# Guardar template de .env
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
    """Configurar sistema de logging"""
    
    # Crear directorios de logs si no existen
    log_dirs = ['logs/trading', 'logs/errors', 'logs/performance']
    for log_dir in log_dirs:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configuración del formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Logger principal
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format
    )
    
    # Logger para trading
    trading_logger = logging.getLogger('trading')
    trading_handler = logging.FileHandler(
        f'logs/trading/trading_{datetime.now().strftime("%Y%m%d")}.log'
    )
    trading_handler.setFormatter(logging.Formatter(log_format, date_format))
    trading_logger.addHandler(trading_handler)
    
    # Logger para errores
    error_logger = logging.getLogger('errors')
    error_handler = logging.FileHandler(
        f'logs/errors/errors_{datetime.now().strftime("%Y%m%d")}.log'
    )
    error_handler.setFormatter(logging.Formatter(log_format, date_format))
    error_logger.addHandler(error_handler)
    
    # Logger para performance
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

# Configurar logging al importar
LOGGERS = setup_logging()
```

## Verificación del Setup

### Script de Verificación
```python
# scripts/verify_setup.py
import sys
import importlib
from config.api_keys import APIKeys
from config.settings import CONFIG

def verify_python_packages():
    """Verificar que todas las librerías estén instaladas"""
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
    """Verificar conexiones a APIs"""
    validation = APIKeys.validate_keys()
    
    print("\nAPI Keys Status:")
    for service, is_valid in validation.items():
        status = "OK" if is_valid else "MISSING"
        print(f"  {status} {service.upper()}")
    
    return validation

def test_data_connection():
    """Test de conexión a datos"""
    try:
        import yfinance as yf
        
        # Test básico con Yahoo Finance
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
    """Función principal de verificación"""
    print("Verifying Quantitative Trading System Setup\n")
    
    # Verificar packages
    print("Verifying Python Packages:")
    missing = verify_python_packages()
    
    if missing:
        print(f"\nInstall missing packages: pip install {' '.join(missing)}")
    
    # Verificar API keys
    api_status = verify_api_connections()
    
    # Test de conexión
    print("\nTesting Data Connection:")
    data_ok = test_data_connection()
    
    # Verificar configuración
    print(f"\nConfiguration:")
    print(f"Account Size: ${CONFIG.ACCOUNT_SIZE:,.2f}")
    print(f"Max Position Size: {CONFIG.MAX_POSITION_SIZE:.1%}")
    print(f"Max Daily Loss: {CONFIG.MAX_DAILY_LOSS:.1%}")
    
    # Resumen final
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

## Primeros Pasos

### 1. Notebook de Exploración Inicial
```python
# notebooks/exploration/first_exploration.ipynb
"""
Notebook para primeros pasos y verificación del setup
"""

# Importar librerías principales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Verificar que todo funciona
print("Starting Initial Exploration")

# 1. Obtener datos de ejemplo
ticker = "AAPL"
data = yf.download(ticker, period="1y")

print(f"Data retrieved for {ticker}: {len(data)} days")
print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")

# 2. Análisis básico
print(f"\nBasic Statistics:")
print(f"Average price: ${data['Close'].mean():.2f}")
print(f"Volatility (std): {data['Close'].pct_change().std():.4f}")
print(f"Total return: {(data['Close'].iloc[-1] / data['Close'].iloc[0] - 1):.2%}")

# 3. Gráfico básico
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'])
plt.title(f'{ticker} - Closing Price (Last Year)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
plt.show()

print("\nSetup verified! Ready to develop strategies.")
```

### 2. Primer Test de Strategy
```python
# scripts/first_strategy_test.py
"""
Test básico de una estrategia simple para verificar que todo funciona
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def simple_moving_average_strategy(symbol="SPY", short_window=20, long_window=50):
    """Estrategia simple de moving average crossover"""
    
    # Obtener datos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calcular moving averages
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    
    # Generar señales
    data['Signal'] = 0
    data['Signal'][short_window:] = np.where(
        data['SMA_short'][short_window:] > data['SMA_long'][short_window:], 1, 0
    )
    
    # Calcular posiciones
    data['Position'] = data['Signal'].diff()
    
    # Calcular retornos
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    
    # Métricas básicas
    total_return = (1 + data['Strategy_Returns']).prod() - 1
    buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
    
    print(f"Results for {symbol}:")
    print(f"Strategy Return: {total_return:.2%}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"Excess Return: {(total_return - buy_hold_return):.2%}")
    
    # Número de trades
    trades = len(data[data['Position'] != 0])
    print(f"Number of trades: {trades}")
    
    return data

if __name__ == "__main__":
    print("Testing Simple Strategy\n")
    results = simple_moving_average_strategy()
    print("\nTest completed successfully!")
```

Con este setup inicial tendrás una base sólida para comenzar a desarrollar y testear estrategias de trading cuantitativo. El siguiente paso será configurar las APIs específicas y comenzar con el development de las estrategias documentadas.
