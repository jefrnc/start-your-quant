# Workflow de Desarrollo

## Jupyter Lab Setup Optimizado

### Configuración de Jupyter Lab
```python
# scripts/setup_jupyter.py
"""
Configuración optimizada de Jupyter Lab para trading development
"""

import os
import json
from pathlib import Path

def setup_jupyter_config():
    """Configurar Jupyter Lab para trading"""
    
    # Configuración de Jupyter Lab
    jupyter_config = {
        "notebook": {
            "Completer": {
                "use_jedi": False  # Mejor autocompletado
            }
        },
        "lab": {
            "shortcuts": [
                {
                    "command": "notebook:run-cell-and-select-next",
                    "keys": ["Shift Enter"],
                    "selector": ".jp-Notebook.jp-mod-editMode"
                },
                {
                    "command": "notebook:run-cell",
                    "keys": ["Ctrl Enter"],
                    "selector": ".jp-Notebook.jp-mod-editMode"
                }
            ]
        }
    }
    
    # Extensiones recomendadas
    extensions = [
        "@jupyter-widgets/jupyterlab-manager",
        "@jupyterlab/toc",
        "@krassowski/jupyterlab-lsp",
        "jupyterlab-plotly",
        "@jupyterlab/git"
    ]
    
    print("⚙️  Configurando Jupyter Lab...")
    
    # Crear directorio de configuración
    jupyter_dir = Path.home() / ".jupyter"
    jupyter_dir.mkdir(exist_ok=True)
    
    # Guardar configuración
    config_file = jupyter_dir / "jupyter_lab_config.json"
    with open(config_file, 'w') as f:
        json.dump(jupyter_config, f, indent=2)
    
    print("✅ Configuración guardada")
    
    # Instalar extensiones
    print("📦 Instalando extensiones...")
    for ext in extensions:
        os.system(f"jupyter labextension install {ext}")
    
    print("🚀 Jupyter Lab configurado para trading!")

# Función para crear template de notebook
def create_notebook_template():
    """Crear template estándar para notebooks de trading"""
    
    template = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Trading Analysis Notebook\n",
                    "\n",
                    "**Date:** {date}\n",
                    "**Strategy:** [Strategy Name]\n",
                    "**Symbol(s):** [Symbols]\n",
                    "**Timeframe:** [Timeframe]\n",
                    "\n",
                    "## Objective\n",
                    "[Describe the analysis objective]\n",
                    "\n",
                    "## Key Questions\n",
                    "1. [Question 1]\n",
                    "2. [Question 2]\n",
                    "3. [Question 3]"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Standard imports for trading analysis\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import yfinance as yf\n",
                    "from datetime import datetime, timedelta\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Custom modules\n",
                    "import sys\n",
                    "sys.path.append('../src')\n",
                    "\n",
                    "# Configuration\n",
                    "plt.style.use('dark_background')\n",
                    "pd.set_option('display.max_columns', None)\n",
                    "pd.set_option('display.width', None)\n",
                    "\n",
                    "print(\"📊 Trading Analysis Environment Ready\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Data Acquisition"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Data acquisition code here\n",
                    "symbol = 'AAPL'  # Replace with target symbol\n",
                    "start_date = '2024-01-01'\n",
                    "end_date = datetime.now().strftime('%Y-%m-%d')\n",
                    "\n",
                    "# Download data\n",
                    "data = yf.download(symbol, start=start_date, end=end_date)\n",
                    "print(f\"📈 Downloaded {len(data)} days of data for {symbol}\")\n",
                    "display(data.head())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Exploratory Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Basic statistics and exploration\n",
                    "print(f\"📊 Basic Statistics for {symbol}:\")\n",
                    "print(f\"Period: {data.index[0].date()} to {data.index[-1].date()}\")\n",
                    "print(f\"Trading days: {len(data)}\")\n",
                    "print(f\"Average daily volume: {data['Volume'].mean():,.0f}\")\n",
                    "print(f\"Price range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}\")\n",
                    "\n",
                    "# Calculate returns\n",
                    "data['Returns'] = data['Close'].pct_change()\n",
                    "total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1\n",
                    "print(f\"Total return: {total_return:.2%}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Technical Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Technical indicators code here\n",
                    "pass"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Strategy Implementation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Strategy code here\n",
                    "pass"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Results and Conclusions\n",
                    "\n",
                    "### Key Findings\n",
                    "- [Finding 1]\n",
                    "- [Finding 2]\n",
                    "- [Finding 3]\n",
                    "\n",
                    "### Next Steps\n",
                    "- [Next step 1]\n",
                    "- [Next step 2]\n",
                    "- [Next step 3]"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return template

# Función para generar notebook desde template
def generate_notebook(strategy_name, symbol=None):
    """Generar notebook desde template"""
    from datetime import datetime
    
    template = create_notebook_template()
    
    # Personalizar template
    date_str = datetime.now().strftime('%Y-%m-%d')
    template['cells'][0]['source'][0] = template['cells'][0]['source'][0].format(date=date_str)
    
    if strategy_name:
        template['cells'][0]['source'][0] = template['cells'][0]['source'][0].replace(
            '[Strategy Name]', strategy_name
        )
    
    if symbol:
        template['cells'][0]['source'][0] = template['cells'][0]['source'][0].replace(
            '[Symbols]', symbol
        )
        template['cells'][3]['source'][0] = template['cells'][3]['source'][0].replace(
            "'AAPL'", f"'{symbol}'"
        )
    
    # Guardar notebook
    filename = f"notebooks/exploration/{strategy_name.lower().replace(' ', '_')}_{date_str}.ipynb"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"📓 Notebook creado: {filename}")
    return filename

if __name__ == "__main__":
    setup_jupyter_config()
    print("\n📓 Creando notebook de ejemplo...")
    generate_notebook("Gap and Go Analysis", "TSLA")
```

## Git Workflow para Trading

### Git Configuration
```bash
# scripts/setup_git.sh
#!/bin/bash

echo "🔧 Configurando Git para proyecto de trading..."

# Configuración básica
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
git config --global init.defaultBranch main

# Configuraciones útiles para desarrollo
git config --global pull.rebase false
git config --global core.autocrlf input
git config --global core.safecrlf true

# Aliases útiles
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
git config --global alias.lg "log --oneline --decorate --all --graph"

echo "✅ Git configurado exitosamente"
```

### Git Hooks para Trading Projects
```python
# scripts/setup_git_hooks.py
"""
Setup Git hooks específicos para proyectos de trading
"""

import os
import stat
from pathlib import Path

def create_pre_commit_hook():
    """Crear pre-commit hook para validaciones"""
    
    hook_content = '''#!/usr/bin/env python3
"""
Pre-commit hook para validar código de trading
"""

import sys
import subprocess
import os

def check_api_keys():
    """Verificar que no se suban API keys"""
    
    # Patrones a buscar
    dangerous_patterns = [
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'secret_key\s*=\s*["\'][^"\']+["\']',
        r'password\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']'
    ]
    
    # Obtener archivos modificados
    result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        return True
    
    modified_files = result.stdout.strip().split('\\n')
    
    # Revisar archivos Python
    for file in modified_files:
        if file.endswith('.py') and os.path.exists(file):
            with open(file, 'r') as f:
                content = f.read()
                
            for pattern in dangerous_patterns:
                import re
                if re.search(pattern, content, re.IGNORECASE):
                    print(f"❌ PELIGRO: Posible API key en {file}")
                    print("   Usa variables de entorno o archivos .env")
                    return False
    
    return True

def check_notebook_outputs():
    """Verificar que notebooks no tengan outputs"""
    
    result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        return True
    
    modified_files = result.stdout.strip().split('\\n')
    
    for file in modified_files:
        if file.endswith('.ipynb') and os.path.exists(file):
            with open(file, 'r') as f:
                import json
                try:
                    notebook = json.load(f)
                    for cell in notebook.get('cells', []):
                        if cell.get('outputs') or cell.get('execution_count'):
                            print(f"⚠️  Notebook {file} tiene outputs")
                            print("   Ejecuta: jupyter nbconvert --clear-output --inplace *.ipynb")
                            return False
                except json.JSONDecodeError:
                    continue
    
    return True

def run_tests():
    """Ejecutar tests básicos"""
    
    if os.path.exists('tests/'):
        print("🧪 Ejecutando tests...")
        result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-v'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Tests fallaron:")
            print(result.stdout)
            print(result.stderr)
            return False
        else:
            print("✅ Tests pasaron")
    
    return True

def main():
    """Función principal del hook"""
    
    print("🔍 Ejecutando validaciones pre-commit...")
    
    checks = [
        ("API Keys", check_api_keys),
        ("Notebook Outputs", check_notebook_outputs),
        ("Tests", run_tests)
    ]
    
    for check_name, check_func in checks:
        print(f"Verificando {check_name}...")
        if not check_func():
            print(f"❌ Falló verificación: {check_name}")
            sys.exit(1)
    
    print("✅ Todas las verificaciones pasaron")
    sys.exit(0)

if __name__ == "__main__":
    main()
'''
    
    # Crear directorio de hooks
    hooks_dir = Path('.git/hooks')
    hooks_dir.mkdir(exist_ok=True)
    
    # Escribir hook
    hook_file = hooks_dir / 'pre-commit'
    with open(hook_file, 'w') as f:
        f.write(hook_content)
    
    # Hacer ejecutable
    hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
    
    print("✅ Pre-commit hook instalado")

def create_commit_msg_hook():
    """Crear commit message hook"""
    
    hook_content = '''#!/usr/bin/env python3
"""
Commit message hook para enforcer convenciones
"""

import sys
import re

def validate_commit_message(msg):
    """Validar formato de commit message"""
    
    # Patrón: tipo(scope): descripción
    # Ejemplo: feat(strategy): add VWAP reclaim strategy
    pattern = r'^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .{1,50}'
    
    if not re.match(pattern, msg):
        print("❌ Formato de commit inválido")
        print("Usar: tipo(scope): descripción")
        print("Tipos: feat, fix, docs, style, refactor, test, chore")
        print("Ejemplo: feat(strategy): add VWAP reclaim strategy")
        return False
    
    return True

def main():
    """Función principal"""
    
    if len(sys.argv) != 2:
        sys.exit(1)
    
    commit_msg_file = sys.argv[1]
    
    with open(commit_msg_file, 'r') as f:
        commit_msg = f.read().strip()
    
    # Ignorar merge commits
    if commit_msg.startswith('Merge'):
        sys.exit(0)
    
    if not validate_commit_message(commit_msg):
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
'''
    
    hooks_dir = Path('.git/hooks')
    hook_file = hooks_dir / 'commit-msg'
    
    with open(hook_file, 'w') as f:
        f.write(hook_content)
    
    hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
    
    print("✅ Commit message hook instalado")

if __name__ == "__main__":
    create_pre_commit_hook()
    create_commit_msg_hook()
```

## Testing Framework

### Unit Testing Setup
```python
# tests/test_indicators.py
"""
Tests para indicadores técnicos
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from indicators.custom_indicators import CustomIndicators

class TestCustomIndicators(unittest.TestCase):
    """Tests para indicadores personalizados"""
    
    def setUp(self):
        """Setup datos de prueba"""
        
        # Crear datos sintéticos
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        np.random.seed(42)  # Para reproducibilidad
        
        self.test_data = pd.DataFrame({
            'open': 100 + np.random.randn(50).cumsum() * 0.5,
            'high': 102 + np.random.randn(50).cumsum() * 0.5,
            'low': 98 + np.random.randn(50).cumsum() * 0.5,
            'close': 100 + np.random.randn(50).cumsum() * 0.5,
            'volume': np.random.randint(100000, 1000000, 50)
        }, index=dates)
        
        # Asegurar que high >= low
        self.test_data['high'] = np.maximum(self.test_data['high'], self.test_data['low'])
    
    def test_vwap_calculation(self):
        """Test cálculo de VWAP"""
        
        vwap = CustomIndicators.vwap(self.test_data)
        
        # VWAP debe ser un Series
        self.assertIsInstance(vwap, pd.Series)
        
        # VWAP debe tener misma longitud que data
        self.assertEqual(len(vwap), len(self.test_data))
        
        # VWAP no debe tener NaN (excepto primeros valores)
        self.assertTrue(pd.notna(vwap.iloc[-1]))
        
        # VWAP debe estar dentro de rango razonable
        min_price = self.test_data[['high', 'low', 'close']].min().min()
        max_price = self.test_data[['high', 'low', 'close']].max().max()
        
        self.assertTrue(vwap.iloc[-1] >= min_price * 0.9)
        self.assertTrue(vwap.iloc[-1] <= max_price * 1.1)
    
    def test_relative_volume(self):
        """Test cálculo de volumen relativo"""
        
        rvol = CustomIndicators.relative_volume(self.test_data, lookback_periods=10)
        
        # Debe ser Series
        self.assertIsInstance(rvol, pd.Series)
        
        # Debe ser positivo
        self.assertTrue((rvol >= 0).all())
        
        # Valores válidos después del período de lookback
        valid_values = rvol.iloc[10:]
        self.assertTrue(pd.notna(valid_values).all())
    
    def test_gap_percentage(self):
        """Test cálculo de gap percentage"""
        
        gap_pct = CustomIndicators.gap_percentage(self.test_data)
        
        # Debe ser Series
        self.assertIsInstance(gap_pct, pd.Series)
        
        # Primer valor debe ser NaN
        self.assertTrue(pd.isna(gap_pct.iloc[0]))
        
        # Valores posteriores deben ser numéricos
        self.assertTrue(pd.notna(gap_pct.iloc[1:]).all())
    
    def test_money_flow_index(self):
        """Test Money Flow Index"""
        
        mfi = CustomIndicators.money_flow_index(self.test_data, period=10)
        
        # Debe ser Series
        self.assertIsInstance(mfi, pd.Series)
        
        # MFI debe estar entre 0 y 100
        valid_mfi = mfi.dropna()
        self.assertTrue((valid_mfi >= 0).all())
        self.assertTrue((valid_mfi <= 100).all())
    
    def test_true_range(self):
        """Test True Range"""
        
        tr = CustomIndicators.true_range(self.test_data)
        
        # Debe ser Series
        self.assertIsInstance(tr, pd.Series)
        
        # True Range debe ser positivo
        valid_tr = tr.dropna()
        self.assertTrue((valid_tr >= 0).all())
    
    def test_squeeze_indicator(self):
        """Test Squeeze Indicator"""
        
        squeeze = CustomIndicators.squeeze_indicator(self.test_data)
        
        # Debe ser Series de booleanos
        self.assertIsInstance(squeeze, pd.Series)
        self.assertTrue(squeeze.dtype == bool)
    
    def test_consecutive_bars(self):
        """Test consecutive bars counter"""
        
        # Test up bars
        consecutive_up = CustomIndicators.consecutive_bars(self.test_data, 'up')
        self.assertIsInstance(consecutive_up, pd.Series)
        self.assertTrue((consecutive_up >= 0).all())
        
        # Test down bars
        consecutive_down = CustomIndicators.consecutive_bars(self.test_data, 'down')
        self.assertIsInstance(consecutive_down, pd.Series)
        self.assertTrue((consecutive_down >= 0).all())
    
    def test_pivot_points(self):
        """Test pivot points calculation"""
        
        # Traditional method
        pivots_trad = CustomIndicators.pivot_points(self.test_data, 'traditional')
        
        required_keys = ['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']
        for key in required_keys:
            self.assertIn(key, pivots_trad)
            self.assertIsInstance(pivots_trad[key], pd.Series)
        
        # Fibonacci method
        pivots_fib = CustomIndicators.pivot_points(self.test_data, 'fibonacci')
        
        for key in required_keys:
            self.assertIn(key, pivots_fib)
            self.assertIsInstance(pivots_fib[key], pd.Series)

class TestDataValidation(unittest.TestCase):
    """Tests para validación de datos"""
    
    def test_data_integrity(self):
        """Test integridad de datos"""
        
        # Datos de ejemplo con problemas
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        
        # Datos con high < low (inválido)
        bad_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [95] * 10,  # High menor que low
            'low': [98] * 10,
            'close': [97] * 10,
            'volume': [1000] * 10
        }, index=dates)
        
        # Función para validar datos
        def validate_ohlc_data(df):
            """Validar datos OHLC"""
            errors = []
            
            # High debe ser >= low
            if (df['high'] < df['low']).any():
                errors.append("High price less than low price")
            
            # High debe ser >= open y close
            if (df['high'] < df['open']).any():
                errors.append("High price less than open price")
            if (df['high'] < df['close']).any():
                errors.append("High price less than close price")
            
            # Low debe ser <= open y close
            if (df['low'] > df['open']).any():
                errors.append("Low price greater than open price")
            if (df['low'] > df['close']).any():
                errors.append("Low price greater than close price")
            
            # Volume debe ser positivo
            if (df['volume'] <= 0).any():
                errors.append("Volume must be positive")
            
            return errors
        
        # Test que detecta errores
        errors = validate_ohlc_data(bad_data)
        self.assertTrue(len(errors) > 0)

if __name__ == '__main__':
    # Ejecutar tests
    unittest.main(verbosity=2)
```

### Integration Testing
```python
# tests/test_integration.py
"""
Tests de integración para trading system
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestTradingSystemIntegration(unittest.TestCase):
    """Tests de integración del sistema completo"""
    
    def test_data_to_signals_pipeline(self):
        """Test pipeline desde datos hasta señales"""
        
        # Mock data provider
        mock_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(100000, 1000000, 100)
        }, index=pd.date_range('2024-01-01', periods=100, freq='D'))
        
        # Ensure valid OHLC
        mock_data['high'] = np.maximum(mock_data['high'], 
                                      np.maximum(mock_data['open'], mock_data['close']))
        mock_data['low'] = np.minimum(mock_data['low'], 
                                     np.minimum(mock_data['open'], mock_data['close']))
        
        # Simular pipeline completo
        def full_pipeline(data):
            """Pipeline completo de datos a señales"""
            
            # 1. Calcular indicadores
            data['sma_20'] = data['close'].rolling(20).mean()
            data['rsi'] = self.calculate_rsi(data['close'])
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # 2. Generar señales
            signals = pd.DataFrame(index=data.index)
            
            # Señal simple: precio > SMA y RSI < 70 y volumen alto
            signals['long_signal'] = (
                (data['close'] > data['sma_20']) &
                (data['rsi'] < 70) &
                (data['volume_ratio'] > 1.5)
            )
            
            # 3. Calcular retornos de estrategia
            signals['strategy_return'] = (
                signals['long_signal'].shift(1) * data['close'].pct_change()
            )
            
            return signals
        
        # Ejecutar pipeline
        signals = full_pipeline(mock_data)
        
        # Validaciones
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('long_signal', signals.columns)
        self.assertIn('strategy_return', signals.columns)
        
        # Verificar que hay señales
        total_signals = signals['long_signal'].sum()
        self.assertGreater(total_signals, 0)
        
        # Verificar retornos son válidos
        valid_returns = signals['strategy_return'].dropna()
        self.assertTrue(len(valid_returns) > 0)
    
    def calculate_rsi(self, prices, period=14):
        """Helper para calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @patch('requests.get')
    def test_data_provider_fallback(self, mock_get):
        """Test fallback entre data providers"""
        
        # Simular primer provider fallando
        mock_get.side_effect = [
            Exception("Connection failed"),  # Primer intento falla
            Mock(status_code=200, json=lambda: {'price': 150.0})  # Segundo éxito
        ]
        
        # Simular UnifiedDataProvider
        class MockUnifiedProvider:
            def __init__(self):
                self.providers = {
                    'primary': Mock(),
                    'secondary': Mock()
                }
            
            def get_quote(self, symbol):
                try:
                    # Intentar primary
                    raise Exception("Primary failed")
                except:
                    # Fallback to secondary
                    return {'price': 150.0, 'source': 'secondary'}
        
        provider = MockUnifiedProvider()
        result = provider.get_quote('AAPL')
        
        self.assertEqual(result['price'], 150.0)
        self.assertEqual(result['source'], 'secondary')
    
    def test_risk_management_integration(self):
        """Test integración de risk management"""
        
        # Mock portfolio state
        portfolio = {
            'cash': 50000,
            'positions': {
                'AAPL': {'shares': 100, 'avg_price': 150.0},
                'TSLA': {'shares': 50, 'avg_price': 200.0}
            }
        }
        
        # Mock position sizer
        class MockPositionSizer:
            def __init__(self, max_position_size=0.2):
                self.max_position_size = max_position_size
            
            def calculate_position_size(self, signal_strength, current_price, stop_price):
                account_value = 50000 + (100 * 150) + (50 * 200)  # 75000
                risk_amount = account_value * 0.02  # 2% risk
                
                risk_per_share = abs(current_price - stop_price)
                if risk_per_share == 0:
                    return 0
                
                max_shares = int(risk_amount / risk_per_share)
                
                # Apply position size limit
                max_position_value = account_value * self.max_position_size
                max_shares_by_exposure = int(max_position_value / current_price)
                
                return min(max_shares, max_shares_by_exposure)
        
        # Test position sizing
        sizer = MockPositionSizer()
        
        # Strong signal, reasonable stop
        position_size = sizer.calculate_position_size(
            signal_strength=0.8,
            current_price=100.0,
            stop_price=95.0
        )
        
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size * 100, 75000 * 0.2)  # Respeta límite de exposición

class TestPerformanceMetrics(unittest.TestCase):
    """Tests para métricas de performance"""
    
    def test_comprehensive_metrics(self):
        """Test cálculo de métricas comprehensivas"""
        
        # Generate synthetic returns
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 252),  # Daily returns for 1 year
            index=pd.date_range('2024-01-01', periods=252, freq='D')
        )
        
        def calculate_all_metrics(returns_series):
            """Calcular todas las métricas de performance"""
            
            metrics = {}
            
            # Basic metrics
            metrics['total_return'] = (1 + returns_series).prod() - 1
            metrics['annual_return'] = (1 + returns_series.mean()) ** 252 - 1
            metrics['annual_volatility'] = returns_series.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_volatility']
            
            # Drawdown metrics
            cumulative = (1 + returns_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
            
            # Win rate
            metrics['win_rate'] = (returns_series > 0).mean()
            
            # VaR
            metrics['var_95'] = returns_series.quantile(0.05)
            
            return metrics
        
        metrics = calculate_all_metrics(returns)
        
        # Validations
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        
        # Sanity checks
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertLessEqual(metrics['win_rate'], 1)
        self.assertLessEqual(metrics['max_drawdown'], 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
```

## Continuous Integration Setup

### GitHub Actions Workflow
```yaml
# .github/workflows/trading-ci.yml
name: Trading System CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8
    
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install security tools
      run: |
        pip install bandit safety
    
    - name: Run security scan with bandit
      run: |
        bandit -r src/ -f json -o bandit-report.json
    
    - name: Check for known security vulnerabilities
      run: |
        safety check --json --output safety-report.json

  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest-benchmark
    
    - name: Run performance tests
      run: |
        pytest tests/test_performance.py -v --benchmark-only
```

Este workflow de desarrollo proporciona una base sólida para desarrollar, testear y mantener sistemas de trading cuantitativos de manera profesional y colaborativa.