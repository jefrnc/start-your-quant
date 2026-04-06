> 🇪🇸 [Leer en Español](README.es.md) | 🇺🇸 **English**

# 🎓 Start Your Quant - Your First Steps into Quantitative Trading

**From zero to professional quant trader with practical, progressive modules.**

Learn quantitative trading in a structured way, at your own pace, with real examples and hands-on exercises. From basic concepts to institutional strategies.

> 🌟 **Completely free and open source** - Available as [GitHub Pages](https://jefrnc.github.io/start-your-quant)

---

## ⚠️ Disclaimer

This repository reflects my personal learning journey in quantitative trading. The ideas and concepts come from **my experience studying courses, attending seminars and talks, reading books, and practicing in the markets**. For several academic and technical topics, I used AI tools (Perplexity, Claude, ChatGPT) to help me develop and articulate my thoughts more completely — but the core ideas, structure, and trading perspective are my own.

It's a starting point, not an absolute truth.

**A note about language:** My native language is Spanish, so you may find occasional grammatical errors or awkward phrasing in the English content. If you spot anything that could be improved — whether it's a translation issue, a technical correction, or an expansion on any topic — **please don't hesitate to send a pull request**. Question the content — the idea is for this to be a useful foundation that grows with community contributions.

If this repo helped you and you think your comment, observation, or contribution could help me or others, **send it without hesitation**. I'd be more than grateful.

---

## 🚀 Where Do I Start?

| Your Level | Start Here | Time |
|----------|--------------|--------|
| **Complete beginner** | [🎯 What Is Being a Quant?](learning-path/fundamentos/f1-que-es-ser-quant/) | 2-4 months |
| **I know some Python** | [🐍 Basic Python Trading](learning-path/fundamentos/f2-python-trading-basico/) | 1-3 months |
| **I already trade manually** | [📊 Technical Indicators](learning-path/fundamentos/f3-indicadores-tecnicos/) | 1-2 months |
| **I want my first strategy** | [🤖 First Strategy](learning-path/fundamentos/f4-primera-estrategia/) | Immediate |
| **Advanced developer** | [🏗️ Infrastructure](infrastructure/) | Immediate |

**Not sure where to start?** → [📖 Getting Started Guide](GETTING-STARTED.md) | [📚 Learning Path](learning-path/)

---

## 📚 Table of Contents

### Core Concepts

- [Introduction](core-concepts/Introduction.md) - What is quantitative trading and why it works
- [Risk Management](core-concepts/Risk-Management.md) - Systematic risk control frameworks
- [Performance Metrics](core-concepts/Performance-Metrics.md) - Key metrics for strategy evaluation
- [Financial Instruments](core-concepts/Financial-Instruments.md) - Types of instruments and their characteristics
- [History of Algo Trading](core-concepts/History-of-Algo-Trading.md) - Evolution of algorithmic trading
- [Cognitive Biases in Algo Trading](core-concepts/Cognitive-Biases-Algo-Trading.md) - How biases affect your decisions
- [Compound Growth and Risk](core-concepts/Compound-Growth-and-Risk.md) - The mathematics of sustainable growth
- [Expected Growth Metrics](core-concepts/Expected-Growth-Metrics-Hierarchy.md) - Hierarchy of performance metrics
- [Trading Plan Framework](core-concepts/Trading-Plan-Framework.md) - How to structure your trading plan
- [Trading Systems Anatomy](core-concepts/Trading-Systems-Anatomy.md) - Components of a trading system

### Technical Practices

- [Strategy Development](technical-practices/Strategy-Development.md) - Systematic approach to strategy creation
- [Backtesting - Three Levels](technical-practices/Backtesting-Three-Levels.md) - Backtesting methodologies by complexity level
- [Common Backtesting Errors](technical-practices/Backtesting-Common-Errors.md) - Common pitfalls and how to avoid them
- [Data Quality and Adjustments](technical-practices/Data-Quality-Adjustments.md) - Cleaning and preparing financial data
- [Filters and Asset Selection](technical-practices/Filters-Asset-Selection-Examples.md) - Practical filtering examples
- [KISS Design Principles](technical-practices/KISS-Design-Principles.md) - Keeping strategies simple and effective
- [Scientific Method in Trading](technical-practices/Scientific-Method-System-Development.md) - Applying the scientific method to system development
- [Entry and Exit Structure](technical-practices/System-Structure-Entries-Exits.md) - Designing entry and exit rules

### Advanced Topics

- [Alternative Data](advanced-topics/Alternative-Data.md) - Non-traditional data sources
- [Portfolio Optimization](advanced-topics/Portfolio-Optimization.md) - Applying modern portfolio theory
- [Dynamic Position Sizing](advanced-topics/Dynamic-Position-Sizing.md) - Dynamic adjustment of position size
- [Execution Algorithms](advanced-topics/Execution-Algorithms.md) - Order execution optimization
- [Regime Detection](advanced-topics/Regime-Detection.md) - Identifying market regime changes

### Detailed Documentation (docs/)

| Category | Content |
|-----------|-----------|
| [Fundamentals](docs/fundamentals/) | What is quant, types of strategies, discretionary vs quantitative |
| [Setup](docs/setup/) | Broker configuration, data providers, getting started |
| [Strategies](docs/strategies/) | Gap & Go, VWAP Reclaim, Low Float Runners, First Green/Red Day, Short Selling |
| [Indicators](docs/indicators/) | Moving Averages, VWAP, Bollinger Bands, Parabolic SAR, Gap/Float, Volume |
| [Backtesting](docs/backtesting/) | Simple engine, metrics, overfitting, walk-forward analysis |
| [Risk](docs/risk/) | Position sizing, stops, portfolio risk, asymmetric risk |
| [Data](docs/data/) | Data sources, cleaning, real-time data, backtesting datasets |
| [Analysis](docs/analysis/) | ML, sentiment, microstructure, fundamental analysis, transformers |
| [Automation](docs/automation/) | System architecture, robo-advisors, automation strategies |
| [Tools](docs/tools/) | Essential libraries, advanced infrastructure, production deployment |
| [Compliance](docs/compliance/) | Regulatory frameworks, ethical AI in trading |
| [Validation](docs/validation/) | Strategy testing, model evaluation, institutional considerations |
| [Quick Reference](docs/QUICK_REFERENCE.md) | Key concepts cheat sheet |

### Templates and Scripts

- [Strategy Templates](templates/strategies/) - Ready-to-use frameworks (momentum, mean-reversion)
- [Metrics Calculators](scripts/strategy-metrics/) - Sharpe ratio, max drawdown, profit factor

### Real-World Examples

- [IBKR Premarket Trader](examples/ibkr-premarket-trader/) - Gap trading system for small caps

### Infrastructure

- [Trading Stack](infrastructure/) - Docker, Kubernetes, monitoring, data pipelines

### Reference Source Code (src/)

```
src/
├── indicators/       # MovingAverages (SMA/EMA/WMA), VWAP with bands
├── strategies/       # Gap and Go strategy
├── backtesting/      # Simple engine + CSV export (TraderVue/TradesViz)
├── risk/             # Position sizing (fixed, Kelly, ATR, risk parity)
├── data/             # Data interfaces (yfinance, Alpha Vantage)
└── examples/         # Complete integrated example
```

## 🛠️ Quick Start for Developers

```bash
# Clone the repository
git clone https://github.com/jefrnc/start-your-quant.git
cd start-your-quant

# Install minimum dependencies for the src/ examples
pip install pandas numpy matplotlib requests

# Run the integrated example
cd src/examples && python complete_strategy_example.py

# Metrics calculators
python scripts/strategy-metrics/sharpe-calculator/calculate_sharpe.py
python scripts/strategy-metrics/max-drawdown/calculate_drawdown.py
python scripts/strategy-metrics/profit-factor/calculate_profit_factor.py
```

### Local Site Preview (Jekyll)

```bash
bundle install
bundle exec jekyll serve
# Site at http://localhost:4000/start-your-quant/
```

## 🛠️ Tech Stack

- **Python** as the main language
- **Pandas/NumPy** for data manipulation
- **Plotly/Matplotlib** for visualization
- **yfinance/Polygon** for market data
- **IBKR TWS** for execution
- **Jekyll** for the GitHub Pages site
- **Docker/Kubernetes** for infrastructure

## 🤝 Contributing

Any improvement, correction, or observation is greatly appreciated:

- 🐛 **Report bugs** in implementations or documentation
- 💡 **Suggest improvements** to strategies or methodologies
- 📚 **Propose new content** based on your experience
- 🔧 **Optimize existing code** or add new features

**How to contribute?** Open an issue or send a pull request directly.

## 📚 Recommended Additional Resources

### Courses and Educational Material

- **[Quantitative Trading in Python](https://github.com/AxelMunguiaQuintero/Trading-Cuantitativo-en-Python)** - Complete course with 15+ practical modules (broker integration, applied ML, sentiment analysis). Material that inspired several implementations in this repository.

### Tools and Platforms

- **[QuantConnect](https://www.quantconnect.com/)** - Cloud backtesting with institutional data
- **[Backtrader](https://github.com/mementum/backtrader)** - Python backtesting framework
- **[TradingView](https://www.tradingview.com/)** - Technical analysis and alerts

### Essential Books

- **"Quantitative Trading"** - Ernest Chan
- **"Algorithmic Trading"** - Ernie Chan
- **"A Man for All Markets"** - Edward Thorp

### Communities

- **[QuantStart](https://www.quantstart.com/)** - Technical articles on quantitative trading
- **[r/algotrading](https://reddit.com/r/algotrading)** - Active community of algorithmic traders

### APIs and Data Providers

- **[Alpha Vantage](https://www.alphavantage.co/)** - Free API
- **[Polygon.io](https://polygon.io/)** - Real-time and historical data

---

> **⚠️ Risk Disclaimer:** This content is purely educational. Trading involves risk of loss. Always trade with proper risk management.
