> 🇪🇸 [Leer en Español](Introduction.es.md) | 🇺🇸 **English**

# Introduction to Quantitative Trading

## What is Quantitative Trading?

Quantitative trading is an investment methodology that uses mathematical models, statistical analysis, and algorithms to identify trading opportunities and execute trades systematically.

### Key Characteristics

- **Data-Driven**: All decisions are backed by statistical evidence
- **Systematic**: Reproducible and automatable processes
- **Objective**: Eliminates emotions and cognitive biases
- **Scalable**: Can handle multiple instruments simultaneously

## Why Does It Work?

### 1. Elimination of Emotional Biases
Humans are susceptible to:
- FOMO (Fear of Missing Out)
- Revenge trading
- Confirmation bias
- Overconfidence

### 2. Superior Information Processing
Algorithms can:
- Analyze thousands of data points simultaneously
- Detect complex patterns
- React in milliseconds
- Operate 24/7 without fatigue

### 3. Systematic Risk Management
- Mathematical position sizing
- Automatic stop losses
- Algorithmic diversification
- Dynamic exposure limits

## Our Focus: Small Caps & Premarket

### Why Small Caps?

**Advantages**:
- **Higher volatility** = greater profit opportunities
- **Less institutional coverage** = market inefficiencies
- **More pronounced movements** in gaps and breakouts
- **Less algorithmic arbitrage** from HFT

**Challenges**:
- **Higher risk** of substantial losses
- **Lower liquidity** = greater slippage
- **Manipulation** and pump & dump schemes
- **Frequent halts and suspensions**

### Premarket Window (5:30 AM - 8:00 AM ET)

**Why it's effective**:
- **Low liquidity** amplifies movements
- **Reaction to overnight news**
- **Significant gaps** vs previous close
- **Less algorithmic competition**

**Key factors**:
- Premarket volume vs average
- Gap magnitude (3-25% ideal)
- Available float for trading
- Fundamental catalyst (news, earnings)

## Playbook Methodology

### 1. Research-Driven Development
```
Hypothesis -> Backtesting -> Validation -> Paper Trading -> Live Implementation
```

### 2. Position Recycling Strategy
Our unique approach where:
- **Initial entry** with optimal size
- **Partial profit taking** on strength
- **Re-entry on pullbacks** to improve average
- **Multiple trades = ONE trading campaign**

### 3. Risk-First Design
- **Maximum risk per trade**: $10
- **Maximum position**: $70
- **Systematic stop loss**: 5-8%
- **Maximum hold time**: 60 minutes

## Tools and Technology Stack

### Core Technologies
- **Python 3.13**: Primary language
- **Pandas/NumPy**: Data manipulation
- **Polygon.io**: Market data
- **PostgreSQL**: Trade storage
- **IBKR TWS**: Trade execution

### Analysis and Backtesting
- **Jupyter Notebooks**: Interactive research
- **Backtrader**: Backtesting engine
- **Plotly**: Visualizations
- **Optuna**: Bayesian optimization

### Monitoring and Alerts
- **Grafana**: Real-time dashboards
- **Prometheus**: System metrics
- **Telegram**: Trading alerts
- **Discord**: Community notifications

## Main Strategies

### 1. Gap & Go (Implemented)
- **Setup**: Gap > 3% + elevated premarket volume
- **Entry**: Breakout above resistance with confirmation
- **Exit**: Trailing stop or profit target

### 2. Early Runner Detection (In Development)
- **ML Model**: Detects penny stocks with runner potential
- **Signals**: Dark pool activity, float rotation, technical setup
- **Score**: 0-100 with HOT/WARM/COLD classification

### 3. VWAP Reclaim (Planned)
- **Setup**: VWAP rejection followed by reclaim
- **Confirmation**: Increasing volume + momentum
- **Risk**: Stop below prior VWAP

## Success Metrics

### Performance Targets
- **Sharpe Ratio**: > 1.5 (target: 2.0+)
- **Max Drawdown**: < 10% (target: < 5%)
- **Win Rate**: > 60% (target: 70%+)
- **Profit Factor**: > 1.5 (target: 2.0+)

### Operational Metrics
- **Trades per day**: 3-8 (quality > quantity)
- **Average hold time**: < 45 minutes
- **Average slippage**: < 0.5%
- **Fill rate**: > 95%

## Trading Philosophy

### Fundamental Principles

1. **The market is probabilistic, not deterministic**
   - We seek statistical edges
   - We accept that not all trades will be winners
   - Focus on positive mathematical expectancy

2. **Consistency beats home runs**
   - We prefer many small gains
   - We avoid large losses at all costs
   - Base hits > grand slams

3. **Constant adaptability**
   - Markets evolve
   - Our models must evolve
   - Continuous learning and improvement

### Psychological Management

**For algorithmic traders**:
- **Trust the process**: Drawdowns are normal
- **Don't intervene manually**: Except in emergencies
- **Analyze post-mortem**: Every trade is a lesson
- **Keep perspective**: Focus on long-term metrics

## Next Steps

### For Beginners
1. Read [Risk Management](./Risk-Management.md)
2. Study [Performance Metrics](./Performance-Metrics.md)
3. Practice with [Strategy Development](../technical-practices/Strategy-Development.md)

### For Experienced Traders
1. Implement [Early Runner Detection](../advanced-topics/Machine-Learning-Trading.md)
2. Optimize [Parameter Optimization](../technical-practices/Parameter-Optimization.md)
3. Scale with [Multi-Broker Integration](../architecture-patterns/Multi-Broker-Integration.md)

---

**Remember**: Quantitative trading is not a magic formula for easy money. It's a discipline that requires:
- Solid technical knowledge
- Emotional discipline
- Continuous improvement
- Strict risk management

But when done correctly, it can provide a systematic and sustainable edge in the markets!
