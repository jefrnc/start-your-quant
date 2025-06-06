# Sistema de Tracking de Performance

## Dashboard de Performance en Tiempo Real

### Real-Time Metrics Engine
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import asyncio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

@dataclass
class PositionMetrics:
    """Métricas de una posición individual"""
    symbol: str
    entry_time: datetime
    entry_price: float
    current_price: float
    quantity: int
    side: str  # 'long' or 'short'
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    time_in_position: timedelta = field(default_factory=lambda: timedelta(0))
    strategy: str = ""
    risk_amount: float = 0.0

@dataclass
class AccountMetrics:
    """Métricas de cuenta agregadas"""
    timestamp: datetime
    total_equity: float
    cash: float
    buying_power: float
    total_unrealized_pnl: float
    total_realized_pnl_today: float
    day_start_equity: float
    positions_count: int
    margin_used: float
    daily_return_pct: float = 0.0
    
    def __post_init__(self):
        if self.day_start_equity > 0:
            self.daily_return_pct = (self.total_equity - self.day_start_equity) / self.day_start_equity

class RealTimePerformanceTracker:
    """Sistema de tracking de performance en tiempo real"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.positions: Dict[str, PositionMetrics] = {}
        self.account_history: List[AccountMetrics] = []
        self.trade_history: List[Dict] = []
        self.daily_metrics: Dict[str, float] = {}
        self.risk_limits = {
            'max_daily_loss_pct': 0.05,    # 5%
            'max_position_size_pct': 0.20,  # 20%
            'max_portfolio_risk_pct': 0.08, # 8%
            'max_positions': 5
        }
        
        # Métricas de tracking
        self.start_of_day_equity = initial_capital
        self.peak_equity_today = initial_capital
        self.current_drawdown = 0.0
        
    def update_position(self, symbol: str, current_price: float, 
                       market_data: Dict = None):
        """Actualizar métricas de posición"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.current_price = current_price
        position.time_in_position = datetime.now() - position.entry_time
        
        # Calcular P&L
        if position.side == 'long':
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        else:  # short
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
        
        position.unrealized_pnl_pct = position.unrealized_pnl / (position.entry_price * position.quantity)
        
        # Actualizar MFE/MAE
        if position.unrealized_pnl > position.max_favorable_excursion:
            position.max_favorable_excursion = position.unrealized_pnl
        
        if position.unrealized_pnl < position.max_adverse_excursion:
            position.max_adverse_excursion = position.unrealized_pnl
        
        return position
    
    def add_position(self, symbol: str, entry_price: float, quantity: int, 
                    side: str, strategy: str = "", risk_amount: float = 0.0):
        """Agregar nueva posición"""
        
        position = PositionMetrics(
            symbol=symbol,
            entry_time=datetime.now(),
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            side=side,
            strategy=strategy,
            risk_amount=risk_amount
        )
        
        self.positions[symbol] = position
        
        # Log trade entry
        self.trade_history.append({
            'timestamp': datetime.now(),
            'type': 'entry',
            'symbol': symbol,
            'price': entry_price,
            'quantity': quantity,
            'side': side,
            'strategy': strategy
        })
        
        return position
    
    def close_position(self, symbol: str, exit_price: float, 
                      exit_reason: str = ""):
        """Cerrar posición"""
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calcular P&L final
        if position.side == 'long':
            realized_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            realized_pnl = (position.entry_price - exit_price) * position.quantity
        
        # Log trade exit
        trade_record = {
            'timestamp': datetime.now(),
            'type': 'exit',
            'symbol': symbol,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'side': position.side,
            'strategy': position.strategy,
            'realized_pnl': realized_pnl,
            'hold_time': datetime.now() - position.entry_time,
            'mfe': position.max_favorable_excursion,
            'mae': position.max_adverse_excursion,
            'exit_reason': exit_reason
        }
        
        self.trade_history.append(trade_record)
        
        # Remover posición
        del self.positions[symbol]
        
        return trade_record
    
    def calculate_account_metrics(self, current_equity: float, 
                                 cash: float, buying_power: float) -> AccountMetrics:
        """Calcular métricas de cuenta"""
        
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Calcular realized P&L today
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = [t for t in self.trade_history 
                       if t['timestamp'] >= today_start and t['type'] == 'exit']
        total_realized_today = sum(t['realized_pnl'] for t in today_trades)
        
        # Margin used (simplified)
        margin_used = sum(pos.entry_price * pos.quantity * 0.5 
                         for pos in self.positions.values() if pos.side == 'short')
        
        metrics = AccountMetrics(
            timestamp=datetime.now(),
            total_equity=current_equity,
            cash=cash,
            buying_power=buying_power,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl_today=total_realized_today,
            day_start_equity=self.start_of_day_equity,
            positions_count=len(self.positions),
            margin_used=margin_used
        )
        
        self.account_history.append(metrics)
        
        # Update peak equity and drawdown
        if current_equity > self.peak_equity_today:
            self.peak_equity_today = current_equity
        
        self.current_drawdown = (self.peak_equity_today - current_equity) / self.peak_equity_today
        
        return metrics
    
    def check_risk_limits(self, current_equity: float) -> List[str]:
        """Verificar límites de riesgo"""
        
        violations = []
        
        # Daily loss limit
        daily_loss = (current_equity - self.start_of_day_equity) / self.start_of_day_equity
        if daily_loss < -self.risk_limits['max_daily_loss_pct']:
            violations.append(f"Daily loss limit exceeded: {daily_loss:.2%}")
        
        # Position size limits
        for symbol, position in self.positions.items():
            position_value = position.current_price * position.quantity
            position_pct = position_value / current_equity
            
            if position_pct > self.risk_limits['max_position_size_pct']:
                violations.append(f"Position size limit exceeded for {symbol}: {position_pct:.2%}")
        
        # Portfolio risk limit
        total_risk = sum(pos.risk_amount for pos in self.positions.values())
        risk_pct = total_risk / current_equity
        
        if risk_pct > self.risk_limits['max_portfolio_risk_pct']:
            violations.append(f"Portfolio risk limit exceeded: {risk_pct:.2%}")
        
        # Max positions
        if len(self.positions) > self.risk_limits['max_positions']:
            violations.append(f"Max positions exceeded: {len(self.positions)}")
        
        return violations
    
    def get_strategy_breakdown(self) -> Dict:
        """Obtener breakdown por estrategia"""
        
        strategy_metrics = {}
        
        # P&L por estrategia de posiciones abiertas
        for position in self.positions.values():
            strategy = position.strategy or 'Unknown'
            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    'open_positions': 0,
                    'unrealized_pnl': 0.0,
                    'total_risk': 0.0
                }
            
            strategy_metrics[strategy]['open_positions'] += 1
            strategy_metrics[strategy]['unrealized_pnl'] += position.unrealized_pnl
            strategy_metrics[strategy]['total_risk'] += position.risk_amount
        
        # P&L por estrategia de trades cerrados hoy
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_exits = [t for t in self.trade_history 
                      if t['timestamp'] >= today_start and t['type'] == 'exit']
        
        for trade in today_exits:
            strategy = trade['strategy'] or 'Unknown'
            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    'open_positions': 0,
                    'unrealized_pnl': 0.0,
                    'total_risk': 0.0,
                    'realized_pnl': 0.0,
                    'completed_trades': 0
                }
            
            if 'realized_pnl' not in strategy_metrics[strategy]:
                strategy_metrics[strategy]['realized_pnl'] = 0.0
                strategy_metrics[strategy]['completed_trades'] = 0
            
            strategy_metrics[strategy]['realized_pnl'] += trade['realized_pnl']
            strategy_metrics[strategy]['completed_trades'] += 1
        
        return strategy_metrics
    
    def calculate_performance_metrics(self, lookback_days: int = 30) -> Dict:
        """Calcular métricas de performance"""
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Filtrar trades del período
        period_trades = [t for t in self.trade_history 
                        if t['timestamp'] >= cutoff_date and t['type'] == 'exit']
        
        if not period_trades:
            return {'error': 'No trades in period'}
        
        # Calcular métricas básicas
        total_trades = len(period_trades)
        winning_trades = len([t for t in period_trades if t['realized_pnl'] > 0])
        
        total_pnl = sum(t['realized_pnl'] for t in period_trades)
        wins = [t['realized_pnl'] for t in period_trades if t['realized_pnl'] > 0]
        losses = [t['realized_pnl'] for t in period_trades if t['realized_pnl'] < 0]
        
        win_rate = winning_trades / total_trades
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calcular Sharpe ratio (aproximado)
        daily_returns = []
        if self.account_history:
            for i in range(1, len(self.account_history)):
                prev_equity = self.account_history[i-1].total_equity
                curr_equity = self.account_history[i].total_equity
                daily_return = (curr_equity - prev_equity) / prev_equity
                daily_returns.append(daily_return)
        
        sharpe_ratio = 0
        if daily_returns:
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.sqrt(252) * np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
        
        # Average hold time
        hold_times = [t['hold_time'].total_seconds() / 3600 for t in period_trades]  # hours
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        return {
            'period_days': lookback_days,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'avg_hold_time_hours': avg_hold_time,
            'largest_win': max(wins) if wins else 0,
            'largest_loss': min(losses) if losses else 0
        }

class PerformanceDashboard:
    """Dashboard interactivo de performance"""
    
    def __init__(self, tracker: RealTimePerformanceTracker):
        self.tracker = tracker
        
    def create_equity_curve_chart(self) -> go.Figure:
        """Crear gráfico de equity curve"""
        
        if not self.tracker.account_history:
            return go.Figure()
        
        timestamps = [h.timestamp for h in self.tracker.account_history]
        equity_values = [h.total_equity for h in self.tracker.account_history]
        
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=equity_values,
            mode='lines',
            name='Total Equity',
            line=dict(color='blue', width=2)
        ))
        
        # Start of day line
        fig.add_hline(
            y=self.tracker.start_of_day_equity,
            line_dash="dash",
            line_color="gray",
            annotation_text="Start of Day"
        )
        
        # Peak equity line
        fig.add_hline(
            y=self.tracker.peak_equity_today,
            line_dash="dash", 
            line_color="green",
            annotation_text="Peak Today"
        )
        
        fig.update_layout(
            title="Equity Curve - Today",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            hovermode='x unified'
        )
        
        return fig
    
    def create_positions_table(self) -> pd.DataFrame:
        """Crear tabla de posiciones"""
        
        if not self.tracker.positions:
            return pd.DataFrame()
        
        positions_data = []
        for symbol, pos in self.tracker.positions.items():
            positions_data.append({
                'Symbol': symbol,
                'Side': pos.side.upper(),
                'Quantity': pos.quantity,
                'Entry Price': f"${pos.entry_price:.2f}",
                'Current Price': f"${pos.current_price:.2f}",
                'Unrealized P&L': f"${pos.unrealized_pnl:.2f}",
                'Unrealized %': f"{pos.unrealized_pnl_pct:.2%}",
                'Time in Position': str(pos.time_in_position).split('.')[0],  # Remove microseconds
                'Strategy': pos.strategy,
                'MFE': f"${pos.max_favorable_excursion:.2f}",
                'MAE': f"${pos.max_adverse_excursion:.2f}"
            })
        
        return pd.DataFrame(positions_data)
    
    def create_pnl_distribution_chart(self) -> go.Figure:
        """Crear gráfico de distribución de P&L"""
        
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = [t for t in self.tracker.trade_history 
                       if t['timestamp'] >= today_start and t['type'] == 'exit']
        
        if not today_trades:
            return go.Figure()
        
        pnl_values = [t['realized_pnl'] for t in today_trades]
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=pnl_values,
            nbinsx=20,
            name="P&L Distribution",
            marker_color="lightblue",
            opacity=0.7
        ))
        
        # Add mean line
        mean_pnl = np.mean(pnl_values)
        fig.add_vline(
            x=mean_pnl,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mean_pnl:.2f}"
        )
        
        fig.update_layout(
            title="P&L Distribution - Today",
            xaxis_title="P&L ($)",
            yaxis_title="Frequency"
        )
        
        return fig
    
    def create_strategy_performance_chart(self) -> go.Figure:
        """Crear gráfico de performance por estrategia"""
        
        strategy_breakdown = self.tracker.get_strategy_breakdown()
        
        if not strategy_breakdown:
            return go.Figure()
        
        strategies = list(strategy_breakdown.keys())
        realized_pnl = [strategy_breakdown[s].get('realized_pnl', 0) for s in strategies]
        unrealized_pnl = [strategy_breakdown[s].get('unrealized_pnl', 0) for s in strategies]
        
        fig = go.Figure()
        
        # Realized P&L
        fig.add_trace(go.Bar(
            name='Realized P&L',
            x=strategies,
            y=realized_pnl,
            marker_color='green'
        ))
        
        # Unrealized P&L  
        fig.add_trace(go.Bar(
            name='Unrealized P&L',
            x=strategies,
            y=unrealized_pnl,
            marker_color='blue'
        ))
        
        fig.update_layout(
            title="P&L by Strategy - Today",
            xaxis_title="Strategy",
            yaxis_title="P&L ($)",
            barmode='group'
        )
        
        return fig
    
    def create_risk_metrics_display(self) -> Dict:
        """Crear display de métricas de riesgo"""
        
        current_equity = self.tracker.account_history[-1].total_equity if self.tracker.account_history else self.tracker.initial_capital
        
        # Current drawdown
        daily_return = (current_equity - self.tracker.start_of_day_equity) / self.tracker.start_of_day_equity
        
        # Portfolio heat (% at risk)
        total_risk = sum(pos.risk_amount for pos in self.tracker.positions.values())
        portfolio_heat = total_risk / current_equity if current_equity > 0 else 0
        
        # Largest position
        largest_position_pct = 0
        largest_position_symbol = ""
        
        for symbol, pos in self.tracker.positions.items():
            position_value = pos.current_price * pos.quantity
            position_pct = position_value / current_equity
            
            if position_pct > largest_position_pct:
                largest_position_pct = position_pct
                largest_position_symbol = symbol
        
        return {
            'daily_return': daily_return,
            'current_drawdown': self.tracker.current_drawdown,
            'portfolio_heat': portfolio_heat,
            'largest_position_pct': largest_position_pct,
            'largest_position_symbol': largest_position_symbol,
            'positions_count': len(self.tracker.positions),
            'risk_violations': self.tracker.check_risk_limits(current_equity)
        }

# Streamlit Dashboard Implementation
def create_streamlit_dashboard():
    """Crear dashboard con Streamlit"""
    
    st.set_page_config(
        page_title="Trading Performance Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Real-Time Trading Performance Dashboard")
    
    # Initialize tracker (in real app, this would be persistent)
    if 'tracker' not in st.session_state:
        st.session_state.tracker = RealTimePerformanceTracker(100000)
        
        # Add some sample data
        st.session_state.tracker.add_position("AAPL", 150.0, 100, "long", "Gap and Go", 1000)
        st.session_state.tracker.add_position("TSLA", 200.0, 50, "long", "VWAP Reclaim", 500)
        
        # Simulate some price movements
        st.session_state.tracker.update_position("AAPL", 152.5)
        st.session_state.tracker.update_position("TSLA", 198.0)
    
    tracker = st.session_state.tracker
    dashboard = PerformanceDashboard(tracker)
    
    # Calculate current metrics
    current_equity = 102000  # Simulated
    account_metrics = tracker.calculate_account_metrics(current_equity, 50000, 80000)
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Equity",
            f"${account_metrics.total_equity:,.2f}",
            f"{account_metrics.daily_return_pct:.2%}"
        )
    
    with col2:
        st.metric(
            "Unrealized P&L",
            f"${account_metrics.total_unrealized_pnl:,.2f}",
            f"{account_metrics.total_unrealized_pnl / tracker.start_of_day_equity:.2%}"
        )
    
    with col3:
        st.metric(
            "Positions",
            account_metrics.positions_count,
            f"Max: {tracker.risk_limits['max_positions']}"
        )
    
    with col4:
        st.metric(
            "Buying Power",
            f"${account_metrics.buying_power:,.2f}"
        )
    
    with col5:
        current_drawdown = tracker.current_drawdown
        st.metric(
            "Current Drawdown",
            f"{current_drawdown:.2%}",
            delta_color="inverse"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            dashboard.create_equity_curve_chart(),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            dashboard.create_strategy_performance_chart(),
            use_container_width=True
        )
    
    # Positions table
    st.subheader("📋 Current Positions")
    positions_df = dashboard.create_positions_table()
    if not positions_df.empty:
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No open positions")
    
    # Risk metrics
    st.subheader("⚠️ Risk Management")
    risk_metrics = dashboard.create_risk_metrics_display()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Portfolio Heat",
            f"{risk_metrics['portfolio_heat']:.2%}",
            f"Limit: {tracker.risk_limits['max_portfolio_risk_pct']:.1%}"
        )
    
    with col2:
        st.metric(
            "Largest Position",
            f"{risk_metrics['largest_position_pct']:.2%}",
            risk_metrics['largest_position_symbol']
        )
    
    with col3:
        st.metric(
            "Daily Return",
            f"{risk_metrics['daily_return']:.2%}",
            f"Limit: {-tracker.risk_limits['max_daily_loss_pct']:.1%}"
        )
    
    # Risk violations
    if risk_metrics['risk_violations']:
        st.error("🚨 Risk Limit Violations:")
        for violation in risk_metrics['risk_violations']:
            st.error(f"• {violation}")
    
    # Performance metrics
    st.subheader("📊 Performance Metrics (30 Days)")
    perf_metrics = tracker.calculate_performance_metrics(30)
    
    if 'error' not in perf_metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", perf_metrics['total_trades'])
        
        with col2:
            st.metric("Win Rate", f"{perf_metrics['win_rate']:.1%}")
        
        with col3:
            st.metric("Profit Factor", f"{perf_metrics['profit_factor']:.2f}")
        
        with col4:
            st.metric("Sharpe Ratio", f"{perf_metrics['sharpe_ratio']:.2f}")

# Demo function
def demo_performance_tracking():
    """Demo del sistema de tracking"""
    
    # Create tracker
    tracker = RealTimePerformanceTracker(100000)
    
    # Add positions
    tracker.add_position("AAPL", 150.0, 100, "long", "Gap and Go", 1000)
    tracker.add_position("TSLA", 200.0, 50, "long", "VWAP Reclaim", 500)
    
    # Update prices
    tracker.update_position("AAPL", 155.0)
    tracker.update_position("TSLA", 195.0)
    
    # Calculate metrics
    account_metrics = tracker.calculate_account_metrics(102500, 50000, 80000)
    
    print("📊 Performance Tracking Demo")
    print(f"Total Equity: ${account_metrics.total_equity:,.2f}")
    print(f"Daily Return: {account_metrics.daily_return_pct:.2%}")
    print(f"Unrealized P&L: ${account_metrics.total_unrealized_pnl:,.2f}")
    print(f"Positions: {account_metrics.positions_count}")
    
    # Check risk limits
    violations = tracker.check_risk_limits(account_metrics.total_equity)
    if violations:
        print("\n⚠️ Risk Violations:")
        for violation in violations:
            print(f"  {violation}")
    else:
        print("\n✅ All risk limits OK")
    
    # Strategy breakdown
    strategy_breakdown = tracker.get_strategy_breakdown()
    print(f"\n📈 Strategy Breakdown:")
    for strategy, metrics in strategy_breakdown.items():
        print(f"  {strategy}: ${metrics['unrealized_pnl']:.2f} unrealized, {metrics['open_positions']} positions")

if __name__ == "__main__":
    demo_performance_tracking()
```

## Alertas y Notificaciones

### Sistema de Alertas Multi-Canal
```python
import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from enum import Enum

class AlertSeverity(Enum):
    """Niveles de severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertChannel(Enum):
    """Canales de alerta disponibles"""
    EMAIL = "email"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SMS = "sms"
    DESKTOP = "desktop"

@dataclass
class Alert:
    """Estructura de alerta"""
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    data: Optional[Dict] = None
    channels: List[AlertChannel] = field(default_factory=list)

class AlertManager:
    """Gestor de alertas multi-canal"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history: List[Alert] = []
        self.channel_handlers = {
            AlertChannel.EMAIL: self._send_email,
            AlertChannel.TELEGRAM: self._send_telegram,
            AlertChannel.DISCORD: self._send_discord,
            AlertChannel.SMS: self._send_sms,
            AlertChannel.DESKTOP: self._send_desktop
        }
        
        # Rate limiting
        self.rate_limits = {
            AlertSeverity.INFO: timedelta(minutes=5),
            AlertSeverity.WARNING: timedelta(minutes=2),
            AlertSeverity.CRITICAL: timedelta(seconds=30),
            AlertSeverity.EMERGENCY: timedelta(seconds=0)
        }
        
        self.last_alert_time = {}
    
    def send_alert(self, alert: Alert):
        """Enviar alerta por canales especificados"""
        
        # Check rate limiting
        if not self._should_send_alert(alert):
            return False
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send to each channel
        success_count = 0
        for channel in alert.channels:
            try:
                handler = self.channel_handlers.get(channel)
                if handler:
                    success = handler(alert)
                    if success:
                        success_count += 1
            except Exception as e:
                print(f"Error sending alert via {channel}: {e}")
        
        # Update rate limiting
        alert_key = f"{alert.title}_{alert.severity.value}"
        self.last_alert_time[alert_key] = alert.timestamp
        
        return success_count > 0
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Verificar si debe enviar alerta (rate limiting)"""
        
        alert_key = f"{alert.title}_{alert.severity.value}"
        last_time = self.last_alert_time.get(alert_key)
        
        if not last_time:
            return True
        
        time_since_last = alert.timestamp - last_time
        rate_limit = self.rate_limits.get(alert.severity, timedelta(minutes=1))
        
        return time_since_last >= rate_limit
    
    def _send_email(self, alert: Alert) -> bool:
        """Enviar alerta por email"""
        
        email_config = self.config.get('email', {})
        if not email_config:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else 'orange' if alert.severity == AlertSeverity.WARNING else 'blue'};">
                    {alert.title}
                </h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong></p>
                <p>{alert.message}</p>
                
                {self._format_alert_data_html(alert.data) if alert.data else ''}
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def _send_telegram(self, alert: Alert) -> bool:
        """Enviar alerta por Telegram"""
        
        telegram_config = self.config.get('telegram', {})
        if not telegram_config:
            return False
        
        try:
            # Format message
            emoji = self._get_severity_emoji(alert.severity)
            message = f"{emoji} *{alert.title}*\n\n"
            message += f"*Severity:* {alert.severity.value.upper()}\n"
            message += f"*Time:* {alert.timestamp.strftime('%H:%M:%S')}\n\n"
            message += f"{alert.message}\n"
            
            if alert.data:
                message += "\n*Details:*\n"
                message += self._format_alert_data_text(alert.data)
            
            # Send via Telegram Bot API
            url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
            
            payload = {
                'chat_id': telegram_config['chat_id'],
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending Telegram: {e}")
            return False
    
    def _send_discord(self, alert: Alert) -> bool:
        """Enviar alerta por Discord webhook"""
        
        discord_config = self.config.get('discord', {})
        if not discord_config:
            return False
        
        try:
            # Format Discord embed
            color = self._get_severity_color(alert.severity)
            
            embed = {
                "title": alert.title,
                "description": alert.message,
                "color": color,
                "timestamp": alert.timestamp.isoformat(),
                "fields": [
                    {
                        "name": "Severity",
                        "value": alert.severity.value.upper(),
                        "inline": True
                    }
                ]
            }
            
            if alert.data:
                embed["fields"].extend(self._format_alert_data_discord(alert.data))
            
            payload = {"embeds": [embed]}
            
            response = requests.post(
                discord_config['webhook_url'],
                json=payload,
                timeout=10
            )
            
            return response.status_code == 204
            
        except Exception as e:
            print(f"Error sending Discord: {e}")
            return False
    
    def _send_sms(self, alert: Alert) -> bool:
        """Enviar alerta por SMS (usando Twilio)"""
        
        sms_config = self.config.get('sms', {})
        if not sms_config:
            return False
        
        try:
            from twilio.rest import Client
            
            client = Client(sms_config['account_sid'], sms_config['auth_token'])
            
            # Format SMS message (keep it short)
            message = f"{alert.severity.value.upper()}: {alert.title}\n{alert.message[:100]}..."
            
            client.messages.create(
                body=message,
                from_=sms_config['from_number'],
                to=sms_config['to_number']
            )
            
            return True
            
        except Exception as e:
            print(f"Error sending SMS: {e}")
            return False
    
    def _send_desktop(self, alert: Alert) -> bool:
        """Enviar notificación de escritorio"""
        
        try:
            import plyer
            
            title = f"[{alert.severity.value.upper()}] {alert.title}"
            message = alert.message[:200] + "..." if len(alert.message) > 200 else alert.message
            
            plyer.notification.notify(
                title=title,
                message=message,
                app_name="Trading System",
                timeout=10
            )
            
            return True
            
        except Exception as e:
            print(f"Error sending desktop notification: {e}")
            return False
    
    def _get_severity_emoji(self, severity: AlertSeverity) -> str:
        """Obtener emoji por severidad"""
        emoji_map = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }
        return emoji_map.get(severity, "📢")
    
    def _get_severity_color(self, severity: AlertSeverity) -> int:
        """Obtener color por severidad (Discord)"""
        color_map = {
            AlertSeverity.INFO: 0x3498db,      # Blue
            AlertSeverity.WARNING: 0xf39c12,   # Orange
            AlertSeverity.CRITICAL: 0xe74c3c,  # Red
            AlertSeverity.EMERGENCY: 0x8e44ad  # Purple
        }
        return color_map.get(severity, 0x95a5a6)  # Gray default
    
    def _format_alert_data_html(self, data: Dict) -> str:
        """Formatear data para HTML"""
        html = "<table border='1' style='border-collapse: collapse;'>"
        for key, value in data.items():
            html += f"<tr><td><b>{key}:</b></td><td>{value}</td></tr>"
        html += "</table>"
        return html
    
    def _format_alert_data_text(self, data: Dict) -> str:
        """Formatear data para texto"""
        lines = []
        for key, value in data.items():
            lines.append(f"• *{key}:* {value}")
        return "\n".join(lines)
    
    def _format_alert_data_discord(self, data: Dict) -> List[Dict]:
        """Formatear data para Discord fields"""
        fields = []
        for key, value in data.items():
            fields.append({
                "name": key,
                "value": str(value),
                "inline": True
            })
        return fields

class PerformanceAlertSystem:
    """Sistema de alertas específico para performance"""
    
    def __init__(self, alert_manager: AlertManager, tracker: RealTimePerformanceTracker):
        self.alert_manager = alert_manager
        self.tracker = tracker
        
        # Thresholds
        self.thresholds = {
            'daily_loss_warning': 0.03,    # 3%
            'daily_loss_critical': 0.05,   # 5%
            'position_size_warning': 0.15,  # 15%
            'position_size_critical': 0.20, # 20%
            'drawdown_warning': 0.05,       # 5%
            'drawdown_critical': 0.08,      # 8%
            'large_profit_notification': 1000,  # $1000
            'large_loss_notification': 500       # $500
        }
    
    def check_alerts(self, current_equity: float):
        """Verificar y enviar alertas necesarias"""
        
        alerts_to_send = []
        
        # Daily P&L alerts
        daily_return = (current_equity - self.tracker.start_of_day_equity) / self.tracker.start_of_day_equity
        
        if daily_return <= -self.thresholds['daily_loss_critical']:
            alerts_to_send.append(Alert(
                title="Critical Daily Loss",
                message=f"Daily loss has reached {daily_return:.2%}, exceeding critical threshold",
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(),
                data={
                    'Daily Return': f"{daily_return:.2%}",
                    'Current Equity': f"${current_equity:,.2f}",
                    'Loss Amount': f"${current_equity - self.tracker.start_of_day_equity:,.2f}"
                },
                channels=[AlertChannel.EMAIL, AlertChannel.TELEGRAM, AlertChannel.SMS]
            ))
        elif daily_return <= -self.thresholds['daily_loss_warning']:
            alerts_to_send.append(Alert(
                title="Daily Loss Warning",
                message=f"Daily loss has reached {daily_return:.2%}",
                severity=AlertSeverity.WARNING,
                timestamp=datetime.now(),
                data={
                    'Daily Return': f"{daily_return:.2%}",
                    'Current Equity': f"${current_equity:,.2f}"
                },
                channels=[AlertChannel.TELEGRAM, AlertChannel.DESKTOP]
            ))
        
        # Position size alerts
        for symbol, position in self.tracker.positions.items():
            position_value = position.current_price * position.quantity
            position_pct = position_value / current_equity
            
            if position_pct >= self.thresholds['position_size_critical']:
                alerts_to_send.append(Alert(
                    title=f"Critical Position Size - {symbol}",
                    message=f"Position in {symbol} is {position_pct:.2%} of portfolio",
                    severity=AlertSeverity.CRITICAL,
                    timestamp=datetime.now(),
                    data={
                        'Symbol': symbol,
                        'Position Size': f"{position_pct:.2%}",
                        'Position Value': f"${position_value:,.2f}",
                        'Current P&L': f"${position.unrealized_pnl:,.2f}"
                    },
                    channels=[AlertChannel.EMAIL, AlertChannel.TELEGRAM]
                ))
        
        # Drawdown alerts
        if self.tracker.current_drawdown >= self.thresholds['drawdown_critical']:
            alerts_to_send.append(Alert(
                title="Critical Drawdown",
                message=f"Current drawdown is {self.tracker.current_drawdown:.2%}",
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(),
                data={
                    'Current Drawdown': f"{self.tracker.current_drawdown:.2%}",
                    'Peak Equity': f"${self.tracker.peak_equity_today:,.2f}",
                    'Current Equity': f"${current_equity:,.2f}"
                },
                channels=[AlertChannel.EMAIL, AlertChannel.TELEGRAM, AlertChannel.SMS]
            ))
        
        # Large P&L notifications
        for symbol, position in self.tracker.positions.items():
            if position.unrealized_pnl >= self.thresholds['large_profit_notification']:
                alerts_to_send.append(Alert(
                    title=f"Large Profit - {symbol}",
                    message=f"Unrealized profit of ${position.unrealized_pnl:,.2f} in {symbol}",
                    severity=AlertSeverity.INFO,
                    timestamp=datetime.now(),
                    data={
                        'Symbol': symbol,
                        'Unrealized P&L': f"${position.unrealized_pnl:,.2f}",
                        'P&L %': f"{position.unrealized_pnl_pct:.2%}",
                        'Time in Position': str(position.time_in_position).split('.')[0]
                    },
                    channels=[AlertChannel.TELEGRAM, AlertChannel.DESKTOP]
                ))
            elif position.unrealized_pnl <= -self.thresholds['large_loss_notification']:
                alerts_to_send.append(Alert(
                    title=f"Large Loss - {symbol}",
                    message=f"Unrealized loss of ${position.unrealized_pnl:,.2f} in {symbol}",
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.now(),
                    data={
                        'Symbol': symbol,
                        'Unrealized P&L': f"${position.unrealized_pnl:,.2f}",
                        'P&L %': f"{position.unrealized_pnl_pct:.2%}",
                        'Time in Position': str(position.time_in_position).split('.')[0]
                    },
                    channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL]
                ))
        
        # Send all alerts
        for alert in alerts_to_send:
            self.alert_manager.send_alert(alert)
        
        return len(alerts_to_send)

# Demo del sistema de alertas
def demo_alert_system():
    """Demo del sistema de alertas"""
    
    # Configuración de alertas
    alert_config = {
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your_email@gmail.com',
            'password': 'your_app_password',
            'from_email': 'your_email@gmail.com',
            'to_email': 'trader@example.com'
        },
        'telegram': {
            'bot_token': 'your_bot_token',
            'chat_id': 'your_chat_id'
        },
        'discord': {
            'webhook_url': 'your_webhook_url'
        }
    }
    
    # Crear alert manager
    alert_manager = AlertManager(alert_config)
    
    # Crear tracker y alert system
    tracker = RealTimePerformanceTracker(100000)
    alert_system = PerformanceAlertSystem(alert_manager, tracker)
    
    # Simular pérdida grande
    tracker.start_of_day_equity = 100000
    current_equity = 95000  # 5% loss
    
    # Check alerts
    alerts_sent = alert_system.check_alerts(current_equity)
    print(f"📨 Sent {alerts_sent} alerts")
    
    # Ejemplo de alerta manual
    manual_alert = Alert(
        title="Trade Execution",
        message="Successfully entered AAPL long position at $150.00",
        severity=AlertSeverity.INFO,
        timestamp=datetime.now(),
        data={
            'Symbol': 'AAPL',
            'Side': 'LONG',
            'Quantity': '100',
            'Entry Price': '$150.00',
            'Strategy': 'Gap and Go'
        },
        channels=[AlertChannel.TELEGRAM, AlertChannel.DESKTOP]
    )
    
    alert_manager.send_alert(manual_alert)
    print("📱 Manual alert sent")

if __name__ == "__main__":
    demo_alert_system()
```

Este sistema de tracking y alertas proporciona monitoreo completo en tiempo real del performance de trading, con alertas inteligentes para ayudar a mantener disciplina y gestionar riesgo efectivamente.