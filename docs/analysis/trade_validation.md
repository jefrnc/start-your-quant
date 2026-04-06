> 🇪🇸 [Leer en Español](trade_validation.es.md) | 🇺🇸 **English**

# Trade Validation and Analysis

## Pre-Trade Validation Framework

### Validation Checklist
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

class ValidationLevel(Enum):
    """Validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"

class ValidationResult(Enum):
    """Validation results"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"

@dataclass
class TradeProposal:
    """Trade proposal for validation"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    strategy_name: str
    setup_score: float
    market_data: Dict
    timestamp: datetime

@dataclass
class ValidationCheck:
    """Result of a specific validation"""
    check_name: str
    result: ValidationResult
    message: str
    score: float  # 0-100
    critical: bool = False

class PreTradeValidator:
    """Pre-trade validation system"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.account_info = {}
        self.current_positions = {}
        self.daily_stats = {}
        
    def set_account_info(self, account_info: Dict):
        """Configure account information"""
        self.account_info = account_info
        
    def set_current_positions(self, positions: Dict):
        """Configure current positions"""
        self.current_positions = positions
        
    def validate_trade(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Validate complete trade proposal"""
        
        validations = []
        
        # Basic validations
        validations.extend(self._validate_basic_checks(proposal))
        
        # Risk management validations
        validations.extend(self._validate_risk_management(proposal))
        
        # Setup quality validations
        validations.extend(self._validate_setup_quality(proposal))
        
        # Market conditions validations
        validations.extend(self._validate_market_conditions(proposal))
        
        # Timing validations
        validations.extend(self._validate_timing(proposal))
        
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            # Additional validations
            validations.extend(self._validate_correlation_risk(proposal))
            validations.extend(self._validate_liquidity(proposal))
        
        if self.validation_level == ValidationLevel.STRICT:
            # Strict validations
            validations.extend(self._validate_news_sentiment(proposal))
            validations.extend(self._validate_technical_confluence(proposal))
        
        return validations
    
    def _validate_basic_checks(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Basic validations obligatorias"""
        
        checks = []
        
        # 1. Symbol validation
        if not proposal.symbol or len(proposal.symbol) < 1:
            checks.append(ValidationCheck(
                "symbol_validation",
                ValidationResult.FAIL,
                "Invalid symbol",
                0,
                critical=True
            ))
        else:
            checks.append(ValidationCheck(
                "symbol_validation",
                ValidationResult.PASS,
                f"Symbol {proposal.symbol} valid",
                100
            ))
        
        # 2. Quantity validation
        if proposal.quantity <= 0:
            checks.append(ValidationCheck(
                "quantity_validation",
                ValidationResult.FAIL,
                "Quantity must be positive",
                0,
                critical=True
            ))
        else:
            checks.append(ValidationCheck(
                "quantity_validation",
                ValidationResult.PASS,
                f"Quantity {proposal.quantity} válida",
                100
            ))
        
        # 3. Price validation
        if proposal.entry_price <= 0:
            checks.append(ValidationCheck(
                "price_validation",
                ValidationResult.FAIL,
                "Entry price must be positive",
                0,
                critical=True
            ))
        else:
            checks.append(ValidationCheck(
                "price_validation",
                ValidationResult.PASS,
                f"Price ${proposal.entry_price:.2f} valid",
                100
            ))
        
        # 4. Stop loss validation
        if proposal.side == 'buy' and proposal.stop_loss >= proposal.entry_price:
            checks.append(ValidationCheck(
                "stop_loss_validation",
                ValidationResult.FAIL,
                "Stop loss must be less than entry price for longs",
                0,
                critical=True
            ))
        elif proposal.side == 'sell' and proposal.stop_loss <= proposal.entry_price:
            checks.append(ValidationCheck(
                "stop_loss_validation",
                ValidationResult.FAIL,
                "Stop loss must be greater than entry price for shorts",
                0,
                critical=True
            ))
        else:
            checks.append(ValidationCheck(
                "stop_loss_validation",
                ValidationResult.PASS,
                f"Stop loss ${proposal.stop_loss:.2f} valid",
                100
            ))
        
        return checks
    
    def _validate_risk_management(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Risk management validations"""
        
        checks = []
        
        # 1. Position size vs account
        account_value = self.account_info.get('total_value', 100000)
        position_value = proposal.quantity * proposal.entry_price
        position_pct = position_value / account_value
        
        max_position_pct = 0.20  # 20% máximo
        
        if position_pct > max_position_pct:
            checks.append(ValidationCheck(
                "position_size_risk",
                ValidationResult.FAIL,
                f"Position {position_pct:.1%} exceeds maximum {max_position_pct:.1%}",
                0,
                critical=True
            ))
        elif position_pct > max_position_pct * 0.8:
            checks.append(ValidationCheck(
                "position_size_risk",
                ValidationResult.WARNING,
                f"Position {position_pct:.1%} is large (max {max_position_pct:.1%})",
                60
            ))
        else:
            score = 100 - (position_pct / max_position_pct * 30)
            checks.append(ValidationCheck(
                "position_size_risk",
                ValidationResult.PASS,
                f"Position {position_pct:.1%} within limits",
                score
            ))
        
        # 2. Risk per trade
        risk_per_share = abs(proposal.entry_price - proposal.stop_loss)
        total_risk = risk_per_share * proposal.quantity
        risk_pct = total_risk / account_value
        
        max_risk_pct = 0.02  # 2% máximo
        
        if risk_pct > max_risk_pct:
            checks.append(ValidationCheck(
                "risk_per_trade",
                ValidationResult.FAIL,
                f"Risk {risk_pct:.2%} exceeds maximum {max_risk_pct:.2%}",
                0,
                critical=True
            ))
        elif risk_pct > max_risk_pct * 0.8:
            checks.append(ValidationCheck(
                "risk_per_trade",
                ValidationResult.WARNING,
                f"Risk {risk_pct:.2%} is high",
                70
            ))
        else:
            score = 100 - (risk_pct / max_risk_pct * 20)
            checks.append(ValidationCheck(
                "risk_per_trade",
                ValidationResult.PASS,
                f"Risk {risk_pct:.2%} appropriate",
                score
            ))
        
        # 3. Daily loss limit
        current_daily_pnl = self.daily_stats.get('unrealized_pnl', 0)
        max_daily_loss = account_value * 0.05  # 5% máximo daily
        
        potential_loss = current_daily_pnl - total_risk
        
        if potential_loss < -max_daily_loss:
            checks.append(ValidationCheck(
                "daily_loss_limit",
                ValidationResult.FAIL,
                f"Potential daily loss exceeds limit",
                0,
                critical=True
            ))
        else:
            checks.append(ValidationCheck(
                "daily_loss_limit",
                ValidationResult.PASS,
                "Within daily loss limit",
                100
            ))
        
        # 4. Maximum number of positions
        current_positions_count = len(self.current_positions)
        max_positions = 5
        
        if current_positions_count >= max_positions:
            checks.append(ValidationCheck(
                "max_positions",
                ValidationResult.FAIL,
                f"You already have {current_positions_count} positions (max {max_positions})",
                0,
                critical=True
            ))
        elif current_positions_count >= max_positions * 0.8:
            checks.append(ValidationCheck(
                "max_positions",
                ValidationResult.WARNING,
                f"Tienes {current_positions_count} positions, near limit",
                70
            ))
        else:
            checks.append(ValidationCheck(
                "max_positions",
                ValidationResult.PASS,
                f"Current positions: {current_positions_count}/{max_positions}",
                100
            ))
        
        return checks
    
    def _validate_setup_quality(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Validate setup quality"""
        
        checks = []
        
        # 1. Setup score validation
        min_setup_score = 60
        
        if proposal.setup_score < min_setup_score:
            checks.append(ValidationCheck(
                "setup_quality",
                ValidationResult.FAIL,
                f"Setup score {proposal.setup_score} below minimum {min_setup_score}",
                proposal.setup_score
            ))
        elif proposal.setup_score < min_setup_score + 20:
            checks.append(ValidationCheck(
                "setup_quality",
                ValidationResult.WARNING,
                f"Setup score {proposal.setup_score} is marginal",
                proposal.setup_score
            ))
        else:
            checks.append(ValidationCheck(
                "setup_quality",
                ValidationResult.PASS,
                f"Setup score {proposal.setup_score} is good",
                proposal.setup_score
            ))
        
        # 2. Risk/Reward ratio
        if proposal.take_profit:
            risk = abs(proposal.entry_price - proposal.stop_loss)
            reward = abs(proposal.take_profit - proposal.entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            min_rr_ratio = 1.5
            
            if rr_ratio < min_rr_ratio:
                checks.append(ValidationCheck(
                    "risk_reward_ratio",
                    ValidationResult.WARNING,
                    f"R/R ratio {rr_ratio:.1f} below minimum {min_rr_ratio}",
                    max(0, (rr_ratio / min_rr_ratio) * 100)
                ))
            else:
                checks.append(ValidationCheck(
                    "risk_reward_ratio",
                    ValidationResult.PASS,
                    f"R/R ratio {rr_ratio:.1f} is good",
                    min(100, (rr_ratio / min_rr_ratio) * 80)
                ))
        
        return checks
    
    def _validate_market_conditions(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Validate market conditions"""
        
        checks = []
        market_data = proposal.market_data
        
        # 1. Volume validation
        current_volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume_20d', 1)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        min_volume_ratio = 0.5  # Minimum 50% of volume average
        
        if volume_ratio < min_volume_ratio:
            checks.append(ValidationCheck(
                "volume_validation",
                ValidationResult.WARNING,
                f"Low volume: {volume_ratio:.1f}x vs average",
                max(0, volume_ratio / min_volume_ratio * 100)
            ))
        else:
            score = min(100, volume_ratio * 50)
            checks.append(ValidationCheck(
                "volume_validation",
                ValidationResult.PASS,
                f"Adequate volume: {volume_ratio:.1f}x vs average",
                score
            ))
        
        # 2. Spread validation
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        
        if bid > 0 and ask > 0:
            spread = ask - bid
            spread_pct = spread / ((bid + ask) / 2)
            
            max_spread_pct = 0.02  # 2% máximo
            
            if spread_pct > max_spread_pct:
                checks.append(ValidationCheck(
                    "spread_validation",
                    ValidationResult.WARNING,
                    f"Wide spread: {spread_pct:.2%}",
                    max(0, (max_spread_pct - spread_pct) / max_spread_pct * 100)
                ))
            else:
                checks.append(ValidationCheck(
                    "spread_validation",
                    ValidationResult.PASS,
                    f"Acceptable spread: {spread_pct:.2%}",
                    100
                ))
        
        # 3. Market hours validation
        market_open = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        current_time = proposal.timestamp
        
        if current_time < market_open or current_time > market_close:
            checks.append(ValidationCheck(
                "market_hours",
                ValidationResult.WARNING,
                "Trade outside regular hours",
                70
            ))
        else:
            checks.append(ValidationCheck(
                "market_hours",
                ValidationResult.PASS,
                "Within market hours",
                100
            ))
        
        return checks
    
    def _validate_timing(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Validate trade timing"""
        
        checks = []
        current_time = proposal.timestamp
        
        # 1. Avoid first/last 30 minutes
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        minutes_from_open = (current_time - market_open).total_seconds() / 60
        minutes_to_close = (market_close - current_time).total_seconds() / 60
        
        if minutes_from_open < 30:
            checks.append(ValidationCheck(
                "opening_timing",
                ValidationResult.WARNING,
                f"Trade too close to open ({minutes_from_open:.0f} min)",
                max(0, minutes_from_open / 30 * 100)
            ))
        elif minutes_to_close < 30:
            checks.append(ValidationCheck(
                "closing_timing",
                ValidationResult.WARNING,
                f"Trade too close to close ({minutes_to_close:.0f} min)",
                max(0, minutes_to_close / 30 * 100)
            ))
        else:
            checks.append(ValidationCheck(
                "market_timing",
                ValidationResult.PASS,
                "Timing de mercado appropriate",
                100
            ))
        
        # 2. Friday afternoon warning
        if current_time.weekday() == 4 and current_time.hour >= 14:  # Friday afternoon
            checks.append(ValidationCheck(
                "friday_timing",
                ValidationResult.WARNING,
                "Trade on Friday afternoon - weekend risk",
                80
            ))
        
        return checks
    
    def _validate_correlation_risk(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Validate correlation risk"""
        
        checks = []
        
        # Check correlation with existing positions
        symbol_sector = self._get_symbol_sector(proposal.symbol)
        
        correlated_positions = 0
        for pos_symbol, position in self.current_positions.items():
            if self._get_symbol_sector(pos_symbol) == symbol_sector:
                correlated_positions += 1
        
        max_sector_positions = 3
        
        if correlated_positions >= max_sector_positions:
            checks.append(ValidationCheck(
                "correlation_risk",
                ValidationResult.WARNING,
                f"You already have {correlated_positions} posiciones en sector {symbol_sector}",
                max(0, (max_sector_positions - correlated_positions) / max_sector_positions * 100)
            ))
        else:
            checks.append(ValidationCheck(
                "correlation_risk",
                ValidationResult.PASS,
                f"Acceptable sector exposure",
                100
            ))
        
        return checks
    
    def _validate_liquidity(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Validate sufficient liquidity"""
        
        checks = []
        market_data = proposal.market_data
        
        # Average daily volume
        avg_daily_volume = market_data.get('avg_volume_20d', 0)
        position_volume = proposal.quantity
        
        # Position should be <10% of average daily volume
        volume_impact = position_volume / avg_daily_volume if avg_daily_volume > 0 else 1
        
        max_volume_impact = 0.10  # 10% máximo
        
        if volume_impact > max_volume_impact:
            checks.append(ValidationCheck(
                "liquidity_risk",
                ValidationResult.WARNING,
                f"Position es {volume_impact:.1%} of average daily volume",
                max(0, (max_volume_impact - volume_impact) / max_volume_impact * 100)
            ))
        else:
            checks.append(ValidationCheck(
                "liquidity_risk",
                ValidationResult.PASS,
                f"Adequate liquidity ({volume_impact:.1%} of volume)",
                100
            ))
        
        return checks
    
    def _validate_news_sentiment(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Validate news sentiment"""
        
        checks = []
        
        # Placeholder - real implementation would query news API
        news_sentiment = market_data.get('news_sentiment', 'neutral')
        
        if proposal.side == 'buy' and news_sentiment == 'negative':
            checks.append(ValidationCheck(
                "news_sentiment",
                ValidationResult.WARNING,
                "Negative news sentiment for long position",
                60
            ))
        elif proposal.side == 'sell' and news_sentiment == 'positive':
            checks.append(ValidationCheck(
                "news_sentiment",
                ValidationResult.WARNING,
                "Positive news sentiment for short position",
                60
            ))
        else:
            checks.append(ValidationCheck(
                "news_sentiment",
                ValidationResult.PASS,
                f"News sentiment: {news_sentiment}",
                100
            ))
        
        return checks
    
    def _validate_technical_confluence(self, proposal: TradeProposal) -> List[ValidationCheck]:
        """Validate technical confluence"""
        
        checks = []
        market_data = proposal.market_data
        
        # Multiple technical confirmations
        confluences = 0
        
        # Check if price is above/below key MAs
        price = proposal.entry_price
        sma_20 = market_data.get('sma_20', price)
        sma_50 = market_data.get('sma_50', price)
        
        if proposal.side == 'buy':
            if price > sma_20:
                confluences += 1
            if price > sma_50:
                confluences += 1
        else:  # sell
            if price < sma_20:
                confluences += 1
            if price < sma_50:
                confluences += 1
        
        # Check RSI
        rsi = market_data.get('rsi', 50)
        if proposal.side == 'buy' and 30 < rsi < 70:
            confluences += 1
        elif proposal.side == 'sell' and 30 < rsi < 70:
            confluences += 1
        
        min_confluences = 2
        
        if confluences < min_confluences:
            checks.append(ValidationCheck(
                "technical_confluence",
                ValidationResult.WARNING,
                f"Few technical confirmations ({confluences}/{min_confluences})",
                (confluences / min_confluences) * 100
            ))
        else:
            checks.append(ValidationCheck(
                "technical_confluence",
                ValidationResult.PASS,
                f"Good technical confluence ({confluences} confirmations)",
                min(100, confluences * 30)
            ))
        
        return checks
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get symbol sector (placeholder)"""
        # In real implementation, would query sector database
        sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology', 
            'GOOGL': 'Technology',
            'TSLA': 'Automotive',
            'SPY': 'ETF'
        }
        return sector_mapping.get(symbol, 'Unknown')
    
    def get_validation_summary(self, validations: List[ValidationCheck]) -> Dict:
        """Get validation summary"""
        
        total_checks = len(validations)
        passed = len([v for v in validations if v.result == ValidationResult.PASS])
        warnings = len([v for v in validations if v.result == ValidationResult.WARNING])
        failed = len([v for v in validations if v.result == ValidationResult.FAIL])
        critical_failures = len([v for v in validations if v.result == ValidationResult.FAIL and v.critical])
        
        average_score = np.mean([v.score for v in validations]) if validations else 0
        
        # Determine overall recommendation
        if critical_failures > 0:
            recommendation = "REJECT"
        elif failed > 0:
            recommendation = "REJECT"
        elif warnings > total_checks * 0.5:
            recommendation = "CAUTION"
        elif average_score >= 80:
            recommendation = "APPROVE"
        else:
            recommendation = "REVIEW"
        
        return {
            'total_checks': total_checks,
            'passed': passed,
            'warnings': warnings,
            'failed': failed,
            'critical_failures': critical_failures,
            'average_score': average_score,
            'recommendation': recommendation,
            'details': validations
        }

# Usage example
def demo_trade_validation():
    """Validation system demo"""
    
    # Configure validator
    validator = PreTradeValidator(ValidationLevel.STANDARD)
    
    # Configure account information
    validator.set_account_info({
        'total_value': 100000,
        'buying_power': 80000,
        'cash': 20000
    })
    
    # Configure current positions
    validator.set_current_positions({
        'AAPL': {'shares': 100, 'avg_price': 150.0}
    })
    
    # Create trade proposal
    proposal = TradeProposal(
        symbol="TSLA",
        side="buy",
        quantity=50,
        entry_price=200.0,
        stop_loss=190.0,
        take_profit=220.0,
        strategy_name="Gap and Go",
        setup_score=75.0,
        market_data={
            'volume': 1000000,
            'avg_volume_20d': 800000,
            'bid': 199.8,
            'ask': 200.2,
            'sma_20': 195.0,
            'sma_50': 190.0,
            'rsi': 55.0
        },
        timestamp=datetime.now().replace(hour=10, minute=30)
    )
    
    # Run validation
    validations = validator.validate_trade(proposal)
    summary = validator.get_validation_summary(validations)
    
    # Show results
    print(f"🔍 Trade Validation: {proposal.symbol}")
    print(f"Recommendation: {summary['recommendation']}")
    print(f"Score average: {summary['average_score']:.1f}")
    print(f"Checks: ✅{summary['passed']} ⚠️{summary['warnings']} ❌{summary['failed']}")
    
    print("\n📋 Details:")
    for validation in validations:
        icon = "✅" if validation.result == ValidationResult.PASS else \
               "⚠️" if validation.result == ValidationResult.WARNING else "❌"
        print(f"{icon} {validation.check_name}: {validation.message} (Score: {validation.score:.0f})")

if __name__ == "__main__":
    demo_trade_validation()
```

## Post-Trade Analysis Framework

### Trade Journal System
```python
class TradeJournal:
    """Trade journaling system"""
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self.trades = []
        
    def log_trade_entry(self, trade_data: Dict):
        """Log trade entry"""
        
        entry_record = {
            'trade_id': self._generate_trade_id(),
            'timestamp': datetime.now(),
            'type': 'entry',
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'quantity': trade_data['quantity'],
            'price': trade_data['price'],
            'strategy': trade_data['strategy'],
            'setup_score': trade_data.get('setup_score'),
            'market_conditions': trade_data.get('market_conditions'),
            'reasoning': trade_data.get('reasoning', ''),
            'pre_trade_plan': trade_data.get('pre_trade_plan'),
            'emotions': trade_data.get('emotions', ''),
            'confidence_level': trade_data.get('confidence_level'),
            'market_context': {
                'spy_price': trade_data.get('spy_price'),
                'vix_level': trade_data.get('vix_level'),
                'sector_performance': trade_data.get('sector_performance')
            }
        }
        
        self.trades.append(entry_record)
        return entry_record['trade_id']
    
    def log_trade_exit(self, trade_id: str, exit_data: Dict):
        """Log trade exit"""
        
        exit_record = {
            'trade_id': trade_id,
            'timestamp': datetime.now(),
            'type': 'exit',
            'exit_price': exit_data['price'],
            'exit_reason': exit_data['reason'],
            'pnl': exit_data['pnl'],
            'pnl_pct': exit_data['pnl_pct'],
            'hold_time': exit_data['hold_time'],
            'max_favorable_excursion': exit_data.get('mfe'),
            'max_adverse_excursion': exit_data.get('mae'),
            'exit_emotions': exit_data.get('emotions', ''),
            'lessons_learned': exit_data.get('lessons', ''),
            'what_went_right': exit_data.get('what_right', ''),
            'what_went_wrong': exit_data.get('what_wrong', ''),
            'would_do_differently': exit_data.get('differently', '')
        }
        
        self.trades.append(exit_record)
        return exit_record
    
    def analyze_trade_performance(self, lookback_days: int = 30) -> Dict:
        """Analyze trade performance"""
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Filter recent trades
        recent_trades = [t for t in self.trades if t['timestamp'] >= cutoff_date]
        
        # Separate entries and exits
        entries = [t for t in recent_trades if t['type'] == 'entry']
        exits = [t for t in recent_trades if t['type'] == 'exit']
        
        # Calculate metrics
        total_trades = len(exits)
        if total_trades == 0:
            return {'error': 'No completed trades in the period'}
        
        winning_trades = len([t for t in exits if t['pnl'] > 0])
        losing_trades = len([t for t in exits if t['pnl'] < 0])
        
        total_pnl = sum([t['pnl'] for t in exits])
        avg_win = np.mean([t['pnl'] for t in exits if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in exits if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Analysis by strategy
        strategy_performance = {}
        for exit_trade in exits:
            # Find corresponding entry
            entry_trade = next((e for e in entries if e['trade_id'] == exit_trade['trade_id']), None)
            if entry_trade:
                strategy = entry_trade['strategy']
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(exit_trade['pnl'])
        
        return {
            'period_days': lookback_days,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'strategy_performance': {
                strategy: {
                    'total_pnl': sum(pnls),
                    'trade_count': len(pnls),
                    'avg_pnl': np.mean(pnls),
                    'win_rate': len([p for p in pnls if p > 0]) / len(pnls)
                }
                for strategy, pnls in strategy_performance.items()
            }
        }
    
    def identify_patterns(self) -> Dict:
        """Identify patterns in trading"""
        
        exits = [t for t in self.trades if t['type'] == 'exit']
        entries = [t for t in self.trades if t['type'] == 'entry']
        
        patterns = {}
        
        # 1. Analysis by day of week
        day_performance = {}
        for exit_trade in exits:
            day_of_week = exit_trade['timestamp'].strftime('%A')
            if day_of_week not in day_performance:
                day_performance[day_of_week] = []
            day_performance[day_of_week].append(exit_trade['pnl'])
        
        patterns['day_of_week'] = {
            day: {
                'avg_pnl': np.mean(pnls),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls),
                'trade_count': len(pnls)
            }
            for day, pnls in day_performance.items()
        }
        
        # 2. Analysis by entry hour
        hour_performance = {}
        for entry_trade in entries:
            # Find corresponding exit
            exit_trade = next((e for e in exits if e['trade_id'] == entry_trade['trade_id']), None)
            if exit_trade:
                hour = entry_trade['timestamp'].hour
                if hour not in hour_performance:
                    hour_performance[hour] = []
                hour_performance[hour].append(exit_trade['pnl'])
        
        patterns['entry_hour'] = {
            hour: {
                'avg_pnl': np.mean(pnls),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls),
                'trade_count': len(pnls)
            }
            for hour, pnls in hour_performance.items()
        }
        
        # 3. Analysis by setup score
        score_buckets = {'low': [], 'medium': [], 'high': []}
        for entry_trade in entries:
            exit_trade = next((e for e in exits if e['trade_id'] == entry_trade['trade_id']), None)
            if exit_trade and entry_trade.get('setup_score'):
                score = entry_trade['setup_score']
                if score < 60:
                    bucket = 'low'
                elif score < 80:
                    bucket = 'medium'
                else:
                    bucket = 'high'
                score_buckets[bucket].append(exit_trade['pnl'])
        
        patterns['setup_score'] = {
            bucket: {
                'avg_pnl': np.mean(pnls) if pnls else 0,
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0,
                'trade_count': len(pnls)
            }
            for bucket, pnls in score_buckets.items()
        }
        
        return patterns
    
    def generate_insights(self) -> List[str]:
        """Generate insights based on patterns"""
        
        patterns = self.identify_patterns()
        insights = []
        
        # Insights by day of week
        if 'day_of_week' in patterns:
            best_day = max(patterns['day_of_week'].items(), 
                          key=lambda x: x[1]['avg_pnl'])
            worst_day = min(patterns['day_of_week'].items(), 
                           key=lambda x: x[1]['avg_pnl'])
            
            insights.append(f"📅 Best day for trading: {best_day[0]} (${best_day[1]['avg_pnl']:.2f} average)")
            insights.append(f"📅 Worst day for trading: {worst_day[0]} (${worst_day[1]['avg_pnl']:.2f} average)")
        
        # Insights by setup score
        if 'setup_score' in patterns:
            high_score_perf = patterns['setup_score']['high']
            low_score_perf = patterns['setup_score']['low']
            
            if high_score_perf['trade_count'] > 0 and low_score_perf['trade_count'] > 0:
                insights.append(f"🎯 Setup scoris highs tienen {high_score_perf['win_rate']:.1%} win rate vs {low_score_perf['win_rate']:.1%} for low scores")
        
        # General insights
        performance = self.analyze_trade_performance(30)
        if 'win_rate' in performance:
            if performance['win_rate'] < 0.5:
                insights.append("⚠️ Win rate below 50% - review entry criteria")
            if performance['profit_factor'] < 1.5:
                insights.append("⚠️ Low profit factor - work on risk/reward ratio")
        
        return insights
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        import uuid
        return str(uuid.uuid4())[:8]

# Journal usage example
def demo_trade_journal():
    """Journaling system demo"""
    
    journal = TradeJournal()
    
    # Log trade entry
    trade_id = journal.log_trade_entry({
        'symbol': 'AAPL',
        'side': 'buy',
        'quantity': 100,
        'price': 150.0,
        'strategy': 'Gap and Go',
        'setup_score': 85,
        'reasoning': 'Strong gap up with volume confirmation',
        'confidence_level': 8,
        'emotions': 'Confident but not overexcited'
    })
    
    print(f"📝 Trade logged with ID: {trade_id}")
    
    # Simulate exit after time
    import time
    time.sleep(1)  # Simulate time in trade
    
    journal.log_trade_exit(trade_id, {
        'price': 155.0,
        'reason': 'Target reached',
        'pnl': 500.0,
        'pnl_pct': 0.033,
        'hold_time': timedelta(hours=2),
        'mfe': 600.0,
        'mae': -100.0,
        'lessons': 'Good patience waiting for setup',
        'what_right': 'Stuck to plan, good entry timing',
        'what_wrong': 'Could have held longer for bigger profit'
    })
    
    # Analyze performance
    performance = journal.analyze_trade_performance(30)
    print(f"\n📊 Performance (30 days):")
    print(f"Total trades: {performance['total_trades']}")
    print(f"Win rate: {performance['win_rate']:.1%}")
    print(f"Total P&L: ${performance['total_pnl']:.2f}")
    
    # Get insights
    insights = journal.generate_insights()
    print(f"\n💡 Insights:")
    for insight in insights:
        print(f"  {insight}")

if __name__ == "__main__":
    demo_trade_journal()
```

This validation and journaling system provides a robust framework for maintaining discipline and continuously improving trading performance through systematic analysis.