# Short Selling Avanzado en Small Caps

## ⚠️ ADVERTENCIA CRÍTICA

Esta es una estrategia extremadamente avanzada y riesgosa. Solo para traders experimentados con:
- 5+ años de experiencia en small caps
- Capital que puedas perder completamente
- Conocimiento profundo de short selling mechanics
- Acceso a borrows y margin suficiente

**El 97% de los traders pierde dinero** con estas estrategias. Una mala operación puede resultar en pérdidas del 200-300%.

## Filosofía del Short Selling en Small Caps

### Por Qué Funciona
- 99% de penny stocks son empresas sin fundamentos
- Dilución constante destruye valor
- Retail traders son predecibles en pánico
- Pump & dump cycles son repetibles

### Anatomía de un Pump & Dump

```python
class PumpDumpCycle:
    def __init__(self):
        self.phases = {
            'accumulation': {
                'duration_days': (-5, -1),
                'volume_multiplier': (2, 3),
                'daily_moves': (5, 10),
                'characteristics': 'Insiders acumulan, volumen sutil'
            },
            'pump': {
                'duration_days': (1, 3),
                'volume_multiplier': (10, 50),
                'daily_moves': (20, 50),
                'characteristics': 'Press releases, FOMO retail, halts'
            },
            'distribution': {
                'duration_days': (3, 5),
                'volume_multiplier': (5, 20),
                'daily_moves': (-5, 15),
                'characteristics': 'Lower highs, volumen alto sin avance'
            },
            'dump': {
                'duration_days': (5, 10),
                'volume_multiplier': (3, 15),
                'daily_moves': (-30, -70),
                'characteristics': 'Colapso, panic selling, shorts cubren'
            }
        }
    
    def identify_phase(self, price_data, volume_data):
        """Identificar en qué fase está el cycle"""
        # Implementar lógica de identificación
        pass
```

## Sistema de Screening Multi-Nivel

### Nivel 1: Filtros Básicos (8000 → 500 stocks)
```python
def basic_filter(universe):
    """Filtro básico para short candidates"""
    criteria = {
        'price_range': (0.50, 20.00),
        'avg_volume_20d': 1_000_000,
        'market_cap_max': 500_000_000,
        'exchange': ['NASDAQ', 'NYSE'],  # Evitar OTC
        'float_max': 50_000_000
    }
    
    filtered = []
    for stock in universe:
        if (criteria['price_range'][0] <= stock.price <= criteria['price_range'][1] and
            stock.avg_volume >= criteria['avg_volume_20d'] and
            stock.market_cap <= criteria['market_cap_max'] and
            stock.exchange in criteria['exchange'] and
            stock.float_shares <= criteria['float_max']):
            
            filtered.append(stock)
    
    return filtered
```

### Nivel 2: Filtros de Actividad (500 → 50 stocks)
```python
def activity_filter(stocks):
    """Filtrar por actividad anormal"""
    filtered = []
    
    for stock in stocks:
        # Calcular métricas
        volume_ratio = stock.volume_today / stock.avg_volume_20d
        price_change = (stock.price - stock.prev_close) / stock.prev_close
        daily_range = (stock.high - stock.low) / stock.low
        
        # Criterios de actividad
        if (volume_ratio > 3 and
            abs(price_change) > 0.20 and
            daily_range > 0.10):
            
            # Verificar días consecutivos verdes (para first red day)
            consecutive_green = count_consecutive_green_days(stock)
            
            stock.activity_score = calculate_activity_score(
                volume_ratio, price_change, daily_range, consecutive_green
            )
            
            if stock.activity_score > 6:  # Threshold
                filtered.append(stock)
    
    return filtered
```

### Nivel 3: Filtros de Calidad (50 → 10 stocks)
```python
def quality_filter(stocks):
    """Filtro final de calidad para shorts"""
    filtered = []
    
    for stock in stocks:
        # Factores de calidad
        has_real_news = check_real_catalyst(stock)
        short_interest = get_short_interest(stock)
        institutional_ownership = get_institutional_ownership(stock)
        pump_history = check_pump_history(stock)
        bid_ask_spread = (stock.ask - stock.bid) / stock.bid
        
        # Score de calidad
        quality_score = 0
        
        if not has_real_news:  # NO debe tener news reales
            quality_score += 30
        if short_interest > 0.15:  # >15% short interest
            quality_score += 20
        if institutional_ownership < 0.20:  # <20% institucional
            quality_score += 20
        if pump_history:  # Historial de pumps
            quality_score += 20
        if bid_ask_spread < 0.03:  # <3% spread
            quality_score += 10
        
        if quality_score >= 70:
            stock.quality_score = quality_score
            filtered.append(stock)
    
    return sorted(filtered, key=lambda x: x.quality_score, reverse=True)
```

## Las 4 Estrategias Principales

### 1. First Red Day Pattern

```python
class FirstRedDayStrategy:
    def __init__(self):
        self.name = "First Red Day"
        self.min_consecutive_green = 2
        self.min_gap_down = -0.05  # -5%
        self.max_gap_down = -0.15  # -15%
        
    def identify_setup(self, stock_data):
        """Identificar setup de First Red Day"""
        # Verificar días verdes consecutivos
        green_days = self.count_consecutive_green_days(stock_data)
        
        if green_days < self.min_consecutive_green:
            return None
        
        # Verificar gap down hoy
        today_gap = (stock_data.open - stock_data.prev_close) / stock_data.prev_close
        
        if not (self.min_gap_down <= today_gap <= self.max_gap_down):
            return None
        
        # Verificar otros criterios
        criteria = {
            'volume_spike': stock_data.volume > stock_data.avg_volume * 3,
            'below_vwap': stock_data.price < stock_data.vwap,
            'pm_high_fade': (stock_data.premarket_high - stock_data.price) / stock_data.premarket_high > 0.05,
            'red_day': (stock_data.price - stock_data.open) / stock_data.open < -0.03
        }
        
        if all(criteria.values()):
            return self.calculate_entry_levels(stock_data)
        
        return None
    
    def calculate_entry_levels(self, stock_data):
        """Calcular niveles de entrada"""
        return {
            'aggressive_entry': stock_data.premarket_low * 0.995,  # Break PM low
            'conservative_entry': stock_data.vwap * 0.99,  # Break VWAP
            'stop_loss': max(stock_data.day_high, stock_data.premarket_high) * 1.02,
            'target_1': stock_data.open * 0.85,  # 15% target
            'target_2': stock_data.open * 0.70,  # 30% target
            'setup_score': self.score_setup(stock_data)
        }
    
    def score_setup(self, stock_data):
        """Score del setup (0-100)"""
        score = 0
        
        # Momentum score (30%)
        consecutive_days = self.count_consecutive_green_days(stock_data)
        if consecutive_days >= 5:
            score += 30
        elif consecutive_days >= 3:
            score += 20
        else:
            score += 10
        
        # Volume score (25%)
        volume_ratio = stock_data.volume / stock_data.avg_volume
        if volume_ratio >= 10:
            score += 25
        elif volume_ratio >= 5:
            score += 20
        elif volume_ratio >= 3:
            score += 15
        
        # Technical score (25%)
        distance_from_ma = (stock_data.price - stock_data.sma_20) / stock_data.sma_20
        if distance_from_ma > 0.5:  # >50% above 20 SMA
            score += 25
        elif distance_from_ma > 0.3:
            score += 15
        
        # Risk score (20%)
        if stock_data.float_shares < 5_000_000:
            score += 20
        elif stock_data.float_shares < 15_000_000:
            score += 15
        
        return min(score, 100)
```

### 2. Parabolic Exhaustion Intraday

```python
class ParabolicExhaustionStrategy:
    def __init__(self):
        self.name = "Parabolic Exhaustion"
        self.min_intraday_gain = 0.60  # 60%
        self.min_volume_spike = 10
        
    def detect_exhaustion_signals(self, intraday_data):
        """Detectar señales de agotamiento"""
        signals = {}
        
        # 1. Lower highs en últimas barras
        recent_highs = intraday_data.tail(12)['high']  # Últimas 12 barras (1 hora)
        signals['lower_highs'] = self.detect_lower_highs(recent_highs)
        
        # 2. Volumen decreciente
        recent_volume = intraday_data.tail(6)['volume']
        signals['decreasing_volume'] = recent_volume.is_monotonic_decreasing
        
        # 3. RSI divergencia
        signals['rsi_divergence'] = self.detect_rsi_divergence(intraday_data)
        
        # 4. Failed breakouts
        signals['failed_breakouts'] = self.count_failed_breakouts(intraday_data)
        
        # 5. Time at high
        signals['time_at_high'] = self.calculate_time_at_high(intraday_data)
        
        # 6. VWAP distance
        current_price = intraday_data.iloc[-1]['close']
        vwap = intraday_data.iloc[-1]['vwap']
        signals['vwap_distance'] = (current_price - vwap) / vwap
        
        return signals
    
    def should_enter_short(self, signals):
        """Determinar si entrar short"""
        entry_criteria = [
            signals['lower_highs'],
            signals['decreasing_volume'],
            signals['rsi_divergence'],
            signals['failed_breakouts'] >= 2,
            signals['time_at_high'] > 30,  # 30 minutes
            signals['vwap_distance'] > 0.10  # 10% above VWAP
        ]
        
        # Necesitamos al menos 4 de 6 señales
        return sum(entry_criteria) >= 4
    
    def calculate_entry_strategy(self, current_price, signals):
        """Calcular estrategia de entrada escalonada"""
        return {
            'entry_1': {
                'price': current_price * 0.98,  # 2% down from current
                'size_pct': 0.25,
                'trigger': 'first_weakness'
            },
            'entry_2': {
                'price': signals.get('vwap_level', current_price * 0.95),
                'size_pct': 0.35,
                'trigger': 'vwap_break'
            },
            'entry_3': {
                'price': current_price * 0.90,  # Opening print level
                'size_pct': 0.40,
                'trigger': 'opening_print_loss'
            },
            'stop_loss': current_price * 1.12,  # 12% stop (wider for volatility)
            'target': current_price * 0.70  # 30% target
        }
```

### 3. Gap and Crap (Fade the Gap)

```python
class GapAndCrapStrategy:
    def __init__(self):
        self.name = "Gap and Crap"
        self.min_gap_size = 0.30  # 30%
        self.max_pm_volume = 500_000
        
    def identify_gap_fade_setup(self, stock_data):
        """Identificar setup de gap fade"""
        # Verificar gap sin catalizador
        gap_size = (stock_data.open - stock_data.prev_close) / stock_data.prev_close
        
        if gap_size < self.min_gap_size:
            return None
        
        # Verificar que NO haya catalizador real
        if self.has_real_catalyst(stock_data):
            return None
        
        # Verificar bajo volumen pre-market
        if stock_data.premarket_volume > self.max_pm_volume:
            return None
        
        # Verificar debilidad en pre-market
        pm_pattern = self.analyze_premarket_pattern(stock_data)
        
        if pm_pattern['is_fading']:
            return self.calculate_fade_levels(stock_data, gap_size)
        
        return None
    
    def calculate_fade_levels(self, stock_data, gap_size):
        """Calcular niveles para fade del gap"""
        prev_close = stock_data.prev_close
        gap_fill_50 = (stock_data.open + prev_close) / 2
        gap_fill_100 = prev_close
        
        return {
            'pm_entry': {
                'price': stock_data.premarket_high * 0.97,
                'size_pct': 0.50,  # Menor size por liquidez PM
                'time_window': '6:00-9:30'
            },
            'market_open_entry': {
                'price': stock_data.premarket_high,
                'size_pct': 1.00,
                'trigger': 'fail_to_break_pm_high'
            },
            'targets': {
                'gap_fill_50': gap_fill_50,
                'gap_fill_100': gap_fill_100,
                'profit_expectation': f"{((stock_data.open - gap_fill_50) / stock_data.open):.1%}"
            },
            'stop_loss': stock_data.premarket_high * 1.05,
            'time_limit': '10:30'  # Si no funciona en 1 hora, salir
        }
```

## Risk Management Específico

### Position Sizing para Shorts
```python
class ShortPositionSizer:
    def __init__(self, account_value, max_portfolio_short_exposure=0.30):
        self.account_value = account_value
        self.max_portfolio_short_exposure = max_portfolio_short_exposure
        
    def calculate_short_size(self, stock_data, strategy_type, setup_score):
        """Calcular tamaño específico para shorts"""
        
        # Base risk según experiencia
        base_risk_by_experience = {
            'beginner': 0.005,    # 0.5% - NO recomendado
            'intermediate': 0.01,  # 1%
            'advanced': 0.015,     # 1.5%
            'expert': 0.02        # 2%
        }
        
        base_risk = base_risk_by_experience['advanced']  # Default
        
        # Ajuste por tipo de estrategia
        strategy_multipliers = {
            'first_red_day': 1.0,
            'parabolic_exhaustion': 0.75,  # Más volátil
            'gap_and_crap': 1.25,  # Más predecible
            'afternoon_breakdown': 1.0
        }
        
        strategy_mult = strategy_multipliers.get(strategy_type, 1.0)
        
        # Ajuste por score del setup
        score_multiplier = 0.5 + (setup_score / 100) * 1.5  # 0.5x a 2.0x
        
        # Ajuste por características del stock
        volatility_mult = self.calculate_volatility_adjustment(stock_data)
        float_mult = self.calculate_float_adjustment(stock_data.float_shares)
        
        # Risk final
        final_risk = (base_risk * strategy_mult * score_multiplier * 
                     volatility_mult * float_mult)
        
        # Cap en máximos
        final_risk = min(final_risk, 0.03)  # Nunca más de 3%
        
        # Calcular shares
        entry_price = stock_data.price
        stop_price = self.calculate_stop_price(stock_data, strategy_type)
        
        risk_per_share = stop_price - entry_price  # Para shorts
        risk_amount = self.account_value * final_risk
        
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        return {
            'shares': shares,
            'risk_amount': risk_amount,
            'risk_pct': final_risk,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'setup_score': setup_score,
            'adjustments': {
                'strategy_mult': strategy_mult,
                'score_mult': score_multiplier,
                'volatility_mult': volatility_mult,
                'float_mult': float_mult
            }
        }
    
    def calculate_volatility_adjustment(self, stock_data):
        """Ajustar por volatilidad"""
        avg_true_range = getattr(stock_data, 'atr_14', 0.1)
        price = stock_data.price
        volatility_pct = avg_true_range / price
        
        if volatility_pct > 0.15:  # >15% diario
            return 0.5  # Reducir size 50%
        elif volatility_pct > 0.10:
            return 0.75
        else:
            return 1.0
    
    def calculate_float_adjustment(self, float_shares):
        """Ajustar por float size"""
        if float_shares < 5_000_000:  # Micro float
            return 0.5  # Muy riesgoso
        elif float_shares < 15_000_000:  # Low float
            return 0.75
        else:
            return 1.0
```

### Stop Loss Management
```python
class ShortStopManager:
    def __init__(self):
        self.max_loss_pct = 0.15  # 15% max loss
        
    def calculate_stop_levels(self, stock_data, strategy_type):
        """Calcular stops específicos por estrategia"""
        
        stops = {}
        
        if strategy_type == 'first_red_day':
            stops['day_high'] = stock_data.day_high * 1.02
            stops['pm_high'] = stock_data.premarket_high * 1.02
            stops['recommended'] = max(stops['day_high'], stops['pm_high'])
            
        elif strategy_type == 'parabolic_exhaustion':
            stops['percentage'] = stock_data.price * 1.12  # 12% stop
            stops['day_high'] = stock_data.day_high * 1.05
            stops['recommended'] = max(stops['percentage'], stops['day_high'])
            
        elif strategy_type == 'gap_and_crap':
            stops['pm_high'] = stock_data.premarket_high * 1.05
            stops['percentage'] = stock_data.price * 1.08  # 8% stop
            stops['recommended'] = stops['pm_high']
            
        # Time-based stops
        stops['time_stop'] = self.calculate_time_stop(strategy_type)
        
        return stops
    
    def should_cut_loss_early(self, current_price, entry_price, unrealized_loss_pct):
        """¿Cortar pérdida antes del stop?"""
        # Cortar si pérdida > 10% y no hay momentum
        if unrealized_loss_pct > 0.10:
            return True
        
        # Cortar si squeeze detectado
        if self.detect_short_squeeze_signs(current_price, entry_price):
            return True
        
        return False
    
    def detect_short_squeeze_signs(self, current_price, entry_price):
        """Detectar signos de short squeeze"""
        # Simplified - en realidad necesitarías más data
        price_increase = (current_price - entry_price) / entry_price
        
        # Si subió >8% rápido, posible squeeze
        return price_increase > 0.08
```

## Monitoreo y Alertas

```python
class ShortMonitoringSystem:
    def __init__(self):
        self.active_shorts = {}
        self.alert_thresholds = {
            'profit_take_1': 0.15,  # 15% gain
            'profit_take_2': 0.25,  # 25% gain
            'stop_loss_warning': 0.08,  # 8% loss warning
            'time_limit_warning': '14:30'  # Warning near close
        }
    
    def monitor_short_positions(self):
        """Monitorear posiciones cortas activas"""
        for ticker, position in self.active_shorts.items():
            current_data = self.get_real_time_data(ticker)
            
            # Calcular P&L
            entry_price = position['entry_price']
            current_price = current_data['price']
            unrealized_pnl = (entry_price - current_price) / entry_price
            
            # Check alerts
            alerts = self.check_position_alerts(ticker, position, current_data, unrealized_pnl)
            
            # Update position
            position['current_price'] = current_price
            position['unrealized_pnl'] = unrealized_pnl
            position['alerts'] = alerts
    
    def check_position_alerts(self, ticker, position, current_data, pnl):
        """Check alertas para la posición"""
        alerts = []
        
        # Profit taking alerts
        if pnl >= self.alert_thresholds['profit_take_2']:
            alerts.append(f"🎯 {ticker}: 25% PROFIT - Consider full exit")
        elif pnl >= self.alert_thresholds['profit_take_1']:
            alerts.append(f"💰 {ticker}: 15% PROFIT - Consider partial exit")
        
        # Loss alerts
        elif pnl <= -self.alert_thresholds['stop_loss_warning']:
            alerts.append(f"⚠️ {ticker}: Approaching stop loss - Monitor closely")
        
        # Time alerts
        current_time = pd.Timestamp.now().time()
        if current_time >= pd.Timestamp(self.alert_thresholds['time_limit_warning']).time():
            alerts.append(f"⏰ {ticker}: Market close approaching - Consider exit")
        
        # Technical alerts
        if current_data['price'] > current_data['vwap'] and position['strategy'] == 'first_red_day':
            alerts.append(f"📈 {ticker}: Above VWAP - Momentum shift warning")
        
        return alerts
```

## Consideraciones Legales y Éticas

### Compliance
```python
# Verificaciones de compliance para short selling
def compliance_check(ticker, intended_size):
    """Verificar compliance antes de short"""
    checks = {
        'uptick_rule': check_uptick_rule_compliance(ticker),
        'locate_available': check_share_locate_availability(ticker),
        'reg_sho_compliance': check_regulation_sho(ticker),
        'position_size_limit': intended_size < MAX_POSITION_SIZE,
        'account_margin': check_margin_requirements()
    }
    
    return all(checks.values()), checks
```

### Advertencias Finales
- **Nunca hacer short selling sin entender completamente los riesgos**
- **Pérdidas ilimitadas** son posibles en shorts
- **Short squeezes** pueden causar pérdidas masivas
- **Costos de borrow** pueden ser extremos en penny stocks
- **Regulaciones** cambian constantemente

## Siguiente Paso

Este contenido es extremadamente avanzado. Para traders menos experimentados, recomiendo empezar con [Gap & Go](gap_and_go.md) en el lado long antes de considerar shorts.