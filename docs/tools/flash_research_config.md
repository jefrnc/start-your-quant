# Configuraciones para Flash Research

## Overview de Flash Research

Flash Research es una plataforma avanzada de screening y análisis que se especializa en market microstructure y detección de patrones institucionales. Es particularmente útil para identificar:

- Flujo de órdenes institucionales
- Actividad inusual de opciones
- Dark pool prints
- Patrones de acumulación/distribución

## Configuraciones Base

### Filtros Universales para Short Selling

```python
# Configuración base para scanner de Flash Research
FLASH_RESEARCH_CONFIG = {
    'base_filters': {
        'exchange': ['NASDAQ', 'NYSE'],
        'price_range': {'min': 0.50, 'max': 20.00},
        'market_cap': {'max': 500_000_000},
        'volume_20d_avg': {'min': 1_000_000},
        'float_shares': {'max': 50_000_000}
    },
    'activity_filters': {
        'volume_ratio': {'min': 3.0},  # 3x average volume
        'price_change_today': {'min_abs': 0.20},  # 20% move
        'daily_range': {'min': 0.10},  # 10% high-low range
        'consecutive_green_days': {'min': 2}
    },
    'quality_filters': {
        'news_catalyst': False,  # NO real news
        'short_interest': {'min': 0.15},  # >15% short interest
        'institutional_ownership': {'max': 0.20},  # <20% institutional
        'bid_ask_spread': {'max': 0.03}  # <3% spread
    }
}
```

### Query Templates por Estrategia

#### First Red Day Pattern
```sql
-- Flash Research Query para First Red Day
SELECT 
    symbol,
    price,
    volume,
    volume_ratio,
    gap_percent,
    premarket_high,
    premarket_volume,
    consecutive_green_days,
    vwap,
    float_shares
FROM market_data 
WHERE 
    exchange IN ('NASDAQ', 'NYSE')
    AND price BETWEEN 0.50 AND 20.00
    AND market_cap < 500000000
    AND volume_20d_avg > 1000000
    AND float_shares < 50000000
    AND volume_ratio > 3
    AND gap_percent BETWEEN -15 AND -5
    AND consecutive_green_days >= 2
    AND price < vwap
    AND premarket_volume > 500000
    AND (price - premarket_high) / premarket_high < -0.05
ORDER BY 
    (consecutive_green_days * volume_ratio * ABS(gap_percent)) DESC
LIMIT 20;
```

#### Parabolic Exhaustion
```sql
-- Query para Parabolic Exhaustion
SELECT 
    symbol,
    price,
    volume,
    volume_ratio,
    intraday_gain_percent,
    time_at_high_minutes,
    failed_breakouts_count,
    vwap_distance_percent,
    rsi_14,
    float_shares
FROM market_data 
WHERE 
    exchange IN ('NASDAQ', 'NYSE')
    AND price BETWEEN 1.00 AND 50.00
    AND intraday_gain_percent > 60
    AND volume_ratio > 10
    AND time_at_high_minutes > 30
    AND failed_breakouts_count >= 2
    AND vwap_distance_percent > 10
    AND rsi_14 > 75
    AND float_shares < 30000000
ORDER BY 
    (intraday_gain_percent * volume_ratio * failed_breakouts_count) DESC
LIMIT 15;
```

#### Gap and Crap
```sql
-- Query para Gap and Crap
SELECT 
    symbol,
    price,
    gap_percent,
    premarket_volume,
    premarket_high,
    current_vs_pm_high,
    first_5min_direction,
    volume_expectation_ratio,
    news_count_today
FROM market_data 
WHERE 
    exchange IN ('NASDAQ', 'NYSE')
    AND price BETWEEN 2.00 AND 25.00
    AND gap_percent > 30
    AND premarket_volume < 500000
    AND news_count_today = 0  -- No real news
    AND (current_vs_pm_high) < -0.05  -- Fading from PM high
    AND first_5min_direction = 'red'
    AND volume_expectation_ratio < 0.8  -- Below expected volume
ORDER BY 
    (gap_percent * (1 - volume_expectation_ratio)) DESC
LIMIT 10;
```

## Sistema de Scoring Avanzado

### Implementación del Scoring
```python
class FlashResearchScorer:
    def __init__(self):
        self.weights = {
            'momentum': 0.30,
            'volume': 0.25,
            'technical': 0.25,
            'risk': 0.20
        }
        
    def calculate_momentum_score(self, data):
        """Score de momentum (30% del total)"""
        score = 0
        
        # Ganancia total desde base
        total_gain = data.get('total_gain_percent', 0)
        if total_gain >= 100:
            score += 40
        elif total_gain >= 60:
            score += 25
        elif total_gain >= 30:
            score += 10
        
        # Días consecutivos verdes
        consecutive_days = data.get('consecutive_green_days', 0)
        if consecutive_days >= 5:
            score += 30
        elif consecutive_days >= 3:
            score += 20
        elif consecutive_days >= 2:
            score += 10
        
        # Aceleración del movimiento
        if data.get('accelerating_momentum', False):
            score += 20
        
        return min(score, 100)
    
    def calculate_volume_score(self, data):
        """Score de volumen (25% del total)"""
        score = 0
        
        # Múltiplo sobre promedio
        volume_ratio = data.get('volume_ratio', 1)
        if volume_ratio >= 10:
            score += 40
        elif volume_ratio >= 5:
            score += 30
        elif volume_ratio >= 3:
            score += 20
        
        # Patrón de volumen
        if data.get('volume_decreasing_today', False):
            score += 30  # Distribución
        
        if data.get('heavy_selling_detected', False):
            score += 20
        
        return min(score, 100)
    
    def calculate_technical_score(self, data):
        """Score técnico (25% del total)"""
        score = 0
        
        # Distancia de SMA 20
        sma_distance = data.get('distance_from_sma20_percent', 0)
        if sma_distance >= 50:
            score += 30  # Muy sobreextendido
        elif sma_distance >= 30:
            score += 20
        elif sma_distance >= 15:
            score += 10
        
        # RSI divergencia
        if data.get('rsi_divergence', False):
            score += 30
        
        # Failed breakout
        if data.get('failed_breakout', False):
            score += 20
        
        # Multiple rejections
        rejections = data.get('resistance_rejections', 0)
        if rejections >= 2:
            score += 20
        
        return min(score, 100)
    
    def calculate_risk_score(self, data):
        """Score de riesgo (20% del total)"""
        score = 0
        
        # Float size
        float_shares = data.get('float_shares', float('inf'))
        if float_shares < 5_000_000:
            score += 30
        elif float_shares < 15_000_000:
            score += 20
        elif float_shares < 30_000_000:
            score += 10
        
        # No halts today
        if not data.get('halts_today', False):
            score += 20
        
        # Locate available
        if data.get('locate_available', False):
            score += 30
        
        # Costo de locate
        locate_cost = data.get('locate_cost_annual_percent', 100)
        if locate_cost < 10:
            score += 20
        elif locate_cost < 25:
            score += 10
        
        return min(score, 100)
    
    def calculate_total_score(self, data):
        """Score total ponderado"""
        momentum_score = self.calculate_momentum_score(data)
        volume_score = self.calculate_volume_score(data)
        technical_score = self.calculate_technical_score(data)
        risk_score = self.calculate_risk_score(data)
        
        total_score = (
            momentum_score * self.weights['momentum'] +
            volume_score * self.weights['volume'] +
            technical_score * self.weights['technical'] +
            risk_score * self.weights['risk']
        )
        
        # Clasificar setup
        if total_score >= 80:
            setup_grade = 'A+'
        elif total_score >= 60:
            setup_grade = 'A'
        elif total_score >= 40:
            setup_grade = 'B'
        else:
            setup_grade = 'C'
        
        return {
            'total_score': total_score,
            'setup_grade': setup_grade,
            'component_scores': {
                'momentum': momentum_score,
                'volume': volume_score,
                'technical': technical_score,
                'risk': risk_score
            },
            'tradeable': total_score >= 40
        }
```

## Configuración de Watchlists

### Criterios de Watchlist
```python
WATCHLIST_CRITERIA = {
    'tier_1_targets': {
        'description': 'Máxima prioridad - Setup perfecto',
        'criteria': {
            'total_score': {'min': 80},
            'consecutive_green_days': {'min': 3},
            'volume_ratio': {'min': 5},
            'float_shares': {'max': 15_000_000},
            'locate_available': True,
            'locate_cost': {'max': 15}
        },
        'max_positions': 3,
        'position_size_multiplier': 1.0
    },
    'tier_2_targets': {
        'description': 'Buena probabilidad - Setup sólido',
        'criteria': {
            'total_score': {'min': 60},
            'consecutive_green_days': {'min': 2},
            'volume_ratio': {'min': 3},
            'float_shares': {'max': 30_000_000},
            'locate_available': True
        },
        'max_positions': 5,
        'position_size_multiplier': 0.75
    },
    'tier_3_targets': {
        'description': 'Monitoreo - Setup marginal',
        'criteria': {
            'total_score': {'min': 40},
            'volume_ratio': {'min': 2},
            'float_shares': {'max': 50_000_000}
        },
        'max_positions': 7,
        'position_size_multiplier': 0.5
    }
}
```

## Alertas y Monitoreo

### Sistema de Alertas Flash Research
```python
class FlashResearchAlerts:
    def __init__(self, api_credentials):
        self.api = api_credentials
        self.alert_history = []
        self.active_targets = {}
        
    def setup_real_time_alerts(self):
        """Configurar alertas en tiempo real"""
        alert_configs = {
            'first_red_day': {
                'trigger_conditions': [
                    'gap_percent BETWEEN -15 AND -5',
                    'volume_ratio > 3',
                    'consecutive_green_days >= 2',
                    'price < vwap'
                ],
                'alert_message': 'FIRST RED DAY SETUP: {symbol} - Gap: {gap_percent}%, RVol: {volume_ratio}x',
                'priority': 'HIGH'
            },
            'parabolic_exhaustion': {
                'trigger_conditions': [
                    'intraday_gain_percent > 60',
                    'volume_ratio > 10',
                    'time_at_high_minutes > 30',
                    'failed_breakouts_count >= 2'
                ],
                'alert_message': 'PARABOLIC EXHAUSTION: {symbol} - Gain: {intraday_gain_percent}%, Time at High: {time_at_high_minutes}min',
                'priority': 'CRITICAL'
            },
            'gap_fade_setup': {
                'trigger_conditions': [
                    'gap_percent > 30',
                    'premarket_volume < 500000',
                    'news_count_today = 0',
                    'current_vs_pm_high < -0.05'
                ],
                'alert_message': 'GAP FADE SETUP: {symbol} - Gap: {gap_percent}%, PM Vol: {premarket_volume}',
                'priority': 'MEDIUM'
            }
        }
        
        return alert_configs
    
    def process_flash_research_data(self, market_data):
        """Procesar data de Flash Research y generar alertas"""
        alerts = []
        
        for symbol, data in market_data.items():
            # Calculate score
            scorer = FlashResearchScorer()
            score_result = scorer.calculate_total_score(data)
            
            # Check if meets criteria
            if score_result['tradeable']:
                # Determine tier
                tier = self.classify_target_tier(score_result['total_score'], data)
                
                # Generate alert
                alert = {
                    'timestamp': pd.Timestamp.now(),
                    'symbol': symbol,
                    'tier': tier,
                    'total_score': score_result['total_score'],
                    'setup_grade': score_result['setup_grade'],
                    'strategy_type': self.identify_strategy_type(data),
                    'entry_price': data.get('current_price'),
                    'suggested_stop': self.calculate_suggested_stop(data),
                    'risk_reward_ratio': self.calculate_risk_reward(data),
                    'market_data': data
                }
                
                alerts.append(alert)
                self.active_targets[symbol] = alert
        
        return alerts
    
    def identify_strategy_type(self, data):
        """Identificar tipo de estrategia más apropiado"""
        if (data.get('consecutive_green_days', 0) >= 2 and 
            data.get('gap_percent', 0) < 0):
            return 'first_red_day'
        
        elif data.get('intraday_gain_percent', 0) > 60:
            return 'parabolic_exhaustion'
        
        elif (data.get('gap_percent', 0) > 30 and 
              data.get('premarket_volume', float('inf')) < 500_000):
            return 'gap_and_crap'
        
        elif data.get('afternoon_weakness', False):
            return 'afternoon_breakdown'
        
        else:
            return 'general_short'
    
    def calculate_suggested_stop(self, data):
        """Calcular stop loss sugerido"""
        current_price = data.get('current_price', 0)
        
        # Stop basado en tipo de setup
        if data.get('premarket_high'):
            pm_high_stop = data['premarket_high'] * 1.02
        else:
            pm_high_stop = current_price * 1.05
        
        day_high_stop = data.get('day_high', current_price) * 1.02
        percentage_stop = current_price * 1.08  # 8% stop
        
        # Usar el más conservador (más alto para shorts)
        suggested_stop = max(pm_high_stop, day_high_stop, percentage_stop)
        
        return round(suggested_stop, 2)
```

## Integración con APIs

### Configuración de API Flash Research
```python
class FlashResearchAPI:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_screening_results(self, strategy_type, limit=50):
        """Obtener resultados de screening"""
        endpoint = f"{self.base_url}/screen/{strategy_type}"
        
        params = {
            'limit': limit,
            'include_scores': True,
            'include_technicals': True,
            'include_fundamentals': True
        }
        
        response = requests.get(endpoint, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def get_real_time_data(self, symbols):
        """Obtener datos en tiempo real"""
        endpoint = f"{self.base_url}/realtime"
        
        payload = {
            'symbols': symbols,
            'include_level2': True,
            'include_options_flow': True,
            'include_dark_pools': True
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def setup_websocket_alerts(self, alert_configs):
        """Configurar alertas vía WebSocket"""
        import websocket
        
        def on_message(ws, message):
            data = json.loads(message)
            self.process_alert(data)
        
        def on_error(ws, error):
            print(f"WebSocket Error: {error}")
        
        def on_close(ws):
            print("WebSocket connection closed")
        
        ws_url = f"wss://api.flashresearch.com/alerts?token={self.api_key}"
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        return ws
```

Esta configuración permite aprovechar completamente las capacidades de Flash Research para identificar oportunidades de short selling con alta precisión.