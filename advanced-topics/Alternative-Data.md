# Integración de Datos Alternativos

## ¿Por qué Datos Alternativos en Small Caps?

Los datos alternativos proporcionan **ventajas informacionales críticas** en small caps porque:

- **Menor cobertura analítica**: Wall Street ignora la mayoría de small caps
- **Information gaps**: Retail e institutional investors tienen información asimétrica
- **News impact amplificado**: Small caps reaccionan más violentamente a noticias
- **Social sentiment relevante**: Retail trading influye significativamente en precios
- **Early warning signals**: Datos alternativos pueden predecir movimientos institucionales

### Edge Único para Small Caps

```python
ALTERNATIVE_DATA_EDGE = {
    'social_sentiment': {
        'edge': 'Retail traders move small cap prices more than large caps',
        'timeframe': 'Minutes to hours before price moves',
        'sources': ['StockTwits', 'Reddit', 'Twitter', 'Discord']
    },
    'sec_filings': {
        'edge': 'Less institutional monitoring = more alpha from filings',
        'timeframe': 'Same day to 3 days post-filing',
        'sources': ['8-K filings', '13D/G', 'Insider transactions', '10-Q/K']
    },
    'options_flow': {
        'edge': 'Unusual options activity often predicts equity moves',
        'timeframe': '1-5 days before earnings or announcements',
        'sources': ['Unusual volume', 'Large block trades', 'IV changes']
    },
    'float_analysis': {
        'edge': 'Float changes dramatically affect price dynamics',
        'timeframe': 'Real-time during offerings/buybacks',
        'sources': ['S-3 registrations', 'Share buyback announcements']
    }
}
```

## Framework de Datos Alternativos

### 1. Social Sentiment Analysis Pipeline

```python
import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta
import json
import re


class SocialSentimentAnalyzer:
    """
    Analyzer completo de social sentiment para small caps

    Integra múltiples fuentes de social data:
    - StockTwits API
    - Reddit API (wallstreetbets, pennystocks, etc.)
    - Twitter API v2
    - Discord webhooks (optional)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.apis = self._initialize_apis()
        self.sentiment_history = {}

    def _initialize_apis(self) -> Dict:
        """Initialize API connections"""
        return {
            'stocktwits': StockTwitsAPI(self.config.get('stocktwits_token')),
            'reddit': RedditAPI(self.config.get('reddit_config')),
            'twitter': TwitterAPI(self.config.get('twitter_bearer_token'))
        }

    def get_comprehensive_sentiment(self, symbol: str,
                                  lookback_hours: int = 24) -> Dict:
        """
        Obtiene sentiment comprehensive para un símbolo
        """

        sentiment_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'lookback_hours': lookback_hours,
            'sources': {}
        }

        # 1. StockTwits sentiment
        try:
            stocktwits_data = self.apis['stocktwits'].get_symbol_sentiment(
                symbol, lookback_hours
            )
            sentiment_data['sources']['stocktwits'] = stocktwits_data
        except Exception as e:
            print(f"StockTwits error for {symbol}: {e}")

        # 2. Reddit sentiment
        try:
            reddit_data = self.apis['reddit'].get_symbol_sentiment(
                symbol, lookback_hours
            )
            sentiment_data['sources']['reddit'] = reddit_data
        except Exception as e:
            print(f"Reddit error for {symbol}: {e}")

        # 3. Twitter sentiment
        try:
            twitter_data = self.apis['twitter'].get_symbol_sentiment(
                symbol, lookback_hours
            )
            sentiment_data['sources']['twitter'] = twitter_data
        except Exception as e:
            print(f"Twitter error for {symbol}: {e}")

        # 4. Aggregate sentiment score
        sentiment_data['aggregated'] = self._aggregate_sentiment_scores(
            sentiment_data['sources']
        )

        # 5. Store for historical analysis
        self.sentiment_history[symbol] = sentiment_data

        return sentiment_data

    def _aggregate_sentiment_scores(self, sources: Dict) -> Dict:
        """
        Agrega scores de múltiples fuentes con weights
        """

        weights = {
            'stocktwits': 0.4,  # Most relevant for stocks
            'reddit': 0.35,     # High influence on small caps
            'twitter': 0.25     # General market sentiment
        }

        total_sentiment = 0
        total_weight = 0
        total_volume = 0

        source_contributions = {}

        for source, data in sources.items():
            if data and 'sentiment_score' in data:
                weight = weights.get(source, 0.33)
                volume = data.get('message_volume', 1)

                # Weight by both source reliability and message volume
                effective_weight = weight * min(volume / 10, 2.0)  # Cap volume influence

                total_sentiment += data['sentiment_score'] * effective_weight
                total_weight += effective_weight
                total_volume += volume

                source_contributions[source] = {
                    'score': data['sentiment_score'],
                    'volume': volume,
                    'weight_used': effective_weight
                }

        if total_weight > 0:
            aggregated_score = total_sentiment / total_weight
        else:
            aggregated_score = 0.5  # Neutral

        # Sentiment velocity (change over time)
        sentiment_velocity = self._calculate_sentiment_velocity(sources)

        return {
            'aggregated_score': aggregated_score,
            'total_volume': total_volume,
            'sentiment_velocity': sentiment_velocity,
            'source_contributions': source_contributions,
            'confidence': min(total_weight / sum(weights.values()), 1.0)
        }

    def _calculate_sentiment_velocity(self, sources: Dict) -> float:
        """
        Calcula velocity of sentiment change (momentum)
        """
        velocity_scores = []

        for source, data in sources.items():
            if data and 'hourly_sentiment' in data:
                hourly_data = data['hourly_sentiment']
                if len(hourly_data) >= 2:
                    recent_avg = np.mean(hourly_data[-2:])
                    previous_avg = np.mean(hourly_data[:-2]) if len(hourly_data) > 2 else recent_avg
                    velocity = recent_avg - previous_avg
                    velocity_scores.append(velocity)

        return np.mean(velocity_scores) if velocity_scores else 0.0

    def generate_sentiment_signal(self, symbol: str) -> Optional[Dict]:
        """
        Genera trading signal basado en sentiment analysis
        """

        sentiment_data = self.get_comprehensive_sentiment(symbol)
        aggregated = sentiment_data['aggregated']

        score = aggregated['aggregated_score']
        volume = aggregated['total_volume']
        velocity = aggregated['sentiment_velocity']
        confidence = aggregated['confidence']

        # Signal generation logic
        signal = None

        # Thresholds (can be optimized)
        high_sentiment_threshold = 0.7
        low_sentiment_threshold = 0.3
        min_volume_threshold = 10
        min_confidence_threshold = 0.3

        if (confidence >= min_confidence_threshold and
            volume >= min_volume_threshold):

            if score >= high_sentiment_threshold and velocity > 0:
                signal = {
                    'action': 'BUY',
                    'signal_strength': score,
                    'confidence': confidence,
                    'reason': 'High positive sentiment with increasing momentum',
                    'sentiment_data': sentiment_data
                }

            elif score <= low_sentiment_threshold and velocity < 0:
                signal = {
                    'action': 'SELL',
                    'signal_strength': 1 - score,  # Invert for sell signal
                    'confidence': confidence,
                    'reason': 'High negative sentiment with decreasing momentum',
                    'sentiment_data': sentiment_data
                }

        return signal


class StockTwitsAPI:
    """
    StockTwits API integration for sentiment analysis
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.base_url = "https://api.stocktwits.com/api/2"

    def get_symbol_sentiment(self, symbol: str, lookback_hours: int = 24) -> Dict:
        """
        Get sentiment data from StockTwits
        """

        # StockTwits streams endpoint
        url = f"{self.base_url}/streams/symbol/{symbol}.json"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            messages = data.get('messages', [])

            if not messages:
                return {'sentiment_score': 0.5, 'message_volume': 0}

            # Analyze sentiment
            sentiments = []
            bullish_count = 0
            bearish_count = 0

            for msg in messages:
                # StockTwits provides sentiment labels
                if msg.get('entities', {}).get('sentiment'):
                    sentiment_label = msg['entities']['sentiment']['basic']
                    if sentiment_label == 'Bullish':
                        sentiments.append(1.0)
                        bullish_count += 1
                    elif sentiment_label == 'Bearish':
                        sentiments.append(0.0)
                        bearish_count += 1
                else:
                    # Use TextBlob for unlabeled messages
                    body = msg.get('body', '')
                    blob = TextBlob(body)
                    # Convert polarity (-1 to 1) to 0-1 scale
                    sentiment_score = (blob.sentiment.polarity + 1) / 2
                    sentiments.append(sentiment_score)

            overall_sentiment = np.mean(sentiments) if sentiments else 0.5

            return {
                'sentiment_score': overall_sentiment,
                'message_volume': len(messages),
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'raw_messages': messages[:10]  # Store sample for analysis
            }

        except Exception as e:
            print(f"StockTwits API error: {e}")
            return {'sentiment_score': 0.5, 'message_volume': 0}


class RedditAPI:
    """
    Reddit API integration for sentiment analysis
    """

    def __init__(self, config: Dict):
        import praw

        self.reddit = praw.Reddit(
            client_id=config.get('client_id'),
            client_secret=config.get('client_secret'),
            user_agent=config.get('user_agent', 'Quant Trading Bot 1.0')
        )

        self.target_subreddits = [
            'wallstreetbets',
            'pennystocks',
            'SecurityAnalysis',
            'investing',
            'stocks'
        ]

    def get_symbol_sentiment(self, symbol: str, lookback_hours: int = 24) -> Dict:
        """
        Search for symbol mentions across relevant subreddits
        """

        all_posts = []
        all_comments = []

        # Search in each subreddit
        for subreddit_name in self.target_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search recent posts
                search_query = f"${symbol} OR {symbol}"
                posts = subreddit.search(search_query, time_filter='day', limit=50)

                for post in posts:
                    if self._is_recent(post.created_utc, lookback_hours):
                        all_posts.append({
                            'title': post.title,
                            'body': post.selftext,
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'created_utc': post.created_utc,
                            'subreddit': subreddit_name
                        })

                        # Get top comments
                        post.comments.replace_more(limit=0)
                        for comment in post.comments[:5]:  # Top 5 comments
                            all_comments.append({
                                'body': comment.body,
                                'score': comment.score,
                                'created_utc': comment.created_utc
                            })

            except Exception as e:
                print(f"Reddit subreddit {subreddit_name} error: {e}")
                continue

        # Analyze sentiment
        if not all_posts and not all_comments:
            return {'sentiment_score': 0.5, 'message_volume': 0}

        # Combine posts and comments for sentiment analysis
        all_text = []

        for post in all_posts:
            text = f"{post['title']} {post['body']}"
            all_text.append({
                'text': text,
                'weight': min(post['score'] / 10, 3.0)  # Weight by upvotes, cap at 3x
            })

        for comment in all_comments:
            all_text.append({
                'text': comment['body'],
                'weight': min(comment['score'] / 5, 2.0)  # Weight by score, cap at 2x
            })

        # Calculate weighted sentiment
        weighted_sentiments = []

        for item in all_text:
            blob = TextBlob(item['text'])
            sentiment = (blob.sentiment.polarity + 1) / 2  # Convert to 0-1
            weight = max(item['weight'], 0.1)  # Minimum weight of 0.1
            weighted_sentiments.extend([sentiment] * int(weight * 10))

        overall_sentiment = np.mean(weighted_sentiments) if weighted_sentiments else 0.5

        return {
            'sentiment_score': overall_sentiment,
            'message_volume': len(all_posts) + len(all_comments),
            'posts_found': len(all_posts),
            'comments_found': len(all_comments),
            'subreddits_searched': self.target_subreddits
        }

    def _is_recent(self, created_utc: float, lookback_hours: int) -> bool:
        """Check if post/comment is within lookback period"""
        cutoff_time = time.time() - (lookback_hours * 3600)
        return created_utc >= cutoff_time


class TwitterAPI:
    """
    Twitter API v2 integration for sentiment analysis
    """

    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"

    def get_symbol_sentiment(self, symbol: str, lookback_hours: int = 24) -> Dict:
        """
        Search for symbol mentions on Twitter
        """

        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }

        # Search query
        query = f"${symbol} OR #{symbol} -is:retweet lang:en"

        # Calculate start time
        start_time = (datetime.now() - timedelta(hours=lookback_hours)).isoformat() + "Z"

        params = {
            'query': query,
            'start_time': start_time,
            'max_results': 100,
            'tweet.fields': 'created_at,public_metrics,context_annotations'
        }

        try:
            response = requests.get(
                f"{self.base_url}/tweets/search/recent",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()

            tweets = data.get('data', [])

            if not tweets:
                return {'sentiment_score': 0.5, 'message_volume': 0}

            # Analyze sentiment with engagement weighting
            weighted_sentiments = []

            for tweet in tweets:
                text = tweet.get('text', '')
                blob = TextBlob(text)
                sentiment = (blob.sentiment.polarity + 1) / 2

                # Weight by engagement (likes + retweets + replies)
                public_metrics = tweet.get('public_metrics', {})
                engagement = (
                    public_metrics.get('like_count', 0) +
                    public_metrics.get('retweet_count', 0) * 2 +  # Retweets worth more
                    public_metrics.get('reply_count', 0)
                )

                weight = min(engagement / 10, 5.0)  # Cap weight at 5x
                weight = max(weight, 0.1)  # Minimum weight

                weighted_sentiments.extend([sentiment] * int(weight * 10))

            overall_sentiment = np.mean(weighted_sentiments) if weighted_sentiments else 0.5

            return {
                'sentiment_score': overall_sentiment,
                'message_volume': len(tweets),
                'total_engagement': sum(
                    tweet.get('public_metrics', {}).get('like_count', 0) +
                    tweet.get('public_metrics', {}).get('retweet_count', 0) +
                    tweet.get('public_metrics', {}).get('reply_count', 0)
                    for tweet in tweets
                )
            }

        except Exception as e:
            print(f"Twitter API error: {e}")
            return {'sentiment_score': 0.5, 'message_volume': 0}


# Ejemplo de uso
def example_sentiment_analysis():
    """
    Ejemplo completo de sentiment analysis para small caps
    """

    # Configurar APIs (usar tus propias keys)
    config = {
        'stocktwits_token': 'your_stocktwits_token',
        'reddit_config': {
            'client_id': 'your_reddit_client_id',
            'client_secret': 'your_reddit_client_secret',
            'user_agent': 'Quant Trading Bot 1.0'
        },
        'twitter_bearer_token': 'your_twitter_bearer_token'
    }

    # Initialize analyzer
    analyzer = SocialSentimentAnalyzer(config)

    # Analyze sentiment for a small cap
    symbol = "AAPL"  # Replace with actual small cap
    sentiment_data = analyzer.get_comprehensive_sentiment(symbol, lookback_hours=12)

    print(f"Sentiment Analysis for ${symbol}")
    print(f"Aggregated Score: {sentiment_data['aggregated']['aggregated_score']:.3f}")
    print(f"Volume: {sentiment_data['aggregated']['total_volume']}")
    print(f"Velocity: {sentiment_data['aggregated']['sentiment_velocity']:.3f}")
    print(f"Confidence: {sentiment_data['aggregated']['confidence']:.3f}")

    # Generate trading signal
    signal = analyzer.generate_sentiment_signal(symbol)
    if signal:
        print(f"\nTrading Signal: {signal['action']}")
        print(f"Strength: {signal['signal_strength']:.3f}")
        print(f"Reason: {signal['reason']}")
    else:
        print("\nNo trading signal generated")

    return sentiment_data

if __name__ == "__main__":
    # Run example (requires API keys)
    # sentiment_data = example_sentiment_analysis()
    print("Social Sentiment Analyzer initialized. Configure API keys to run.")
```

### 2. SEC Filings Monitor

```python
import requests
import pandas as pd
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import re


class SECFilingsMonitor:
    """
    Monitor SEC filings para small caps con automated analysis

    Key filings para small caps:
    - 8-K: Current events (often price-moving)
    - 10-Q/K: Quarterly/Annual reports
    - 13D/G: Beneficial ownership (5%+ positions)
    - 4: Insider transactions
    - S-3: Shelf registrations (dilution risk)
    """

    def __init__(self):
        self.base_url = "https://www.sec.gov/Archives/edgar"
        self.headers = {
            'User-Agent': 'QuantTrading research@example.com'  # SEC requires identification
        }

        # Filing types que típicamente mueven precios en small caps
        self.price_moving_filings = {
            '8-K': 'Current Report - often contains material news',
            '8-K/A': 'Amended Current Report',
            '10-Q': 'Quarterly Report',
            '10-K': 'Annual Report',
            '13D': 'Beneficial Ownership Report (>5%)',
            '13G': 'Beneficial Ownership Report (passive)',
            'SC 13D': 'Schedule 13D',
            'SC 13G': 'Schedule 13G',
            '4': 'Insider Trading Report',
            'S-3': 'Registration Statement (dilution risk)',
            'S-1': 'Registration Statement',
            'DEF 14A': 'Proxy Statement'
        }

    def get_recent_filings(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """
        Obtiene recent filings para un símbolo específico
        """

        # Get CIK (Central Index Key) for the symbol
        cik = self._get_cik_for_symbol(symbol)
        if not cik:
            return []

        # Search for recent filings
        filings = self._search_company_filings(cik, days_back)

        # Filter for price-moving filings
        relevant_filings = []
        for filing in filings:
            if filing['form_type'] in self.price_moving_filings:
                # Add detailed analysis
                filing['analysis'] = self._analyze_filing(filing)
                filing['price_impact_probability'] = self._estimate_price_impact(filing)
                relevant_filings.append(filing)

        return sorted(relevant_filings,
                     key=lambda x: x['price_impact_probability'],
                     reverse=True)

    def _get_cik_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Get CIK number for stock symbol using SEC company tickers API
        """
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            data = response.json()

            for entry in data.values():
                if entry['ticker'].upper() == symbol.upper():
                    # Return CIK with leading zeros (10 digits)
                    return str(entry['cik_str']).zfill(10)

            return None

        except Exception as e:
            print(f"Error getting CIK for {symbol}: {e}")
            return None

    def _search_company_filings(self, cik: str, days_back: int) -> List[Dict]:
        """
        Search company filings usando SEC EDGAR API
        """
        try:
            # Company facts endpoint provides recent filings
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            recent_filings = data.get('filings', {}).get('recent', {})

            filings = []
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Parse filings data
            form_types = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])

            for i, (form_type, filing_date, accession) in enumerate(
                zip(form_types, filing_dates, accession_numbers)
            ):
                filing_datetime = datetime.strptime(filing_date, '%Y-%m-%d')

                if filing_datetime >= cutoff_date:
                    filings.append({
                        'form_type': form_type,
                        'filing_date': filing_date,
                        'accession_number': accession,
                        'cik': cik,
                        'filing_url': f"{self.base_url}/data/{cik.replace('-', '')}/{accession.replace('-', '')}"
                    })

            return filings

        except Exception as e:
            print(f"Error searching filings for CIK {cik}: {e}")
            return []

    def _analyze_filing(self, filing: Dict) -> Dict:
        """
        Analyze filing content para extract key information
        """
        analysis = {
            'form_type': filing['form_type'],
            'urgency': 'medium',
            'key_events': [],
            'dilution_risk': False,
            'insider_activity': False
        }

        # Form-specific analysis
        if filing['form_type'] == '8-K':
            analysis.update(self._analyze_8k_filing(filing))
        elif filing['form_type'] in ['13D', '13G', 'SC 13D', 'SC 13G']:
            analysis.update(self._analyze_13d_filing(filing))
        elif filing['form_type'] == '4':
            analysis.update(self._analyze_form4_filing(filing))
        elif filing['form_type'] in ['S-3', 'S-1']:
            analysis.update(self._analyze_registration_filing(filing))

        return analysis

    def _analyze_8k_filing(self, filing: Dict) -> Dict:
        """
        Analyze 8-K filings (current reports) - often most price-sensitive
        """

        # 8-K items that typically move stock prices
        price_moving_items = {
            '1.01': 'Entry into Material Agreement',
            '1.02': 'Termination of Material Agreement',
            '2.01': 'Completion of Acquisition or Disposition',
            '2.02': 'Results of Operations and Financial Condition',
            '3.01': 'Notice of Delisting',
            '3.02': 'Unregistered Sales of Equity Securities',
            '5.02': 'Departure of Directors or Officers',
            '8.01': 'Other Events',
            '9.01': 'Financial Statements and Exhibits'
        }

        return {
            'urgency': 'high',  # 8-K filings are typically urgent
            'key_events': ['Material event reported'],
            'potential_items': list(price_moving_items.values())
        }

    def _analyze_13d_filing(self, filing: Dict) -> Dict:
        """
        Analyze 13D/G filings (beneficial ownership)
        """
        return {
            'urgency': 'high',
            'key_events': ['Large position disclosure (>5%)'],
            'insider_activity': True,
            'note': '13D = active investor, 13G = passive investor'
        }

    def _analyze_form4_filing(self, filing: Dict) -> Dict:
        """
        Analyze Form 4 (insider transactions)
        """
        return {
            'urgency': 'medium',
            'key_events': ['Insider buying/selling activity'],
            'insider_activity': True
        }

    def _analyze_registration_filing(self, filing: Dict) -> Dict:
        """
        Analyze S-3/S-1 registration statements (potential dilution)
        """
        return {
            'urgency': 'medium',
            'key_events': ['Securities registration - potential dilution'],
            'dilution_risk': True
        }

    def _estimate_price_impact(self, filing: Dict) -> float:
        """
        Estimate probability of price impact (0-1 score)
        """

        base_scores = {
            '8-K': 0.8,     # High probability of price movement
            '13D': 0.9,     # Very high - large investor disclosure
            '13G': 0.6,     # Medium-high - passive position
            '4': 0.4,       # Medium - insider trading
            'S-3': 0.7,     # High - dilution risk
            'S-1': 0.8,     # High - new issuance
            '10-Q': 0.5,    # Medium - earnings
            '10-K': 0.6     # Medium-high - annual results
        }

        base_score = base_scores.get(filing['form_type'], 0.3)

        # Adjust based on analysis
        analysis = filing.get('analysis', {})

        if analysis.get('urgency') == 'high':
            base_score *= 1.2
        elif analysis.get('urgency') == 'low':
            base_score *= 0.8

        if analysis.get('dilution_risk'):
            base_score *= 1.1  # Slightly higher impact for dilution

        return min(base_score, 1.0)

    def generate_filing_alert(self, symbol: str) -> Optional[Dict]:
        """
        Generate trading alert basado en recent SEC filings
        """

        recent_filings = self.get_recent_filings(symbol, days_back=3)

        if not recent_filings:
            return None

        # Focus on highest probability filings
        high_impact_filings = [
            f for f in recent_filings
            if f['price_impact_probability'] >= 0.6
        ]

        if not high_impact_filings:
            return None

        most_recent = high_impact_filings[0]

        alert = {
            'symbol': symbol,
            'alert_type': 'SEC_FILING',
            'urgency': most_recent['analysis']['urgency'],
            'filing_type': most_recent['form_type'],
            'filing_date': most_recent['filing_date'],
            'price_impact_probability': most_recent['price_impact_probability'],
            'key_events': most_recent['analysis']['key_events'],
            'action_required': self._determine_action(most_recent),
            'filing_url': most_recent['filing_url']
        }

        return alert

    def _determine_action(self, filing: Dict) -> str:
        """
        Determine recommended action basado en filing type
        """

        form_type = filing['form_type']
        analysis = filing['analysis']

        if form_type == '8-K':
            return "MONITOR_CLOSELY - Review 8-K content for material events"
        elif form_type in ['13D', 'SC 13D']:
            return "BULLISH_BIAS - Large investor taking active position"
        elif form_type in ['13G', 'SC 13G']:
            return "NEUTRAL_TO_BULLISH - Large passive position disclosed"
        elif form_type in ['S-3', 'S-1'] or analysis.get('dilution_risk'):
            return "BEARISH_BIAS - Potential dilution from new shares"
        elif form_type == '4' and analysis.get('insider_activity'):
            return "ANALYZE_INSIDER_TRADES - Check if buying or selling"
        else:
            return "MONITOR - Review filing for material information"


# Ejemplo de uso
def example_sec_monitor():
    """
    Ejemplo de monitoring de SEC filings
    """

    monitor = SECFilingsMonitor()

    # Monitor filings for a small cap
    symbol = "AAPL"  # Replace with actual small cap

    # Get recent filings
    filings = monitor.get_recent_filings(symbol, days_back=7)

    print(f"Recent SEC Filings for ${symbol}:")
    for filing in filings:
        print(f"\nForm: {filing['form_type']}")
        print(f"Date: {filing['filing_date']}")
        print(f"Impact Probability: {filing['price_impact_probability']:.1%}")
        print(f"Events: {', '.join(filing['analysis']['key_events'])}")

    # Generate alert
    alert = monitor.generate_filing_alert(symbol)
    if alert:
        print(f"\n🚨 SEC FILING ALERT for ${symbol}")
        print(f"Type: {alert['filing_type']}")
        print(f"Urgency: {alert['urgency']}")
        print(f"Action: {alert['action_required']}")

    return filings

if __name__ == "__main__":
    # Run example
    print("SEC Filings Monitor initialized.")
    # filings = example_sec_monitor()
```

### 3. Integrated Alternative Data Signal Generator

```python
class AlternativeDataSignalGenerator:
    """
    Integrated signal generator que combina múltiples fuentes de alternative data
    """

    def __init__(self, config: Dict):
        self.sentiment_analyzer = SocialSentimentAnalyzer(config['sentiment'])
        self.sec_monitor = SECFilingsMonitor()

        # Signal weights
        self.weights = {
            'sentiment': 0.4,
            'sec_filings': 0.35,
            'options_flow': 0.25  # Placeholder for future implementation
        }

    def generate_comprehensive_signal(self, symbol: str) -> Optional[Dict]:
        """
        Generate comprehensive trading signal from all alternative data sources
        """

        signals = {}

        # 1. Social sentiment signal
        sentiment_signal = self.sentiment_analyzer.generate_sentiment_signal(symbol)
        if sentiment_signal:
            signals['sentiment'] = sentiment_signal

        # 2. SEC filings signal
        sec_alert = self.sec_monitor.generate_filing_alert(symbol)
        if sec_alert:
            signals['sec_filings'] = self._convert_sec_alert_to_signal(sec_alert)

        # 3. Combine signals if multiple sources
        if len(signals) == 0:
            return None
        elif len(signals) == 1:
            # Single source signal
            return list(signals.values())[0]
        else:
            # Multi-source signal combination
            return self._combine_alternative_signals(signals, symbol)

    def _convert_sec_alert_to_signal(self, alert: Dict) -> Dict:
        """
        Convert SEC alert to trading signal format
        """

        action_mapping = {
            'BULLISH_BIAS': 'BUY',
            'BEARISH_BIAS': 'SELL',
            'NEUTRAL_TO_BULLISH': 'BUY',
            'MONITOR': 'HOLD'
        }

        action_required = alert['action_required']
        action = 'HOLD'

        for bias, trading_action in action_mapping.items():
            if bias in action_required:
                action = trading_action
                break

        signal_strength = alert['price_impact_probability']

        return {
            'action': action,
            'signal_strength': signal_strength,
            'confidence': signal_strength,  # Use impact probability as confidence
            'reason': f"SEC Filing: {alert['filing_type']} - {alert['action_required']}",
            'source': 'sec_filings',
            'filing_data': alert
        }

    def _combine_alternative_signals(self, signals: Dict, symbol: str) -> Dict:
        """
        Combine multiple alternative data signals into one coherent signal
        """

        # Calculate weighted scores for each action
        buy_score = 0
        sell_score = 0
        total_weight = 0

        signal_details = []

        for source, signal in signals.items():
            weight = self.weights.get(source, 0.33)
            strength = signal['signal_strength']
            confidence = signal['confidence']

            effective_weight = weight * confidence

            if signal['action'] == 'BUY':
                buy_score += strength * effective_weight
            elif signal['action'] == 'SELL':
                sell_score += strength * effective_weight

            total_weight += effective_weight

            signal_details.append({
                'source': source,
                'action': signal['action'],
                'strength': strength,
                'confidence': confidence,
                'reason': signal['reason']
            })

        # Determine final action
        if total_weight == 0:
            return None

        if buy_score > sell_score and buy_score / total_weight > 0.6:
            final_action = 'BUY'
            final_strength = buy_score / total_weight
        elif sell_score > buy_score and sell_score / total_weight > 0.6:
            final_action = 'SELL'
            final_strength = sell_score / total_weight
        else:
            final_action = 'HOLD'
            final_strength = max(buy_score, sell_score) / total_weight

        return {
            'symbol': symbol,
            'action': final_action,
            'signal_strength': final_strength,
            'confidence': total_weight / sum(self.weights.values()),
            'reason': f"Multi-source alternative data signal ({len(signals)} sources)",
            'source_signals': signal_details,
            'timestamp': datetime.now()
        }


# Example implementation
def example_alternative_data_integration():
    """
    Ejemplo completo de alternative data integration
    """

    config = {
        'sentiment': {
            'stocktwits_token': 'your_token',
            'reddit_config': {
                'client_id': 'your_id',
                'client_secret': 'your_secret'
            },
            'twitter_bearer_token': 'your_token'
        }
    }

    generator = AlternativeDataSignalGenerator(config)

    # Test symbols (small caps)
    test_symbols = ['AAPL', 'TSLA', 'AMD']  # Replace with actual small caps

    for symbol in test_symbols:
        print(f"\n{'='*50}")
        print(f"Alternative Data Analysis for ${symbol}")
        print(f"{'='*50}")

        signal = generator.generate_comprehensive_signal(symbol)

        if signal:
            print(f"Action: {signal['action']}")
            print(f"Strength: {signal['signal_strength']:.3f}")
            print(f"Confidence: {signal['confidence']:.3f}")
            print(f"Reason: {signal['reason']}")

            if 'source_signals' in signal:
                print(f"\nContributing Sources:")
                for source_signal in signal['source_signals']:
                    print(f"  - {source_signal['source']}: {source_signal['action']} "
                          f"(strength: {source_signal['strength']:.2f})")
        else:
            print("No alternative data signals found")

if __name__ == "__main__":
    print("Alternative Data Integration Framework initialized.")
    # example_alternative_data_integration()
```

## Integration con Trading Strategies

Los datos alternativos se integran con nuestras estrategias existentes:

### 1. **Gap & Go con Sentiment Overlay**
```python
# Add sentiment confirmation to gap and go signals
if gap_signal and sentiment_score > 0.7:
    confidence_multiplier = 1.2  # Increase position size
elif gap_signal and sentiment_score < 0.3:
    confidence_multiplier = 0.7  # Reduce position size or skip
```

### 2. **SEC Filing-Based Entries**
```python
# Use SEC filings as catalyst for position initiation
if sec_filing_alert and alert['urgency'] == 'high':
    initiate_position_monitoring(symbol)
    # Wait for technical setup confirmation
```

### 3. **Multi-Data Confirmation System**
```python
# Require multiple data sources for high-conviction trades
if (technical_signal and
    sentiment_signal and
    no_negative_sec_filings):
    position_size *= 1.5  # High conviction trade
```

---

**Next Steps**:
- Configurar APIs para datos alternativos
- Integrar con [Strategy Development](../technical-practices/Strategy-Development.md)
- Crear dashboards para monitoreo en tiempo real
- Backtest strategies con alternative data overlays

Esta framework de datos alternativos proporciona **ventajas informacionales críticas** que son especialmente valiosas en el trading de small caps donde la información asimétrica puede generar alpha significativo.