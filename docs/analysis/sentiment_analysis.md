> 🇪🇸 [Leer en Español](sentiment_analysis.es.md) | 🇺🇸 **English**

# Sentiment Analysis in Financial Markets

## Introduction

Sentiment analysis quantifies the emotions and opinions expressed in text, providing an additional dimension for market analysis. In trading, sentiment can anticipate price movements before they are reflected in traditional technical data.

## Core Concepts

### Why Does Sentiment Analysis Work?

**Psychological Impact on Markets:**
- News directly influences investment decisions
- Retail sentiment can create momentum in small caps
- Social media amplifies the impact of sentiment
- Institutional algorithms now incorporate sentiment data

**Sentiment Data Sources:**
- Financial news (Bloomberg, Reuters, FinViz)
- Social media (Twitter, Reddit, StockTwits)
- Analyst reports
- Earnings call transcripts
- Specialized investment forums

## Implementation with VADER Sentiment

VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically designed to analyze sentiment in social media texts and news.

### Base Analysis Framework

```python
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
import string
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fake_useragent import UserAgent
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import yfinance as yf
from warnings import filterwarnings
filterwarnings("ignore")

# Download required resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class SentimentAnalyzer:
    """
    Sentiment analyzer for financial markets
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stemmer = nltk.stem.PorterStemmer()
        
        # Financial market-specific words
        self.financial_keywords = {
            'bullish': ['bull', 'bullish', 'rally', 'moon', 'rocket', 'pump', 'surge', 'soar'],
            'bearish': ['bear', 'bearish', 'crash', 'dump', 'plunge', 'tank', 'drop', 'fall'],
            'neutral': ['hold', 'sideways', 'flat', 'consolidate', 'range']
        }
    
    def preprocess_text(self, text, advanced=True):
        """
        Preprocess text for sentiment analysis
        
        Parameters
        ----------
        text : str
            Original text
        advanced : bool
            Whether to apply advanced preprocessing
            
        Returns
        -------
        str
            Processed text
        """
        if not advanced:
            return text.lower().strip()
        
        # 1. Tokenization
        tokens = nltk.tokenize.word_tokenize(text.lower())
        
        # 2. Lemmatization (convert to base form)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # 3. Stemming (reduce to root)
        stemmed_tokens = [self.stemmer.stem(token) for token in lemmatized_tokens]
        
        # 4. Remove stop words
        filtered_tokens = [token for token in stemmed_tokens if token not in self.stop_words]
        
        # 5. Normalization (remove punctuation)
        normalized_tokens = [token for token in filtered_tokens if token not in string.punctuation]
        
        # 6. Reassemble processed text
        processed_text = " ".join(normalized_tokens)
        
        return processed_text
    
    def analyze_sentiment(self, text, method='vader'):
        """
        Analyze sentiment of a text
        
        Parameters
        ----------
        text : str
            Text to analyze
        method : str
            Analysis method ('vader', 'textblob', 'both')
            
        Returns
        -------
        dict
            Sentiment scores
        """
        results = {}
        
        if method in ['vader', 'both']:
            # VADER Analysis
            vader_scores = self.vader.polarity_scores(text)
            results['vader'] = {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'classification': 'positive' if vader_scores['compound'] >= 0.05 else 'negative' if vader_scores['compound'] <= -0.05 else 'neutral'
            }
        
        if method in ['textblob', 'both']:
            # TextBlob Analysis
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'classification': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
            }
        
        # Financial keyword analysis
        results['financial_sentiment'] = self.analyze_financial_keywords(text)
        
        return results
    
    def analyze_financial_keywords(self, text):
        """
        Analyze financial market-specific keywords
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in self.financial_keywords['bullish'] if word in text_lower)
        bearish_count = sum(1 for word in self.financial_keywords['bearish'] if word in text_lower)
        neutral_count = sum(1 for word in self.financial_keywords['neutral'] if word in text_lower)
        
        total_keywords = bullish_count + bearish_count + neutral_count
        
        if total_keywords == 0:
            return {'score': 0, 'classification': 'neutral', 'keywords_found': 0}
        
        # Score based on proportion of bullish vs bearish words
        score = (bullish_count - bearish_count) / total_keywords
        
        if score > 0.2:
            classification = 'bullish'
        elif score < -0.2:
            classification = 'bearish'
        else:
            classification = 'neutral'
        
        return {
            'score': score,
            'classification': classification,
            'keywords_found': total_keywords,
            'bullish_words': bullish_count,
            'bearish_words': bearish_count
        }
    
    def batch_analyze(self, texts, preprocess=True):
        """
        Analyze sentiment of multiple texts
        """
        results = []
        
        for text in texts:
            if preprocess:
                processed_text = self.preprocess_text(text, advanced=True)
                raw_sentiment = self.analyze_sentiment(text, method='both')
                processed_sentiment = self.analyze_sentiment(processed_text, method='both')
                
                results.append({
                    'original_text': text,
                    'processed_text': processed_text,
                    'raw_sentiment': raw_sentiment,
                    'processed_sentiment': processed_sentiment
                })
            else:
                sentiment = self.analyze_sentiment(text, method='both')
                results.append({
                    'text': text,
                    'sentiment': sentiment
                })
        
        return results

class NewsScraperFinViz:
    """
    FinViz news scraper for sentiment analysis
    """
    
    def __init__(self):
        self.base_url = "https://finviz.com/quote.ashx?t={}&p=d"
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def scrape_news(self, tickers, max_retries=3):
        """
        Extract news from FinViz for multiple tickers
        
        Parameters
        ----------
        tickers : list
            List of stock symbols
        max_retries : int
            Maximum number of attempts per ticker
            
        Returns
        -------
        pd.DataFrame
            DataFrame with news and sentiment
        """
        news_data = []
        
        for ticker in tickers:
            print(f"Scraping news for {ticker}...")
            
            for attempt in range(max_retries):
                try:
                    # Random user agent to avoid blocks
                    ua = UserAgent()
                    headers = {"User-Agent": str(ua.chrome)}
                    
                    # Make request
                    response = requests.get(
                        self.base_url.format(ticker), 
                        headers=headers,
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    # Parse HTML
                    soup = BeautifulSoup(response.content, "html.parser")
                    news_table = soup.find(id="news-table")
                    
                    if news_table is None:
                        print(f"No news table found for {ticker}")
                        break
                    
                    # Extract individual news items
                    news_rows = news_table.findAll("tr")
                    
                    for row in news_rows:
                        try:
                            # Extract headline
                            news_link = row.find("a", class_="tab-link-news")
                            if news_link is None:
                                continue
                            
                            headline = news_link.text.strip()
                            
                            # Extract date and time
                            time_data = row.find("td").text.replace("\\n", "").strip().split()
                            
                            if len(time_data) == 2:
                                date_str = time_data[0]
                                time_str = time_data[1]
                                
                                # Handle "Today"
                                if date_str.lower() == "today":
                                    date_str = datetime.now().strftime("%b-%d-%y")
                                    
                            elif len(time_data) == 1:
                                # Time only, use current date
                                time_str = time_data[0]
                                date_str = datetime.now().strftime("%b-%d-%y")
                            else:
                                continue
                            
                            # Convert date
                            try:
                                news_date = datetime.strptime(date_str, "%b-%d-%y")
                            except:
                                news_date = datetime.now()
                            
                            # Analyze sentiment
                            sentiment_result = self.sentiment_analyzer.analyze_sentiment(headline, method='both')
                            
                            news_data.append({
                                'ticker': ticker,
                                'date': news_date,
                                'time': time_str,
                                'headline': headline,
                                'vader_compound': sentiment_result['vader']['compound'],
                                'vader_classification': sentiment_result['vader']['classification'],
                                'textblob_polarity': sentiment_result['textblob']['polarity'],
                                'financial_sentiment': sentiment_result['financial_sentiment']['score'],
                                'financial_classification': sentiment_result['financial_sentiment']['classification'],
                                'keywords_found': sentiment_result['financial_sentiment']['keywords_found']
                            })
                            
                        except Exception as e:
                            print(f"Error processing news row for {ticker}: {e}")
                            continue
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                    if attempt == max_retries - 1:
                        print(f"Failed to scrape {ticker} after {max_retries} attempts")
        
        # Convert to DataFrame
        if news_data:
            df = pd.DataFrame(news_data)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            return pd.DataFrame()

def sentiment_trading_strategy(price_data, sentiment_data, 
                             sentiment_threshold=0.1, 
                             lookback_days=3):
    """
    Trading strategy based on sentiment analysis
    
    Parameters
    ----------
    price_data : pd.DataFrame
        Historical price data
    sentiment_data : pd.DataFrame
        Sentiment data with dates
    sentiment_threshold : float
        Threshold for generating signals
    lookback_days : int
        Days to look back for aggregating sentiment
    """
    # Aggregate sentiment by day
    daily_sentiment = sentiment_data.groupby('date').agg({
        'vader_compound': 'mean',
        'financial_sentiment': 'mean',
        'keywords_found': 'sum'
    }).reset_index()
    
    # Create trading signals
    signals = pd.DataFrame(index=price_data.index)
    signals['price'] = price_data['Close']
    signals['signal'] = 0
    signals['sentiment_score'] = np.nan
    signals['confidence'] = 0
    
    for i, date in enumerate(price_data.index):
        # Look for sentiment in the last N days
        start_date = date - timedelta(days=lookback_days)
        end_date = date
        
        period_sentiment = daily_sentiment[
            (daily_sentiment['date'] >= start_date) & 
            (daily_sentiment['date'] <= end_date)
        ]
        
        if len(period_sentiment) > 0:
            # Calculate weighted average score (more weight to recent days)
            weights = np.linspace(0.5, 1.0, len(period_sentiment))
            
            avg_vader = np.average(period_sentiment['vader_compound'], weights=weights)
            avg_financial = np.average(period_sentiment['financial_sentiment'], weights=weights)
            total_keywords = period_sentiment['keywords_found'].sum()
            
            # Combined score
            combined_score = (avg_vader * 0.6 + avg_financial * 0.4)
            
            # Adjust by news volume (more news = more confidence)
            confidence = min(total_keywords / 10.0, 1.0)  # Normalize to 0-1
            
            signals.loc[date, 'sentiment_score'] = combined_score
            signals.loc[date, 'confidence'] = confidence
            
            # Generate signals only with minimum confidence
            if confidence > 0.3:
                if combined_score > sentiment_threshold:
                    signals.loc[date, 'signal'] = 1  # Buy
                elif combined_score < -sentiment_threshold:
                    signals.loc[date, 'signal'] = -1  # Sell
    
    return signals

def analyze_sentiment_correlation(price_data, sentiment_data, ticker):
    """
    Analyze correlation between sentiment and price movements
    """
    # Prepare daily data
    daily_sentiment = sentiment_data.groupby('date').agg({
        'vader_compound': 'mean',
        'financial_sentiment': 'mean',
        'keywords_found': 'count'
    }).reset_index()
    
    # Add price returns
    price_returns = price_data['Close'].pct_change()
    daily_data = pd.DataFrame({
        'date': price_data.index,
        'return': price_returns.values,
        'price': price_data['Close'].values
    })
    
    # Combine data
    combined_data = daily_data.merge(daily_sentiment, on='date', how='inner')
    
    if len(combined_data) == 0:
        return {'error': 'No matching dates between price and sentiment data'}
    
    # Calculate correlations
    correlations = {
        'vader_sentiment_correlation': combined_data['vader_compound'].corr(combined_data['return']),
        'financial_sentiment_correlation': combined_data['financial_sentiment'].corr(combined_data['return']),
        'news_volume_correlation': combined_data['keywords_found'].corr(abs(combined_data['return'])),
    }
    
    # Lead/lag analysis
    lead_lag_analysis = {}
    for lag in range(-3, 4):  # -3 to +3 days
        if lag == 0:
            continue
        
        if lag > 0:
            # Sentiment predicts future returns
            shifted_returns = combined_data['return'].shift(-lag)
            lead_lag_analysis[f'sentiment_leads_{lag}d'] = combined_data['vader_compound'].corr(shifted_returns)
        else:
            # Returns predict future sentiment
            shifted_sentiment = combined_data['vader_compound'].shift(lag)
            lead_lag_analysis[f'price_leads_{abs(lag)}d'] = combined_data['return'].corr(shifted_sentiment)
    
    return {
        'correlations': correlations,
        'lead_lag_analysis': lead_lag_analysis,
        'data_points': len(combined_data),
        'date_range': f"{combined_data['date'].min()} to {combined_data['date'].max()}"
    }

def create_sentiment_dashboard(tickers, sentiment_data):
    """
    Create visual sentiment analysis dashboard
    """
    # Configure subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sentiment Score by Ticker
    daily_sentiment = sentiment_data.groupby(['ticker', 'date']).agg({
        'vader_compound': 'mean',
        'financial_sentiment': 'mean'
    }).reset_index()
    
    for ticker in tickers:
        ticker_data = daily_sentiment[daily_sentiment['ticker'] == ticker]
        axes[0, 0].plot(ticker_data['date'], ticker_data['vader_compound'], label=ticker, marker='o')
    
    axes[0, 0].set_title('VADER Sentiment Score Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sentiment Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Sentiment Distribution
    axes[0, 1].hist(sentiment_data['vader_compound'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution of Sentiment Scores')
    axes[0, 1].set_xlabel('VADER Compound Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sentiment by Ticker (Box plot)
    sentiment_by_ticker = [sentiment_data[sentiment_data['ticker'] == ticker]['vader_compound'] 
                          for ticker in tickers]
    axes[1, 0].boxplot(sentiment_by_ticker, labels=tickers)
    axes[1, 0].set_title('Sentiment Distribution by Ticker')
    axes[1, 0].set_ylabel('VADER Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 4. Keywords found per day
    keywords_by_date = sentiment_data.groupby('date')['keywords_found'].sum()
    axes[1, 1].plot(keywords_by_date.index, keywords_by_date.values, color='purple', linewidth=2)
    axes[1, 1].set_title('Financial Keywords Found Over Time')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Keywords Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Complete usage example
def sentiment_analysis_example():
    """
    Complete sentiment analysis example for trading
    """
    # Tickers to analyze
    tickers = ["AAPL", "TSLA", "NVDA", "AMZN"]
    
    print("=== FINANCIAL SENTIMENT ANALYSIS ===\\n")
    
    # 1. News scraping
    print("Extracting news...")
    scraper = NewsScraperFinViz()
    news_data = scraper.scrape_news(tickers)
    
    if news_data.empty:
        print("Could not extract news")
        return
    
    print(f"Extracted {len(news_data)} news items")
    
    # 2. Statistical analysis
    print(f"\\nGENERAL STATISTICS:")
    for ticker in tickers:
        ticker_news = news_data[news_data['ticker'] == ticker]
        if len(ticker_news) > 0:
            avg_sentiment = ticker_news['vader_compound'].mean()
            total_news = len(ticker_news)
            positive_news = (ticker_news['vader_compound'] > 0.05).sum()
            negative_news = (ticker_news['vader_compound'] < -0.05).sum()
            
            print(f"   {ticker}:")
            print(f"      Total News: {total_news}")
            print(f"      Average Sentiment: {avg_sentiment:.3f}")
            print(f"      Positive News: {positive_news} ({positive_news/total_news:.1%})")
            print(f"      Negative News: {negative_news} ({negative_news/total_news:.1%})")
    
    # 3. Correlation analysis with prices
    print(f"\\nCORRELATION ANALYSIS:")
    for ticker in tickers:
        try:
            # Get price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            price_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            
            ticker_sentiment = news_data[news_data['ticker'] == ticker]
            
            if len(ticker_sentiment) > 0 and len(price_data) > 0:
                correlation_analysis = analyze_sentiment_correlation(price_data, ticker_sentiment, ticker)
                
                if 'error' not in correlation_analysis:
                    print(f"   {ticker}:")
                    print(f"      Sentiment-Return Correlation: {correlation_analysis['correlations']['vader_sentiment_correlation']:.3f}")
                    print(f"      Data Points: {correlation_analysis['data_points']}")
        
        except Exception as e:
            print(f"   {ticker}: Error in analysis - {e}")
    
    # 4. Generate example strategy
    print(f"\\nSTRATEGY EXAMPLE:")
    ticker = "AAPL"  # Use Apple as example
    try:
        price_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        ticker_sentiment = news_data[news_data['ticker'] == ticker]
        
        if len(ticker_sentiment) > 0:
            strategy_signals = sentiment_trading_strategy(price_data, ticker_sentiment)
            
            total_signals = strategy_signals['signal'].abs().sum()
            buy_signals = (strategy_signals['signal'] == 1).sum()
            sell_signals = (strategy_signals['signal'] == -1).sum()
            avg_confidence = strategy_signals[strategy_signals['confidence'] > 0]['confidence'].mean()
            
            print(f"   Ticker: {ticker}")
            print(f"   Total Signals: {total_signals}")
            print(f"   Buy Signals: {buy_signals}")
            print(f"   Sell Signals: {sell_signals}")
            print(f"   Average Confidence: {avg_confidence:.1%}")
    
    except Exception as e:
        print(f"   Error generating strategy: {e}")
    
    # 5. Create visualization
    print(f"\\nGenerating dashboard...")
    try:
        create_sentiment_dashboard(tickers, news_data)
    except Exception as e:
        print(f"Error creating dashboard: {e}")
    
    return news_data

# Sentiment analysis for small caps
def small_cap_sentiment_strategy(ticker, sentiment_threshold=0.15):
    """
    Small cap-specific sentiment strategy
    """
    # Small caps are more sensitive to sentiment
    scraper = NewsScraperFinViz()
    sentiment_data = scraper.scrape_news([ticker])
    
    if sentiment_data.empty:
        return {'error': 'No sentiment data available'}
    
    # Get price data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    price_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Parameters adjusted for small caps
    signals = sentiment_trading_strategy(
        price_data, 
        sentiment_data,
        sentiment_threshold=sentiment_threshold,  # Higher threshold
        lookback_days=1  # Faster reaction
    )
    
    # Add small cap-specific filters
    signals['volume_filter'] = price_data['Volume'] > price_data['Volume'].rolling(20).mean()
    signals['volatility_filter'] = price_data['Close'].pct_change().rolling(5).std() > 0.02
    
    # Only generate signals when there is volume and volatility
    signals['final_signal'] = np.where(
        signals['volume_filter'] & signals['volatility_filter'],
        signals['signal'],
        0
    )
    
    return {
        'signals': signals,
        'sentiment_data': sentiment_data,
        'price_data': price_data
    }

if __name__ == "__main__":
    sentiment_analysis_example()
```

## Integration with Trading Strategies

### 1. Sentiment + Gap & Go
```python
def sentiment_gap_strategy(ticker, gap_threshold=0.03):
    """
    Combine sentiment analysis with Gap & Go strategy
    """
    # Get data
    scraper = NewsScraperFinViz()
    sentiment_data = scraper.scrape_news([ticker])
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    price_data = yf.download(ticker, start=start_date, end=end_date)
    
    signals = pd.DataFrame(index=price_data.index)
    signals['price'] = price_data['Close']
    signals['gap_pct'] = (price_data['Open'] / price_data['Close'].shift(1)) - 1
    signals['volume_ratio'] = price_data['Volume'] / price_data['Volume'].rolling(20).mean()
    signals['signal'] = 0
    
    # Get previous day sentiment
    for i, date in enumerate(price_data.index[1:], 1):
        prev_date = price_data.index[i-1]
        
        # Look for previous day sentiment
        day_sentiment = sentiment_data[
            sentiment_data['date'].dt.date == prev_date.date()
        ]
        
        if len(day_sentiment) > 0:
            avg_sentiment = day_sentiment['vader_compound'].mean()
            
            # Gap up with positive sentiment
            if (signals.loc[date, 'gap_pct'] > gap_threshold and 
                avg_sentiment > 0.1 and
                signals.loc[date, 'volume_ratio'] > 2):
                signals.loc[date, 'signal'] = 1
            
            # Gap down with very negative sentiment (potential reversal)
            elif (signals.loc[date, 'gap_pct'] < -gap_threshold and
                  avg_sentiment < -0.2 and
                  signals.loc[date, 'volume_ratio'] > 2):
                signals.loc[date, 'signal'] = 1  # Contrarian play
    
    return signals
```

### 2. Sentiment + VWAP
```python
def sentiment_vwap_strategy(ticker):
    """
    Combine sentiment with VWAP strategy
    """
    # Get intraday data if possible
    price_data = yf.download(ticker, period="5d", interval="1h")
    
    # Calculate VWAP
    price_data['vwap'] = (price_data['Close'] * price_data['Volume']).cumsum() / price_data['Volume'].cumsum()
    
    # Get sentiment
    scraper = NewsScraperFinViz()
    sentiment_data = scraper.scrape_news([ticker])
    
    # Generate signals
    signals = pd.DataFrame(index=price_data.index)
    signals['price'] = price_data['Close']
    signals['vwap'] = price_data['vwap']
    signals['signal'] = 0
    
    # Current day sentiment
    current_date = datetime.now().date()
    today_sentiment = sentiment_data[
        sentiment_data['date'].dt.date == current_date
    ]
    
    if len(today_sentiment) > 0:
        avg_sentiment = today_sentiment['vader_compound'].mean()
        
        for i, date in enumerate(price_data.index):
            # Long: price near VWAP + positive sentiment
            if (signals.loc[date, 'price'] > signals.loc[date, 'vwap'] * 0.999 and
                signals.loc[date, 'price'] < signals.loc[date, 'vwap'] * 1.001 and
                avg_sentiment > 0.05):
                signals.loc[date, 'signal'] = 1
            
            # Short: price rejected at VWAP + negative sentiment
            elif (signals.loc[date, 'price'] < signals.loc[date, 'vwap'] and
                  avg_sentiment < -0.05):
                signals.loc[date, 'signal'] = -1
    
    return signals
```

## Best Practices

### 1. Sentiment Data Validation
```python
def validate_sentiment_data(sentiment_df):
    """
    Validate sentiment data quality
    """
    validation_results = {
        'total_articles': len(sentiment_df),
        'date_range': (sentiment_df['date'].min(), sentiment_df['date'].max()),
        'sentiment_distribution': sentiment_df['vader_compound'].describe(),
        'missing_data': sentiment_df.isnull().sum(),
        'duplicate_headlines': sentiment_df['headline'].duplicated().sum()
    }
    
    # Detect potential issues
    warnings = []
    
    if validation_results['total_articles'] < 10:
        warnings.append("Too few news items for reliable analysis")
    
    if abs(sentiment_df['vader_compound'].mean()) > 0.5:
        warnings.append("Extremely biased sentiment")
    
    if validation_results['duplicate_headlines'] > len(sentiment_df) * 0.1:
        warnings.append("Many duplicate news items")
    
    validation_results['warnings'] = warnings
    
    return validation_results
```

### 2. Temporal Normalization
```python
def normalize_sentiment_by_time(sentiment_df, method='zscore'):
    """
    Normalize sentiment by time period
    """
    sentiment_df = sentiment_df.copy()
    
    if method == 'zscore':
        # Z-score normalization
        sentiment_df['normalized_sentiment'] = (
            sentiment_df['vader_compound'] - sentiment_df['vader_compound'].mean()
        ) / sentiment_df['vader_compound'].std()
    
    elif method == 'rolling_zscore':
        # Rolling z-score (30-day window)
        rolling_mean = sentiment_df['vader_compound'].rolling(30).mean()
        rolling_std = sentiment_df['vader_compound'].rolling(30).std()
        sentiment_df['normalized_sentiment'] = (
            sentiment_df['vader_compound'] - rolling_mean
        ) / rolling_std
    
    elif method == 'percentile':
        # Percentile ranking
        sentiment_df['normalized_sentiment'] = sentiment_df['vader_compound'].rank(pct=True)
    
    return sentiment_df
```

### 3. Quality Filters
```python
def apply_quality_filters(sentiment_df, min_keywords=1, confidence_threshold=0.5):
    """
    Apply quality filters to sentiment data
    """
    filtered_df = sentiment_df.copy()
    
    # Filter by financial keywords found
    filtered_df = filtered_df[filtered_df['keywords_found'] >= min_keywords]
    
    # Filter very short headlines (probably not informative)
    filtered_df = filtered_df[filtered_df['headline'].str.len() > 20]
    
    # Remove exact duplicates
    filtered_df = filtered_df.drop_duplicates(subset=['headline'])
    
    # Filter by classification confidence
    abs_sentiment = abs(filtered_df['vader_compound'])
    filtered_df = filtered_df[abs_sentiment > confidence_threshold * abs_sentiment.std()]
    
    return filtered_df
```

## Limitations and Considerations

### 1. Limitations of Sentiment Analysis
- **Sarcasm and context**: Models may not detect sarcasm
- **Financial jargon**: Sector-specific words may be misinterpreted
- **News volume**: Small caps may have few news articles
- **Timing**: The impact of sentiment can be immediate or delayed

### 2. Implementation Best Practices
```python
SENTIMENT_BEST_PRACTICES = {
    'data_quality': {
        'min_articles_per_day': 3,
        'max_sentiment_abs': 0.8,  # Avoid suspiciously extreme sentiments
        'min_headline_length': 20,
        'duplicate_threshold': 0.1
    },
    'trading_integration': {
        'sentiment_weight': 0.3,  # No more than 30% weight in decisions
        'confirmation_required': True,  # Confirm with technical indicators
        'volume_filter': True,  # Only trade with confirming volume
        'time_decay': 24  # Hours before sentiment loses relevance
    },
    'risk_management': {
        'max_position_sentiment': 0.05,  # Maximum 5% of capital in sentiment trades
        'stop_loss_tight': True,  # Tighter stops for sentiment trades
        'sentiment_correlation_limit': 0.7  # Avoid too much correlation with sentiment
    }
}
```

## Alternative Data Sources

### 1. Reddit/Twitter Integration
```python
def reddit_sentiment_analysis(ticker, subreddit='wallstreetbets'):
    """
    Placeholder for Reddit sentiment analysis
    (Requires Reddit API)
    """
    # Implementation requires praw library and API keys
    pass

def twitter_sentiment_analysis(ticker):
    """
    Placeholder for Twitter sentiment analysis
    (Requires Twitter API)
    """
    # Implementation requires tweepy library and API keys
    pass
```

### 2. StockTwits Integration
```python
def stocktwits_sentiment(ticker):
    """
    Placeholder for StockTwits sentiment
    (Requires StockTwits API)
    """
    pass
```

## Next Step

With Sentiment Analysis implemented, let's continue with [Fundamental Analysis](fundamental_analysis.md) to complete the quantitative tools arsenal.
