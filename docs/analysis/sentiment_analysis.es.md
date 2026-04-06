> 🇺🇸 [Read in English](sentiment_analysis.md) | 🇪🇸 **Español**

# Análisis de Sentimiento en Mercados Financieros

## Introducción

El análisis de sentimiento cuantifica las emociones y opiniones expresadas en texto, proporcionando una dimensión adicional para el análisis de mercados. En trading, el sentimiento puede anticipar movimientos de precios antes de que se reflejen en datos técnicos tradicionales.

## Conceptos Fundamentales

### ¿Por Qué Funciona el Análisis de Sentimiento?

**Impacto Psicológico en Mercados:**
- Las noticias influyen directamente en las decisiones de inversión
- El sentimiento retail puede crear momentum en small caps
- Las redes sociales amplifican el impacto del sentimiento
- Los algoritmos institucionales ahora incorporan datos de sentimiento

**Fuentes de Datos de Sentimiento:**
- Noticias financieras (Bloomberg, Reuters, FinViz)
- Redes sociales (Twitter, Reddit, StockTwits)
- Informes de analistas
- Transcripciones de earnings calls
- Foros de inversión especializados

## Implementación con VADER Sentiment

VADER (Valence Aware Dictionary and sEntiment Reasoner) está específicamente diseñado para analizar sentimiento en textos de redes sociales y noticias.

### Framework Base de Análisis

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

# Descargar recursos necesarios
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class SentimentAnalyzer:
    """
    Analizador de sentimiento para mercados financieros
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stemmer = nltk.stem.PorterStemmer()
        
        # Palabras específicas de mercados financieros
        self.financial_keywords = {
            'bullish': ['bull', 'bullish', 'rally', 'moon', 'rocket', 'pump', 'surge', 'soar'],
            'bearish': ['bear', 'bearish', 'crash', 'dump', 'plunge', 'tank', 'drop', 'fall'],
            'neutral': ['hold', 'sideways', 'flat', 'consolidate', 'range']
        }
    
    def preprocess_text(self, text, advanced=True):
        """
        Preprocesar texto para análisis de sentimiento
        
        Parámetros
        ----------
        text : str
            Texto original
        advanced : bool
            Si aplicar preprocesamiento avanzado
            
        Returns
        -------
        str
            Texto procesado
        """
        if not advanced:
            return text.lower().strip()
        
        # 1. Tokenización
        tokens = nltk.tokenize.word_tokenize(text.lower())
        
        # 2. Lematización (convertir a forma base)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # 3. Stemming (reducir a raíz)
        stemmed_tokens = [self.stemmer.stem(token) for token in lemmatized_tokens]
        
        # 4. Eliminar stop words
        filtered_tokens = [token for token in stemmed_tokens if token not in self.stop_words]
        
        # 5. Normalización (eliminar puntuación)
        normalized_tokens = [token for token in filtered_tokens if token not in string.punctuation]
        
        # 6. Reunir texto procesado
        processed_text = " ".join(normalized_tokens)
        
        return processed_text
    
    def analyze_sentiment(self, text, method='vader'):
        """
        Analizar sentimiento de un texto
        
        Parámetros
        ----------
        text : str
            Texto a analizar
        method : str
            Método de análisis ('vader', 'textblob', 'both')
            
        Returns
        -------
        dict
            Scores de sentimiento
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
        Analizar palabras clave específicas de mercados financieros
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in self.financial_keywords['bullish'] if word in text_lower)
        bearish_count = sum(1 for word in self.financial_keywords['bearish'] if word in text_lower)
        neutral_count = sum(1 for word in self.financial_keywords['neutral'] if word in text_lower)
        
        total_keywords = bullish_count + bearish_count + neutral_count
        
        if total_keywords == 0:
            return {'score': 0, 'classification': 'neutral', 'keywords_found': 0}
        
        # Score basado en proporción de palabras bullish vs bearish
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
        Analizar sentimiento de múltiples textos
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
    Scraper de noticias de FinViz para análisis de sentimiento
    """
    
    def __init__(self):
        self.base_url = "https://finviz.com/quote.ashx?t={}&p=d"
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def scrape_news(self, tickers, max_retries=3):
        """
        Extraer noticias de FinViz para múltiples tickers
        
        Parámetros
        ----------
        tickers : list
            Lista de símbolos de acciones
        max_retries : int
            Número máximo de intentos por ticker
            
        Returns
        -------
        pd.DataFrame
            DataFrame con noticias y sentimiento
        """
        news_data = []
        
        for ticker in tickers:
            print(f"Scraping news for {ticker}...")
            
            for attempt in range(max_retries):
                try:
                    # User agent aleatorio para evitar bloqueos
                    ua = UserAgent()
                    headers = {"User-Agent": str(ua.chrome)}
                    
                    # Realizar petición
                    response = requests.get(
                        self.base_url.format(ticker), 
                        headers=headers,
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    # Parsear HTML
                    soup = BeautifulSoup(response.content, "html.parser")
                    news_table = soup.find(id="news-table")
                    
                    if news_table is None:
                        print(f"No news table found for {ticker}")
                        break
                    
                    # Extraer noticias individuales
                    news_rows = news_table.findAll("tr")
                    
                    for row in news_rows:
                        try:
                            # Extraer titular
                            news_link = row.find("a", class_="tab-link-news")
                            if news_link is None:
                                continue
                            
                            headline = news_link.text.strip()
                            
                            # Extraer fecha y hora
                            time_data = row.find("td").text.replace("\n", "").strip().split()
                            
                            if len(time_data) == 2:
                                date_str = time_data[0]
                                time_str = time_data[1]
                                
                                # Manejar "Today"
                                if date_str.lower() == "today":
                                    date_str = datetime.now().strftime("%b-%d-%y")
                                    
                            elif len(time_data) == 1:
                                # Solo hora, usar fecha actual
                                time_str = time_data[0]
                                date_str = datetime.now().strftime("%b-%d-%y")
                            else:
                                continue
                            
                            # Convertir fecha
                            try:
                                news_date = datetime.strptime(date_str, "%b-%d-%y")
                            except:
                                news_date = datetime.now()
                            
                            # Analizar sentimiento
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
                    
                    break  # Éxito, salir del loop de reintentos
                    
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                    if attempt == max_retries - 1:
                        print(f"Failed to scrape {ticker} after {max_retries} attempts")
        
        # Convertir a DataFrame
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
    Estrategia de trading basada en análisis de sentimiento
    
    Parámetros
    ----------
    price_data : pd.DataFrame
        Datos de precios históricos
    sentiment_data : pd.DataFrame
        Datos de sentimiento con fechas
    sentiment_threshold : float
        Umbral para generar señales
    lookback_days : int
        Días hacia atrás para agregar sentimiento
    """
    # Agregar sentimiento por día
    daily_sentiment = sentiment_data.groupby('date').agg({
        'vader_compound': 'mean',
        'financial_sentiment': 'mean',
        'keywords_found': 'sum'
    }).reset_index()
    
    # Crear señales de trading
    signals = pd.DataFrame(index=price_data.index)
    signals['price'] = price_data['Close']
    signals['signal'] = 0
    signals['sentiment_score'] = np.nan
    signals['confidence'] = 0
    
    for i, date in enumerate(price_data.index):
        # Buscar sentimiento en los últimos N días
        start_date = date - timedelta(days=lookback_days)
        end_date = date
        
        period_sentiment = daily_sentiment[
            (daily_sentiment['date'] >= start_date) & 
            (daily_sentiment['date'] <= end_date)
        ]
        
        if len(period_sentiment) > 0:
            # Calcular score promedio ponderado (más peso a días recientes)
            weights = np.linspace(0.5, 1.0, len(period_sentiment))
            
            avg_vader = np.average(period_sentiment['vader_compound'], weights=weights)
            avg_financial = np.average(period_sentiment['financial_sentiment'], weights=weights)
            total_keywords = period_sentiment['keywords_found'].sum()
            
            # Score combinado
            combined_score = (avg_vader * 0.6 + avg_financial * 0.4)
            
            # Ajustar por cantidad de noticias (más noticias = más confianza)
            confidence = min(total_keywords / 10.0, 1.0)  # Normalizar a 0-1
            
            signals.loc[date, 'sentiment_score'] = combined_score
            signals.loc[date, 'confidence'] = confidence
            
            # Generar señales solo con confianza mínima
            if confidence > 0.3:
                if combined_score > sentiment_threshold:
                    signals.loc[date, 'signal'] = 1  # Comprar
                elif combined_score < -sentiment_threshold:
                    signals.loc[date, 'signal'] = -1  # Vender
    
    return signals

def analyze_sentiment_correlation(price_data, sentiment_data, ticker):
    """
    Analizar correlación entre sentimiento y movimientos de precios
    """
    # Preparar datos diarios
    daily_sentiment = sentiment_data.groupby('date').agg({
        'vader_compound': 'mean',
        'financial_sentiment': 'mean',
        'keywords_found': 'count'
    }).reset_index()
    
    # Agregar retornos de precios
    price_returns = price_data['Close'].pct_change()
    daily_data = pd.DataFrame({
        'date': price_data.index,
        'return': price_returns.values,
        'price': price_data['Close'].values
    })
    
    # Combinar datos
    combined_data = daily_data.merge(daily_sentiment, on='date', how='inner')
    
    if len(combined_data) == 0:
        return {'error': 'No matching dates between price and sentiment data'}
    
    # Calcular correlaciones
    correlations = {
        'vader_sentiment_correlation': combined_data['vader_compound'].corr(combined_data['return']),
        'financial_sentiment_correlation': combined_data['financial_sentiment'].corr(combined_data['return']),
        'news_volume_correlation': combined_data['keywords_found'].corr(abs(combined_data['return'])),
    }
    
    # Análisis de lead/lag
    lead_lag_analysis = {}
    for lag in range(-3, 4):  # -3 a +3 días
        if lag == 0:
            continue
        
        if lag > 0:
            # Sentimiento predice retornos futuros
            shifted_returns = combined_data['return'].shift(-lag)
            lead_lag_analysis[f'sentiment_leads_{lag}d'] = combined_data['vader_compound'].corr(shifted_returns)
        else:
            # Retornos predicen sentimiento futuro
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
    Crear dashboard visual de análisis de sentimiento
    """
    # Configurar subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sentiment Score por Ticker
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
    
    # 2. Distribución de Sentimiento
    axes[0, 1].hist(sentiment_data['vader_compound'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution of Sentiment Scores')
    axes[0, 1].set_xlabel('VADER Compound Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sentimiento por Ticker (Box plot)
    sentiment_by_ticker = [sentiment_data[sentiment_data['ticker'] == ticker]['vader_compound'] 
                          for ticker in tickers]
    axes[1, 0].boxplot(sentiment_by_ticker, labels=tickers)
    axes[1, 0].set_title('Sentiment Distribution by Ticker')
    axes[1, 0].set_ylabel('VADER Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 4. Keywords encontradas por día
    keywords_by_date = sentiment_data.groupby('date')['keywords_found'].sum()
    axes[1, 1].plot(keywords_by_date.index, keywords_by_date.values, color='purple', linewidth=2)
    axes[1, 1].set_title('Financial Keywords Found Over Time')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Keywords Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Ejemplo de uso completo
def sentiment_analysis_example():
    """
    Ejemplo completo de análisis de sentimiento para trading
    """
    # Tickers para analizar
    tickers = ["AAPL", "TSLA", "NVDA", "AMZN"]
    
    print("=== ANÁLISIS DE SENTIMIENTO FINANCIERO ===\n")
    
    # 1. Scraping de noticias
    print("📰 Extrayendo noticias...")
    scraper = NewsScraperFinViz()
    news_data = scraper.scrape_news(tickers)
    
    if news_data.empty:
        print("❌ No se pudieron extraer noticias")
        return
    
    print(f"✅ Extraídas {len(news_data)} noticias")
    
    # 2. Análisis estadístico
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    for ticker in tickers:
        ticker_news = news_data[news_data['ticker'] == ticker]
        if len(ticker_news) > 0:
            avg_sentiment = ticker_news['vader_compound'].mean()
            total_news = len(ticker_news)
            positive_news = (ticker_news['vader_compound'] > 0.05).sum()
            negative_news = (ticker_news['vader_compound'] < -0.05).sum()
            
            print(f"   {ticker}:")
            print(f"      Total Noticias: {total_news}")
            print(f"      Sentimiento Promedio: {avg_sentiment:.3f}")
            print(f"      Noticias Positivas: {positive_news} ({positive_news/total_news:.1%})")
            print(f"      Noticias Negativas: {negative_news} ({negative_news/total_news:.1%})")
    
    # 3. Análisis de correlación con precios
    print(f"\n🔍 ANÁLISIS DE CORRELACIÓN:")
    for ticker in tickers:
        try:
            # Obtener datos de precios
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            price_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            
            ticker_sentiment = news_data[news_data['ticker'] == ticker]
            
            if len(ticker_sentiment) > 0 and len(price_data) > 0:
                correlation_analysis = analyze_sentiment_correlation(price_data, ticker_sentiment, ticker)
                
                if 'error' not in correlation_analysis:
                    print(f"   {ticker}:")
                    print(f"      Correlación Sentimiento-Retorno: {correlation_analysis['correlations']['vader_sentiment_correlation']:.3f}")
                    print(f"      Puntos de Datos: {correlation_analysis['data_points']}")
        
        except Exception as e:
            print(f"   {ticker}: Error en análisis - {e}")
    
    # 4. Generar estrategia de ejemplo
    print(f"\n📈 EJEMPLO DE ESTRATEGIA:")
    ticker = "AAPL"  # Usar Apple como ejemplo
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
            print(f"   Total Señales: {total_signals}")
            print(f"   Señales de Compra: {buy_signals}")
            print(f"   Señales de Venta: {sell_signals}")
            print(f"   Confianza Promedio: {avg_confidence:.1%}")
    
    except Exception as e:
        print(f"   Error generando estrategia: {e}")
    
    # 5. Crear visualización
    print(f"\n📊 Generando dashboard...")
    try:
        create_sentiment_dashboard(tickers, news_data)
    except Exception as e:
        print(f"Error creando dashboard: {e}")
    
    return news_data

# Análisis de sentimiento para small caps
def small_cap_sentiment_strategy(ticker, sentiment_threshold=0.15):
    """
    Estrategia específica de sentimiento para small caps
    """
    # Small caps son más sensibles al sentimiento
    scraper = NewsScraperFinViz()
    sentiment_data = scraper.scrape_news([ticker])
    
    if sentiment_data.empty:
        return {'error': 'No sentiment data available'}
    
    # Obtener datos de precio
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    price_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Parámetros ajustados para small caps
    signals = sentiment_trading_strategy(
        price_data, 
        sentiment_data,
        sentiment_threshold=sentiment_threshold,  # Umbral más alto
        lookback_days=1  # Reacción más rápida
    )
    
    # Agregar filtros específicos para small caps
    signals['volume_filter'] = price_data['Volume'] > price_data['Volume'].rolling(20).mean()
    signals['volatility_filter'] = price_data['Close'].pct_change().rolling(5).std() > 0.02
    
    # Solo generar señales cuando hay volumen y volatilidad
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

## Integración con Estrategias de Trading

### 1. Sentimiento + Gap & Go
```python
def sentiment_gap_strategy(ticker, gap_threshold=0.03):
    """
    Combinar análisis de sentimiento con estrategia Gap & Go
    """
    # Obtener datos
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
    
    # Obtener sentimiento del día anterior
    for i, date in enumerate(price_data.index[1:], 1):
        prev_date = price_data.index[i-1]
        
        # Buscar sentimiento del día anterior
        day_sentiment = sentiment_data[
            sentiment_data['date'].dt.date == prev_date.date()
        ]
        
        if len(day_sentiment) > 0:
            avg_sentiment = day_sentiment['vader_compound'].mean()
            
            # Gap up con sentimiento positivo
            if (signals.loc[date, 'gap_pct'] > gap_threshold and 
                avg_sentiment > 0.1 and
                signals.loc[date, 'volume_ratio'] > 2):
                signals.loc[date, 'signal'] = 1
            
            # Gap down con sentimiento muy negativo (potencial reversal)
            elif (signals.loc[date, 'gap_pct'] < -gap_threshold and
                  avg_sentiment < -0.2 and
                  signals.loc[date, 'volume_ratio'] > 2):
                signals.loc[date, 'signal'] = 1  # Contrarian play
    
    return signals
```

### 2. Sentimiento + VWAP
```python
def sentiment_vwap_strategy(ticker):
    """
    Combinar sentimiento con estrategia VWAP
    """
    # Obtener datos intraday si es posible
    price_data = yf.download(ticker, period="5d", interval="1h")
    
    # Calcular VWAP
    price_data['vwap'] = (price_data['Close'] * price_data['Volume']).cumsum() / price_data['Volume'].cumsum()
    
    # Obtener sentimiento
    scraper = NewsScraperFinViz()
    sentiment_data = scraper.scrape_news([ticker])
    
    # Generar señales
    signals = pd.DataFrame(index=price_data.index)
    signals['price'] = price_data['Close']
    signals['vwap'] = price_data['vwap']
    signals['signal'] = 0
    
    # Sentimiento del día actual
    current_date = datetime.now().date()
    today_sentiment = sentiment_data[
        sentiment_data['date'].dt.date == current_date
    ]
    
    if len(today_sentiment) > 0:
        avg_sentiment = today_sentiment['vader_compound'].mean()
        
        for i, date in enumerate(price_data.index):
            # Long: precio cerca de VWAP + sentimiento positivo
            if (signals.loc[date, 'price'] > signals.loc[date, 'vwap'] * 0.999 and
                signals.loc[date, 'price'] < signals.loc[date, 'vwap'] * 1.001 and
                avg_sentiment > 0.05):
                signals.loc[date, 'signal'] = 1
            
            # Short: precio rechaza VWAP + sentimiento negativo
            elif (signals.loc[date, 'price'] < signals.loc[date, 'vwap'] and
                  avg_sentiment < -0.05):
                signals.loc[date, 'signal'] = -1
    
    return signals
```

## Mejores Prácticas

### 1. Validación de Datos de Sentimiento
```python
def validate_sentiment_data(sentiment_df):
    """
    Validar calidad de datos de sentimiento
    """
    validation_results = {
        'total_articles': len(sentiment_df),
        'date_range': (sentiment_df['date'].min(), sentiment_df['date'].max()),
        'sentiment_distribution': sentiment_df['vader_compound'].describe(),
        'missing_data': sentiment_df.isnull().sum(),
        'duplicate_headlines': sentiment_df['headline'].duplicated().sum()
    }
    
    # Detectar posibles problemas
    warnings = []
    
    if validation_results['total_articles'] < 10:
        warnings.append("Muy pocas noticias para análisis confiable")
    
    if abs(sentiment_df['vader_compound'].mean()) > 0.5:
        warnings.append("Sentimiento extremadamente sesgado")
    
    if validation_results['duplicate_headlines'] > len(sentiment_df) * 0.1:
        warnings.append("Muchas noticias duplicadas")
    
    validation_results['warnings'] = warnings
    
    return validation_results
```

### 2. Normalización Temporal
```python
def normalize_sentiment_by_time(sentiment_df, method='zscore'):
    """
    Normalizar sentimiento por período de tiempo
    """
    sentiment_df = sentiment_df.copy()
    
    if method == 'zscore':
        # Z-score normalization
        sentiment_df['normalized_sentiment'] = (
            sentiment_df['vader_compound'] - sentiment_df['vader_compound'].mean()
        ) / sentiment_df['vader_compound'].std()
    
    elif method == 'rolling_zscore':
        # Rolling z-score (ventana de 30 días)
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

### 3. Filtros de Calidad
```python
def apply_quality_filters(sentiment_df, min_keywords=1, confidence_threshold=0.5):
    """
    Aplicar filtros de calidad a datos de sentimiento
    """
    filtered_df = sentiment_df.copy()
    
    # Filtrar por keywords financieras encontradas
    filtered_df = filtered_df[filtered_df['keywords_found'] >= min_keywords]
    
    # Filtrar headlines muy cortas (probablemente no informativas)
    filtered_df = filtered_df[filtered_df['headline'].str.len() > 20]
    
    # Remover duplicados exactos
    filtered_df = filtered_df.drop_duplicates(subset=['headline'])
    
    # Filtrar por confianza en clasificación
    abs_sentiment = abs(filtered_df['vader_compound'])
    filtered_df = filtered_df[abs_sentiment > confidence_threshold * abs_sentiment.std()]
    
    return filtered_df
```

## Limitaciones y Consideraciones

### 1. Limitaciones del Análisis de Sentimiento
- **Sarcasmo y contexto**: Los modelos pueden no detectar sarcasmo
- **Jerga financiera**: Palabras específicas del sector pueden ser malinterpretadas
- **Volumen de noticias**: Small caps pueden tener pocas noticias
- **Timing**: El impacto del sentimiento puede ser inmediato o retrasado

### 2. Mejores Prácticas de Implementación
```python
SENTIMENT_BEST_PRACTICES = {
    'data_quality': {
        'min_articles_per_day': 3,
        'max_sentiment_abs': 0.8,  # Evitar sentimientos extremos sospechosos
        'min_headline_length': 20,
        'duplicate_threshold': 0.1
    },
    'trading_integration': {
        'sentiment_weight': 0.3,  # No más del 30% del peso en decisiones
        'confirmation_required': True,  # Confirmar con indicadores técnicos
        'volume_filter': True,  # Solo operar con volumen confirmatorio
        'time_decay': 24  # Horas antes de que el sentimiento pierda relevancia
    },
    'risk_management': {
        'max_position_sentiment': 0.05,  # Máximo 5% del capital en trades sentimiento
        'stop_loss_tight': True,  # Stops más ajustados para trades sentimiento
        'sentiment_correlation_limit': 0.7  # Evitar demasiada correlación con sentimiento
    }
}
```

## Fuentes de Datos Alternativas

### 1. Integración con Reddit/Twitter
```python
def reddit_sentiment_analysis(ticker, subreddit='wallstreetbets'):
    """
    Placeholder para análisis de sentimiento de Reddit
    (Requiere API de Reddit)
    """
    # Implementación requiere praw library y API keys
    pass

def twitter_sentiment_analysis(ticker):
    """
    Placeholder para análisis de sentimiento de Twitter
    (Requiere Twitter API)
    """
    # Implementación requiere tweepy library y API keys
    pass
```

### 2. StockTwits Integration
```python
def stocktwits_sentiment(ticker):
    """
    Placeholder para StockTwits sentiment
    (Requiere StockTwits API)
    """
    pass
```

## Siguiente Paso

Con Análisis de Sentimiento implementado, continuemos con [Análisis Fundamental](fundamental_analysis.md) para completar el arsenal de herramientas cuantitativas.
