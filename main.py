import os
import re
import sqlite3
import logging
import random
import requests
import pandas as pd
import nltk
import yfinance as yf
from time import sleep
from datetime import datetime, timedelta
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
from difflib import get_close_matches
from googlesearch import search
import numpy as np

# Configuration
CONFIG = {
    'DB_PATH': "articles.db",
    'CSV_INPUT': "finviz.csv", 
    'CSV_OUTPUT': "finviz_first5.csv",
    'MAX_TICKERS': 5,
    'DAYS_BACK': 7,
    'USER_AGENT': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# Comprehensive sentiment keywords
SENTIMENT_KEYWORDS = {
    # Positive financial terms
    "merger": "positive", "acquisition": "positive", "buyout": "positive", "takeover": "positive",
    "stimulus": "positive", "monetary stimulus": "positive", "fiscal stimulus": "positive",
    "rate cuts": "positive", "big cuts": "positive", "rate reduction": "positive", "cut rates": "positive",
    "pboc": "positive", "fed support": "positive", "central bank support": "positive",
    "liquidity": "positive", "liquidity injection": "positive", "cash injection": "positive",
    "qe": "positive", "quantitative easing": "positive", "asset purchases": "positive",
    "dovish": "positive", "accommodative": "positive", "loose monetary": "positive",
    "earnings beat": "positive", "beat expectations": "positive", "strong earnings": "positive",
    "revenue growth": "positive", "profit growth": "positive", "margin expansion": "positive",
    "dividend increase": "positive", "dividend hike": "positive", "share buyback": "positive",
    "stock split": "positive", "bullish": "positive", "rally": "positive", "surge": "positive",
    "breakthrough": "positive", "innovation": "positive", "patent": "positive", "approval": "positive",
    "expansion": "positive", "growth": "positive", "recovery": "positive", "rebound": "positive",
    "upgrade": "positive", "outperform": "positive", "buy rating": "positive", "strong buy": "positive",
    "partnership": "positive", "alliance": "positive", "joint venture": "positive",
    "contract win": "positive", "deal secured": "positive", "order backlog": "positive",
    "market share": "positive", "competitive advantage": "positive", "moat": "positive",
    "free cash flow": "positive", "cash generation": "positive", "debt reduction": "positive",
    
    # Negative financial terms
    "cpi": "negative", "inflation": "negative", "high inflation": "negative", "rising prices": "negative",
    "qt": "negative", "quantitative tightening": "negative", "qe taper": "negative", "tapering": "negative",
    "tightening": "negative", "hawkish": "negative", "restrictive policy": "negative",
    "rate hikes": "negative", "interest rate increase": "negative", "higher rates": "negative",
    "recession": "negative", "downturn": "negative", "contraction": "negative", "slowdown": "negative",
    "bearish": "negative", "decline": "negative", "crash": "negative", "plunge": "negative",
    "earnings miss": "negative", "miss expectations": "negative", "weak earnings": "negative",
    "revenue decline": "negative", "profit warning": "negative", "margin compression": "negative",
    "dividend cut": "negative", "dividend suspension": "negative", "share dilution": "negative",
    "bankruptcy": "negative", "insolvency": "negative", "default": "negative", "restructuring": "negative",
    "layoffs": "negative", "job cuts": "negative", "workforce reduction": "negative",
    "downgrade": "negative", "underperform": "negative", "sell rating": "negative", "avoid": "negative",
    "investigation": "negative", "lawsuit": "negative", "regulatory action": "negative", "fine": "negative",
    "supply chain": "negative", "shortage": "negative", "disruption": "negative", "delay": "negative",
    "competition": "negative", "market pressure": "negative", "pricing pressure": "negative",
    "debt burden": "negative", "cash burn": "negative", "liquidity concerns": "negative",
    "guidance cut": "negative", "outlook reduced": "negative", "forecast lowered": "negative",
    "volatility": "negative", "uncertainty": "negative", "risk": "negative", "concern": "negative",
    "sell-off": "negative", "correction": "negative", "bear market": "negative", "loss": "negative",
    "weak demand": "negative", "declining sales": "negative", "market share loss": "negative",
    
    # Sector-specific terms
    "gold demand": "positive", "gold shortage": "positive", "safe haven": "positive",
    "mining output": "positive", "ore grade": "positive", "resource expansion": "positive",
    "commodity prices": "positive", "precious metals": "positive", "inflation hedge": "positive",
    "geopolitical tension": "positive", "currency debasement": "positive", "dollar weakness": "positive",
    "mining costs": "negative", "environmental concerns": "negative", "permit delays": "negative",
    "labor strikes": "negative", "operational issues": "negative", "production cuts": "negative",
    
    # Market sentiment terms
    "optimism": "positive", "confidence": "positive", "momentum": "positive", "strength": "positive",
    "resilience": "positive", "robust": "positive", "solid": "positive", "stable": "positive",
    "pessimism": "negative", "fear": "negative", "panic": "negative", "weakness": "negative",
    "fragile": "negative", "unstable": "negative", "vulnerable": "negative", "risk-off": "negative"
}

# Global setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class NewsProcessor:
    def __init__(self):
        self.session = self._create_session()
        self.valid_tickers = self._load_tickers()
        self.stopwords = self._setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        self.headers = {"User-Agent": CONFIG['USER_AGENT'], "Accept-Language": "en-US,en;q=0.9"}
        self.price_cache = {}  # Cache for price data to avoid repeated API calls
    
    def _create_session(self):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[403, 500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session
    
    def _load_tickers(self):
        try:
            df = pd.read_csv(CONFIG['CSV_INPUT'])
            tickers = set(df['Ticker'].dropna().str.upper()) if 'Ticker' in df.columns else set()
            logger.info(f"Loaded {len(tickers)} valid tickers")
            return tickers
        except Exception as e:
            logger.error(f"Failed to load tickers: {e}")
            return set()
    
    def _setup_nltk(self):
        for name in ['stopwords', 'punkt', 'wordnet']:
            try:
                nltk.data.find(f'tokenizers/{name}' if name == 'punkt' else f'corpora/{name}')
            except LookupError:
                nltk.download(name, quiet=True)
        return set(stopwords.words("english"))
    
    def get_stock_price_data(self, ticker, article_datetime):
        """Fetch historical stock price data around article publication time"""
        try:
            # Create cache key
            cache_key = f"{ticker}_{article_datetime.date()}"
            if cache_key in self.price_cache:
                return self.price_cache[cache_key]
            
            # Get extended date range for price data (before and after article)
            start_date = article_datetime.date() - timedelta(days=2)
            end_date = article_datetime.date() + timedelta(days=8)  # Extended for end-of-week calculation
            
            # Fetch stock data
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date, end=end_date, interval='1h')
            
            if hist_data.empty:
                logger.warning(f"No price data found for {ticker} around {article_datetime}")
                return None
            
            # Get the baseline price (closest to article time, before publication)
            article_time = article_datetime
            before_article = hist_data[hist_data.index <= article_time]
            
            if before_article.empty:
                # If no data before article, use the first available price
                baseline_price = hist_data['Close'].iloc[0]
                baseline_time = hist_data.index[0]
            else:
                baseline_price = before_article['Close'].iloc[-1]
                baseline_time = before_article.index[-1]
            
            # Calculate time intervals after article
            one_hour_after = article_time + timedelta(hours=1)
            four_hours_after = article_time + timedelta(hours=4)
            
            # End of day (market close at 4 PM ET)
            eod_time = article_time.replace(hour=21, minute=0, second=0, microsecond=0)  # 4 PM ET in UTC
            if article_time.hour >= 21:  # If article is after market close, use next day
                eod_time += timedelta(days=1)
            
            # End of week (Friday market close)
            days_until_friday = (4 - article_time.weekday()) % 7  # 0=Monday, 4=Friday
            if days_until_friday == 0 and article_time.hour >= 21:
                days_until_friday = 7  # Next Friday if it's Friday after market close
            eow_time = (article_time + timedelta(days=days_until_friday)).replace(hour=21, minute=0, second=0, microsecond=0)
            
            # Get prices at each interval
            def get_closest_price(target_time, data):
                """Get the closest available price to target time"""
                if data.empty:
                    return None, None
                    
                # Find closest timestamp
                time_diffs = abs(data.index - target_time)
                closest_idx = time_diffs.argmin()
                closest_time = data.index[closest_idx]
                closest_price = data['Close'].iloc[closest_idx]
                
                return closest_price, closest_time
            
            # Get prices for each interval
            price_1h, time_1h = get_closest_price(one_hour_after, hist_data)
            price_4h, time_4h = get_closest_price(four_hours_after, hist_data)
            price_eod, time_eod = get_closest_price(eod_time, hist_data)
            price_eow, time_eow = get_closest_price(eow_time, hist_data)
            
            # Calculate percentage changes
            def calc_pct_change(current_price, base_price):
                if current_price is None or base_price is None or base_price == 0:
                    return None
                return ((current_price - base_price) / base_price) * 100
            
            pct_change_1h = calc_pct_change(price_1h, baseline_price)
            pct_change_4h = calc_pct_change(price_4h, baseline_price)
            pct_change_eod = calc_pct_change(price_eod, baseline_price)
            pct_change_eow = calc_pct_change(price_eow, baseline_price)
            
            # Determine overall price direction (using end-of-day as primary indicator)
            if pct_change_eod is not None:
                price_direction = "Positive ↑" if pct_change_eod > 0 else "Negative ↓"
            elif pct_change_4h is not None:
                price_direction = "Positive ↑" if pct_change_4h > 0 else "Negative ↓"
            elif pct_change_1h is not None:
                price_direction = "Positive ↑" if pct_change_1h > 0 else "Negative ↓"
            else:
                price_direction = "Unknown"
            
            # Create comprehensive price data dictionary
            price_data = {
                'baseline_price': round(baseline_price, 2),
                'baseline_time': baseline_time,
                'price_1h': round(price_1h, 2) if price_1h else None,
                'price_4h': round(price_4h, 2) if price_4h else None,
                'price_eod': round(price_eod, 2) if price_eod else None,
                'price_eow': round(price_eow, 2) if price_eow else None,
                'pct_change_1h': round(pct_change_1h, 2) if pct_change_1h else None,
                'pct_change_4h': round(pct_change_4h, 2) if pct_change_4h else None,
                'pct_change_eod': round(pct_change_eod, 2) if pct_change_eod else None,
                'pct_change_eow': round(pct_change_eow, 2) if pct_change_eow else None,
                'price_direction': price_direction,
                'time_1h': time_1h,
                'time_4h': time_4h,
                'time_eod': time_eod,
                'time_eow': time_eow
            }
            
            # Cache the result
            self.price_cache[cache_key] = price_data
            
            logger.info(f"Price data for {ticker}: {price_direction} | 1H: {pct_change_1h}% | 4H: {pct_change_4h}% | EOD: {pct_change_eod}% | EOW: {pct_change_eow}%")
            
            return price_data
            
        except Exception as e:
            logger.error(f"Failed to fetch price data for {ticker}: {e}")
            return None
    
    def analyze_sentiment_vs_price(self, sentiment_score, price_data):
        """Analyze correlation between sentiment and price movement"""
        if not price_data:
            return {"accuracy": None, "prediction": "Unknown"}
        
        # Predict direction based on sentiment
        predicted_direction = "Positive ↑" if sentiment_score > 0 else "Negative ↓" if sentiment_score < 0 else "Neutral"
        actual_direction = price_data.get('price_direction', 'Unknown')
        
        # Calculate accuracy for each time interval
        accuracy_metrics = {}
        for interval in ['1h', '4h', 'eod', 'eow']:
            pct_change = price_data.get(f'pct_change_{interval}')
            if pct_change is not None:
                actual_positive = pct_change > 0
                predicted_positive = sentiment_score > 0
                accuracy_metrics[f'accuracy_{interval}'] = actual_positive == predicted_positive
        
        return {
            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,
            "prediction_correct": predicted_direction == actual_direction,
            **accuracy_metrics
        }
    
    def parse_datetime(self, s):
        """Parse datetime with common patterns"""
        if not s:
            return None
        
        now = datetime.now()
        s = s.strip().lower()
        
        # Relative dates
        if s.startswith("today"):
            return now.replace(second=0, microsecond=0)
        if s.startswith("yesterday"):
            return (now - timedelta(days=1)).replace(second=0, microsecond=0)
        
        # Standard formats
        for fmt in ["%b-%d-%y %I:%M%p", "%Y-%m-%d %I:%M%p", "%m/%d/%Y %I:%M%p"]:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        
        # Time only
        time_match = re.search(r"(\d{1,2}:\d{2}[ap]m)", s)
        if time_match:
            try:
                time_obj = datetime.strptime(time_match.group(1).upper(), "%I:%M%p").time()
                return datetime.combine(now.date(), time_obj)
            except ValueError:
                pass
        
        return None
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Clean and tokenize
        text = re.sub(r"http\S+|www\S+|https\S+", "", text.lower())
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(re.sub(r"\s+", " ", text).strip())
        
        # Filter and lemmatize
        return " ".join(
            self.lemmatizer.lemmatize(token) for token in tokens 
            if token not in self.stopwords and len(token) > 1
        )
    
    def extract_mentions_and_sentiment(self, text, exclude_ticker=None):
        """Extract ticker mentions and analyze sentiment"""
        # Extract ticker mentions
        potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
        mentions = list(dict.fromkeys([  # Remove duplicates preserving order
            t for t in potential_tickers 
            if t in self.valid_tickers and t != exclude_ticker
        ]))
        
        # Sentiment analysis
        processed_text = self.preprocess_text(text)
        try:
            keywords = self.kw_model.extract_keywords(
                processed_text, keyphrase_ngram_range=(1, 3), 
                stop_words='english', top_n=15, use_mmr=True, diversity=0.3
            )
            
            pos_keywords, neg_keywords = [], []
            for keyword, score in keywords:
                kw_clean = keyword.lower().strip()
                sentiment = SENTIMENT_KEYWORDS.get(kw_clean)
                
                # Fuzzy matching if no direct match
                if not sentiment:
                    matches = get_close_matches(kw_clean, SENTIMENT_KEYWORDS.keys(), n=1, cutoff=0.8)
                    sentiment = SENTIMENT_KEYWORDS.get(matches[0]) if matches else None
                
                if sentiment == "positive":
                    pos_keywords.append(keyword)
                elif sentiment == "negative":
                    neg_keywords.append(keyword)
            
            return mentions, pos_keywords, neg_keywords, processed_text
            
        except Exception as e:
            logger.debug(f"Keyword extraction failed: {e}")
            return mentions, [], [], processed_text
    
    def scrape_article(self, url):
        """Scrape article content from URL"""
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract content from paragraphs
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            
            return text if len(text) >= 200 else ""
        except Exception as e:
            logger.debug(f"Scrape failed for {url}: {e}")
            return ""
    
    def fallback_search(self, headline):
        """Fallback scraping via Google search"""
        try:
            search_results = list(search(headline, num_results=3, lang="en"))
            cutoff_date = datetime.now() - timedelta(days=CONFIG['DAYS_BACK'])
            
            for url in search_results:
                text = self.scrape_article(url)
                if text:
                    logger.info(f"Fallback succeeded: {url}")
                    return text, url
                    
        except Exception as e:
            logger.warning(f"Fallback search failed: {e}")
        
        return "", ""
    
    def fetch_finviz_news(self, ticker):
        """Fetch news for ticker from Finviz with price analysis"""
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch Finviz for {ticker}: {e}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        news_table = soup.select_one("table.fullview-news-outer") or soup.select_one("#news-table")
        
        if not news_table:
            logger.warning(f"No news table for {ticker}")
            return []
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=CONFIG['DAYS_BACK'])
        last_date = None
        
        for row in news_table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            
            # Parse datetime
            datetime_text = cols[0].get_text(strip=True)
            parsed_dt = self.parse_datetime(datetime_text)
            
            if parsed_dt:
                # Handle time inheritance
                is_time_only = bool(re.search(r"^\s*\d{1,2}:\d{2}[ap]m\s*$", datetime_text.lower()))
                if is_time_only and last_date:
                    final_datetime = datetime.combine(last_date, parsed_dt.time())
                else:
                    final_datetime = parsed_dt
                    last_date = parsed_dt.date()
            else:
                continue
            
            if final_datetime < cutoff_date:
                continue
            
            # Extract article info
            link_element = cols[1].find("a")
            if not link_element:
                continue
            
            article_url = urljoin("https://finviz.com/", link_element["href"])
            headline = link_element.get_text(strip=True)
            
            # Get article text
            article_text = self.scrape_article(article_url)
            if not article_text:
                article_text, fallback_url = self.fallback_search(headline)
                if article_text:
                    article_url = fallback_url
            
            if not article_text:
                continue
            
            # Process article sentiment
            mentions, pos_kw, neg_kw, tokens = self.extract_mentions_and_sentiment(article_text, ticker)
            sentiment_score = len(pos_kw) - len(neg_kw)
            
            # Get price data for this article
            price_data = self.get_stock_price_data(ticker, final_datetime)
            
            # Analyze sentiment vs price correlation
            prediction_analysis = self.analyze_sentiment_vs_price(sentiment_score, price_data)
            
            # Create comprehensive article data
            article_entry = {
                "ticker": ticker,
                "datetime": datetime_text,
                "parsed_datetime": final_datetime,
                "headline": headline,
                "url": article_url,
                "text": article_text,
                "tokens": tokens,
                "pos_keywords": ", ".join(pos_kw),
                "neg_keywords": ", ".join(neg_kw),
                "sentiment_score": sentiment_score,
                "total_keywords": len(pos_kw) + len(neg_kw),
                "mentions": ", ".join(mentions)
            }
            
            # Add price data if available
            if price_data:
                article_entry.update({
                    "baseline_price": price_data['baseline_price'],
                    "price_1h": price_data['price_1h'],
                    "price_4h": price_data['price_4h'],
                    "price_eod": price_data['price_eod'],
                    "price_eow": price_data['price_eow'],
                    "pct_change_1h": price_data['pct_change_1h'],
                    "pct_change_4h": price_data['pct_change_4h'],
                    "pct_change_eod": price_data['pct_change_eod'],
                    "pct_change_eow": price_data['pct_change_eow'],
                    "price_direction": price_data['price_direction']
                })
            
            # Add prediction analysis
            article_entry.update({
                "predicted_direction": prediction_analysis['predicted_direction'],
                "prediction_correct": prediction_analysis.get('prediction_correct', False),
                "accuracy_1h": prediction_analysis.get('accuracy_1h'),
                "accuracy_4h": prediction_analysis.get('accuracy_4h'),
                "accuracy_eod": prediction_analysis.get('accuracy_eod'),
                "accuracy_eow": prediction_analysis.get('accuracy_eow')
            })
            
            articles.append(article_entry)
        
        logger.info(f"Found {len(articles)} articles for {ticker} with price analysis")
        return articles

def init_database():
    """Initialize SQLite database with price columns"""
    conn = sqlite3.connect(CONFIG['DB_PATH'])
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            ticker TEXT, datetime TEXT, parsed_datetime TIMESTAMP, headline TEXT,
            url TEXT UNIQUE, text TEXT, tokens TEXT, pos_keywords TEXT, 
            neg_keywords TEXT, sentiment_score INTEGER, total_keywords INTEGER, mentions TEXT,
            baseline_price REAL, price_1h REAL, price_4h REAL, price_eod REAL, price_eow REAL,
            pct_change_1h REAL, pct_change_4h REAL, pct_change_eod REAL, pct_change_eow REAL,
            price_direction TEXT, predicted_direction TEXT, prediction_correct BOOLEAN,
            accuracy_1h BOOLEAN, accuracy_4h BOOLEAN, accuracy_eod BOOLEAN, accuracy_eow BOOLEAN
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized with price analysis columns: {CONFIG['DB_PATH']}")

def save_articles(articles):
    """Save articles to CSV and database"""
    if not articles:
        logger.warning("No articles to save")
        return
    
    df = pd.DataFrame(articles)
    
    # Save to CSV
    df.to_csv(CONFIG['CSV_OUTPUT'], index=False)
    logger.info(f"Saved {len(articles)} articles to {CONFIG['CSV_OUTPUT']}")
    
    # Calculate and display summary statistics
    if 'sentiment_score' in df.columns and 'price_direction' in df.columns:
        total_articles = len(df)
        correct_predictions = df['prediction_correct'].sum() if 'prediction_correct' in df.columns else 0
        accuracy_rate = (correct_predictions / total_articles * 100) if total_articles > 0 else 0
        
        logger.info(f"Prediction Accuracy: {correct_predictions}/{total_articles} ({accuracy_rate:.1f}%)")
        
        # Show average price changes by sentiment
        positive_sentiment = df[df['sentiment_score'] > 0]
        negative_sentiment = df[df['sentiment_score'] < 0]
        
        if not positive_sentiment.empty:
            avg_pos_change = positive_sentiment['pct_change_eod'].mean()
            logger.info(f"Average EOD price change for positive sentiment: {avg_pos_change:.2f}%")
        
        if not negative_sentiment.empty:
            avg_neg_change = negative_sentiment['pct_change_eod'].mean()
            logger.info(f"Average EOD price change for negative sentiment: {avg_neg_change:.2f}%")
    
    # Save to database
    conn = sqlite3.connect(CONFIG['DB_PATH'])
    try:
        existing_urls = pd.read_sql("SELECT url FROM articles", conn)["url"].tolist()
        new_articles = df[~df["url"].isin(existing_urls)]
        
        if not new_articles.empty:
            new_articles.to_sql("articles", conn, if_exists="append", index=False)
            logger.info(f"Inserted {len(new_articles)} new articles to database")
    except Exception as e:
        logger.error(f"Database save failed: {e}")
    finally:
        conn.close()

def main():
    """Main execution"""
    logger.info("Starting financial news scraper with price analysis")
    
    # Initialize
    init_database()
    processor = NewsProcessor()
    
    # Load tickers
    try:
        df = pd.read_csv(CONFIG['CSV_INPUT'])
        tickers = df["Ticker"].dropna().str.upper().unique()[:CONFIG['MAX_TICKERS']]
        logger.info(f"Processing {len(tickers)} tickers: {list(tickers)}")
    except Exception as e:
        logger.error(f"Failed to read {CONFIG['CSV_INPUT']}: {e}")
        return
    
    # Process tickers
    all_articles = []
    for ticker in tickers:
        logger.info(f"Processing: {ticker}")
        try:
            articles = processor.fetch_finviz_news(ticker)
            all_articles.extend(articles)
            sleep(random.uniform(2, 4))  # Slightly longer delay for price API calls
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
    
    save_articles(all_articles)
    logger.info("Scraper with price analysis completed")

if __name__ == "__main__":
    main()