import os, re, sqlite3, logging, random, requests, pandas as pd, nltk, yfinance as yf, pytz
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
import json
from word_analysis_framework import NewsAnalysisFramework

# Configuration
CONFIG = {
    'DB_PATH': "articles.db",
    'CSV_INPUT': "finviz.csv", 
    'CSV_OUTPUT': "finviz_first5.csv",
    'MAX_TICKERS': 5,
    'DAYS_BACK': 7,
    'USER_AGENT': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Suppress yfinance warnings
yf_logger = logging.getLogger("yfinance")
yf_logger.setLevel(logging.ERROR)

# Load sentiment keywords from CSV
sentiment_df = pd.read_csv("sentiment_keywords.csv")
SENTIMENT_KEYWORDS = dict(zip(sentiment_df["keyword"], sentiment_df["sentiment"]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class SimplifiedPriceAnalyzer:
    def __init__(self):
        self.price_cache = {}
        self.eastern_tz = pytz.timezone('US/Eastern')
    
    def get_historical_stock_price_simplified(self, ticker, article_datetime):
        try:
            # Ensure timezone
            if article_datetime.tzinfo is None:
                article_datetime = self.eastern_tz.localize(article_datetime)
            else:
                article_datetime = article_datetime.astimezone(self.eastern_tz)
            
            # Check cache
            cache_key = f"{ticker}_{article_datetime.strftime('%Y%m%d_%H%M')}"
            if cache_key in self.price_cache:
                return self.price_cache[cache_key]
            
            # Fetch data
            start_date = (article_datetime - timedelta(days=5)).date()
            end_date = (article_datetime + timedelta(days=10)).date()
            
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date, end=end_date, interval='1d')
            if hist_data.empty:
                return None
            
            # Fix timezone
            if hist_data.index.tz is None:
                hist_data.index = hist_data.index.tz_localize('UTC').tz_convert(self.eastern_tz)
            else:
                hist_data.index = hist_data.index.tz_convert(self.eastern_tz)
            
            # Get baseline price
            baseline_price = self._get_baseline_price(hist_data, article_datetime)
            if baseline_price is None:
                return None
            
            # Calculate intervals
            intervals = {
                '1h': timedelta(hours=1),
                '4h': timedelta(hours=4),
                'eod': self._get_end_of_day_delta(article_datetime),
                'eow': self._get_end_of_week_delta(article_datetime)
            }
            
            result = {
                'ticker': ticker,
                'article_datetime': article_datetime,
                'baseline_price': round(baseline_price, 2),
                'data_interval': '1d',
                'data_points': len(hist_data)
            }
            
            # Process each interval
            for interval_name, delta in intervals.items():
                target_time = article_datetime + delta
                
                if target_time > datetime.now(self.eastern_tz):
                    result[f'price_{interval_name}'] = None
                    result[f'pct_change_{interval_name}'] = None
                    result[f'direction_{interval_name}'] = "Future"
                    continue
                
                target_price = self._get_price_at_time(hist_data, target_time)
                pct_change = self._calculate_pct_change(baseline_price, target_price)
                
                result[f'price_{interval_name}'] = round(target_price, 2) if target_price is not None else None
                result[f'pct_change_{interval_name}'] = round(pct_change, 2) if pct_change is not None else None
                result[f'direction_{interval_name}'] = self._get_direction_label(pct_change)
            
            self.price_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Price analysis failed for {ticker}: {e}")
            return None
    
    def _get_baseline_price(self, hist_data, article_datetime):
        try:
            if hist_data.empty:
                return None
            
            market_close_time = article_datetime.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if article_datetime.time() > market_close_time.time():
                next_day = article_datetime + timedelta(days=1)
                target_date = next_day.date()
            else:
                target_date = article_datetime.date()
            
            available_dates = [d.date() for d in hist_data.index]
            target_dates = pd.bdate_range(start=target_date, periods=5).date
            
            for date in target_dates:
                if date in available_dates:
                    return float(hist_data.loc[hist_data.index.date == date, 'Open'].iloc[0])
            
            return float(hist_data['Open'].iloc[0])
            
        except Exception as e:
            logger.error(f"Error getting baseline price: {e}")
            return None
    
    def _get_price_at_time(self, hist_data, target_datetime):
        try:
            if hist_data.empty:
                return None
            
            target_date = target_datetime.date()
            available_dates = [d.date() for d in hist_data.index]
            
            future_dates = [d for d in available_dates if d >= target_date]
            if future_dates:
                closest_date = min(future_dates)
                return float(hist_data.loc[hist_data.index.date == closest_date, 'Close'].iloc[0])
            
            return float(hist_data['Close'].iloc[-1])
            
        except Exception as e:
            logger.error(f"Error getting price at time: {e}")
            return None
    
    def _calculate_pct_change(self, baseline_price, target_price):
        if baseline_price is None or target_price is None:
            return None
        try:
            return ((target_price - baseline_price) / baseline_price) * 100
        except (ZeroDivisionError, TypeError):
            return None
    
    def _get_direction_label(self, pct_change):
        if pct_change is None:
            return "No Data"
        elif pct_change > 0:
            return "Positive ↑"
        elif pct_change < 0:
            return "Negative ↓"
        else:
            return "Neutral →"
    
    def _get_end_of_day_delta(self, article_datetime):
        eod_time = article_datetime.replace(hour=16, minute=0, second=0, microsecond=0)
        if article_datetime.hour >= 16:
            eod_time += timedelta(days=1)
        return eod_time - article_datetime
    
    def _get_end_of_week_delta(self, article_datetime):
        days_until_friday = (4 - article_datetime.weekday()) % 7
        if days_until_friday == 0:
            if article_datetime.hour >= 16:
                days_until_friday = 7
        elif days_until_friday == 0:
            days_until_friday = 0
        
        eow_time = article_datetime + timedelta(days=days_until_friday)
        eow_time = eow_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return eow_time - article_datetime

class NewsProcessor:
    def __init__(self):
        self.session = self._create_session()
        self.valid_tickers = self._load_tickers()
        self.stopwords = self._setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        except:
            self.kw_model = None
            logger.warning("KeyBERT model not available")
        self.headers = {"User-Agent": CONFIG['USER_AGENT']}
        self.price_analyzer = SimplifiedPriceAnalyzer()
        self.word_analyzer = NewsAnalysisFramework()
        self.sentiment_weights = self.load_sentiment_weights()
    
    def load_sentiment_weights(self):
        """Load dynamic sentiment weights from analysis"""
        try:
            with open("word_analysis_results.json", 'r') as f:
                results = json.load(f)
                return results.get('sentiment_weights', {})
        except FileNotFoundError:
            logger.warning("No word analysis results found, using basic sentiment")
            return {}
    
    def calculate_dynamic_sentiment(self, text):
        """Calculate sentiment using learned weights"""
        if not self.sentiment_weights:
            return 0
        
        words = text.split()
        sentiment_score = 0
        total_weight = 0
        
        for word in words:
            if word in self.sentiment_weights:
                weight = self.sentiment_weights[word]['weight']
                confidence = self.sentiment_weights[word]['confidence']
                sentiment_score += weight * confidence
                total_weight += confidence
        
        # Check bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in self.sentiment_weights:
                weight = self.sentiment_weights[bigram]['weight']
                confidence = self.sentiment_weights[bigram]['confidence']
                sentiment_score += weight * confidence * 1.5  # Boost bigrams
                total_weight += confidence
        
        return sentiment_score / (total_weight + 1e-6)  # Avoid division by zero
    
    def _create_session(self):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session
    
    def _load_tickers(self):
        try:
            df = pd.read_csv(CONFIG['CSV_INPUT'])
            tickers = set(df['Ticker'].dropna().str.upper()) if 'Ticker' in df.columns else set()
            logger.info(f"Loaded {len(tickers)} tickers")
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
    
    def get_price_data(self, ticker, article_datetime):
        price_data = self.price_analyzer.get_historical_stock_price_simplified(ticker, article_datetime)
        if not price_data:
            return None
        
        return {
            'baseline_price': price_data['baseline_price'],
            'eod_price': price_data.get('price_eod'),
            'pct_change_eod': price_data.get('pct_change_eod'),
            'price_direction': price_data.get('direction_eod', 'No Data'),
            'price_1h': price_data.get('price_1h'),
            'price_4h': price_data.get('price_4h'),
            'price_eow': price_data.get('price_eow'),
            'pct_change_1h': price_data.get('pct_change_1h'),
            'pct_change_4h': price_data.get('pct_change_4h'),
            'pct_change_eow': price_data.get('pct_change_eow'),
            'direction_1h': price_data.get('direction_1h'),
            'direction_4h': price_data.get('direction_4h'),
            'direction_eow': price_data.get('direction_eow'),
            'data_interval': price_data['data_interval'],
            'data_points': price_data['data_points']
        }
    
    def parse_datetime(self, s):
        if not s:
            return None
        
        now = datetime.now()
        s = s.strip().lower()
        
        if s.startswith("today"):
            return now.replace(second=0, microsecond=0)
        if s.startswith("yesterday"):
            return (now - timedelta(days=1)).replace(second=0, microsecond=0)
        
        formats = [
            "%b-%d-%y %I:%M%p", "%Y-%m-%d %I:%M%p", "%m/%d/%Y %I:%M%p",
            "%b %d %I:%M%p", "%m-%d-%y %H:%M"
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(s, fmt)
                if parsed.year == 1900:
                    parsed = parsed.replace(year=now.year)
                return parsed
            except ValueError:
                continue
        return None
    
    def preprocess_text(self, text):
        if not text:
            return ""
        
        text = re.sub(r"http\S+|www\S+|https\S+", "", text.lower())
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(re.sub(r"\s+", " ", text).strip())
        
        return " ".join(
            self.lemmatizer.lemmatize(token) for token in tokens 
            if token not in self.stopwords and len(token) > 1
        )
    
    def extract_mentions_and_sentiment(self, text, exclude_ticker=None):
        potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
        mentions = list(dict.fromkeys([
            t for t in potential_tickers 
            if t in self.valid_tickers and t != exclude_ticker
        ]))
        
        processed_text = self.preprocess_text(text)
        pos_keywords, neg_keywords = [], []
        
        if self.kw_model and SENTIMENT_KEYWORDS:
            try:
                keywords = self.kw_model.extract_keywords(
                    processed_text, keyphrase_ngram_range=(1, 3), 
                    stop_words='english', top_n=15, use_mmr=True, diversity=0.3
                )
                
                for keyword, score in keywords:
                    kw_clean = keyword.lower().strip()
                    sentiment = SENTIMENT_KEYWORDS.get(kw_clean)
                    
                    if not sentiment:
                        matches = get_close_matches(kw_clean, SENTIMENT_KEYWORDS.keys(), n=1, cutoff=0.8)
                        sentiment = SENTIMENT_KEYWORDS.get(matches[0]) if matches else None
                    
                    if sentiment == "positive":
                        pos_keywords.append(keyword)
                    elif sentiment == "negative":
                        neg_keywords.append(keyword)
                        
            except Exception as e:
                logger.debug(f"Keyword extraction failed: {e}")
        
        return mentions, pos_keywords, neg_keywords, processed_text
    
    def scrape_article(self, url):
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            
            return text if len(text) >= 200 else ""
        except Exception as e:
            logger.debug(f"Scrape failed for {url}: {e}")
            return ""
    
    def fallback_search(self, headline):
        try:
            search_results = list(search(headline, num_results=3, lang="en"))
            
            for url in search_results:
                text = self.scrape_article(url)
                if text:
                    logger.info(f"Fallback succeeded: {url}")
                    return text, url
                    
        except Exception as e:
            logger.warning(f"Fallback search failed: {e}")
        
        return "", ""
    
    def fetch_finviz_news(self, ticker):
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Finviz fetch failed for {ticker}: {e}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        news_table = soup.select_one("table.fullview-news-outer") or soup.select_one("#news-table")
        
        if not news_table:
            return []
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=CONFIG['DAYS_BACK'])
        
        for row in news_table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            
            parsed_dt = self.parse_datetime(cols[0].get_text(strip=True))
            if not parsed_dt or parsed_dt < cutoff_date:
                continue
            
            link_element = cols[1].find("a")
            if not link_element:
                continue
            
            article_url = urljoin("https://finviz.com/", link_element["href"])
            headline = link_element.get_text(strip=True)
            
            article_text = self.scrape_article(article_url)
            if not article_text:
                article_text, fallback_url = self.fallback_search(headline)
                if article_text:
                    article_url = fallback_url
            
            if not article_text:
                continue
            
            mentions, pos_kw, neg_kw, tokens = self.extract_mentions_and_sentiment(article_text, ticker)
            sentiment_score = len(pos_kw) - len(neg_kw)
            
            price_data = self.get_price_data(ticker, parsed_dt)
            
            article_entry = {
                "ticker": ticker,
                "datetime": cols[0].get_text(strip=True),
                "headline": headline,
                "url": article_url,
                "text": article_text,
                "tokens": tokens,
                "sentiment_score": sentiment_score,
                "pos_keywords": ", ".join(pos_kw),
                "neg_keywords": ", ".join(neg_kw),
                "total_keywords": len(pos_kw) + len(neg_kw),
                "mentions": ", ".join(mentions),
                "predicted_direction": "Positive" if sentiment_score > 0 else "Negative"
            }
            
            if price_data:
                article_entry.update(price_data)
                
                is_sentiment_positive = sentiment_score > 0
                intervals = ['1h', '4h', 'eod', 'eow']
                
                for interval in intervals:
                    pct_change_key = f'pct_change_{interval}'
                    if pct_change_key in price_data:
                        article_entry[f'prediction_correct_{interval}'] = self._check_prediction_accuracy(
                            is_sentiment_positive, price_data[pct_change_key]
                        )
            
            articles.append(article_entry)
        
        logger.info(f"Found {len(articles)} articles for {ticker}")
        return articles
    
    def _check_prediction_accuracy(self, is_sentiment_positive, pct_change):
        if pct_change is None:
            return None
        return (is_sentiment_positive and pct_change > 0) or (not is_sentiment_positive and pct_change < 0)

def init_database():
    conn = sqlite3.connect(CONFIG['DB_PATH'])
    cursor = conn.cursor()
    
    all_required_columns = {
        'ticker': 'TEXT', 'datetime': 'TEXT', 'headline': 'TEXT', 'url': 'TEXT UNIQUE', 'text': 'TEXT',
        'tokens': 'TEXT', 'sentiment_score': 'INTEGER', 'pos_keywords': 'TEXT', 'neg_keywords': 'TEXT', 
        'total_keywords': 'INTEGER', 'mentions': 'TEXT', 'baseline_price': 'REAL', 'eod_price': 'REAL',
        'pct_change_eod': 'REAL', 'price_direction': 'TEXT', 'predicted_direction': 'TEXT',
        'prediction_correct': 'BOOLEAN', 'price_1h': 'REAL', 'price_4h': 'REAL', 'price_eow': 'REAL',
        'pct_change_1h': 'REAL', 'pct_change_4h': 'REAL', 'pct_change_eow': 'REAL',
        'direction_1h': 'TEXT', 'direction_4h': 'TEXT', 'direction_eow': 'TEXT',
        'data_interval': 'TEXT', 'data_points': 'INTEGER', 'prediction_correct_1h': 'BOOLEAN',
        'prediction_correct_4h': 'BOOLEAN', 'prediction_correct_eod': 'BOOLEAN', 'prediction_correct_eow': 'BOOLEAN'
    }
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        cursor.execute("PRAGMA table_info(articles)")
        existing_columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        missing_columns = set(all_required_columns.keys()) - set(existing_columns.keys())
        
        if missing_columns:
            for col_name in missing_columns:
                try:
                    col_type = all_required_columns[col_name]
                    cursor.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type}")
                except Exception as e:
                    logger.warning(f"Could not add column {col_name}: {e}")
    else:
        column_defs = [f"{col} {dtype}" for col, dtype in all_required_columns.items()]
        create_sql = f"CREATE TABLE articles ({', '.join(column_defs)})"
        cursor.execute(create_sql)
        logger.info("Created new articles table")
    
    conn.commit()
    conn.close()

def save_articles(articles):
    if not articles:
        logger.warning("No articles to save")
        return
    
    df = pd.DataFrame(articles)
    df.to_csv(CONFIG['CSV_OUTPUT'], index=False)
    logger.info(f"Saved {len(articles)} articles to {CONFIG['CSV_OUTPUT']}")
    
    # Show accuracy stats
    intervals = ['1h', '4h', 'eod', 'eow']
    for interval in intervals:
        col = f'prediction_correct_{interval}'
        if col in df.columns:
            valid = df[col].dropna()
            if not valid.empty:
                correct = valid.sum()
                total = len(valid)
                logger.info(f"Prediction accuracy {interval.upper()}: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # Show average price changes
    for interval in intervals:
        col = f'pct_change_{interval}'
        if col in df.columns:
            avg_change = df[col].mean()
            if not pd.isna(avg_change):
                logger.info(f"Average price change {interval.upper()}: {avg_change:+.2f}%")
    
    # Save to database
    conn = sqlite3.connect(CONFIG['DB_PATH'])
    try:
        try:
            existing_urls = pd.read_sql("SELECT url FROM articles", conn)["url"].tolist()
            new_articles = df[~df["url"].isin(existing_urls)]
        except Exception as e:
            logger.warning(f"Could not check existing URLs: {e}")
            new_articles = df
        
        if not new_articles.empty:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(articles)")
            db_columns = [row[1] for row in cursor.fetchall()]
            
            save_columns = [col for col in new_articles.columns if col in db_columns]
            if save_columns:
                new_articles[save_columns].to_sql("articles", conn, if_exists="append", index=False)
                logger.info(f"Added {len(new_articles)} new articles to database")
            else:
                logger.error("No matching columns found")
        else:
            logger.info("No new articles to add")
            
    except Exception as e:
        logger.error(f"Database save failed: {e}")
    finally:
        conn.close()

def main():
    logger.info("Starting financial news scraper")
    
    init_database()
    processor = NewsProcessor()
    
    try:
        df = pd.read_csv(CONFIG['CSV_INPUT'])
        tickers = df["Ticker"].dropna().str.upper().unique()[:CONFIG['MAX_TICKERS']]
        logger.info(f"Processing {len(tickers)} tickers")
    except Exception as e:
        logger.error(f"Failed to read {CONFIG['CSV_INPUT']}: {e}")
        return
    
    all_articles = []
    for ticker in tickers:
        logger.info(f"Processing: {ticker}")
        try:
            articles = processor.fetch_finviz_news(ticker)
            all_articles.extend(articles)
            sleep(random.uniform(1, 3))
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
    
    save_articles(all_articles)
    logger.info("Scraper completed successfully")

if __name__ == "__main__":
    main()