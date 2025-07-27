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
from word_analysis_framework import DynamicSentimentLearner, EnhancedNewsProcessor
from feature_engineering import feature_engineering_pipeline

# Configuration
CONFIG = {
    'DB_PATH': "articles.db",
    'CSV_INPUT': "finviz.csv", 
    'CSV_OUTPUT': "scraped_articles.csv",
    'MAX_TICKERS': None,
    'BATCH_SIZE': 50,
    'BATCH_START': 0,
    'DAYS_BACK': 7,
    'USER_AGENT': "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    'NO_NEWS_CACHE': "tickers_with_no_news.json",  
    'REFRESH_NO_NEWS_CACHE': False                 
}

# No news cache loading
def load_no_news_cache():
    if os.path.exists(CONFIG['NO_NEWS_CACHE']) and not CONFIG['REFRESH_NO_NEWS_CACHE']:
        with open(CONFIG['NO_NEWS_CACHE'], 'r') as f:
            return set(json.load(f))
    return set()

def save_no_news_cache(tickers):
    try:
        with open(CONFIG['NO_NEWS_CACHE'], 'w') as f:
            json.dump(list(tickers), f)
    except Exception as e:
        logger.warning(f"Failed to save no-news cache: {e}")

# Suppress yfinance warnings
yf_logger = logging.getLogger("yfinance")
yf_logger.setLevel(logging.ERROR)

# Load sentiment keywords from CSV
try:
    sentiment_df = pd.read_csv("sentiment_keywords.csv")
    SENTIMENT_KEYWORDS = dict(zip(sentiment_df["keyword"], sentiment_df["sentiment"]))
except FileNotFoundError:
    SENTIMENT_KEYWORDS = {}
    print("Warning: sentiment_keywords.csv not found, using empty sentiment dictionary")

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
            
            # Calculate intervals - FIXED: Removed the problematic time offsets
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
            return "Positive"
        elif pct_change < 0:
            return "Negative"
        else:
            return "Neutral"
    
    def _get_end_of_day_delta(self, article_datetime):
        eod_time = article_datetime.replace(hour=16, minute=0, second=0, microsecond=0)
        if article_datetime.hour >= 16:
            eod_time += timedelta(days=1)
        return eod_time - article_datetime
    
    def _get_end_of_week_delta(self, article_datetime):
        # FIXED: Simplified end of week calculation
        days_until_friday = (4 - article_datetime.weekday()) % 7
        if days_until_friday == 0 and article_datetime.hour >= 16:
            days_until_friday = 7
        
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
        except Exception as e:
            self.kw_model = None
            logger.warning(f"KeyBERT model not available: {e}")
        self.headers = {"User-Agent": CONFIG['USER_AGENT']}
        self.price_analyzer = SimplifiedPriceAnalyzer()
        
        # Initialize enhanced sentiment analysis
        self.sentiment_learner = DynamicSentimentLearner()
        self.enhanced_processor = EnhancedNewsProcessor()
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
                if name == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                else:
                    nltk.data.find(f'corpora/{name}')
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
        
        # FIXED: Better handling of relative dates
        if s.startswith("today"):
            return now.replace(hour=9, minute=30, second=0, microsecond=0)
        if s.startswith("yesterday"):
            return (now - timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Handle relative time strings like "1h ago", "2 hours ago", etc.
        time_ago_patterns = [
            (r'(\d+)h ago', lambda m: now - timedelta(hours=int(m.group(1)))),
            (r'(\d+) hours? ago', lambda m: now - timedelta(hours=int(m.group(1)))),
            (r'(\d+)m ago', lambda m: now - timedelta(minutes=int(m.group(1)))),
            (r'(\d+) minutes? ago', lambda m: now - timedelta(minutes=int(m.group(1)))),
        ]
        
        for pattern, func in time_ago_patterns:
            match = re.search(pattern, s)
            if match:
                return func(match).replace(second=0, microsecond=0)
        
        formats = [
            "%b-%d-%y %I:%M%p", "%Y-%m-%d %I:%M%p", "%m/%d/%Y %I:%M%p",
            "%b %d %I:%M%p", "%m-%d-%y %H:%M", "%b-%d-%y", "%m/%d/%Y"
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(s, fmt)
                if parsed.year == 1900:
                    parsed = parsed.replace(year=now.year)
                # If no time specified, assume market open
                if parsed.hour == 0 and parsed.minute == 0:
                    parsed = parsed.replace(hour=9, minute=30)
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
            
            # FIXED: Better content extraction
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content areas first
            content_selectors = [
                'article', '.article-content', '.content', '.post-content',
                '.entry-content', 'main', '[role="main"]'
            ]
            
            text = ""
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    paragraphs = content.find_all("p")
                    text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                    break
            
            # Fallback to all paragraphs
            if not text:
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
            logger.warning(f"No news table found for {ticker}")
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
            
            # Use enhanced sentiment analysis
            full_text = f"{headline} {article_text}"
            enhanced_sentiment = self.enhanced_processor.calculate_enhanced_sentiment(full_text)
            
            price_data = self.get_price_data(ticker, parsed_dt)
            
            # Article entry with tokens included
            article_entry = {
                "ticker": ticker,
                "datetime": cols[0].get_text(strip=True),
                "headline": headline,
                "url": article_url,
                "text": article_text,
                "tokens": tokens,  # Add the preprocessed tokens
                
                # Sentiment signals
                "sentiment_dynamic": enhanced_sentiment.get('dynamic_weights', 0),
                "sentiment_ml": enhanced_sentiment.get('ml_prediction', 0),
                "sentiment_keyword": enhanced_sentiment.get('keyword_based', 0),
                "sentiment_combined": enhanced_sentiment.get('combined', 0),
                "prediction_confidence": abs(enhanced_sentiment.get('combined', 0)),
                
                # Additional extracted features
                "mentions": ', '.join(mentions) if mentions else '',
                "pos_keywords": ', '.join(pos_kw) if pos_kw else '',
                "neg_keywords": ', '.join(neg_kw) if neg_kw else '',
                "total_keywords": len(pos_kw) + len(neg_kw)
            }
            
            # Add price data if available
            if price_data:
                article_entry.update({
                    'pct_change_1h': price_data.get('pct_change_1h'),
                    'pct_change_4h': price_data.get('pct_change_4h'),
                    'pct_change_eod': price_data.get('pct_change_eod'),
                    'pct_change_eow': price_data.get('pct_change_eow'),
                    'direction_1h': price_data.get('direction_1h'),
                    'direction_4h': price_data.get('direction_4h'),
                    'direction_eod': price_data.get('direction_eod'),
                    'direction_eow': price_data.get('direction_eow')
                })
            
            articles.append(article_entry)
        
        logger.info(f"Found {len(articles)} articles for {ticker}")
        return articles
    
    def _check_prediction_accuracy(self, is_sentiment_positive, pct_change):
        if pct_change is None:
            return None
        return (is_sentiment_positive and pct_change > 0) or (not is_sentiment_positive and pct_change < 0)
    
    def filter_tickers_with_news(self, tickers):
        """Filter tickers that have news - simplified version"""
        # Simply return the sorted tickers without checking news table
        return sorted(tickers)
    
def init_database():
    """Create (or patch) a lean 'articles' table that matches the minimal feature set."""
    conn = sqlite3.connect(CONFIG["DB_PATH"])
    cursor = conn.cursor()

    # ── keep only what the slim FE produces ──────────────────────────────
    all_required_columns = {
    "ticker": "TEXT",
    "datetime": "TEXT",
    "headline": "TEXT",
    "url": "TEXT",
    "text": "TEXT",
    "tokens": "TEXT",
    "sentiment_dynamic": "REAL",
    "sentiment_ml": "REAL",
    "sentiment_keyword": "REAL",
    "sentiment_combined": "REAL",
    "prediction_confidence": "REAL",
    "mentions": "INTEGER",
    "pos_keywords": "INTEGER",
    "neg_keywords": "INTEGER",
    "total_keywords": "INTEGER",

    # price changes and directions
    "pct_change_1h": "REAL",
    "pct_change_4h": "REAL",
    "pct_change_eod": "REAL",
    "pct_change_eow": "REAL",
    "direction_1h": "TEXT",
    "direction_4h": "TEXT",
    "direction_eod": "TEXT",
    "direction_eow": "TEXT",

    # time-based features
    "day_of_week": "INTEGER",
    "hour_of_day": "INTEGER",
    "day_of_month": "INTEGER",
    "month": "INTEGER",
    "quarter": "INTEGER",
    "year": "INTEGER",
    "is_weekend": "BOOLEAN",
    "is_market_hours": "BOOLEAN",
    "is_premarket": "BOOLEAN",
    "is_aftermarket": "BOOLEAN",
    "is_opening_hour": "BOOLEAN",
    "is_closing_hour": "BOOLEAN",

    # cyclical features
    "hour_sin": "REAL",
    "hour_cos": "REAL",
    "day_sin": "REAL",
    "day_cos": "REAL",

    # advanced sentiment
    "sentiment_combined_strength": "REAL",
    "sentiment_combined_positive": "REAL",
    "sentiment_combined_negative": "REAL",
    "sentiment_combined_neutral": "REAL",
    "sentiment_combined_very_positive": "REAL",
    "sentiment_combined_very_negative": "REAL",
    "sentiment_combined_confidence": "REAL",

    # derived movement features
    "pct_change_1h_abs": "REAL",
    "pct_change_1h_positive": "BOOLEAN",
    "pct_change_1h_negative": "BOOLEAN",
    "pct_change_1h_significant": "BOOLEAN",
    "pct_change_4h_abs": "REAL",
    "pct_change_4h_positive": "BOOLEAN",
    "pct_change_4h_negative": "BOOLEAN",
    "pct_change_4h_significant": "BOOLEAN",
    "pct_change_eod_abs": "REAL",
    "pct_change_eod_positive": "BOOLEAN",
    "pct_change_eod_negative": "BOOLEAN",
    "pct_change_eod_significant": "BOOLEAN",
    "pct_change_eow_abs": "REAL",
    "pct_change_eow_positive": "BOOLEAN",
    "pct_change_eow_negative": "BOOLEAN",
    "pct_change_eow_significant": "BOOLEAN",

    # target labels
    "target_pct_change_1h_up_0_02": "BOOLEAN",
    "target_pct_change_1h_down_0_02": "BOOLEAN",
    "target_pct_change_1h_direction_0_02": "TEXT",
    "target_pct_change_4h_up_0_02": "BOOLEAN",
    "target_pct_change_4h_down_0_02": "BOOLEAN",
    "target_pct_change_4h_direction_0_02": "TEXT",
    "target_pct_change_eod_up_0_02": "BOOLEAN",
    "target_pct_change_eod_down_0_02": "BOOLEAN",
    "target_pct_change_eod_direction_0_02": "TEXT",
    "target_pct_change_eow_up_0_02": "BOOLEAN",
    "target_pct_change_eow_down_0_02": "BOOLEAN",
    "target_pct_change_eow_direction_0_02": "TEXT"
}
    # ─────────────────────────────────────────────────────────────────────

    # does the table already exist?
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
    )
    if cursor.fetchone():
        # patch any missing columns
        cursor.execute("PRAGMA table_info(articles)")
        existing = {row[1] for row in cursor.fetchall()}
        for col, col_type in all_required_columns.items():
            if col not in existing:
                cursor.execute(
                    f"ALTER TABLE articles ADD COLUMN {col} {col_type}"
                )
                logger.info(f"Added column: {col}")
    else:
        # brand‑new table
        column_defs = ", ".join(f"{c} {t}" for c, t in all_required_columns.items())
        cursor.execute(f"CREATE TABLE articles ({column_defs})")
        logger.info("Created new, slim articles table")

    conn.commit()
    conn.close()

def save_articles(articles):
    if not articles:
        logger.warning("No articles to save")
        return
    
    df = pd.DataFrame(articles)
    df.to_csv(CONFIG['CSV_OUTPUT'], index=False)
    logger.info(f"Saved {len(articles)} articles to {CONFIG['CSV_OUTPUT']}")
    
    # Show accuracy stats for different sentiment methods
    methods = ['dynamic', 'ml', 'keyword', 'combined']
    intervals = ['1h', '4h', 'eod', 'eow']
    
    for method in methods:
        for interval in intervals:
            col = f'accuracy_{method}_{interval}'
            if col in df.columns:
                valid = df[col].dropna()
                if not valid.empty:
                    correct = valid.sum()
                    total = len(valid)
                    logger.info(f"Prediction accuracy {method.upper()} {interval.upper()}: {correct}/{total} ({correct/total*100:.1f}%)")
    
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
            # Fill NaN values with appropriate defaults
            new_articles = new_articles.fillna({
                'text': '',
                'tokens': '',
                'mentions': '',
                'pos_keywords': '',
                'neg_keywords': '',
                'total_keywords': 0,
                'sentiment_score': 0,
                'sentiment_dynamic': 0.0,
                'sentiment_ml': 0.0,
                'sentiment_keyword': 0.0,
                'sentiment_combined': 0.0,
                'prediction_confidence': 0.0
            })

            new_articles.to_sql("articles", conn, if_exists="append", index=False)
            logger.info(f"Inserted {len(new_articles)} new articles into the database")
        else:
            logger.info("No new articles to insert")

    except Exception as e:
        logger.error(f"Error saving articles to database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("Starting News Sentiment Analysis Pipeline")

    init_database()
    processor = NewsProcessor()

    filtered_tickers = processor.filter_tickers_with_news(processor.valid_tickers)

    start = CONFIG['BATCH_START']
    end = start + CONFIG['BATCH_SIZE']
    batch_tickers = filtered_tickers[start:end]

    all_articles = []  # Collect all articles first

    for ticker in batch_tickers:
        try:
            logger.info(f"Processing ticker: {ticker}")

            articles = processor.fetch_finviz_news(ticker)

            if not articles:
                logger.warning(f"No articles processed for {ticker}")
                continue

            all_articles.extend(articles)

        except Exception as e:
            logger.error(f"Failed to process ticker {ticker}: {e}")
            continue

    # Now apply feature engineering to ALL articles at once
    if all_articles:
        df = pd.DataFrame(all_articles)
        
        # Save the articles (with additional features)
        df = feature_engineering_pipeline(df)
        save_articles(df.to_dict(orient="records"))
        
        logger.info(f"Final dataset has {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"Columns: {list(df.columns)}")
    else:
        logger.warning("No articles to process")

    logger.info("Pipeline complete.")



