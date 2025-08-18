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
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keybert import KeyBERT
from difflib import get_close_matches
from googlesearch import search
import numpy as np
import json
import time
from word_analysis_framework import DynamicSentimentLearner, EnhancedNewsProcessor, run_sentiment_analysis
from feature_engineering import FinancialNewsFeatureEngineer
from ticker_filter import get_tickers_with_news 
import traceback
import concurrent.futures
import traceback

# List of rotating User-Agent headers
USER_AGENTS = [ 
    "Mozilla/5.0 (Linux; Android 10; SM-G975F)",
    "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
    "Mozilla/5.0 (Windows NT 10.0; WOW64)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
]

# Configuration
CONFIG = {
    'DB_PATH': "articles.db",
    'CSV_INPUT': "finviz.csv", 
    'CSV_OUTPUT': "scraped_articles.csv",
    'MAX_TICKERS': 2500,
    'BATCH_SIZE': 100,
    'BATCH_START': 0,
    'DAYS_BACK': 14,          
}

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
        # Initialize session and core components
        self.session = self._create_session()
        self.valid_tickers = self._load_tickers()
        self.stopwords = self._setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        
        # === Eagerly load NLTK data to prevent thread race conditions ===
        try:
            logger.info("Priming NLTK resources for multithreading...")
            # These dummy calls force the lazy-loaded resources to be loaded into memory now.
            word_tokenize('priming call')
            self.lemmatizer.lemmatize('priming')
            logger.info("NLTK resources primed successfully.")
        except Exception as e:
            logger.error(f"Failed to prime NLTK resources: {e}. Multithreading may be unstable.")

        # Initialize KeyBERT model
        try:
            self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        except Exception as e:
            self.kw_model = None
            logger.warning(f"KeyBERT model not available: {e}")
        
        # Initialize analyzers
        self.price_analyzer = SimplifiedPriceAnalyzer()
        
        # Initialize ENHANCED sentiment analysis
        self.sentiment_learner = DynamicSentimentLearner()
        self.enhanced_processor = EnhancedNewsProcessor()
        self.sentiment_weights = self.load_sentiment_weights()
    
    def load_sentiment_weights(self):
        """Load enhanced sentiment weights from analysis"""
        try:
            # Try enhanced results first
            with open("enhanced_analysis_results.json", 'r') as f:
                results = json.load(f)
                weights = results.get('sentiment_weights', {})
                logger.info(f"Loaded {len(weights)} enhanced sentiment weights")
                return weights
        except FileNotFoundError:
            try:
                # Fallback to original results
                with open("word_analysis_results.json", 'r') as f:
                    results = json.load(f)
                    weights = results.get('sentiment_weights', {})
                    logger.info(f"Loaded {len(weights)} original sentiment weights")
                    return weights
            except FileNotFoundError:
                logger.warning("No sentiment analysis results found, using basic sentiment")
                return {}
    
    def calculate_dynamic_sentiment(self, text):
        """Calculate sentiment using enhanced learned weights"""
        if not self.sentiment_weights:
            return 0
        
        # Use the enhanced processor's method if available
        if hasattr(self.enhanced_processor, 'calculate_enhanced_sentiment'):
            try:
                enhanced_results = self.enhanced_processor.calculate_enhanced_sentiment(text)
                return enhanced_results.get('combined', 0)
            except Exception as e:
                logger.warning(f"Enhanced sentiment calculation failed: {e}")
        
        # Fallback to improved keyword-based calculation
        return self._calculate_improved_keyword_sentiment(text)

    def _calculate_improved_keyword_sentiment(self, text):
        """Improved keyword-based sentiment with negation handling"""
        if not text or not self.sentiment_weights:
            return 0
        
        negation_words = {'not', 'no', 'never', "n't", 'none', 'cannot', 'won\'t', 'don\'t', 'isn\'t', 'aren\'t'}
        words = re.findall(r'\b\w+\b', text.lower())
        
        total_score = 0
        total_weight = 0
        
        i = 0
        while i < len(words):
            # Check for negation in preceding 3 words
            negated = any(w in negation_words for w in words[max(0, i-3):i])
            
            # Check bigrams first (higher priority)
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in self.sentiment_weights:
                    weight_info = self.sentiment_weights[bigram]
                    score = weight_info['weight'] * weight_info['confidence']
                    if negated:
                        score *= -0.8  # Slightly reduce negation impact
                    total_score += score
                    total_weight += weight_info['confidence']
                    i += 2
                    continue
            
            # Check individual words
            word = words[i]
            if word in self.sentiment_weights:
                weight_info = self.sentiment_weights[word]
                score = weight_info['weight'] * weight_info['confidence']
                if negated:
                    score *= -0.8
                total_score += score
                total_weight += weight_info['confidence']
            
            i += 1
        
        if total_weight == 0:
            return 0
        
        normalized_score = total_score / total_weight
        return max(-1, min(1, normalized_score))  # Clamp to [-1, 1]
    
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

        # Handle explicit "15 min" style
        if re.match(r"^\d+\s*min$", s):
            minutes = int(re.findall(r"\d+", s)[0])
            return (now - timedelta(minutes=minutes)).replace(second=0, microsecond=0)
        
        # Better handling of relative dates
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
    
    def _process_article_text_thread_safe(self, text, exclude_ticker):
        """
        A thread-safe method to process text by initializing NLTK within the method.
        """
        # Each thread creates its own lemmatizer to avoid conflicts
        lemmatizer = WordNetLemmatizer()

        # Preprocess text (lemmatize, tokenize, etc.)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text.lower())
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(re.sub(r"\s+", " ", text).strip())
        processed_text = " ".join(
            lemmatizer.lemmatize(token) for token in tokens 
            if token not in self.stopwords and len(token) > 1
        )

        # The rest of the original extract_mentions_and_sentiment logic
        potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
        mentions = list(dict.fromkeys([
            t for t in potential_tickers 
            if t in self.valid_tickers and t != exclude_ticker
        ]))

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
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            
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
        """
        Fetch news for a ticker from Finviz, with:
        - Database check to skip already-scraped tickers
        - Scraping + fallback search
        - Enhanced sentiment and price data
        """
        try:
            # ==== 0. Skip if already processed ====
            conn = sqlite3.connect(CONFIG['DB_PATH'])
            exists_df = pd.read_sql(
                "SELECT 1 FROM articles WHERE ticker = ? LIMIT 1",
                conn,
                params=(ticker,)
            )
            conn.close()
            if not exists_df.empty:
                logger.info(f"Skipping {ticker}: already exists in DB")
                return []
        except Exception as e:
            logger.warning(f"Could not check ticker {ticker} in DB: {e}")

        # Randomized short sleep to avoid server-side rate limiting
        time.sleep(random.uniform(0.2, 1.0))

        base_url = f"https://finviz.com/quote.ashx?t={ticker}"
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            response = self.session.get(base_url, headers=headers, timeout=10)
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

            # Try scraping the article
            article_text = self.scrape_article(article_url)
            if not article_text:
                # Fallback search if direct scrape failed
                article_text, fallback_url = self.fallback_search(headline)
                if article_text:
                    article_url = fallback_url
            if not article_text:
                continue

            mentions, pos_kw, neg_kw, tokens = self._process_article_text_thread_safe(article_text, ticker)

            # --- Enhanced Sentiment ---
            full_text = f"{headline} {article_text}"
            enhanced_sentiment = self.enhanced_processor.calculate_enhanced_sentiment(full_text)

            ml_prediction = 0.5
            if hasattr(self.sentiment_learner, 'predict_sentiment') and self.sentiment_learner.sentiment_model:
                try:
                    ml_prediction = self.sentiment_learner.predict_sentiment(full_text)
                except Exception as e:
                    logger.debug(f"ML prediction failed: {e}")

            sentiment_combined_score = enhanced_sentiment.get('combined', 0)
            prediction_confidence = abs(sentiment_combined_score)

            if sentiment_combined_score > 0.1 and prediction_confidence > 0.3:
                sentiment_category = "Bullish"
            elif sentiment_combined_score < -0.1 and prediction_confidence > 0.3:
                sentiment_category = "Bearish"
            elif prediction_confidence > 0.5:
                sentiment_category = "Strong Neutral"
            else:
                sentiment_category = "Weak Signal"

            # --- Price Data ---
            price_data = self.get_price_data(ticker, parsed_dt)

            # --- Article Record ---
            article_entry = {
                "ticker": ticker,
                "datetime": cols[0].get_text(strip=True),
                "headline": headline,
                "url": article_url,
                "text": article_text,
                "tokens": tokens,
                "sentiment_dynamic": enhanced_sentiment.get('dynamic_weights', 0),
                "sentiment_ml": ml_prediction,
                "sentiment_keyword": enhanced_sentiment.get('keyword_based', 0),
                "sentiment_combined": sentiment_combined_score,
                "prediction_confidence": prediction_confidence,
                "sentiment_category": sentiment_category,
                "mentions": ', '.join(mentions) if mentions else '',
                "pos_keywords": ', '.join(pos_kw) if pos_kw else '',
                "neg_keywords": ', '.join(neg_kw) if neg_kw else '',
                "total_keywords": len(pos_kw) + len(neg_kw),
                "text_length": len(article_text.split()),
                "headline_sentiment": self.calculate_dynamic_sentiment(headline),
                "keyword_density": (len(pos_kw) + len(neg_kw)) / len(article_text.split()) if article_text else 0
            }

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

        logger.info(f"Found {len(articles)} new articles for {ticker}")
        return articles

    def _check_prediction_accuracy(self, is_sentiment_positive, pct_change):
        if pct_change is None:
            return None
        return (is_sentiment_positive and pct_change > 0) or (not is_sentiment_positive and pct_change < 0)
    
    def filter_tickers_with_news(self, tickers):
        """Filter tickers that have news available (fallback method)"""
        tickers_with_news = []
        
        for ticker in tickers[:20]:  # Limit for testing
            try:
                url = f"https://finviz.com/quote.ashx?t={ticker}"
                headers = {"User-Agent": random.choice(USER_AGENTS)}
                response = self.session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "html.parser")
                news_table = soup.select_one("table.fullview-news-outer") or soup.select_one("#news-table")
                
                if news_table and news_table.find_all("tr"):
                    tickers_with_news.append(ticker)
                    logger.info(f"✓ {ticker} has news")
                else:
                    logger.debug(f"✗ {ticker} no news")
                    
                time.sleep(random.uniform(1, 2))  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error checking {ticker}: {e}")
                continue
        
        return tickers_with_news

def get_processed_tickers():
    """Return a set of tickers already stored in the articles DB."""
    try:
        conn = sqlite3.connect(CONFIG['DB_PATH'])
        df = pd.read_sql("SELECT DISTINCT ticker FROM articles", conn)
        conn.close()
        existing = set(df['ticker'].dropna().str.upper())
        logger.info(f"Found {len(existing)} tickers already processed in DB")
        return existing
    except Exception as e:
        logger.warning(f"Could not retrieve processed tickers: {e}")
        return set()

def init_database():
    """Create (or patch) the 'articles' table to match all required features."""

    conn = sqlite3.connect(CONFIG["DB_PATH"])
    cursor = conn.cursor()

    all_required_columns = {
        # --- Basic Info ---
        "ticker": "TEXT", "datetime": "TEXT", "headline": "TEXT", "url": "TEXT", "text": "TEXT",
        
        # --- Text & Keyword Features ---
        "tokens": "TEXT", "mentions": "TEXT", "pos_keywords": "TEXT", "neg_keywords": "TEXT",
        "total_keywords": "INTEGER", "text_length": "INTEGER", "keyword_density": "REAL",

        # --- Sentiment Features ---
        "sentiment_dynamic": "REAL", "sentiment_ml": "REAL", "sentiment_keyword": "REAL",
        "sentiment_combined": "REAL", "headline_sentiment": "REAL", "prediction_confidence": "REAL",
        "sentiment_category": "TEXT", "ml_confidence": "REAL", "sentiment_strength": "REAL",
        "sentiment_score": "REAL",

        # --- Advanced Sentiment Features ---
        'sentiment_combined_strength': 'REAL', 'sentiment_combined_positive': 'REAL', 'sentiment_combined_negative': 'REAL',
        'sentiment_combined_very_positive': 'REAL', 'sentiment_combined_very_negative': 'REAL',
        'sentiment_combined_confidence': 'REAL', 'sentiment_combined_neutral': 'REAL',

        # --- OHLCV Data ---
        "Close": "REAL", "Open": "REAL", "High": "REAL", "Low": "REAL", "Volume": "REAL",
        "Adj_Close": "REAL", 

        # --- Raw Price Change Features ---
        "pct_change_1h": "REAL", "pct_change_4h": "REAL", "pct_change_eod": "REAL", "pct_change_eow": "REAL",
        "direction_1h": "TEXT", "direction_4h": "TEXT", "direction_eod": "TEXT", "direction_eow": "TEXT",

        # --- Derived Price Movement Features ---
        'pct_change_1h_abs': 'REAL', 'pct_change_4h_abs': 'REAL', 'pct_change_eod_abs': 'REAL', 'pct_change_eow_abs': 'REAL',
        'pct_change_1h_positive': 'BOOLEAN', 'pct_change_1h_negative': 'BOOLEAN', 'pct_change_1h_significant': 'BOOLEAN',
        'pct_change_4h_positive': 'BOOLEAN', 'pct_change_4h_negative': 'BOOLEAN', 'pct_change_4h_significant': 'BOOLEAN',
        'pct_change_eod_positive': 'BOOLEAN', 'pct_change_eod_negative': 'BOOLEAN', 'pct_change_eod_significant': 'BOOLEAN',
        'pct_change_eow_positive': 'BOOLEAN', 'pct_change_eow_negative': 'BOOLEAN', 'pct_change_eow_significant': 'BOOLEAN',
        
        # --- Time Features ---
        "day_of_week": "INTEGER", "hour_of_day": "INTEGER", "day_of_month": "INTEGER", "month": "INTEGER",
        "quarter": "INTEGER", "year": "INTEGER", "is_weekend": "BOOLEAN", "is_market_hours": "BOOLEAN",
        "is_premarket": "BOOLEAN", "is_aftermarket": "BOOLEAN", "is_opening_hour": "BOOLEAN", "is_closing_hour": "BOOLEAN",
        "hour_sin": "REAL", "hour_cos": "REAL", "day_sin": "REAL", "day_cos": "REAL",

        # --- Technical Indicators (from pandas-ta) ---
        "rsi_14": "REAL", "macd": "REAL", "macd_signal": "REAL", "macd_hist": "REAL",
        "price_vs_sma50": "REAL", "price_vs_sma200": "REAL", "MACD_12_26_9": "REAL",
        "MACDh_12_26_9": "REAL", "MACDs_12_26_9": "REAL", "ema_12": "REAL", "ema_26": "REAL",
        "sma_50": "REAL", "sma_200": "REAL", "bb_mid": "REAL", "bb_upper": "REAL", "bb_lower": "REAL",

        # --- Market Context ---
        "vix_close": "REAL", "spy_daily_return": "REAL",
        
        # --- Target Labels for ML ---
        'target_pct_change_1h_up_0_02': 'BOOLEAN', 'target_pct_change_1h_down_0_02': 'BOOLEAN',
        'target_pct_change_4h_up_0_02': 'BOOLEAN', 'target_pct_change_4h_down_0_02': 'BOOLEAN',
        'target_pct_change_eod_up_0_02': 'BOOLEAN', 'target_pct_change_eod_down_0_02': 'BOOLEAN',
        'target_pct_change_eow_up_0_02': 'BOOLEAN', 'target_pct_change_eow_down_0_02': 'BOOLEAN',
        'target_pct_change_1h_direction_0_02': 'TEXT', 'target_pct_change_4h_direction_0_02': 'TEXT',
        'target_pct_change_eod_direction_0_02': 'TEXT', 'target_pct_change_eow_direction_0_02': 'TEXT'
    }

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
    if cursor.fetchone():
        # Patch missing columns
        cursor.execute("PRAGMA table_info(articles)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        for col, col_type in all_required_columns.items():
            if col not in existing_cols:
                try:
                    # Use quotes to handle column names with spaces
                    cursor.execute(f'ALTER TABLE articles ADD COLUMN "{col}" {col_type}')
                    logger.info(f"Added missing column: {col}")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Could not add column {col}: {e}")
    else:
        # Create table from scratch
        # Use quotes to handle column names with spaces
        cols_sql = ", ".join(f'"{col}" {ctype}' for col, ctype in all_required_columns.items())
        cursor.execute(f"CREATE TABLE articles ({cols_sql})")
        logger.info("Created 'articles' table with full schema")

    conn.commit()
    conn.close()

def save_enhanced_articles(articles_df):
    """Save articles with ALL feature engineering outputs to database."""
    if articles_df.empty:
        logger.warning("No articles to save")
        return

    logger.info(f"Processing {len(articles_df)} articles for enhanced saving...")
    
    # --- Save to CSV with all features ---
    try:
        output_file = CONFIG['CSV_OUTPUT']
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            # Check if 'url' column exists in both DataFrames
            if 'url' in existing_df.columns and 'url' in articles_df.columns:
                new_df = articles_df[~articles_df['url'].isin(existing_df['url'])]
                
                if not new_df.empty:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_csv(output_file, index=False)
                    logger.info(f"Appended {len(new_df)} new articles to {output_file}")
                else:
                    logger.info("No new unique articles to add to CSV.")
            else:
                # If URL column missing, append all articles
                combined_df = pd.concat([existing_df, articles_df], ignore_index=True)
                combined_df.to_csv(output_file, index=False)
                logger.info(f"Appended {len(articles_df)} articles to {output_file}")
        else:
            articles_df.to_csv(output_file, index=False)
            logger.info(f"Created new CSV {output_file} with {len(articles_df)} articles")
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")

    # --- Save to Database with ALL engineered features ---
    conn = None
    try:
        conn = sqlite3.connect(CONFIG['DB_PATH'])
        
        # Check if articles table exists and has url column
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists and 'url' in articles_df.columns:
            try:
                existing_urls_query = "SELECT url FROM articles WHERE url IS NOT NULL"
                existing_urls = pd.read_sql(existing_urls_query, conn)["url"].tolist()
                new_to_db_df = articles_df[~articles_df["url"].isin(existing_urls)]
            except Exception as e:
                logger.warning(f"Could not check existing URLs: {e}. Proceeding with all articles.")
                new_to_db_df = articles_df
        else:
            new_to_db_df = articles_df

        if not new_to_db_df.empty:
            # Create a copy to avoid modifying the original DataFrame
            new_articles_filled = new_to_db_df.copy()
            
            # Only fill NaN values for columns that actually exist in the DataFrame
            fillna_dict = {}
            
            # Basic text fields
            for col in ['text', 'tokens', 'mentions', 'pos_keywords', 'neg_keywords']:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = ''
            if 'sentiment_category' in new_articles_filled.columns:
                fillna_dict['sentiment_category'] = 'Neutral'
            
            # Numeric sentiment fields (match your main.py structure)
            numeric_sentiment_cols = [
                'sentiment_dynamic', 'sentiment_ml', 'sentiment_keyword', 'sentiment_combined',
                'prediction_confidence', 'text_length', 'headline_sentiment', 'keyword_density',
                'total_keywords', 'ml_confidence', 'sentiment_strength'
            ]
            for col in numeric_sentiment_cols:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = 0.0
            
            # Stock price fields (OHLCV data)
            price_data_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
            for col in price_data_cols:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = 0.0
            
            # Price change fields
            price_change_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
            for col in price_change_cols:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = 0.0
            
            direction_cols = ['direction_1h', 'direction_4h', 'direction_eod', 'direction_eow']
            for col in direction_cols:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = 'No Data'
            
            # Technical indicators
            if 'rsi_14' in new_articles_filled.columns:
                fillna_dict['rsi_14'] = 50.0
            for col in ['macd', 'macd_signal', 'macd_hist', 'price_vs_sma50', 'price_vs_sma200']: # <-- ADD 'macd_hist' HERE
                if col in new_articles_filled.columns:
                    fillna_dict[col] = 0.0
            
            # Market context
            if 'vix_close' in new_articles_filled.columns:
                fillna_dict['vix_close'] = 20.0
            if 'spy_daily_return' in new_articles_filled.columns:
                fillna_dict['spy_daily_return'] = 0.0
            
            # Time-based features (match your init_database structure)
            time_defaults = {
                'day_of_week': 1, 'hour_of_day': 9, 'day_of_month': 1,
                'month': 1, 'quarter': 1, 'year': 2024
            }
            for col, default_val in time_defaults.items():
                if col in new_articles_filled.columns:
                    fillna_dict[col] = default_val
            
            boolean_time_cols = {
                'is_weekend': False, 'is_market_hours': True, 'is_premarket': False,
                'is_aftermarket': False, 'is_opening_hour': False, 'is_closing_hour': False
            }
            for col, default_val in boolean_time_cols.items():
                if col in new_articles_filled.columns:
                    fillna_dict[col] = default_val
            
            # Cyclical features
            cyclical_defaults = {'hour_sin': 0.0, 'hour_cos': 1.0, 'day_sin': 0.0, 'day_cos': 1.0}
            for col, default_val in cyclical_defaults.items():
                if col in new_articles_filled.columns:
                    fillna_dict[col] = default_val
            
            # Advanced sentiment features
            sentiment_advanced_cols = [
                'sentiment_combined_strength', 'sentiment_combined_positive', 'sentiment_combined_negative',
                'sentiment_combined_very_positive', 'sentiment_combined_very_negative',
                'sentiment_combined_confidence', 'sentiment_score'
            ]
            for col in sentiment_advanced_cols:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = 0.0
            
            if 'sentiment_combined_neutral' in new_articles_filled.columns:
                fillna_dict['sentiment_combined_neutral'] = 1.0
            
            # Derived movement features
            movement_numeric_cols = [
                'pct_change_1h_abs', 'pct_change_4h_abs', 'pct_change_eod_abs', 'pct_change_eow_abs'
            ]
            for col in movement_numeric_cols:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = 0.0
            
            movement_boolean_cols = [
                'pct_change_1h_positive', 'pct_change_1h_negative', 'pct_change_1h_significant',
                'pct_change_4h_positive', 'pct_change_4h_negative', 'pct_change_4h_significant',
                'pct_change_eod_positive', 'pct_change_eod_negative', 'pct_change_eod_significant',
                'pct_change_eow_positive', 'pct_change_eow_negative', 'pct_change_eow_significant'
            ]
            for col in movement_boolean_cols:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = False
            
            # Target labels (from your init_database)
            target_boolean_cols = [
                'target_pct_change_1h_up_0_02', 'target_pct_change_1h_down_0_02',
                'target_pct_change_4h_up_0_02', 'target_pct_change_4h_down_0_02',
                'target_pct_change_eod_up_0_02', 'target_pct_change_eod_down_0_02',
                'target_pct_change_eow_up_0_02', 'target_pct_change_eow_down_0_02'
            ]
            for col in target_boolean_cols:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = False
            
            target_direction_cols = [
                'target_pct_change_1h_direction_0_02', 'target_pct_change_4h_direction_0_02',
                'target_pct_change_eod_direction_0_02', 'target_pct_change_eow_direction_0_02'
            ]
            for col in target_direction_cols:
                if col in new_articles_filled.columns:
                    fillna_dict[col] = 'neutral'
            
            # Apply the fillna only for columns that exist
            new_articles_filled = new_articles_filled.fillna(fillna_dict)
            
            new_articles_filled.to_sql("articles", conn, if_exists="append", index=False)
            logger.info(f"Successfully inserted {len(new_articles_filled)} new articles with ALL features into database")
            
            # Call show_article_statistics if the function exists (from your main.py)
            try:
                show_article_statistics(new_articles_filled)
            except NameError:
                logger.info("show_article_statistics function not available")
        else:
            logger.info("No new articles to insert into database (all duplicates).")
    except Exception as e:
        logger.error(f"Error saving enhanced articles to database: {e}")
        logger.error(traceback.format_exc())
    finally:
        if conn:
            conn.close()

def show_article_statistics(df):
    """Show statistics about the processed articles"""
    if df.empty:
        return
        
    # Show accuracy stats for different sentiment methods
    methods = ['dynamic', 'ml', 'keyword', 'combined']
    intervals = ['1h', '4h', 'eod', 'eow']
    
    logger.info("=== Article Processing Statistics ===")
    
    # Basic stats
    logger.info(f"Total articles: {len(df)}")
    logger.info(f"Unique tickers: {df['ticker'].nunique()}")
    
    # Price change statistics
    for interval in intervals:
        col = f'pct_change_{interval}'
        if col in df.columns:
            valid_changes = df[col].dropna()
            if not valid_changes.empty:
                avg_change = valid_changes.mean()
                pos_changes = (valid_changes > 0).sum()
                neg_changes = (valid_changes < 0).sum()
                logger.info(f"Price changes {interval.upper()}: avg={avg_change:+.2f}%, pos={pos_changes}, neg={neg_changes}")
    
    # Sentiment distribution
    if 'sentiment_combined' in df.columns:
        sentiment_scores = df['sentiment_combined'].dropna()
        if not sentiment_scores.empty:
            avg_sentiment = sentiment_scores.mean()
            pos_sentiment = (sentiment_scores > 0).sum()
            neg_sentiment = (sentiment_scores < 0).sum()
            logger.info(f"Sentiment: avg={avg_sentiment:.3f}, positive={pos_sentiment}, negative={neg_sentiment}")

def run_enhanced_sentiment_training():
    """Train the enhanced sentiment model with ALL engineered features."""
    logger.info("Training enhanced sentiment model with feature-engineered data...")
    
    try:
        # Load existing data from database
        conn = sqlite3.connect(CONFIG['DB_PATH'])
        articles_df = pd.read_sql("SELECT * FROM articles", conn)
        conn.close()
        
        if articles_df.empty:
            logger.warning("No articles in database for training")
            return False
        
        # Apply feature engineering to ensure we have all features
        logger.info("Applying feature engineering to training data...")
        engineer = FinancialNewsFeatureEngineer()
        enhanced_df = engineer.feature_engineering_pipeline(articles_df)
        
        if enhanced_df.empty:
            logger.error("Feature engineering returned empty DataFrame")
            return False
        
        results = run_sentiment_analysis()
        
        model_perf = results.get('model_performance', {})
        
        if results and model_perf.get('model_trained'):
            # --- THIS IS THE CORRECTED LOGIC ---
            # Get the raw test data and predictions
            y_test = model_perf.get('y_test_for_metrics', [])
            y_pred = model_perf.get('y_pred_optimized_for_metrics', [])
            
            # Calculate metrics only if data is available
            if y_test and y_pred:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                # Get the F1 score from the correct key
                f1_score = model_perf.get('test_f1_optimized', 0)
            else:
                accuracy, precision, recall, f1_score = 0.0, 0.0, 0.0, 0.0

            logger.info("Enhanced model trained successfully with engineered features:")
            logger.info(f"  - Accuracy: {accuracy:.3f}")
            logger.info(f"  - Precision: {precision:.3f}")
            logger.info(f"  - Recall: {recall:.3f}")
            logger.info(f"  - F1 Score: {f1_score:.3f}")
            logger.info(f"  - Training samples: {model_perf.get('training_samples', 'N/A')}")
            logger.info(f"  - Features used: {model_perf.get('feature_count', 'N/A')}")
            
            return True
        else:
            error_message = model_perf.get('error', 'Unknown error during training')
            logger.error("--- MODEL TRAINING FAILED ---")
            logger.error(f"Reason: {error_message}")
            logger.warning("Proceeding with existing model if available.")
            return False
            
    except Exception as e:
        logger.error(f"Training process crashed: {e}", exc_info=True)
        return False

# === NEW UNIFIED SAVE FUNCTION ===
def save_synchronized_files(raw_df, engineered_df):
    """
    Saves both raw and engineered data to their respective CSVs,
    ensuring they remain perfectly in sync by using the same duplicate check.
    """
    if raw_df.empty:
        logger.warning("Received empty raw dataframe, nothing to save.")
        return

    # --- Save scraped_articles.csv (Raw Data) ---
    raw_output_file = "scraped_articles.csv"
    try:
        new_raw_df = raw_df
        if os.path.exists(raw_output_file):
            existing_raw_df = pd.read_csv(raw_output_file)
            new_raw_df = raw_df[~raw_df['url'].isin(existing_raw_df['url'])]
            if not new_raw_df.empty:
                combined_df = pd.concat([existing_raw_df, new_raw_df], ignore_index=True)
                combined_df.to_csv(raw_output_file, index=False)
        else:
            new_raw_df.to_csv(raw_output_file, index=False)
        
        if not new_raw_df.empty:
            logger.info(f"Saved/Appended {len(new_raw_df)} new rows to {raw_output_file}")

    except Exception as e:
        logger.error(f"Error saving to {raw_output_file}: {e}")

    # --- Save cleaned_engineered_features.csv (Engineered Data) ---
    # Only proceed if there were new raw articles to add
    if not new_raw_df.empty and not engineered_df.empty:
        engineered_output_file = "cleaned_engineered_features.csv"
        try:
            # Filter the engineered data to only include the newly added raw articles
            new_engineered_df = engineered_df[engineered_df['url'].isin(new_raw_df['url'])]
            
            if os.path.exists(engineered_output_file):
                existing_eng_df = pd.read_csv(engineered_output_file)
                if not new_engineered_df.empty:
                    combined_df = pd.concat([existing_eng_df, new_engineered_df], ignore_index=True)
                    combined_df.to_csv(engineered_output_file, index=False)
            else:
                new_engineered_df.to_csv(engineered_output_file, index=False)
            
            if not new_engineered_df.empty:
                logger.info(f"Saved/Appended {len(new_engineered_df)} new rows to {engineered_output_file}")

        except Exception as e:
            logger.error(f"Error saving to {engineered_output_file}: {e}")

# Main execution logic
def main():
    """Main execution function with complete feature engineering pipeline."""
    logger.info("Starting Enhanced News Sentiment Analysis Pipeline")
    
    init_database()
    processor = NewsProcessor()

    # Initialize feature engineer
    engineer = FinancialNewsFeatureEngineer()

    # Ticker Loading
    logger.info("Loading pre-filtered tickers...")
    try:
        filtered_tickers = get_tickers_with_news()
        if not filtered_tickers:
            raise ValueError("Pre-filtered list is empty.")
        logger.info(f"Loaded {len(filtered_tickers)} pre-filtered tickers")
    except Exception as e:
        logger.warning(f"Failed to load pre-filtered tickers: {e}. Using fallback.")
        filtered_tickers = sorted(list(processor.valid_tickers))

    if not filtered_tickers:
        logger.error("No tickers available to process. Exiting.")
        return

    # Ticker Sampling
    if len(filtered_tickers) > CONFIG['MAX_TICKERS']:
        random.shuffle(filtered_tickers)
        filtered_tickers = filtered_tickers[:CONFIG['MAX_TICKERS']]

    # NEW: Skip already processed tickers
    processed_tickers = get_processed_tickers()
    filtered_tickers = [t for t in filtered_tickers if t not in processed_tickers]

    if not filtered_tickers:
        logger.info("All tickers have already been processed. Exiting.")
        return

    total_tickers = len(filtered_tickers)
    logger.info(f"Processing a total of {total_tickers} NEW tickers.")

    # Batch Processing with Complete Pipeline
    batch_size = CONFIG['BATCH_SIZE']
    total_batches = (total_tickers + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_tickers)
        batch_tickers = filtered_tickers[start_idx:end_idx]
        
        logger.info(f"--- Starting Batch {batch_num + 1}/{total_batches} ({len(batch_tickers)} tickers) ---")
        
        batch_articles = []
        
        # Parallel Scraping
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(processor.fetch_finviz_news, batch_tickers)
            
            for article_list in results:
                if article_list:
                    batch_articles.extend(article_list)
        
        logger.info(f"Batch {batch_num + 1} collected {len(batch_articles)} raw articles.")
        
        # COMPLETE PIPELINE: Raw -> Engineered -> Save -> Train
        if batch_articles:
            try:
                # Step 1: Convert to DataFrame
                raw_df = pd.DataFrame(batch_articles)
                logger.info(f"Created DataFrame with {len(raw_df)} articles and {len(raw_df.columns)} columns")
                
                # Step 2: Apply complete feature engineering
                logger.info("Applying feature engineering pipeline...")
                engineered_df = engineer.feature_engineering_pipeline(raw_df)
                
                if engineered_df.empty:
                    logger.warning("Feature engineering produced empty DataFrame, skipping batch")
                    continue
                
                logger.info(f"Feature engineering complete: {len(engineered_df)} articles with {len(engineered_df.columns)} features")
                
                # Step 3: Save both the raw and the engineered CSV files
                save_synchronized_files(raw_df, engineered_df)

                # Step 4: Also save the engineered data to the database
                save_enhanced_articles(engineered_df)

                # Step 5: Retrain model with enhanced data
                training_success = run_enhanced_sentiment_training()
                
                if training_success:
                    logger.info("✓ Batch processing complete: scraped -> engineered -> saved -> retrained")
                else:
                    logger.warning("✗ Training failed, but data was saved successfully")
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num + 1}: {e}", exc_info=True)
                continue
        else:
            logger.warning(f"No articles collected in batch {batch_num + 1}")

    logger.info("=== Enhanced Pipeline Completed Successfully ===")
    
    # Final summary
    try:
        conn = sqlite3.connect(CONFIG['DB_PATH'])
        total_count = pd.read_sql("SELECT COUNT(*) as count FROM articles", conn).iloc[0]['count']
        conn.close()
        logger.info(f"Total articles in database: {total_count}")
    except Exception as e:
        logger.error(f"Could not get final count: {e}")

if __name__ == "__main__":
    print("Script started, calling main()")
    main()
    print("Script completed")