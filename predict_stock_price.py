import pandas as pd
import numpy as np
import logging
import sys
import os
import joblib
from datetime import datetime, time, timedelta
import pytz
import re
from bs4 import BeautifulSoup
from pyfinviz.news import News
from main import SimplifiedPriceAnalyzer, NewsProcessor
from feature_engineering import FinancialNewsFeatureEngineer
from train_price_regressor import create_advanced_features
from word_analysis_framework import EnhancedNewsProcessor
import json
import requests
import time as time_module
import threading
from collections import defaultdict
import signal

# === Market Hours Configuration ===
MARKET_TIMEZONE = pytz.timezone("US/Eastern")
MARKET_OPEN_TIME = time(9, 30)  # 9:30 AM ET
MARKET_CLOSE_TIME = time(16, 0)  # 4:00 PM ET
EXTENDED_HOURS_START = time(4, 0)  # 4:00 AM ET (pre-market start)
EXTENDED_HOURS_END = time(20, 0)   # 8:00 PM ET (after-hours end)

# Different intervals for different periods
MARKET_HOURS_INTERVAL = 300      # 5 minutes during market hours
AFTER_HOURS_INTERVAL = 900       # 15 minutes during after hours
OVERNIGHT_INTERVAL = 1800        # 30 minutes overnight
WEEKEND_INTERVAL = 3600          # 1 hour on weekends

NEWS_CACHE_DURATION = 3600       # 1 hour cache for news articles (extended for after-hours)
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

# After-hours trading significance
PROCESS_AFTER_HOURS = True       # Set to False to disable after-hours processing
PROCESS_WEEKENDS = True          # Process weekend news for Monday predictions

# === Logging Setup ===
def setup_logging():
    log_filename = f"stock_predictions_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename)
        ]
    )
    return logging.getLogger("MarketHoursPredictor")

logger = setup_logging()

# === Market Hours Utilities ===
class MarketSchedule:
    def __init__(self):
        self.market_tz = MARKET_TIMEZONE
        # Common market holidays (you may want to expand this)
        self.holidays_2025 = [
            "2025-01-01",  # New Year's Day
            "2025-01-20",  # MLK Day
            "2025-02-17",  # Presidents Day
            "2025-04-18",  # Good Friday
            "2025-05-26",  # Memorial Day
            "2025-07-04",  # Independence Day
            "2025-09-01",  # Labor Day
            "2025-11-27",  # Thanksgiving
            "2025-12-25",  # Christmas
        ]
    
    def is_market_open(self, dt=None):
        """Check if market is currently open"""
        if dt is None:
            dt = datetime.now(self.market_tz)
        else:
            dt = dt.astimezone(self.market_tz)
        
        # Check if it's a weekend
        if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        date_str = dt.strftime('%Y-%m-%d')
        if date_str in self.holidays_2025:
            return False
        
        # Check if it's within market hours
        current_time = dt.time()
        return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME
    
    def is_extended_hours(self, dt=None):
        """Check if it's extended hours (pre-market + after-hours)"""
        if dt is None:
            dt = datetime.now(self.market_tz)
        else:
            dt = dt.astimezone(self.market_tz)
        
        # Check if it's a weekend
        if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        date_str = dt.strftime('%Y-%m-%d')
        if date_str in self.holidays_2025:
            return False
        
        # Check if it's within extended hours but not regular market hours
        current_time = dt.time()
        in_extended = EXTENDED_HOURS_START <= current_time <= EXTENDED_HOURS_END
        in_market = MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME
        
        return in_extended and not in_market
    
    def is_weekend(self, dt=None):
        """Check if it's weekend"""
        if dt is None:
            dt = datetime.now(self.market_tz)
        else:
            dt = dt.astimezone(self.market_tz)
        return dt.weekday() >= 5
    
    def get_market_session(self, dt=None):
        """Get current market session type"""
        if dt is None:
            dt = datetime.now(self.market_tz)
        else:
            dt = dt.astimezone(self.market_tz)
        
        if self.is_market_open(dt):
            return "MARKET_HOURS"
        elif self.is_extended_hours(dt):
            current_time = dt.time()
            if current_time < MARKET_OPEN_TIME:
                return "PRE_MARKET"
            else:
                return "AFTER_HOURS"
        elif self.is_weekend(dt):
            return "WEEKEND"
        else:
            return "CLOSED"
    
    def time_until_market_open(self, dt=None):
        """Calculate time until next market open"""
        if dt is None:
            dt = datetime.now(self.market_tz)
        else:
            dt = dt.astimezone(self.market_tz)
        
        # If market is currently open, return 0
        if self.is_market_open(dt):
            return timedelta(0)
        
        # Find next market open
        next_open = dt.replace(hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute, second=0, microsecond=0)
        
        # If we're past today's market open, move to next business day
        if dt.time() > MARKET_OPEN_TIME or dt.weekday() >= 5:
            next_open += timedelta(days=1)
            
            # Skip weekends
            while next_open.weekday() >= 5 or next_open.strftime('%Y-%m-%d') in self.holidays_2025:
                next_open += timedelta(days=1)
        
        return next_open - dt
    
    def time_until_market_close(self, dt=None):
        """Calculate time until market close"""
        if dt is None:
            dt = datetime.now(self.market_tz)
        else:
            dt = dt.astimezone(self.market_tz)
        
        if not self.is_market_open(dt):
            return timedelta(0)
        
        market_close = dt.replace(hour=MARKET_CLOSE_TIME.hour, minute=MARKET_CLOSE_TIME.minute, second=0, microsecond=0)
        return market_close - dt

# === News Cache System ===
class NewsCache:
    def __init__(self, cache_duration=NEWS_CACHE_DURATION):
        self.cache = {}
        self.cache_duration = cache_duration
        self.processed_articles = set()  # Track processed article URLs to avoid duplicates
    
    def is_fresh(self, timestamp):
        """Check if cached data is still fresh"""
        return (datetime.now() - timestamp).seconds < self.cache_duration
    
    def add_article(self, url, data):
        """Add article to cache"""
        self.cache[url] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def get_article(self, url):
        """Get article from cache if fresh"""
        if url in self.cache and self.is_fresh(self.cache[url]['timestamp']):
            return self.cache[url]['data']
        return None
    
    def is_processed(self, article_id):
        """Check if article was already processed"""
        return article_id in self.processed_articles
    
    def mark_processed(self, article_id):
        """Mark article as processed"""
        self.processed_articles.add(article_id)
    
    def cleanup_old_entries(self):
        """Remove old cached entries"""
        current_time = datetime.now()
        old_entries = [url for url, data in self.cache.items() 
                      if (current_time - data['timestamp']).seconds > self.cache_duration]
        for url in old_entries:
            del self.cache[url]

# === Enhanced Prediction Engine ===
class ContinuousPredictionEngine:
    def __init__(self):
        self.market_schedule = MarketSchedule()
        self.news_cache = NewsCache()
        self.load_components()
        self.prediction_history = []
        self.running = False
        self.performance_stats = defaultdict(list)
        
    def load_components(self):
        """Load all required models and processors"""
        try:
            # Load model
            self.model_package = self.load_model()
            self.model = self.model_package["model"]
            self.selected_features = self.model_package["selected_features"]
            self.use_scaling = self.model_package.get("use_scaling", False)
            self.scaler = self.model_package.get("scaler", None)
            self.original_feature_columns = self.model_package.get("original_feature_columns", self.selected_features)
            
            # Load processors
            self.full_processor = NewsProcessor()
            self.feature_engineer = FinancialNewsFeatureEngineer(live_mode=True)
            self.price_analyzer = SimplifiedPriceAnalyzer()
            self.enhanced_processor_inst = EnhancedNewsProcessor()
            
            # Load tickers
            self.valid_tickers = self.load_valid_tickers_from_json()
            self.banned_tickers = {"IBD", "CNBC", "WSJ", "MARKETWATCH", "BARRONS", "YAHOO", "THE", "NASDAQ", "NYSE"}
            
            logger.info("All components loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            raise
    
    def load_model(self, model_path="models/stock_price_regressor.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        logger.info(f"Loading trained regression model from {model_path}")
        obj = joblib.load(model_path)
        if isinstance(obj, dict) and "model" in obj:
            return obj
        raise ValueError("Model file does not contain expected keys.")
    
    def load_valid_tickers_from_json(self, json_path="tickers_with_news.json"):
        try:
            with open(json_path, "r") as f:
                tickers = json.load(f)
            return set(t.upper().strip() for t in tickers if t and isinstance(t, str))
        except Exception as e:
            logger.warning(f"Could not load tickers from JSON: {e}")
            return set()
    
    def fetch_news_with_retry(self):
        """Fetch news with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                news = News(view_option=News.ViewOption.STOCKS_NEWS)
                news_items = news.news_df
                
                if news_items is not None and not news_items.empty:
                    logger.info(f"Fetched {len(news_items)} news articles (attempt {attempt + 1})")
                    return news_items
                else:
                    logger.warning(f"No news data retrieved (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.error(f"Failed to fetch news (attempt {attempt + 1}): {e}")
                
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time_module.sleep(RETRY_DELAY)
        
        logger.error("Failed to fetch news after all retries")
        return pd.DataFrame()
    
    def parse_tickers_from_finviz_news_html(self, html):
        """Parse tickers from Finviz HTML"""
        soup = BeautifulSoup(html, "html.parser")
        news_rows = soup.select("table.styled-table-new tr.styled-row")
        news_ticker_map = []
        
        for row in news_rows:
            headline_link = row.select_one("td.news_link-cell a.nn-tab-link")
            if not headline_link:
                continue
            headline = headline_link.get_text(strip=True)

            ticker_links = row.select("a[href^='/quote.ashx?t=']")
            tickers = []
            for a in ticker_links:
                href = a['href']
                match = re.search(r"/quote\.ashx\?t=([A-Z]{1,5})", href)
                if match:
                    ticker = match.group(1).upper()
                    if ticker not in self.banned_tickers:
                        tickers.append(ticker)
            
            news_ticker_map.append({
                "headline": headline,
                "tickers": list(set(tickers))
            })
        
        return news_ticker_map
    
    def process_single_prediction_cycle(self):
        """Process one complete prediction cycle"""
        cycle_start = datetime.now()
        logger.info("Starting prediction cycle...")
        
        try:
            # Fetch news
            news_items = self.fetch_news_with_retry()
            if news_items.empty:
                logger.warning("No news items to process")
                return []
            
            # Get Finviz HTML for ticker mapping
            try:
                finviz_html = requests.get(
                    "https://finviz.com/news.ashx?v=3",
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10
                ).text
                news_ticker_map = self.parse_tickers_from_finviz_news_html(finviz_html)
            except Exception as e:
                logger.warning(f"Failed to fetch Finviz HTML: {e}")
                news_ticker_map = []
            
            # Process articles
            articles_for_processing = []
            processed_count = 0
            
            for _, row in news_items.iterrows():
                try:
                    headline = str(row.get('Headline', "")).strip()
                    ts_str = str(row.get('Time', ""))
                    url = row.get('URL', '')
                    
                    # Create unique article ID
                    article_id = f"{headline}_{url}"
                    
                    # Skip if already processed
                    if self.news_cache.is_processed(article_id):
                        continue
                    
                    parsed_dt = self.full_processor.parse_datetime(ts_str)
                    if parsed_dt is None:
                        continue
                    
                    # Find tickers for this headline
                    tickers = []
                    for item in news_ticker_map:
                        if item["headline"] == headline:
                            tickers = item["tickers"]
                            break
                    
                    if not tickers:
                        continue
                    
                    # Check cache first
                    cached_article = self.news_cache.get_article(url)
                    if cached_article:
                        article_html = cached_article
                    else:
                        # Scrape article
                        article_html = self.full_processor.scrape_article(url)
                        if article_html:
                            self.news_cache.add_article(url, article_html)
                    
                    # Process for each ticker
                    for ticker in tickers:
                        if ticker not in self.valid_tickers:
                            continue
                        
                        full_text = f"{headline} {article_html}" if article_html else headline
                        mentions, pos_kw, neg_kw, tokens = self.full_processor.extract_mentions_and_sentiment(full_text, ticker)
                        enhanced_sentiment = self.enhanced_processor_inst.calculate_enhanced_sentiment(full_text)
                        
                        processed = {
                            "ticker": ticker,
                            "datetime": parsed_dt,
                            "headline": headline,
                            "url": url,
                            "text": article_html,
                            "tokens": tokens,
                            "pos_keywords": len(pos_kw),
                            "neg_keywords": len(neg_kw),
                            "mentions": len(mentions),
                            "keyword_activity": len(pos_kw) + len(neg_kw),
                            "pos_keywords_str": ', '.join(pos_kw),
                            "neg_keywords_str": ', '.join(neg_kw),
                            "sentiment_dynamic": enhanced_sentiment.get('dynamic_weights', 0),
                            "sentiment_ml": enhanced_sentiment.get('ml_prediction', 0),
                            "sentiment_keyword": enhanced_sentiment.get('keyword_based', 0),
                            "sentiment_combined": enhanced_sentiment.get('combined', 0),
                            "prediction_confidence": abs(enhanced_sentiment.get('combined', 0)),
                            "market_session": self.market_schedule.get_market_session(parsed_dt),
                            "news_age_minutes": (datetime.now(pytz.UTC) - parsed_dt.astimezone(pytz.UTC)).total_seconds() / 60,
                            "pct_change_1h": None,
                            "pct_change_4h": None,
                            "pct_change_eod": None,
                            "pct_change_eow": None,
                            "direction_1h": "No Data",
                            "direction_4h": "No Data",
                            "direction_eod": "No Data",
                            "direction_eow": "No Data"
                        }
                        articles_for_processing.append(processed)
                    
                    # Mark as processed
                    self.news_cache.mark_processed(article_id)
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing article: {e}")
                    continue
            
            logger.info(f"Processed {processed_count} new articles, generated {len(articles_for_processing)} ticker-article pairs")
            
            # Generate predictions
            predictions = self.generate_predictions(articles_for_processing)
            
            # Save results
            if predictions:
                self.save_predictions(predictions)
                self.prediction_history.extend(predictions)
            
            # Cleanup cache
            self.news_cache.cleanup_old_entries()
            
            cycle_duration = (datetime.now() - cycle_start).seconds
            self.performance_stats['cycle_duration'].append(cycle_duration)
            logger.info(f"Prediction cycle completed in {cycle_duration} seconds")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction cycle: {e}")
            return []
    
    def generate_predictions(self, articles_for_processing):
        """Generate ML predictions from processed articles"""
        if not articles_for_processing:
            return []
        
        try:
            input_df = pd.DataFrame(articles_for_processing)
            
            # Feature engineering
            engineered_df = self.feature_engineer.feature_engineering_pipeline(input_df, save_cleaned=False)
            
            # Ensure required columns
            for col in ["pos_keywords", "neg_keywords", "mentions"]:
                if col in engineered_df.columns:
                    engineered_df[col] = pd.to_numeric(engineered_df[col], errors="coerce").fillna(0).astype(int)
                else:
                    engineered_df[col] = 0
            
            if "keyword_activity" not in engineered_df.columns or engineered_df["keyword_activity"].sum() == 0:
                engineered_df["keyword_activity"] = engineered_df["pos_keywords"] + engineered_df["neg_keywords"]
            
            # Add missing string columns
            if "pos_keywords_str" not in engineered_df.columns:
                engineered_df["pos_keywords_str"] = input_df.get("pos_keywords_str", "")
            if "neg_keywords_str" not in engineered_df.columns:
                engineered_df["neg_keywords_str"] = input_df.get("neg_keywords_str", "")
            
            # Advanced features
            engineered_df, all_features_list = create_advanced_features(
                engineered_df,
                self.original_feature_columns,
                target_col=None
            )
            
            # Ensure all selected features exist
            for feature in self.selected_features:
                if feature not in engineered_df.columns:
                    logger.warning(f"Selected feature '{feature}' not found. Creating with default value 0.")
                    engineered_df[feature] = 0
                engineered_df[feature] = engineered_df[feature].fillna(0)
            
            # Generate predictions
            predictions = []
            current_time = datetime.now(MARKET_TIMEZONE)
            
            for _, row in engineered_df.iterrows():
                try:
                    fv = row[self.selected_features].values.reshape(1, -1)
                    if self.use_scaling and self.scaler:
                        fv = self.scaler.transform(fv)
                    pred = self.model.predict(fv)[0]
                    
                    predictions.append({
                        "prediction_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "news_datetime": row['datetime'].strftime("%Y-%m-%d %H:%M:%S"),
                        "ticker": row['ticker'],
                        "headline": row['headline'],
                        "market_session": row.get('market_session', 'UNKNOWN'),
                        "news_age_minutes": round(float(row.get('news_age_minutes', 0)), 1),
                        "sentiment_combined": round(float(row['sentiment_combined']), 4),
                        "keyword_activity": float(row.get('keyword_activity', 0)),
                        "mentions": float(row.get('mentions', 0)),
                        "neg_keywords": float(row.get('neg_keywords', 0)),
                        "predicted_pct_change_1h": round(float(pred), 4),
                        "prediction_confidence": round(float(row.get('prediction_confidence', 0)), 4)
                    })
                except Exception as e:
                    logger.error(f"Prediction failed for row: {e}")
                    continue
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []
    
    def save_predictions(self, predictions):
        """Save predictions to timestamped CSV"""
        try:
            if not predictions:
                return
            
            df_output = pd.DataFrame(predictions).sort_values(
                by=["predicted_pct_change_1h", "prediction_time"],
                ascending=[False, False]
            )
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.csv"
            
            df_output.to_csv(filename, index=False)
            logger.info(f"Saved {len(predictions)} predictions to {filename}")
            
            # Also save session-specific files for after-hours analysis
            session = self.market_schedule.get_market_session()
            if session in ["AFTER_HOURS", "PRE_MARKET", "WEEKEND"]:
                session_filename = f"predictions_{session.lower()}_{timestamp}.csv"
                df_output.to_csv(session_filename, index=False)
                logger.info(f"Saved {session} predictions to {session_filename}")
            
            # Also maintain a continuous log file
            continuous_filename = "continuous_predictions.csv"
            file_exists = os.path.exists(continuous_filename)
            
            df_output.to_csv(continuous_filename, mode='a', header=not file_exists, index=False)
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def print_status(self):
        """Print current status information"""
        current_time = datetime.now(MARKET_TIMEZONE)
        session = self.market_schedule.get_market_session(current_time)
        market_open = self.market_schedule.is_market_open(current_time)
        
        print(f"\n{'='*60}")
        print(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Market Session: {session}")
        
        if market_open:
            time_to_close = self.market_schedule.time_until_market_close(current_time)
            hours, remainder = divmod(time_to_close.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            print(f"Time until close: {hours}h {minutes}m")
        elif session in ["PRE_MARKET", "AFTER_HOURS"]:
            if session == "PRE_MARKET":
                market_open_today = current_time.replace(hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute, second=0, microsecond=0)
                time_diff = market_open_today - current_time
                hours, remainder = divmod(time_diff.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                print(f"Time until market open: {hours}h {minutes}m")
            else:  # AFTER_HOURS
                print("Processing after-hours news for next trading day")
        elif session == "WEEKEND":
            time_to_open = self.market_schedule.time_until_market_open(current_time)
            if time_to_open.days > 0:
                print(f"Time until market open: {time_to_open.days} days, {time_to_open.seconds//3600}h {(time_to_open.seconds%3600)//60}m")
        else:
            time_to_open = self.market_schedule.time_until_market_open(current_time)
            if time_to_open.days > 0:
                print(f"Time until open: {time_to_open.days} days, {time_to_open.seconds//3600}h {(time_to_open.seconds%3600)//60}m")
            else:
                hours, remainder = divmod(time_to_open.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                print(f"Time until open: {hours}h {minutes}m")
        
        print(f"Total predictions made today: {len(self.prediction_history)}")
        
        # Show session breakdown
        session_counts = {}
        for pred in self.prediction_history:
            sess = pred.get('market_session', 'UNKNOWN')
            session_counts[sess] = session_counts.get(sess, 0) + 1
        
        if session_counts:
            print("Session breakdown:", end=" ")
            for sess, count in session_counts.items():
                print(f"{sess}:{count}", end=" ")
            print()
        
        if self.performance_stats['cycle_duration']:
            avg_cycle_time = sum(self.performance_stats['cycle_duration']) / len(self.performance_stats['cycle_duration'])
            print(f"Average cycle time: {avg_cycle_time:.1f} seconds")
        
        print(f"Cache size: {len(self.news_cache.cache)} articles")
        print(f"{'='*60}\n")
    
    def run_continuous_prediction(self):
        """Main continuous prediction loop with extended hours support"""
        logger.info("Starting continuous prediction engine with extended hours support...")
        self.running = True
        
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while self.running:
            try:
                current_time = datetime.now(MARKET_TIMEZONE)
                session = self.market_schedule.get_market_session(current_time)
                
                # Determine if we should process news in current session
                should_process = False
                wait_interval = OVERNIGHT_INTERVAL  # default
                
                if session == "MARKET_HOURS":
                    should_process = True
                    wait_interval = MARKET_HOURS_INTERVAL
                elif session in ["PRE_MARKET", "AFTER_HOURS"] and PROCESS_AFTER_HOURS:
                    should_process = True
                    wait_interval = AFTER_HOURS_INTERVAL
                elif session == "WEEKEND" and PROCESS_WEEKENDS:
                    should_process = True
                    wait_interval = WEEKEND_INTERVAL
                elif session == "CLOSED":
                    # Check how long until next session
                    time_to_open = self.market_schedule.time_until_market_open(current_time)
                    if time_to_open.days == 0 and time_to_open.seconds < 3600:  # Less than 1 hour
                        logger.info(f"Market opens soon. Waiting {time_to_open.seconds//60} minutes...")
                        self.print_status()
                        time_module.sleep(60)  # Check every minute when close to open
                        continue
                
                if should_process:
                    # Process news and make predictions
                    self.print_status()
                    logger.info(f"Processing news for {session} session...")
                    
                    predictions = self.process_single_prediction_cycle()
                    
                    if predictions:
                        logger.info(f"Generated {len(predictions)} predictions for {session}")
                        
                        # Show top predictions with session context
                        top_predictions = sorted(predictions, key=lambda x: abs(x['predicted_pct_change_1h']), reverse=True)[:5]
                        print(f"\nTop 5 {session} Predictions by Magnitude:")
                        for i, pred in enumerate(top_predictions, 1):
                            age_str = f"({pred['news_age_minutes']:.0f}m ago)" if pred['news_age_minutes'] < 60 else f"({pred['news_age_minutes']/60:.1f}h ago)"
                            print(f"{i}. {pred['ticker']}: {pred['predicted_pct_change_1h']:+.2f}% {age_str}")
                            print(f"   {pred['headline'][:70]}...")
                        
                        # Special handling for after-hours/pre-market predictions
                        if session in ["AFTER_HOURS", "PRE_MARKET"]:
                            high_impact_preds = [p for p in predictions if abs(p['predicted_pct_change_1h']) > 2.0]
                            if high_impact_preds:
                                print(f"\n⚠️  HIGH IMPACT {session} NEWS ({len(high_impact_preds)} predictions >2%):")
                                for pred in sorted(high_impact_preds, key=lambda x: abs(x['predicted_pct_change_1h']), reverse=True)[:3]:
                                    print(f"   {pred['ticker']}: {pred['predicted_pct_change_1h']:+.2f}% - Confidence: {pred['prediction_confidence']:.2f}")
                    
                else:
                    # Market closed and not processing extended hours
                    time_to_next = self.market_schedule.time_until_market_open(current_time)
                    if time_to_next.days > 0:
                        logger.info(f"Market closed until next trading day. Next check in {wait_interval//60} minutes...")
                    else:
                        logger.info(f"Market closed. Next check in {wait_interval//60} minutes...")
                    self.print_status()
                
                # Wait for next cycle
                logger.info(f"Next cycle in {wait_interval//60} minutes...")
                time_module.sleep(wait_interval)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                logger.info("Continuing after error...")
                time_module.sleep(30)  # Short wait before retrying
        
        logger.info("Continuous prediction engine stopped.")
        
        # Print final summary
        if self.prediction_history:
            session_summary = {}
            for pred in self.prediction_history:
                sess = pred.get('market_session', 'UNKNOWN')
                session_summary[sess] = session_summary.get(sess, 0) + 1
            
            print(f"\n📊 Final Session Summary:")
            for session, count in session_summary.items():
                print(f"   {session}: {count} predictions")
            
            total_high_impact = len([p for p in self.prediction_history if abs(p['predicted_pct_change_1h']) > 2.0])
            print(f"   High impact predictions (>2%): {total_high_impact}")


# === Main Execution ===
if __name__ == "__main__":
    try:
        engine = ContinuousPredictionEngine()
        engine.run_continuous_prediction()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)