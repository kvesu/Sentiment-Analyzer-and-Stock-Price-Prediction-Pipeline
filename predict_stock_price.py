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
from train_regressor import engineer_features
from word_analysis_framework import EnhancedNewsProcessor
import json
import requests
import time as time_module
from collections import defaultdict
import signal
import threading

# === Horizon Mapping ===
TARGET_MAP = {
    "1hr": "pct_change_1h",
    "4hr": "pct_change_4h",
    "eod": "pct_change_eod"
}

# === Market Hours Configuration ===
MARKET_TIMEZONE = pytz.timezone("US/Eastern")
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)
EXTENDED_HOURS_START = time(4, 0)
EXTENDED_HOURS_END = time(20, 0)

MARKET_HOURS_INTERVAL = 300
AFTER_HOURS_INTERVAL = 900
OVERNIGHT_INTERVAL = 1800
WEEKEND_INTERVAL = 3600

NEWS_CACHE_DURATION = 3600
MAX_RETRIES = 3
RETRY_DELAY = 10

PROCESS_AFTER_HOURS = True
PROCESS_WEEKENDS = True

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

# === Market Schedule ===
class MarketSchedule:
    def __init__(self):
        self.market_tz = MARKET_TIMEZONE
        self.holidays_2025 = [
            "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
            "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"
        ]

    def is_market_open(self, dt=None):
        dt = datetime.now(self.market_tz) if dt is None else dt.astimezone(self.market_tz)
        if dt.weekday() >= 5 or dt.strftime('%Y-%m-%d') in self.holidays_2025:
            return False
        return MARKET_OPEN_TIME <= dt.time() <= MARKET_CLOSE_TIME

    def is_extended_hours(self, dt=None):
        dt = datetime.now(self.market_tz) if dt is None else dt.astimezone(self.market_tz)
        if dt.weekday() >= 5 or dt.strftime('%Y-%m-%d') in self.holidays_2025:
            return False
        return (EXTENDED_HOURS_START <= dt.time() <= EXTENDED_HOURS_END) and not self.is_market_open(dt)

    def is_weekend(self, dt=None):
        dt = datetime.now(self.market_tz) if dt is None else dt.astimezone(self.market_tz)
        return dt.weekday() >= 5

    def get_market_session(self, dt=None):
        dt = datetime.now(self.market_tz) if dt is None else dt.astimezone(self.market_tz)
        if self.is_market_open(dt):
            return "MARKET_HOURS"
        elif self.is_extended_hours(dt):
            return "PRE_MARKET" if dt.time() < MARKET_OPEN_TIME else "AFTER_HOURS"
        elif self.is_weekend(dt):
            return "WEEKEND"
        else:
            return "CLOSED"

    def time_until_market_open(self, dt=None):
        if dt is None: dt = datetime.now(self.market_tz)
        else: dt = dt.astimezone(self.market_tz)
        if self.is_market_open(dt): return timedelta(0)
        next_open = dt.replace(hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute, second=0, microsecond=0)
        if dt.time() > MARKET_OPEN_TIME or dt.weekday() >= 5:
            next_open += timedelta(days=1)
            while next_open.weekday() >= 5 or next_open.strftime('%Y-%m-%d') in self.holidays_2025:
                next_open += timedelta(days=1)
        return next_open - dt

    def time_until_market_close(self, dt=None):
        if dt is None: dt = datetime.now(self.market_tz)
        else: dt = dt.astimezone(self.market_tz)
        if not self.is_market_open(dt): return timedelta(0)
        market_close = dt.replace(hour=MARKET_CLOSE_TIME.hour, minute=MARKET_CLOSE_TIME.minute, second=0, microsecond=0)
        return market_close - dt

# === News Cache System ===
class NewsCache:
    def __init__(self, cache_duration=NEWS_CACHE_DURATION):
        self.cache = {}
        self.cache_duration = cache_duration
        self.processed_articles = set()

    def is_fresh(self, timestamp):
        return (datetime.now() - timestamp).seconds < self.cache_duration

    def add_article(self, url, data):
        self.cache[url] = {'data': data, 'timestamp': datetime.now()}

    def get_article(self, url):
        if url in self.cache and self.is_fresh(self.cache[url]['timestamp']):
            return self.cache[url]['data']
        return None

    def is_processed(self, article_id):
        return article_id in self.processed_articles

    def mark_processed(self, article_id):
        self.processed_articles.add(article_id)

    def cleanup_old_entries(self):
        current_time = datetime.now()
        old_entries = [url for url, data in self.cache.items()
                       if (current_time - data['timestamp']).seconds > self.cache_duration]
        for url in old_entries:
            del self.cache[url]

# === Enhanced Prediction Engine ===
class ContinuousPredictionEngine:
    def __init__(self, target_horizon):
        self.target_horizon = target_horizon
        self.target_column = TARGET_MAP[target_horizon]
        self.regressor_model_path = f"models/stock_price_regressor_{target_horizon}.pkl"
        self.classifier_model_path = f"models/stock_move_classifier_{target_horizon}.pkl"

        self.market_schedule = MarketSchedule()
        self.news_cache = NewsCache()
        self.load_components()
        self.prediction_history = []
        self.running = False
        self.performance_stats = defaultdict(list)

    def load_components(self):
        try:
            # Load regressor model
            logger.info(f"Loading regression model from {self.regressor_model_path}")
            self.model_package = joblib.load(self.regressor_model_path)
            self.model = self.model_package["model"]
            self.selected_features = self.model_package["selected_features"]
            self.use_scaling = self.model_package.get("use_scaling", False)
            self.scaler = self.model_package.get("scaler", None)
            self.original_feature_columns = self.model_package.get("original_feature_columns", self.selected_features)

            # Load Gatekeeper classifier model
            logger.info(f"Loading Gatekeeper classifier from {self.classifier_model_path}")
            gk_package = joblib.load(self.classifier_model_path)
            self.gk_model = gk_package["model"]
            self.gk_features = gk_package["selected_features"]
            self.gk_scaler = gk_package["scaler"]
            self.gk_threshold = gk_package.get("optimal_threshold", 0.5)

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

    def load_valid_tickers_from_json(self, json_path="tickers_with_news.json"):
        try:
            with open(json_path, "r") as f:
                tickers = json.load(f)
            return set(t.upper().strip() for t in tickers if t and isinstance(t, str))
        except Exception as e:
            logger.warning(f"Could not load tickers from JSON: {e}")
            return set()

    def fetch_news_with_retry(self):
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
        soup = BeautifulSoup(html, "html.parser")
        news_rows = soup.select("table.styled-table-new tr.styled-row")
        news_ticker_map = []
        for row in news_rows:
            headline_link = row.select_one("td.news_link-cell a.nn-tab-link")
            if not headline_link: continue
            headline = headline_link.get_text(strip=True)
            ticker_links = row.select("a[href^='/quote.ashx?t=']")
            tickers = []
            for a in ticker_links:
                href = a['href']
                match = re.search(r"/quote\.ashx\?t=([A-Z]{1,5})", href)
                if match:
                    ticker = match.group(1).upper()
                    if ticker not in self.banned_tickers: tickers.append(ticker)
            news_ticker_map.append({"headline": headline, "tickers": list(set(tickers))})
        return news_ticker_map

    def process_single_prediction_cycle(self):
        cycle_start = datetime.now()
        logger.info("Starting prediction cycle...")
        try:
            news_items = self.fetch_news_with_retry()
            if news_items.empty:
                logger.warning("No news items to process")
                return []
            try:
                finviz_html = requests.get("https://finviz.com/news.ashx?v=3", headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
                news_ticker_map = self.parse_tickers_from_finviz_news_html(finviz_html)
            except Exception as e:
                logger.warning(f"Failed to fetch Finviz HTML: {e}")
                news_ticker_map = []
            articles_for_processing = []
            processed_count = 0
            for _, row in news_items.iterrows():
                try:
                    headline = str(row.get('Headline', "")).strip()
                    ts_str = str(row.get('Time', ""))
                    url = row.get('URL', '')
                    article_id = f"{headline}_{url}"
                    if self.news_cache.is_processed(article_id): continue
                    parsed_dt = self.full_processor.parse_datetime(ts_str)
                    if parsed_dt is None: continue
                    tickers = []
                    for item in news_ticker_map:
                        if item["headline"] == headline:
                            tickers = item["tickers"]
                            break
                    if not tickers: continue
                    cached_article = self.news_cache.get_article(url)
                    if cached_article:
                        article_html = cached_article
                    else:
                        article_html = self.full_processor.scrape_article(url)
                        if article_html: self.news_cache.add_article(url, article_html)
                    for ticker in tickers:
                        if ticker not in self.valid_tickers: continue
                        full_text = f"{headline} {article_html}" if article_html else headline
                        mentions, pos_kw, neg_kw, tokens = self.full_processor.extract_mentions_and_sentiment(full_text, ticker)
                        enhanced_sentiment = self.enhanced_processor_inst.calculate_enhanced_sentiment(full_text)
                        processed = {
                            "ticker": ticker, "datetime": parsed_dt, "headline": headline, "url": url,
                            "text": article_html, "tokens": tokens, "pos_keywords": len(pos_kw),
                            "neg_keywords": len(neg_kw), "mentions": len(mentions),
                            "keyword_activity": len(pos_kw) + len(neg_kw), "pos_keywords_str": ', '.join(pos_kw),
                            "neg_keywords_str": ', '.join(neg_kw), "sentiment_dynamic": enhanced_sentiment.get('dynamic_weights', 0),
                            "sentiment_ml": enhanced_sentiment.get('ml_prediction', 0),
                            "sentiment_keyword": enhanced_sentiment.get('keyword_based', 0),
                            "sentiment_combined": enhanced_sentiment.get('combined', 0),
                            "prediction_confidence": abs(enhanced_sentiment.get('combined', 0)),
                            "total_keywords": len(pos_kw) + len(neg_kw), 
                            "headline_sentiment": self.enhanced_processor_inst.calculate_enhanced_sentiment(headline).get('combined', 0),
                            "keyword_density": (len(pos_kw) + len(neg_kw)) / len(full_text.split()) if full_text else 0,
                            "market_session": self.market_schedule.get_market_session(parsed_dt),
                            "news_age_minutes": (datetime.now(pytz.UTC) - parsed_dt.astimezone(pytz.UTC)).total_seconds() / 60,
                            "pct_change_1hr": None, "pct_change_4hr": None, "pct_change_eod": None,
                            "direction_1hr": "No Data", "direction_4hr": "No Data", "direction_eod": "No Data"
                        }
                        articles_for_processing.append(processed)
                    self.news_cache.mark_processed(article_id)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing article: {e}")
                    continue
            logger.info(f"Processed {processed_count} new articles, generated {len(articles_for_processing)} ticker-article pairs")
            predictions = self.generate_predictions(articles_for_processing)
            if predictions:
                self.save_predictions(predictions)
                self.prediction_history.extend(predictions)
            self.news_cache.cleanup_old_entries()
            cycle_duration = (datetime.now() - cycle_start).seconds
            self.performance_stats['cycle_duration'].append(cycle_duration)
            logger.info(f"Prediction cycle completed in {cycle_duration} seconds")
            return predictions
        except Exception as e:
            logger.error(f"Error in prediction cycle: {e}")
            return []

    def generate_predictions(self, articles_for_processing):
        if not articles_for_processing: return []
        try:
            input_df = pd.DataFrame(articles_for_processing)
            engineered_df = self.feature_engineer.feature_engineering_pipeline(input_df)
            # Make sure: from train_regressor import engineer_features is at the top of your file
            engineered_df, _ = engineer_features(engineered_df, engineered_df.columns.tolist())
            all_required_features = list(set(self.gk_features + self.selected_features))
            for feature in all_required_features:
                if feature not in engineered_df.columns:
                    logger.warning(f"Feature '{feature}' not found. Creating with default value 0.")
                    engineered_df[feature] = 0
                engineered_df[feature] = pd.to_numeric(engineered_df[feature], errors="coerce").fillna(0)
            
            # --- Gatekeeper Filter ---
            X_gk = engineered_df[self.gk_features]
            X_gk_scaled = self.gk_scaler.transform(X_gk)
            gk_probs = self.gk_model.predict_proba(X_gk_scaled)[:, 1]
            engineered_df["gk_prob"] = gk_probs
            filtered_df = engineered_df[engineered_df["gk_prob"] >= self.gk_threshold]
            if filtered_df.empty:
                logger.info("No articles passed the Gatekeeper filter. No predictions generated.")
                return []
            logger.info(f"Gatekeeper filtered out {len(engineered_df) - len(filtered_df)} of {len(engineered_df)} articles.")
            
            # --- Regression Prediction on Filtered Data ---
            X_reg = filtered_df[self.selected_features].values
            if self.use_scaling and self.scaler: X_reg = self.scaler.transform(X_reg)
            preds = self.model.predict(X_reg)
            
            predictions = []
            current_time = datetime.now(MARKET_TIMEZONE)
            for row, pred in zip(filtered_df.to_dict("records"), preds):
                predictions.append({
                    "prediction_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "news_datetime": row['datetime'].strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": row['ticker'],
                    "headline": row['headline'],
                    "market_session": row.get('market_session', 'UNKNOWN'),
                    "news_age_minutes": round(float(row.get('news_age_minutes', 0)), 1),
                    "sentiment_combined": round(float(row.get('sentiment_combined', 0)), 4),
                    f"predicted_{self.target_column}": round(float(pred), 6),
                    "prediction_confidence": round(float(row.get('gk_prob', 0)), 4),
                    "horizon": self.target_horizon
                })
            logger.info(f"Generated {len(predictions)} predictions for horizon '{self.target_horizon}'")
            return predictions
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []

    def save_predictions(self, predictions):
        try:
            if not predictions: return
            continuous_filename = f"continuous_predictions_{self.target_horizon}.csv"
            new_predictions_df = pd.DataFrame(predictions)
            if os.path.exists(continuous_filename):
                try:
                    existing_predictions_df = pd.read_csv(continuous_filename)
                    combined_df = pd.concat([new_predictions_df, existing_predictions_df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    combined_df = new_predictions_df
            else:
                combined_df = new_predictions_df
            combined_df = combined_df.sort_values(by=["prediction_time", "news_datetime"], ascending=[False, False])
            combined_df.drop_duplicates(subset=['ticker', 'headline', 'news_datetime'], keep='first', inplace=True)
            combined_df.to_csv(continuous_filename, index=False)
            logger.info(f"Updated '{continuous_filename}' with {len(new_predictions_df)} new predictions. Total entries: {len(combined_df)}.")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")

    def print_status(self):
        current_time = datetime.now(MARKET_TIMEZONE)
        session = self.market_schedule.get_market_session(current_time)
        market_open = self.market_schedule.is_market_open(current_time)
        
        print(f"\n{'='*60}")
        print(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Prediction Horizon: {self.target_horizon.upper()}")
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
            else:
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
        
        print(f"Total predictions made today for {self.target_horizon}: {len(self.prediction_history)}")
        
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
        logger.info(f"Starting continuous prediction engine for horizon '{self.target_horizon}'...")
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
                should_process = False
                wait_interval = OVERNIGHT_INTERVAL

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
                    time_to_open = self.market_schedule.time_until_market_open(current_time)
                    if time_to_open.days == 0 and time_to_open.seconds < 3600:
                        logger.info(f"Market opens soon. Waiting {time_to_open.seconds//60} minutes...")
                        self.print_status()
                        time_module.sleep(60)
                        continue

                if should_process:
                    self.print_status()
                    logger.info(f"Processing news for {session} session...")
                    predictions = self.process_single_prediction_cycle()

                    if predictions:
                        logger.info(f"Generated {len(predictions)} predictions for {session}")
                        predicted_col = f"predicted_{self.target_column}"

                    # === NEW: ensure only one entry per ticker in Top 5 display ===
                    unique_ticker_preds = {}
                    for pred in sorted(predictions, key=lambda x: abs(x[predicted_col]), reverse=True):
                        if pred['ticker'] not in unique_ticker_preds:
                            unique_ticker_preds[pred['ticker']] = pred

                    top_predictions = list(unique_ticker_preds.values())[:5]

                    print(f"\nTop 5 {session} Predictions by Magnitude ({self.target_horizon.upper()} Horizon):")
                    for i, pred in enumerate(top_predictions, 1):
                        age_str = (f"({pred['news_age_minutes']:.0f}m ago)"
                                if pred['news_age_minutes'] < 60
                                else f"({pred['news_age_minutes']/60:.1f}h ago)")
                        print(f"{i}. {pred['ticker']}: {pred[predicted_col]:+.2f}% {age_str}")
                        print(f"   Headline: {pred['headline'][:70]}...")
                        print(f"   Confidence: {pred['prediction_confidence']:.2f}")
                else:
                    time_to_next = self.market_schedule.time_until_market_open(current_time)
                    if time_to_next.days > 0:
                        logger.info(f"Market closed until next trading day. Next check in {wait_interval//60} minutes...")
                    else:
                        logger.info(f"Market closed. Next check in {wait_interval//60} minutes...")
                    self.print_status()

                logger.info(f"Next cycle in {wait_interval//60} minutes...")
                time_module.sleep(wait_interval)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                logger.info("Continuing after error...")
                time_module.sleep(30)

        logger.info("Continuous prediction engine stopped.")

        if self.prediction_history:
            session_summary = {}
            for pred in self.prediction_history:
                sess = pred.get('market_session', 'UNKNOWN')
                session_summary[sess] = session_summary.get(sess, 0) + 1
            print(f"\nFinal Session Summary for {self.target_horizon.upper()} Horizon:")
            for session, count in session_summary.items():
                print(f"   {session}: {count} predictions")
            predicted_col = f"predicted_{self.target_column}"
            total_high_impact = len([p for p in self.prediction_history if abs(p[predicted_col]) > 0.02])
            print(f"   High impact predictions (>2%): {total_high_impact}")

# === Main Execution ===
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in TARGET_MAP:
        horizon = sys.argv[1].lower()
    else:
        horizon = "eod"
        logger.info(f"No prediction horizon specified. Defaulting to '{horizon}'.")
    try:
        engine = ContinuousPredictionEngine(horizon)
        engine.run_continuous_prediction()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)