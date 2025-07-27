import pandas as pd
import logging
import sys
import os
import joblib
from datetime import datetime, timedelta
import pytz
import re
from pyfinviz.news import News
from word_analysis_framework import EnhancedNewsProcessor
from feature_engineering import FinancialNewsFeatureEngineer
from main import SimplifiedPriceAnalyzer

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("StockNewsRegressor")

# === Load Model ===
def load_model(model_path="models/stock_price_regressor.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    logger.info(f"Loading trained regression model from {model_path}")
    return joblib.load(model_path)

model = load_model()

# === Initialize Components ===
sentiment_processor = EnhancedNewsProcessor()
feature_engineer = FinancialNewsFeatureEngineer(live_mode=True)  # Enable live mode
price_analyzer = SimplifiedPriceAnalyzer()

# === Feature Columns will be determined dynamically from the feature engineering pipeline ===
FEATURE_COLUMNS = None  # Will be set after running feature engineering

# === Load Valid Tickers from finviz.csv ===
def load_valid_tickers(csv_path="finviz.csv"):
    """Load valid ticker symbols from finviz.csv"""
    try:
        ticker_df = pd.read_csv(csv_path)
        logger.info(f"Loaded ticker CSV with columns: {ticker_df.columns.tolist()}")
        
        ticker_columns = ['Ticker', 'ticker', 'Symbol', 'symbol', 'TICKER', 'SYMBOL']
        ticker_col = None
        
        for col in ticker_columns:
            if col in ticker_df.columns:
                ticker_col = col
                break
        
        if ticker_col is None:
            ticker_col = ticker_df.columns[0]
            logger.warning(f"No standard ticker column found. Using first column: {ticker_col}")
        
        valid_tickers = set(ticker_df[ticker_col].astype(str).str.upper().str.strip())
        logger.info(f"Loaded {len(valid_tickers)} valid tickers from {csv_path}")
        return valid_tickers
        
    except Exception as e:
        logger.warning(f"Could not load tickers from {csv_path}: {e}")
        return set()

VALID_TICKERS = load_valid_tickers()

# === Optional: Filter to These Tickers Only ===
TARGET_TICKERS = None  # or e.g., ["AAPL", "TSLA", "GOOGL", "NVDA"]

# === Fetch Real-Time Finviz News ===
logger.info("Fetching real-time stock-specific news from Finviz...")
news = News(view_option=News.ViewOption.STOCKS_NEWS)
news_items = news.news_df  # This is a DataFrame

logger.info("News DataFrame columns: %s", news_items.columns.tolist())
logger.info("News DataFrame shape: %s", news_items.shape)
logger.info("First few rows:\n%s", news_items.head())

# === Handle Different Possible Column Names ===
timestamp_columns = ['Timestamp', 'timestamp', 'Time', 'time', 'Date', 'date', 'DateTime', 'datetime']
ticker_columns = ['Ticker', 'ticker', 'Symbol', 'symbol', 'Stock', 'stock']
headline_columns = ['Headline', 'headline', 'Title', 'title', 'News', 'news']

timestamp_col = None
ticker_col = None
headline_col = None

for col in timestamp_columns:
    if col in news_items.columns:
        timestamp_col = col
        break

for col in ticker_columns:
    if col in news_items.columns:
        ticker_col = col
        break

for col in headline_columns:
    if col in news_items.columns:
        headline_col = col
        break

logger.info(f"Found timestamp column: {timestamp_col}")
logger.info(f"Found ticker column: {ticker_col}")
logger.info(f"Found headline column: {headline_col}")

# === Enhanced DateTime Processing ===
eastern = pytz.timezone("US/Eastern")

def process_timestamp(timestamp_value):
    """Enhanced timestamp processing that matches the feature engineering expectations"""
    if pd.isna(timestamp_value) or timestamp_value is None:
        return datetime.now(eastern)
    
    # Convert to string for processing
    timestamp_str = str(timestamp_value).strip()
    
    # If it's already a datetime object, just ensure timezone
    if isinstance(timestamp_value, (datetime, pd.Timestamp)):
        if hasattr(timestamp_value, 'tz') and timestamp_value.tz is None:
            return eastern.localize(timestamp_value.replace(tzinfo=None))
        elif hasattr(timestamp_value, 'tz') and timestamp_value.tz is not None:
            return timestamp_value.astimezone(eastern)
        else:
            return timestamp_value
    
    # Handle relative time strings that match what finviz might return
    current_time = datetime.now(eastern)
    
    if 'min' in timestamp_str.lower() and 'ago' in timestamp_str.lower():
        match = re.search(r'(\d+)', timestamp_str)
        if match:
            minutes = int(match.group(1))
            return current_time - timedelta(minutes=minutes)
    
    if 'hour' in timestamp_str.lower() and 'ago' in timestamp_str.lower():
        match = re.search(r'(\d+)', timestamp_str)
        if match:
            hours = int(match.group(1))
            return current_time - timedelta(hours=hours)
    
    if 'day' in timestamp_str.lower() and 'ago' in timestamp_str.lower():
        match = re.search(r'(\d+)', timestamp_str)
        if match:
            days = int(match.group(1))
            return current_time - timedelta(days=days)
    
    # Handle "Today" and "Yesterday" strings
    if 'today' in timestamp_str.lower():
        return current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    
    if 'yesterday' in timestamp_str.lower():
        return (current_time - timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
    
    # Try standard datetime parsing
    try:
        parsed = pd.to_datetime(timestamp_str, errors='coerce')
        if pd.notna(parsed):
            if parsed.tz is None:
                return eastern.localize(parsed.to_pydatetime())
            else:
                return parsed.tz_convert(eastern).to_pydatetime()
    except:
        pass
    
    # Default fallback
    return current_time

# Process timestamps if column exists
if timestamp_col is None:
    logger.warning("No timestamp column found. Using current time for all news items.")
    current_time = datetime.now(eastern)
    news_items['datetime'] = current_time
    timestamp_col = 'datetime'
else:
    # Apply enhanced timestamp processing
    news_items['datetime'] = news_items[timestamp_col].apply(process_timestamp)
    timestamp_col = 'datetime'

if headline_col is None:
    logger.error(f"Headline column not found. Available columns: {news_items.columns.tolist()}")
    logger.error("Please check the pyfinviz library documentation for the correct column names.")
    sys.exit(1)

def extract_ticker_from_headline(headline, valid_tickers=None):
    if valid_tickers is None:
        valid_tickers = set()
    
    match = re.search(r'\(([A-Z]{1,5})\)', headline)
    if match:
        ticker = match.group(1)
        if not valid_tickers or ticker in valid_tickers:
            return ticker
    
    matches = re.findall(r'\$([A-Z]{1,5})', headline)
    for ticker in matches:
        if not valid_tickers or ticker in valid_tickers:
            return ticker
    
    match = re.search(r'^([A-Z]{2,5}):', headline)
    if match:
        ticker = match.group(1)
        if not valid_tickers or ticker in valid_tickers:
            return ticker
    
    if valid_tickers:
        words = re.findall(r'\b[A-Z]{1,5}\b', headline)
        for word in words:
            if word in valid_tickers:
                return word
    
    if not valid_tickers:
        match = re.search(r'(?:^|\s)([A-Z]{2,5})(?:\s|:|$)', headline)
        if match:
            ticker = match.group(1)
            excluded_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'DOW', 'CEO', 'CFO', 'CTO', 'USA', 'SEC', 'FDA', 'IPO', 'ETF', 'ESG', 'NYSE', 'NASDAQ'}
            if ticker not in excluded_words:
                return ticker
    
    return None

# === Collect all articles for batch processing ===
articles_for_processing = []

for _, row in news_items.iterrows():
    headline = row.get(headline_col, "").strip() if headline_col else ""
    timestamp = row.get('datetime', datetime.now(eastern))
    
    if ticker_col:
        ticker = row.get(ticker_col, "").strip().upper()
    else:
        ticker = extract_ticker_from_headline(headline, VALID_TICKERS)
        ticker = ticker.upper() if ticker else ""

    if not ticker or not headline:
        logger.debug(f"Skipping row due to missing ticker or headline: ticker='{ticker}', headline='{headline[:50] if headline else 'None'}...'")
        continue
    if TARGET_TICKERS and ticker not in TARGET_TICKERS:
        continue

    try:
        # === Sentiment Calculation ===
        sentiment = sentiment_processor.calculate_enhanced_sentiment(headline.lower())

        # === Prepare dict for feature engineering ===
        processed = {
            "ticker": ticker,
            "datetime": timestamp,  # Pass the processed datetime object directly
            "headline": headline,
            "text": headline,
            "sentiment_combined": sentiment.get("combined", 0.0),
            "sentiment_dynamic": sentiment.get("dynamic", 0.0),
            "sentiment_ml": sentiment.get("ml", 0.0),
            "sentiment_keyword": sentiment.get("keyword", 0.0),
        }
        
        articles_for_processing.append(processed)

    except Exception as e:
        logger.error(f"Error preparing article '{headline[:50]}...': {e}")
        continue

# === Batch Process Articles ===
if articles_for_processing:
    logger.info(f"Processing {len(articles_for_processing)} articles in batch...")
    
    # Create DataFrame from all articles
    input_df = pd.DataFrame(articles_for_processing)
    
    try:
        # Run feature engineering pipeline
        engineered_df = feature_engineer.feature_engineering_pipeline(input_df, save_cleaned=False)
        
        if len(engineered_df) == 0:
            logger.error("No articles remained after feature engineering. Check datetime formats.")
            sys.exit(1)
        
        # Get feature columns
        if FEATURE_COLUMNS is None:
            FEATURE_COLUMNS = feature_engineer.feature_columns
            logger.info(f"Determined feature columns: {len(FEATURE_COLUMNS)} features")
        
        # Generate predictions for all processed articles
        predictions = []
        
        for idx, row in engineered_df.iterrows():
            try:
                # Check if all required features are present
                missing_cols = [col for col in FEATURE_COLUMNS if col not in engineered_df.columns]
                if missing_cols:
                    logger.warning(f"Skipping article due to missing features {missing_cols[:5]}...")
                    continue

                feature_vector = row[FEATURE_COLUMNS].tolist()
                
                # --- PREDICTION ---
                predicted_pct_change = model.predict([feature_vector])[0]

                predictions.append({
                    "datetime": row['datetime'].strftime("%Y-%m-%d %H:%M:%S") if hasattr(row['datetime'], 'strftime') else str(row['datetime']),
                    "ticker": row['ticker'],
                    "headline": row['headline'],
                    "sentiment_combined": round(row['sentiment_combined'], 4),
                    "predicted_pct_change_1h": round(predicted_pct_change, 4)
                })

            except Exception as e:
                logger.error(f"Error generating prediction for article: {e}")
                continue
        
        # === Output Results ===
        if predictions:
            df_output = pd.DataFrame(predictions)
            df_output = df_output.sort_values(by="datetime", ascending=False)

            output_path = "predicted_price_change_news.csv"
            df_output.to_csv(output_path, index=False)
            logger.info(f"Saved {len(predictions)} live predictions to {output_path}")
            
            logger.info("Summary of predictions:")
            logger.info(f"Total articles fetched: {len(news_items)}")
            logger.info(f"Articles with valid tickers: {len(articles_for_processing)}")
            logger.info(f"Articles successfully processed: {len(engineered_df)}")
            logger.info(f"Predictions generated: {len(predictions)}")
            
            if predictions:
                df_summary = df_output.groupby('ticker').size().sort_values(ascending=False)
                logger.info(f"Top tickers by article count: {dict(df_summary.head(10))}")
            
            logger.info("Top 10 predictions by sentiment:")
            top_predictions = df_output.nlargest(10, 'sentiment_combined')
            for _, pred in top_predictions.iterrows():
                logger.info(f"{pred['ticker']}: {pred['predicted_pct_change_1h']:.4f}% - {pred['headline'][:60]}...")
        else:
            logger.warning("No predictions generated after feature engineering.")

    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {e}")
        import traceback
        traceback.print_exc()

else:
    logger.warning("No valid articles found for processing. Check ticker extraction logic.")