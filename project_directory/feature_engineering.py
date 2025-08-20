import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from typing import List, Optional
import warnings
import re
from datetime import datetime, timedelta
import pytz
import os
from dateutil import parser as date_parser

warnings.filterwarnings('ignore')

class FinancialNewsFeatureEngineer:
    def __init__(self, datetime_col: str = 'datetime', group_col: str = 'ticker', live_mode: bool = False):
        self.datetime_col = datetime_col
        self.group_col = group_col
        self.eastern_tz = pytz.timezone('US/Eastern') # Timezone used for market and article datetimes
        self.live_mode = live_mode
        self.feature_columns = [] # List to keep track of generated feature columns
        self.price_cache = {} # In-memory cache for ticker price data
        self.market_cache = None # Market-wide context data (e.g., SPY, VIX)

    def _fetch_and_cache_data(self, df: pd.DataFrame):
        """Fetches and caches all required historical price and market data using persistent files."""
        
        # Create a directory for the cache if it doesn't exist
        cache_dir = "price_data_cache"
        os.makedirs(cache_dir, exist_ok=True)

        tickers = df[self.group_col].unique().tolist()
        if not tickers:
            return
        
        # Define date range with lookback buffer (1 year before earliest article date)
        min_date = df[self.datetime_col].min().date() - timedelta(days=365)
        max_date = df[self.datetime_col].max().date() + timedelta(days=1)
        
        for ticker in tickers:
            # Define the path for this ticker's cache file
            cache_file = os.path.join(cache_dir, f"{ticker}.parquet")

            # First, check the in-memory cache
            if ticker in self.price_cache:
                continue

            # Second, check the persistent file cache
            if os.path.exists(cache_file):
                print(f"Loading cached data for {ticker} from disk...")
                try:
                    cached_data = pd.read_parquet(cache_file)
                    # Ensure Date column is timezone-aware and matches article timezone
                    if 'Date' in cached_data.columns:
                        cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                        if cached_data['Date'].dt.tz is None:
                            # If timezone-naive, assume it's US/Eastern market data
                            cached_data['Date'] = cached_data['Date'].dt.tz_localize(self.eastern_tz)
                        else:
                            # If timezone-aware, convert to Eastern
                            cached_data['Date'] = cached_data['Date'].dt.tz_convert(self.eastern_tz)
                    self.price_cache[ticker] = cached_data
                    continue
                except Exception as e:
                    print(f"Error loading cached data for {ticker}: {e}")
                
            # If not in any cache, download the data
            print(f"Fetching historical data for {ticker} from API...")
            try:
                stock_hist = yf.download(ticker, start=min_date, end=max_date, progress=False)
                if not stock_hist.empty:
                    # Reset index to make Date a regular column
                    stock_hist = stock_hist.reset_index()
                    
                    # Flatten multi-level columns if they exist
                    if isinstance(stock_hist.columns, pd.MultiIndex):
                        stock_hist.columns = [col[0] if isinstance(col, tuple) else col for col in stock_hist.columns]
                    
                    # Ensure Date column is timezone-aware
                    if 'Date' in stock_hist.columns:
                        stock_hist['Date'] = pd.to_datetime(stock_hist['Date'])
                        if stock_hist['Date'].dt.tz is None:
                            # Assume market data is in US/Eastern timezone
                            stock_hist['Date'] = stock_hist['Date'].dt.tz_localize(self.eastern_tz)
                        else:
                            stock_hist['Date'] = stock_hist['Date'].dt.tz_convert(self.eastern_tz)
                    
                    # Store in in-memory cache for the current run
                    self.price_cache[ticker] = stock_hist
                    # Save to disk for future runs (convert to UTC for storage)
                    try:
                        stock_hist_utc = stock_hist.copy()
                        stock_hist_utc['Date'] = stock_hist_utc['Date'].dt.tz_convert('UTC')
                        stock_hist_utc.to_parquet(cache_file)
                    except Exception as e:
                        print(f"Warning: Could not save cache for {ticker}: {e}")
                else:
                    print(f"Warning: No data returned for {ticker}")
                    self.price_cache[ticker] = pd.DataFrame() # Store empty DataFrame
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                self.price_cache[ticker] = pd.DataFrame()

        # Also fetch market context data (SPY, VIX)
        self._fetch_market_context_data(min_date, max_date)

    def _fetch_market_context_data(self, min_date, max_date):
        """Fetch SPY and VIX data for market context."""
        cache_file = os.path.join("price_data_cache", "market_context.parquet")
        
        # Check if market data is already cached
        if os.path.exists(cache_file):
            print("Loading cached market context data...")
            try:
                self.market_cache = pd.read_parquet(cache_file)
                # Ensure timezone consistency
                if 'Date' in self.market_cache.columns:
                    self.market_cache['Date'] = pd.to_datetime(self.market_cache['Date'])
                    if self.market_cache['Date'].dt.tz is None:
                        self.market_cache['Date'] = self.market_cache['Date'].dt.tz_localize(self.eastern_tz)
                    else:
                        self.market_cache['Date'] = self.market_cache['Date'].dt.tz_convert(self.eastern_tz)
                return
            except Exception as e:
                print(f"Error loading cached market data: {e}")
            
        print("Fetching market context data (SPY, VIX)...")
        try:
            # Fetch SPY data
            spy_data = yf.download('SPY', start=min_date, end=max_date, progress=False)
            if not spy_data.empty:
                spy_data = spy_data.reset_index()
                
                # Flatten columns if multi-level
                if isinstance(spy_data.columns, pd.MultiIndex):
                    spy_data.columns = [col[0] if isinstance(col, tuple) else col for col in spy_data.columns]
                
                spy_data['spy_daily_return'] = spy_data['Close'].pct_change()
                spy_data = spy_data[['Date', 'spy_daily_return']]
                
                # Fetch VIX data
                vix_data = yf.download('^VIX', start=min_date, end=max_date, progress=False)
                if not vix_data.empty:
                    vix_data = vix_data.reset_index()
                    # Flatten columns if multi-level
                    if isinstance(vix_data.columns, pd.MultiIndex):
                        vix_data.columns = [col[0] if isinstance(col, tuple) else col for col in vix_data.columns]
                    vix_data = vix_data[['Date', 'Close']].rename(columns={'Close': 'vix_close'})
                    
                    # Merge SPY and VIX data
                    market_data = pd.merge(spy_data, vix_data, on='Date', how='outer')
                else:
                    market_data = spy_data
                    market_data['vix_close'] = np.nan
                    
                # Ensure timezone consistency
                market_data['Date'] = pd.to_datetime(market_data['Date'])
                if market_data['Date'].dt.tz is None:
                    market_data['Date'] = market_data['Date'].dt.tz_localize(self.eastern_tz)
                else:
                    market_data['Date'] = market_data['Date'].dt.tz_convert(self.eastern_tz)
                    
                self.market_cache = market_data
                # Save to disk (convert to UTC for storage)
                try:
                    market_data_utc = market_data.copy()
                    market_data_utc['Date'] = market_data_utc['Date'].dt.tz_convert('UTC')
                    market_data_utc.to_parquet(cache_file)
                except Exception as e:
                    print(f"Warning: Could not save market cache: {e}")
            else:
                print("Warning: No SPY data available")
                self.market_cache = pd.DataFrame()
        except Exception as e:
            print(f"Error fetching market context data: {e}")
            self.market_cache = pd.DataFrame()

    def calculate_ta_features_for_one_row(self, past_data: pd.DataFrame) -> dict:
        """Calculate technical analysis features for a single row using only past data."""
        features = {
            'rsi_14': np.nan,
            'macd': np.nan,
            'macd_signal': np.nan,
            'price_vs_sma50': np.nan,
            'price_vs_sma200': np.nan
        }
        
        if past_data is None or past_data.empty or len(past_data) < 2:
            return features
            
        try:
            # Determine price column
            price_col = None
            if 'Close' in past_data.columns:
                price_col = 'Close'
            elif 'Adj Close' in past_data.columns:
                price_col = 'Adj Close'
            else:
                return features
                
            # Make a copy to avoid modifying original data
            past_data_copy = past_data.copy()

            # --- FIX: Drop any rows where the price is missing ---
            past_data_copy.dropna(subset=[price_col], inplace=True)
            
            # RSI - need at least 14 periods
            if len(past_data_copy) >= 14:
                try:
                    rsi_series = ta.rsi(past_data_copy[price_col], length=14)
                    if rsi_series is not None and not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]):
                        features['rsi_14'] = rsi_series.iloc[-1]
                except Exception as e:
                    print(f"Error calculating RSI: {e}")
            
            # MACD - need at least 26 periods
            if len(past_data_copy) >= 26:
                try:
                    macd_data = ta.macd(past_data_copy[price_col], fast=12, slow=26, signal=9)
                    if macd_data is not None and not macd_data.empty:
                        # Check for the correct column names
                        macd_col = None
                        signal_col = None
                        
                        for col in macd_data.columns:
                            if 'MACD_' in col and 'MACDs_' not in col:
                                macd_col = col
                            elif 'MACDs_' in col:
                                signal_col = col
                        
                        if macd_col and not pd.isna(macd_data[macd_col].iloc[-1]):
                            features['macd'] = macd_data[macd_col].iloc[-1]
                        if signal_col and not pd.isna(macd_data[signal_col].iloc[-1]):
                            features['macd_signal'] = macd_data[signal_col].iloc[-1]
                except Exception as e:
                    print(f"Error calculating MACD: {e}")
            
            # SMAs and price vs SMA
            current_price = past_data_copy[price_col].iloc[-1]
            
            if len(past_data_copy) >= 50:
                try:
                    sma_50 = ta.sma(past_data_copy[price_col], length=50)
                    if sma_50 is not None and not sma_50.empty and not pd.isna(sma_50.iloc[-1]):
                        sma_50_val = sma_50.iloc[-1]
                        features['price_vs_sma50'] = (current_price - sma_50_val) / sma_50_val * 100
                except Exception as e:
                    print(f"Error calculating SMA50: {e}")
                    
            if len(past_data_copy) >= 200:
                try:
                    sma_200 = ta.sma(past_data_copy[price_col], length=200)
                    if sma_200 is not None and not sma_200.empty and not pd.isna(sma_200.iloc[-1]):
                        sma_200_val = sma_200.iloc[-1]
                        features['price_vs_sma200'] = (current_price - sma_200_val) / sma_200_val * 100
                except Exception as e:
                    print(f"Error calculating SMA200: {e}")
                    
        except Exception as e:
            print(f"Error calculating TA features: {e}")
            
        return features

    def calculate_market_context_for_one_row(self, row_datetime: pd.Timestamp) -> dict:
        """Calculate market context features for a single row."""
        features = {
            'spy_daily_return': np.nan,
            'vix_close': np.nan
        }
        
        if self.market_cache is None or self.market_cache.empty:
            return features
            
        try:
            # Find the most recent market data before or on the article datetime
            # Convert row_datetime to same timezone as market data
            row_date = row_datetime.tz_convert(self.eastern_tz).date()
            market_data_before = self.market_cache[
                self.market_cache['Date'].dt.date < row_date
            ].sort_values('Date')
            
            if not market_data_before.empty:
                latest_market = market_data_before.iloc[-1]
                if 'spy_daily_return' in latest_market and not pd.isna(latest_market['spy_daily_return']):
                    features['spy_daily_return'] = latest_market['spy_daily_return']
                if 'vix_close' in latest_market and not pd.isna(latest_market['vix_close']):
                    features['vix_close'] = latest_market['vix_close']
                    
        except Exception as e:
            print(f"Error calculating market context: {e}")
            
        return features

    def _parse_today_time(self, time_part: str, now: datetime) -> Optional[datetime]:
        if not time_part or time_part.strip() == '': 
            return now.replace(hour=9, minute=30, second=0, microsecond=0)
        time_formats = ['%I:%M%p', '%I:%M %p', '%H:%M', '%I%p', '%I %p']
        time_part = time_part.strip()
        for fmt in time_formats:
            try:
                time_obj = datetime.strptime(time_part, fmt).time()
                return now.replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)
            except ValueError: 
                continue
        return now.replace(hour=9, minute=30, second=0, microsecond=0)

    def _parse_yesterday_time(self, time_part: str, now: datetime) -> Optional[datetime]:
        yesterday = now - timedelta(days=1)
        if not time_part or time_part.strip() == '': 
            return yesterday.replace(hour=9, minute=30, second=0, microsecond=0)
        time_formats = ['%I:%M%p', '%I:%M %p', '%H:%M', '%I%p', '%I %p']
        time_part = time_part.strip()
        for fmt in time_formats:
            try:
                time_obj = datetime.strptime(time_part, fmt).time()
                return yesterday.replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)
            except ValueError: 
                continue
        return yesterday.replace(hour=9, minute=30, second=0, microsecond=0)

    def parse_and_standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        def parse_datetime_enhanced(s):
            if pd.isna(s) or s == '' or s is None: 
                return None
            if isinstance(s, (datetime, pd.Timestamp)): 
                return s.astimezone(self.eastern_tz) if hasattr(s, 'tz') and s.tz is not None else self.eastern_tz.localize(s.replace(tzinfo=None))
            s = str(s).strip()
            if not s or s.lower() in ['nan', 'none', 'null']: 
                return None
            now = datetime.now(pytz.utc).astimezone(self.eastern_tz)
            s_lower = s.lower()
            relative_patterns = [
                (r'^today\s*(.*)$', lambda m: self._parse_today_time(m.group(1), now)),
                (r'^yesterday\s*(.*)$', lambda m: self._parse_yesterday_time(m.group(1), now)),
                (r'^(\d+)\s*h(?:ours?)?\s+ago$', lambda m: now - timedelta(hours=int(m.group(1)))),
                (r'^(\d+)\s*hr?s?\s+ago$', lambda m: now - timedelta(hours=int(m.group(1)))),
                (r'^(\d+)\s*m(?:in|ins|inutes?)?\s+ago$', lambda m: now - timedelta(minutes=int(m.group(1)))),
                (r'^(\d+)\s*d(?:ays?)?\s+ago$', lambda m: now - timedelta(days=int(m.group(1)))),
            ]
            for pattern, func in relative_patterns:
                match = re.search(pattern, s_lower)
                if match:
                    try:
                        result = func(match)
                        if result: 
                            return self.eastern_tz.localize(result.replace(tzinfo=None)) if result.tzinfo is None else result.astimezone(self.eastern_tz)
                    except Exception: 
                        continue
            try:
                parsed = date_parser.parse(s, fuzzy=True)
                return parsed.astimezone(self.eastern_tz) if parsed.tzinfo is not None else self.eastern_tz.localize(parsed)
            except Exception: 
                pass
            return None
            
        df[self.datetime_col] = df[self.datetime_col].apply(parse_datetime_enhanced)
        df = df.dropna(subset=[self.datetime_col])
        if len(df) > 0: 
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], utc=True).dt.tz_convert(self.eastern_tz)
        return df

    def clean_comma_separated_columns(self, df: pd.DataFrame, columns_to_clean: Optional[List[str]] = None) -> pd.DataFrame:
        if columns_to_clean is None: 
            columns_to_clean = ["pos_keywords", "neg_keywords", "mentions"]
        existing_columns = [col for col in columns_to_clean if col in df.columns]
        for col in existing_columns: 
            df[col] = df[col].fillna("").astype(str).apply(lambda x: len([item for item in x.split(',') if item.strip()]))
        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dt_col = df[self.datetime_col]
        df['day_of_week'] = dt_col.dt.dayofweek
        df['hour_of_day'] = dt_col.dt.hour
        is_weekday = (df['day_of_week'] < 5)
        df['is_market_hours'] = (is_weekday & (dt_col.dt.time >= pd.to_datetime('09:30').time()) & (dt_col.dt.time <= pd.to_datetime('16:00').time())).astype(int)
        df['is_premarket'] = (is_weekday & (dt_col.dt.time >= pd.to_datetime('04:00').time()) & (dt_col.dt.time < pd.to_datetime('09:30').time())).astype(int)
        df['is_aftermarket'] = (is_weekday & (dt_col.dt.time > pd.to_datetime('16:00').time()) & (dt_col.dt.time <= pd.to_datetime('20:00').time())).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        return df

    def feature_engineering_pipeline(self, df):
        """Complete feature engineering pipeline with timezone fixes."""
        print("Starting feature engineering pipeline...")
        
        # Step 1: Parse dates and get a clean DataFrame
        df = self.parse_and_standardize_datetime(df)
        if df.empty:
            print("No valid datetimes found after parsing.")
            return df
        
        # Step 2: Clean comma-separated columns
        df = self.clean_comma_separated_columns(df)
        
        # Step 3: Create time features
        df = self.create_time_features(df)
        
        # Step 4: Fetch and cache all raw data first
        self._fetch_and_cache_data(df)
        
        # Step 5: Process each article individually with proper timezone handling
        processed_articles = []
        
        for idx, row in df.iterrows():
            ticker = row[self.group_col]
            article_datetime = row[self.datetime_col]
            
            # Start with the original row
            new_row = row.to_dict()
            
            # Get ticker data
            ticker_data = self.price_cache.get(ticker, pd.DataFrame())
            
            if not ticker_data.empty:
                # Filter to only past data - both datetimes should be timezone-aware now
                try:
                    past_data = ticker_data[ticker_data['Date'].dt.date < article_datetime.date()].copy()                    
                    # Calculate TA features using only past data
                    ta_features = self.calculate_ta_features_for_one_row(past_data)
                    new_row.update(ta_features)
                    
                except Exception as e:
                    print(f"Error processing TA features for {ticker} at {article_datetime}: {e}")
                    # Add NaN values for TA features
                    new_row.update({
                        'rsi_14': np.nan,
                        'macd': np.nan,
                        'macd_signal': np.nan,
                        'price_vs_sma50': np.nan,
                        'price_vs_sma200': np.nan
                    })
            else:
                # Add NaN values for TA features if no ticker data
                new_row.update({
                    'rsi_14': np.nan,
                    'macd': np.nan,
                    'macd_signal': np.nan,
                    'price_vs_sma50': np.nan,
                    'price_vs_sma200': np.nan
                })
            
            # Calculate market context features
            market_context_features = self.calculate_market_context_for_one_row(article_datetime)
            new_row.update(market_context_features)
            
            processed_articles.append(new_row)
        
        result_df = pd.DataFrame(processed_articles)
        print(f"Feature engineering completed. Shape: {result_df.shape}")
        return result_df

def check_sentiment_price_alignment(df: pd.DataFrame, live_mode: bool = False) -> pd.DataFrame:
    df['sentiment_score'] = pd.to_numeric(df['sentiment_combined'], errors='coerce')
    price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
    if live_mode:
        price_cols = [col for col in price_cols if col in df.columns]
        if not price_cols:
            return df.groupby('ticker').agg({'sentiment_score': 'mean'}).reset_index()

    agg_dict = {'sentiment_score': 'mean', **{col: 'mean' for col in price_cols if col in df.columns}}
    grouped = df.groupby('ticker').agg(agg_dict).reset_index()

    def alignment_logic(row):
        result = {}
        for col in price_cols:
            if col in df.columns:
                sentiment, price = row['sentiment_score'], row[col]
                if pd.isna(sentiment) or pd.isna(price): 
                    result[col] = 'Missing'
                elif sentiment * price > 0: 
                    result[col] = 'Aligned'
                elif sentiment == 0 or price == 0: 
                    result[col] = 'Neutral'
                else: 
                    result[col] = 'Not Aligned'
        return pd.Series(result)

    if any(col in df.columns for col in price_cols):
        return pd.concat([grouped[['ticker', 'sentiment_score']], grouped.apply(alignment_logic, axis=1)], axis=1)
    return grouped[['ticker', 'sentiment_score']]
    
if __name__ == "__main__":
    try:
        df = pd.read_csv("scraped_articles.csv")
        print("Successfully loaded data from scraped_articles.csv")
    except FileNotFoundError:
        print("scraped_articles.csv not found, creating dummy data.")
        data = {
            'datetime': ['Today 10:00AM', 'Yesterday 3:30PM', 'Dec-25-23 11:00AM', '2024-01-15 09:00AM', '1h ago', '30m ago', '2023-07-20'],
            'ticker': ['AAPL', 'GOOG', 'AAPL', 'GOOG', 'MSFT', 'MSFT', 'AAPL'],
            'sentiment_combined': [0.6, -0.3, 0.8, 0.1, -0.7, 0.2, 0.5],
            'pct_change_1h': [0.015, -0.008, 0.021, 0.005, -0.030, 0.010, 0.012],
            'pct_change_4h': [0.020, -0.012, 0.030, 0.008, -0.045, 0.005, 0.018],
            'pct_change_eod': [0.030, -0.020, 0.045, 0.010, -0.050, 0.003, 0.025],
            'pct_change_eow': [0.050, -0.035, 0.070, 0.015, -0.080, 0.007, 0.040],
            'pos_keywords': ['good, excellent, great', 'positive', '', 'amazing, wonderful', None, 'bullish', 'strong, growth'],
            'neg_keywords': ['bad, terrible', '', 'poor, weak', None, 'bearish, decline', 'risky', ''],
            'mentions': ['AAPL, Apple', 'GOOG, Google, Alphabet', 'AAPL', 'GOOG', 'MSFT, Microsoft', 'MSFT', 'AAPL, Apple Inc']
        }
        df = pd.DataFrame(data)

    print(f"Original data shape: {df.shape}")
    
    # Initialize the feature engineer
    engineer = FinancialNewsFeatureEngineer()
    
    # Run the feature engineering pipeline
    df_processed = engineer.feature_engineering_pipeline(df.copy())
    
    if not df_processed.empty:
        print("\n--- Feature Engineering Summary ---")
        print(f"Final data shape: {df_processed.shape}")
        print("\nFeature Categories:")
        
        # Time-based features
        time_features = ['day_of_week', 'hour_of_day', 'is_market_hours', 'is_premarket', 'is_aftermarket', 'hour_sin', 'hour_cos']
        existing_time_features = [f for f in time_features if f in df_processed.columns]
        if existing_time_features:
            print(f"  - Time features: {existing_time_features}")
        
        # Technical analysis features
        ta_features = ['rsi_14', 'macd', 'macd_signal', 'price_vs_sma50', 'price_vs_sma200']
        existing_ta_features = [f for f in ta_features if f in df_processed.columns]
        if existing_ta_features:
            print(f"  - Technical Analysis: {existing_ta_features}")
        
        # Market context features
        market_features = ['spy_daily_return', 'vix_close']
        existing_market_features = [f for f in market_features if f in df_processed.columns]
        if existing_market_features:
            print(f"  - Market Context: {existing_market_features}")
        
        # Cleaned keyword features
        keyword_features = ['pos_keywords', 'neg_keywords', 'mentions']
        existing_keyword_features = [f for f in keyword_features if f in df_processed.columns]
        if existing_keyword_features:
            print(f"  - Keyword counts: {existing_keyword_features}")
        
        # Show sample of processed data
        print(f"\nSample of processed data:")
        display_cols = ['datetime', 'ticker', 'sentiment_combined', 'is_market_hours']
        display_cols.extend(existing_ta_features[:2])
        display_cols.extend(existing_market_features[:1])
        
        print(df_processed[display_cols].head())
        
        # Check for missing values in key features
        key_features = existing_ta_features + existing_market_features
        if key_features:
            missing_counts = df_processed[key_features].isnull().sum()
            if missing_counts.sum() > 0:
                print(f"\nMissing values in key features:")
                for feature, count in missing_counts.items():
                    if count > 0:
                        print(f"  {feature}: {count}/{len(df_processed)} ({count/len(df_processed)*100:.1f}%)")
        
        # Run sentiment-price alignment analysis if price data exists
        price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        existing_price_cols = [col for col in price_cols if col in df_processed.columns]
        
        if existing_price_cols and 'sentiment_combined' in df_processed.columns:
            print(f"\n--- Sentiment-Price Alignment Analysis ---")
            alignment_df = check_sentiment_price_alignment(df_processed)
            print(alignment_df)
        
        # Save processed data
        output_file = "processed_financial_news_features.csv"
        df_processed.to_csv(output_file, index=False)
        print(f"\nProcessed data saved to: {output_file}")
        
    else:
        print("Processing resulted in an empty DataFrame.")
        print("Please check your input data and datetime parsing.")