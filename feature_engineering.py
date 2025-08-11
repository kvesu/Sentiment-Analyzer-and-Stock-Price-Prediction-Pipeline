import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import warnings
import re
import os
from datetime import datetime, timedelta
import pytz
from dateutil import parser as date_parser

warnings.filterwarnings('ignore')

class FinancialNewsFeatureEngineer:
    def __init__(self, datetime_col: str = 'datetime', group_col: str = 'ticker', live_mode: bool = False):
        self.datetime_col = datetime_col
        self.group_col = group_col
        self.feature_columns = []
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.live_mode = live_mode

    def _parse_today_time(self, time_part: str, now: datetime) -> Optional[datetime]:
        """Parse time part from 'Today ...' strings."""
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
        """Parse time part from 'Yesterday ...' strings."""
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
        """Enhanced datetime parser that handles more formats and provides better debugging."""
        print(f"Analyzing datetime column '{self.datetime_col}'...")
        if self.datetime_col not in df.columns:
            datetime_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if datetime_candidates:
                self.datetime_col = datetime_candidates[0]
                print(f"Using '{self.datetime_col}' as datetime column")
            else:
                raise ValueError(f"No datetime column found. Available columns: {list(df.columns)}")

        sample_values = df[self.datetime_col].dropna().head(10).tolist()
        print(f"Sample datetime values: {sample_values}")

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

        original_series = df[self.datetime_col]
        df[self.datetime_col] = df[self.datetime_col].apply(parse_datetime_enhanced)
        parsed_count = df[self.datetime_col].notna().sum()
        failed_count = df[self.datetime_col].isna().sum()

        print(f"Parsing results: {parsed_count} successful, {failed_count} failed")
        if failed_count > 0:
            failed_indices = original_series[df[self.datetime_col].isna()].index
            print("Failed to parse values for:")
            for i in failed_indices.head(10):
                print(f"  Index {i}: '{original_series.loc[i]}'")
            df = df.dropna(subset=[self.datetime_col])
            print(f"Dropped {failed_count} rows due to unparseable datetime values")

        if len(df) > 0:
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], utc=True).dt.tz_convert(self.eastern_tz)
        return df

    def clean_comma_separated_columns(self, df: pd.DataFrame, columns_to_clean: Optional[List[str]] = None) -> pd.DataFrame:
        if columns_to_clean is None:
            columns_to_clean = ["pos_keywords", "neg_keywords", "mentions"]

        existing_columns = [col for col in columns_to_clean if col in df.columns]
        if existing_columns:
            print(f"Cleaning comma-separated columns: {existing_columns}")
            for col in existing_columns:
                print(f"Processing column '{col}'...")
                df[col] = df[col].fillna("").astype(str).apply(lambda x: len([item for item in x.split(',') if item.strip()]))
        else:
            print("No comma-separated columns found to clean.")
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates time-based and market session features."""
        print("Creating time and market session features...")
        dt_col = df[self.datetime_col]
        df['day_of_week'] = dt_col.dt.dayofweek
        df['hour_of_day'] = dt_col.dt.hour
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        is_weekday = (df['day_of_week'] < 5)

        # Define market session hours (US/Eastern)
        # Market hours are 9:30 AM to 4:00 PM (16:00)
        df['is_market_hours'] = (
            is_weekday &
            (dt_col.dt.time >= pd.to_datetime('09:30').time()) &
            (dt_col.dt.time <= pd.to_datetime('16:00').time())
        ).astype(int)
        
        # Pre-market hours are 4:00 AM to 9:30 AM
        df['is_premarket'] = (
            is_weekday &
            (dt_col.dt.time >= pd.to_datetime('04:00').time()) &
            (dt_col.dt.time < pd.to_datetime('09:30').time())
        ).astype(int)

        # After-market hours are 4:00 PM (16:00) to 8:00 PM (20:00)
        df['is_aftermarket'] = (
            is_weekday &
            (dt_col.dt.time > pd.to_datetime('16:00').time()) &
            (dt_col.dt.time <= pd.to_datetime('20:00').time())
        ).astype(int)
        
        # Cyclical features for time of day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        # Add all new feature names to the list
        time_features = [
            'day_of_week', 'hour_of_day', 'is_market_hours', 
            'is_premarket', 'is_aftermarket', 'hour_sin', 'hour_cos'
        ]
        self.feature_columns.extend(time_features)
        return df

    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'sentiment_combined' in df.columns:
            col = 'sentiment_combined'
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[f'{col}_strength'] = df[col].abs()
            df[f'{col}_positive'] = (df[col] > 0).astype(int)
            df[f'{col}_very_positive'] = (df[col] > 0.5).astype(int)
            sentiment_features = [f'{col}_strength', f'{col}_positive', f'{col}_very_positive']
            self.feature_columns.extend(sentiment_features)
        return df

    def create_price_features(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[f'{col}_abs'] = df[col].abs()
                df[f'{col}_significant'] = (df[f'{col}_abs'] > 0.02).astype(int) # 2% change
                self.feature_columns.extend([f'{col}_abs', f'{col}_significant'])
        return df

    def create_target_variables(self, df: pd.DataFrame, price_change_cols: Optional[List[str]] = None, thresholds: List[float] = [0.02]) -> pd.DataFrame:
        if self.live_mode: return df
        if price_change_cols is None:
            price_change_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        
        for col in price_change_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                for threshold in thresholds:
                    threshold_str = str(threshold).replace(".", "_")
                    df[f'target_{col}_direction_{threshold_str}'] = np.where(df[col] > threshold, 1, np.where(df[col] < -threshold, -1, 0))
        return df

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        critical_cols = ['sentiment_combined', self.datetime_col, self.group_col]
        df.dropna(subset=[col for col in critical_cols if col in df.columns], inplace=True)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def feature_engineering_pipeline(self, df: pd.DataFrame, clean_columns: Optional[List[str]] = None, save_cleaned: bool = True) -> pd.DataFrame:
        print(f"Starting enhanced feature engineering with {len(df)} articles")
        self.feature_columns = []
        df = self.clean_comma_separated_columns(df, clean_columns)
        df = self.parse_and_standardize_datetime(df)
        if df.empty:
            print("ERROR: No data remaining after datetime parsing!")
            return df
        
        price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        df = self.create_time_features(df)
        df = self.create_sentiment_features(df)
        df = self.create_price_features(df, price_cols)
        df = self.create_target_variables(df, price_cols)
        df = self.preprocess_features(df)

        print(f"Enhanced feature engineering complete. Final shape: {df.shape}")
        print(f"Created {len(self.feature_columns)} features")

        if save_cleaned and not df.empty:
            output_filename = "cleaned_engineered_features.csv"
            try:
                if os.path.exists(output_filename):
                    existing_df = pd.read_csv(output_filename)
                    new_df = df[~df["url"].isin(existing_df["url"])]
                    if not new_df.empty:
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df.to_csv(output_filename, index=False)
                        print(f"Appended {len(new_df)} new rows to '{output_filename}'")
                    else:
                        print("No new articles to append.")
                else:
                    df.to_csv(output_filename, index=False)
                    print(f"Created '{output_filename}' with {len(df)} rows")
            except Exception as e:
                print(f"Warning: Failed to save CSV due to: {e}")

        if not df.empty:
            alignment_summary = check_sentiment_price_alignment(df, live_mode=self.live_mode)
            print("\nSentiment-Price Alignment Summary:")
            print(alignment_summary)

        return df

    def get_feature_groups(self) -> dict:
        return {
            'time': [f for f in self.feature_columns if any(key in f for key in ['day_', 'hour_', 'is_', '_sin', '_cos'])],
            'sentiment': [f for f in self.feature_columns if 'sentiment_combined' in f],
            'price': [f for f in self.feature_columns if 'pct_change' in f]
        }

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
                if pd.isna(sentiment) or pd.isna(price): result[col] = 'Missing'
                elif sentiment * price > 0: result[col] = 'Aligned'
                elif sentiment == 0 or price == 0: result[col] = 'Neutral'
                else: result[col] = 'Not Aligned'
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
    columns_to_clean = ["pos_keywords", "neg_keywords", "mentions"]
    engineer = FinancialNewsFeatureEngineer()
    df_processed = engineer.feature_engineering_pipeline(df.copy(), clean_columns=columns_to_clean, save_cleaned=True)
    
    if not df_processed.empty:
        print("\n--- Feature Engineering Summary ---")
        print(f"Final data shape: {df_processed.shape}")
        print(f"Total features tracked: {len(engineer.feature_columns)}")
        
        print("\nFeature Categories:")
        feature_groups = engineer.get_feature_groups()
        for group, features in feature_groups.items():
            print(f"- {group.capitalize()} features ({len(features)}): {features[:3]}")
            
    else:
        print("Processing resulted in an empty DataFrame.")