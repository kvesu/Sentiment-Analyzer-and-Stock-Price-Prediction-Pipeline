import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import warnings
import re
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
        self.live_mode = live_mode  # Add live mode support
        
    def parse_and_standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced datetime parser that handles more formats and provides better debugging."""
        df = df.copy()
        
        # First, let's see what we're working with
        print(f"Analyzing datetime column '{self.datetime_col}'...")
        if self.datetime_col not in df.columns:
            print(f"Available columns: {list(df.columns)}")
            # Try to find a datetime-like column
            datetime_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if datetime_candidates:
                print(f"Found potential datetime columns: {datetime_candidates}")
                self.datetime_col = datetime_candidates[0]
                print(f"Using '{self.datetime_col}' as datetime column")
            else:
                raise ValueError(f"No datetime column found. Available columns: {list(df.columns)}")
        
        # Show sample values for debugging
        sample_values = df[self.datetime_col].dropna().head(10).tolist()
        print(f"Sample datetime values: {sample_values}")
        
        def parse_datetime_enhanced(s):
            """Enhanced datetime parser with more flexible handling."""
            if pd.isna(s) or s == '' or s is None:
                return None
                
            # If it's already a datetime object, return it
            if isinstance(s, (datetime, pd.Timestamp)):
                if hasattr(s, 'tz') and s.tz is None:
                    return self.eastern_tz.localize(s.replace(tzinfo=None))
                elif hasattr(s, 'tz') and s.tz is not None:
                    return s.astimezone(self.eastern_tz)
                else:
                    return s
                
            # Convert to string if not already
            s = str(s).strip()
            if not s or s.lower() in ['nan', 'none', 'null']:
                return None
            
            now = datetime.now(pytz.utc).astimezone(self.eastern_tz)
            s_lower = s.lower()
            
            # Handle relative time strings (more comprehensive)
            relative_patterns = [
                # Today/Yesterday patterns
                (r'^today\s*(.*)$', lambda m: self._parse_today_time(m.group(1), now)),
                (r'^yesterday\s*(.*)$', lambda m: self._parse_yesterday_time(m.group(1), now)),
                
                # Hours ago patterns
                (r'^(\d+)\s*h(?:ours?)?\s+ago$', lambda m: now - timedelta(hours=int(m.group(1)))),
                (r'^(\d+)\s*hr?s?\s+ago$', lambda m: now - timedelta(hours=int(m.group(1)))),
                
                # Minutes ago patterns
                (r'^(\d+)\s*m(?:in|ins|inutes?)?\s+ago$', lambda m: now - timedelta(minutes=int(m.group(1)))),
                
                # Days ago patterns
                (r'^(\d+)\s*d(?:ays?)?\s+ago$', lambda m: now - timedelta(days=int(m.group(1)))),
            ]
            
            for pattern, func in relative_patterns:
                match = re.search(pattern, s_lower)
                if match:
                    try:
                        result = func(match)
                        if result:
                            return self.eastern_tz.localize(result.replace(tzinfo=None)) if result.tzinfo is None else result.astimezone(self.eastern_tz)
                    except:
                        continue
            
            # Try dateutil parser first (most flexible)
            try:
                parsed = date_parser.parse(s, fuzzy=True)
                # If no timezone info, assume Eastern
                if parsed.tzinfo is None:
                    parsed = self.eastern_tz.localize(parsed)
                else:
                    parsed = parsed.astimezone(self.eastern_tz)
                return parsed
            except:
                pass
            
            # Try pandas to_datetime
            try:
                parsed = pd.to_datetime(s, infer_datetime_format=True, errors='coerce')
                if pd.notna(parsed):
                    if parsed.tz is None:
                        return self.eastern_tz.localize(parsed.to_pydatetime())
                    else:
                        return parsed.tz_convert(self.eastern_tz).to_pydatetime()
            except:
                pass
            
            # Manual format attempts for common patterns
            manual_formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %I:%M:%S %p",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %I:%M:%S %p",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %I:%M %p",
                "%m/%d/%Y %H:%M",
                "%m/%d/%Y %I:%M %p",
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%b %d, %Y %I:%M %p",
                "%B %d, %Y %I:%M %p",
                "%b %d %Y %I:%M %p",
                "%b-%d-%y %I:%M%p",
                "%b-%d-%Y %I:%M%p",
                "%d-%b-%Y %H:%M",
                "%d/%m/%Y %H:%M",
            ]
            
            for fmt in manual_formats:
                try:
                    parsed = datetime.strptime(s, fmt)
                    return self.eastern_tz.localize(parsed)
                except:
                    continue
            
            return None
        
        def _parse_today_time(self, time_part, now):
            """Parse time part from 'Today ...' strings."""
            if not time_part or time_part.strip() == '':
                return now.replace(hour=9, minute=30, second=0, microsecond=0)
            
            time_formats = ['%I:%M%p', '%I:%M %p', '%H:%M', '%I%p', '%I %p']
            time_part = time_part.strip()
            
            for fmt in time_formats:
                try:
                    time_obj = datetime.strptime(time_part, fmt).time()
                    return now.replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)
                except:
                    continue
            return now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        def _parse_yesterday_time(self, time_part, now):
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
                except:
                    continue
            return yesterday.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Bind helper methods to self
        self._parse_today_time = _parse_today_time
        self._parse_yesterday_time = _parse_yesterday_time
        
        # Apply the enhanced parser
        print("Parsing datetime values...")
        df[self.datetime_col] = df[self.datetime_col].apply(parse_datetime_enhanced)
        
        # Count and report parsing results
        initial_rows = df.shape[0]
        parsed_count = df[self.datetime_col].notna().sum()
        failed_count = df[self.datetime_col].isna().sum()
        
        print(f"Parsing results: {parsed_count} successful, {failed_count} failed")
        
        if failed_count > 0:
            print("Failed to parse values:")
            failed_values = df[df[self.datetime_col].isna()][self.datetime_col].head(10)
            for val in failed_values:
                print(f"  '{val}'")
        
        # Drop rows with unparseable datetime values
        if failed_count > 0:
            df = df.dropna(subset=[self.datetime_col])
            print(f"Dropped {failed_count} rows due to unparseable datetime values")
        
        # Ensure all datetimes are timezone-aware
        if len(df) > 0:
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
            if df[self.datetime_col].dt.tz is None:
                df[self.datetime_col] = df[self.datetime_col].dt.tz_localize(self.eastern_tz, ambiguous='infer', nonexistent='shift_forward')
            else:
                df[self.datetime_col] = df[self.datetime_col].dt.tz_convert(self.eastern_tz)

        return df
    
    def clean_comma_separated_columns(self, df: pd.DataFrame, 
                                      columns_to_clean: List[str] = None) -> pd.DataFrame:
        """Clean comma-separated string columns and convert to counts."""
        df = df.copy()
        
        if columns_to_clean is None:
            columns_to_clean = ["pos_keywords", "neg_keywords", "mentions"]
        
        # Filter to only existing columns
        existing_columns = [col for col in columns_to_clean if col in df.columns]
        
        if existing_columns:
            print(f"Cleaning comma-separated columns: {existing_columns}")
            
            for col in existing_columns:
                print(f"Processing column '{col}'...")
                original_sample = df[col].head(3).tolist()
                print(f"  Sample original values: {original_sample}")
                
                # Convert comma-separated strings to counts
                def count_comma_separated(x):
                    if pd.isna(x):
                        return 0
                    if isinstance(x, str):
                        # Handle empty strings or strings with just whitespace
                        x = x.strip()
                        if not x:
                            return 0
                        # Count commas + 1, but handle edge cases
                        items = [item.strip() for item in x.split(",") if item.strip()]
                        return len(items)
                    else:
                        # If it's already a number, convert to int
                        try:
                            return int(x)
                        except (ValueError, TypeError):
                            return 0
                
                df[col] = df[col].apply(count_comma_separated)
                
                processed_sample = df[col].head(3).tolist()
                print(f"  Sample processed values: {processed_sample}")
                print(f"  Column '{col}' stats: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
        else:
            print("No comma-separated columns found to clean.")
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features for financial markets."""
        df = df.copy()
        
        # Basic time features
        df['day_of_week'] = df[self.datetime_col].dt.dayofweek  # 0=Monday
        df['hour_of_day'] = df[self.datetime_col].dt.hour
        df['day_of_month'] = df[self.datetime_col].dt.day
        df['month'] = df[self.datetime_col].dt.month
        df['quarter'] = df[self.datetime_col].dt.quarter
        df['year'] = df[self.datetime_col].dt.year
        
        # Financial market specific features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_market_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 16)).astype(int)
        df['is_premarket'] = ((df['hour_of_day'] >= 4) & (df['hour_of_day'] < 9)).astype(int)
        df['is_aftermarket'] = ((df['hour_of_day'] > 16) & (df['hour_of_day'] <= 20)).astype(int)
        
        # Market session features
        df['is_opening_hour'] = (df['hour_of_day'] == 9).astype(int)
        df['is_closing_hour'] = (df['hour_of_day'] == 16).astype(int)
        
        # Cyclical encoding for better ML performance
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        time_features = [
            'day_of_week', 'hour_of_day', 'day_of_month', 'month', 'quarter', 'year',
            'is_weekend', 'is_market_hours', 'is_premarket', 'is_aftermarket',
            'is_opening_hour', 'is_closing_hour',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        self.feature_columns.extend(time_features)
        return df
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment features from sentiment_combined only."""
        df = df.copy()
        
        # Focus only on sentiment_combined
        if 'sentiment_combined' in df.columns:
            col = 'sentiment_combined'
            
            # Sentiment strength (absolute value)
            df[f'{col}_strength'] = abs(df[col])
            
            # Sentiment direction
            df[f'{col}_positive'] = (df[col] > 0).astype(int)
            df[f'{col}_negative'] = (df[col] < 0).astype(int)
            df[f'{col}_neutral'] = (df[col] == 0).astype(int)
            
            # Sentiment extremes (based on thresholds)
            df[f'{col}_very_positive'] = (df[col] > 0.5).astype(int)
            df[f'{col}_very_negative'] = (df[col] < -0.5).astype(int)
            
            # Sentiment confidence (distance from neutral)
            df[f'{col}_confidence'] = abs(df[col])
            
            sentiment_features = [
                f'{col}_strength', f'{col}_positive', f'{col}_negative', f'{col}_neutral',
                f'{col}_very_positive', f'{col}_very_negative', f'{col}_confidence'
            ]
            
            self.feature_columns.extend(sentiment_features)
        
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features for 1h, 4h, eod, eow only."""
        df = df.copy()
        
        # Only the specified price change columns
        price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        
        # Skip price features in live mode if columns don't exist
        if self.live_mode:
            price_cols = [col for col in price_cols if col in df.columns]
            if not price_cols:
                print("Live mode: No price change columns found, skipping price features")
                return df
        
        for col in price_cols:
            if col in df.columns:
                # Ensure the column is numeric before performing operations
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                df[f'{col}_abs'] = abs(df[col])
                df[f'{col}_positive'] = (df[col] > 0).astype(int)
                df[f'{col}_negative'] = (df[col] < 0).astype(int)
                df[f'{col}_significant'] = (abs(df[col]) > 2).astype(int)  # >2% change
                
                price_features = [
                    f'{col}_abs', f'{col}_positive', f'{col}_negative', f'{col}_significant'
                ]
                
                self.feature_columns.extend(price_features)
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame, 
                                 price_change_cols: List[str] = None,
                                 thresholds: List[float] = [0.02]) -> pd.DataFrame:
        """Create target variables for the specified price changes."""
        df = df.copy()
        if price_change_cols is None:
            price_change_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        
        # Skip target creation in live mode if columns don't exist
        if self.live_mode:
            price_change_cols = [col for col in price_change_cols if col in df.columns]
            if not price_change_cols:
                print("Live mode: No price change columns found, skipping target variables")
                return df
        
        for col in price_change_cols:
            if col in df.columns:
                # Ensure the column is numeric before creating targets
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                for threshold in thresholds:
                    # Binary targets
                    threshold_str = str(threshold).replace(".", "_")
                    df[f'target_{col}_up_{threshold_str}'] = (df[col] > threshold).astype(int)
                    df[f'target_{col}_down_{threshold_str}'] = (df[col] < -threshold).astype(int)
                    
                    # Multi-class targets
                    df[f'target_{col}_direction_{threshold_str}'] = np.where(
                        df[col] > threshold, 1,
                        np.where(df[col] < -threshold, -1, 0)
                    )
                    
                    target_features = [
                        f'target_{col}_up_{threshold_str}',
                        f'target_{col}_down_{threshold_str}',
                        f'target_{col}_direction_{threshold_str}'
                    ]
                    
                    self.feature_columns.extend(target_features)
        return df
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess features."""
        df = df.copy()
        
        # Drop rows with missing critical features
        critical_cols = ['sentiment_combined', self.datetime_col, self.group_col] 
        existing_critical = [col for col in critical_cols if col in df.columns]
        if existing_critical:
            initial_rows = df.shape[0]
            df = df.dropna(subset=existing_critical)
            if df.shape[0] < initial_rows:
                print(f"Dropped {initial_rows - df.shape[0]} rows due to missing critical columns after feature creation.")
        
        # Fill missing values for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols_to_fill = [col for col in numeric_cols if col not in existing_critical]
        df[numeric_cols_to_fill] = df[numeric_cols_to_fill].fillna(0)
        
        # Remove infinite values and fill with 0
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def feature_engineering_pipeline(self, df: pd.DataFrame, 
                                     clean_columns: List[str] = None,
                                     save_cleaned: bool = True) -> pd.DataFrame:
        """Main feature engineering pipeline with integrated column cleaning."""
        print(f"Starting enhanced feature engineering with {len(df)} articles")
        
        self.feature_columns = []
        df = df.copy()

        # Step 1: Clean comma-separated columns first (before other processing)
        df = self.clean_comma_separated_columns(df, clean_columns)

        # Step 2: Parse and standardize datetime
        df = self.parse_and_standardize_datetime(df)
        
        if len(df) == 0:
            print("ERROR: No data remaining after datetime parsing!")
            return df
        
        # Ensure the grouping column exists
        if self.group_col not in df.columns or df[self.group_col].isnull().all():
            print(f"Warning: Grouping column '{self.group_col}' is missing or entirely null.")
            df[self.group_col] = df[self.group_col].fillna('UNKNOWN_TICKER')

        # Step 3: Apply streamlined feature creation methods
        df = self.create_time_features(df)
        df = self.create_sentiment_features(df)
        df = self.create_price_features(df)
        df = self.create_target_variables(df)
        df = self.preprocess_features(df)

        print(f"Enhanced feature engineering complete. Final shape: {df.shape}")
        print(f"Created {len(self.feature_columns)} features")

        # Step 4: Save cleaned and processed data
        if save_cleaned and len(df) > 0:
            output_filename = "cleaned_engineered_features.csv"
            df.to_csv(output_filename, index=False)
            print(f"Cleaned and engineered features saved as '{output_filename}'")

        # Step 5: Evaluate whether average sentiment aligns with average price changes for each ticker
        if len(df) > 0:  # Only run if we have data left
            alignment_summary = check_sentiment_price_alignment(df, live_mode=self.live_mode)
            print("\nSentiment-Price Alignment Summary:")
            print(alignment_summary)
        
        return df

    def get_feature_groups(self) -> dict:
        """Return a dictionary grouping features by category."""
        groups = {
            'time': [f for f in self.feature_columns if any(key in f for key in ['day_', 'hour_', 'month_', 'year_', 'is_', '_sin', '_cos'])],
            'sentiment': [f for f in self.feature_columns if 'sentiment_combined' in f],
            'price': [f for f in self.feature_columns if 'pct_change' in f and not f.startswith('target_')],
            'target': [f for f in self.feature_columns if f.startswith('target_')]
        }
        return groups

def check_sentiment_price_alignment(df: pd.DataFrame, live_mode: bool = False) -> pd.DataFrame:
    """Check alignment between sentiment scores and price changes by ticker."""
    df = df.copy()
    df['sentiment_score'] = pd.to_numeric(df['sentiment_combined'], errors='coerce')

    # Define price columns to check
    price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
    
    # In live mode, only use existing price columns
    if live_mode:
        price_cols = [col for col in price_cols if col in df.columns]
        if not price_cols:
            print("Live mode: No price columns for alignment check, returning sentiment summary only")
            return df.groupby('ticker').agg({'sentiment_score': 'mean'}).reset_index()

    # Build aggregation dictionary dynamically
    agg_dict = {'sentiment_score': 'mean'}
    for col in price_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'

    grouped = df.groupby('ticker').agg(agg_dict).reset_index()

    def alignment_logic(row):
        result = {}
        for col in price_cols:
            if col in df.columns:
                sentiment = row['sentiment_score']
                price = row[col]
                if pd.isna(sentiment) or pd.isna(price):
                    result[col] = 'Missing'
                elif sentiment * price > 0:
                    result[col] = 'Aligned'
                elif sentiment == 0 or price == 0:
                    result[col] = 'Neutral'
                else:
                    result[col] = 'Not Aligned'
        return pd.Series(result)

    alignment_df = grouped[['ticker', 'sentiment_score']].copy()
    if len([col for col in price_cols if col in df.columns]) > 0:
        alignment_results = grouped.apply(alignment_logic, axis=1)
        return pd.concat([alignment_df, alignment_results], axis=1)
    else:
        return alignment_df

# Main execution function
def feature_engineering_pipeline(df: pd.DataFrame, 
                                 clean_columns: List[str] = None) -> pd.DataFrame:
    """Main pipeline function for integration with existing script."""
    engineer = FinancialNewsFeatureEngineer()
    return engineer.feature_engineering_pipeline(df, clean_columns=clean_columns)

if __name__ == "__main__":
    # Example usage
    try:
        # Create a dummy DataFrame for demonstration if scraped_articles.csv doesn't exist
        try:
            df = pd.read_csv("scraped_articles.csv")
        except FileNotFoundError:
            print("scraped_articles.csv not found, creating dummy data.")
            data = {
                'datetime': [
                    'Today 10:00AM', 'Yesterday 3:30PM', 'Dec-25-23 11:00AM', 
                    '2024-01-15 09:00AM', '1h ago', '30m ago', '2023-07-20'
                ],
                'ticker': ['AAPL', 'GOOG', 'AAPL', 'GOOG', 'MSFT', 'MSFT', 'AAPL'],
                'sentiment_combined': [0.6, -0.3, 0.8, 0.1, -0.7, 0.2, 0.5],
                'pct_change_1h': [1.5, -0.8, 2.1, 0.5, -3.0, 1.0, 1.2],
                'pct_change_4h': [2.0, -1.2, 3.0, 0.8, -4.5, 0.5, 1.8],
                'pct_change_eod': [3.0, -2.0, 4.5, 1.0, -5.0, 0.3, 2.5],
                'pct_change_eow': [5.0, -3.5, 7.0, 1.5, -8.0, 0.7, 4.0],
                # Add sample comma-separated columns for testing
                'pos_keywords': ['good, excellent, great', 'positive', '', 'amazing, wonderful', None, 'bullish', 'strong, growth'],
                'neg_keywords': ['bad, terrible', '', 'poor, weak', None, 'bearish, decline', 'risky', ''],
                'mentions': ['AAPL, Apple', 'GOOG, Google, Alphabet', 'AAPL', 'GOOG', 'MSFT, Microsoft', 'MSFT', 'AAPL, Apple Inc']
            }
            df = pd.DataFrame(data)

        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Specify which columns to clean (will default to pos_keywords, neg_keywords, mentions if None)
        columns_to_clean = ["pos_keywords", "neg_keywords", "mentions"]
        
        engineer = FinancialNewsFeatureEngineer()
        df_processed = engineer.feature_engineering_pipeline(
            df.copy(), 
            clean_columns=columns_to_clean,
            save_cleaned=True
        )
        
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Created features: {len(engineer.feature_columns)}")
        
        # Only display results if we have data
        if len(df_processed) > 0:
            print("Feature engineering complete!")
            
            # Display feature summary
            print("\nFeature Categories:")
            feature_groups = engineer.get_feature_groups()
            for group, features in feature_groups.items():
                print(f"- {group.capitalize()} features: {len(features)}")
                if features:  # Only show first few features as examples
                    print(f"  Examples: {features[:3]}")
            
            print(f"\nTotal features created: {len(engineer.feature_columns)}")
            print(f"Original columns: {df.shape[1]}")
            print(f"Final columns: {df_processed.shape[1]}")
            
            # Show sample of cleaned columns
            if any(col in df_processed.columns for col in columns_to_clean):
                print("\nSample of cleaned columns:")
                cleaned_cols = [col for col in columns_to_clean if col in df_processed.columns]
                print(df_processed[cleaned_cols].head())
        else:
            print("No data remaining after datetime parsing. Check your datetime column format.")

    except Exception as e:
        print(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()