import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import warnings
import re
from datetime import datetime, timedelta
import pytz
warnings.filterwarnings('ignore')

class FinancialNewsFeatureEngineer:
    def __init__(self, datetime_col: str = 'datetime', group_col: str = 'ticker'):
        self.datetime_col = datetime_col
        self.group_col = group_col
        self.feature_columns = []
        self.eastern_tz = pytz.timezone('US/Eastern')
        
    def parse_and_standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse various datetime formats from the main script and standardize them."""
        df = df.copy()
        
        def parse_datetime_flexible(s):
            """Parse datetime strings in various formats from finviz"""
            if pd.isna(s) or not s:
                return None
            
            now = datetime.now()
            s = str(s).strip().lower()
            
            # Handle relative time strings
            if s.startswith("today"):
                return now.replace(hour=9, minute=30, second=0, microsecond=0)
            if s.startswith("yesterday"):
                return (now - timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Handle "X hours ago" format
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
            
            # Standard datetime formats
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
        
        # Parse datetime column
        df[self.datetime_col] = df[self.datetime_col].apply(parse_datetime_flexible)
        df = df.dropna(subset=[self.datetime_col])
        
        # Convert to pandas datetime
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time-based features for financial markets."""
        df = df.copy()
        
        # Ensure datetime is properly formatted
        df = self.parse_and_standardize_datetime(df)
        
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
        df['is_overnight'] = ((df['hour_of_day'] > 20) | (df['hour_of_day'] < 4)).astype(int)
        
        # Market session features
        df['is_opening_hour'] = (df['hour_of_day'] == 9).astype(int)
        df['is_closing_hour'] = (df['hour_of_day'] == 16).astype(int)
        df['is_lunch_hour'] = (df['hour_of_day'] == 12).astype(int)
        
        # Cyclical encoding for better ML performance
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time until market events
        df['minutes_to_market_open'] = df.apply(self._minutes_to_market_open, axis=1)
        df['minutes_to_market_close'] = df.apply(self._minutes_to_market_close, axis=1)
        
        time_features = [
            'day_of_week', 'hour_of_day', 'day_of_month', 'month', 'quarter', 'year',
            'is_weekend', 'is_market_hours', 'is_premarket', 'is_aftermarket', 'is_overnight',
            'is_opening_hour', 'is_closing_hour', 'is_lunch_hour',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'minutes_to_market_open', 'minutes_to_market_close'
        ]
        
        self.feature_columns.extend(time_features)
        return df
    
    def _minutes_to_market_open(self, row):
        """Calculate minutes until next market open."""
        dt = row[self.datetime_col]
        if dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):
            market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
            return int((market_open - dt).total_seconds() / 60)
        else:
            next_day = dt + timedelta(days=1)
            market_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
            return int((market_open - dt).total_seconds() / 60)
    
    def _minutes_to_market_close(self, row):
        """Calculate minutes until market close."""
        dt = row[self.datetime_col]
        if dt.hour < 16:
            market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
            return int((market_close - dt).total_seconds() / 60)
        else:
            next_day = dt + timedelta(days=1)
            market_close = next_day.replace(hour=16, minute=0, second=0, microsecond=0)
            return int((market_close - dt).total_seconds() / 60)
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced sentiment features from multiple sentiment scores."""
        df = df.copy()
        
        # Multiple sentiment methods from main script
        sentiment_cols = ['sentiment_dynamic', 'sentiment_ml', 'sentiment_keyword', 'sentiment_combined']
        
        for col in sentiment_cols:
            if col in df.columns:
                # Sentiment strength (absolute value)
                df[f'{col}_strength'] = abs(df[col])
                
                # Sentiment direction
                df[f'{col}_positive'] = (df[col] > 0).astype(int)
                df[f'{col}_negative'] = (df[col] < 0).astype(int)
                df[f'{col}_neutral'] = (df[col] == 0).astype(int)
                
                # Sentiment extremes (based on thresholds)
                df[f'{col}_very_positive'] = (df[col] > 0.5).astype(int)
                df[f'{col}_very_negative'] = (df[col] < -0.5).astype(int)
                df[f'{col}_moderate'] = ((df[col] >= -0.5) & (df[col] <= 0.5)).astype(int)
                
                # Sentiment confidence (distance from neutral)
                df[f'{col}_confidence'] = abs(df[col])
                
                self.feature_columns.extend([
                    f'{col}_strength', f'{col}_positive', f'{col}_negative', f'{col}_neutral',
                    f'{col}_very_positive', f'{col}_very_negative', f'{col}_moderate', f'{col}_confidence'
                ])
        
        # Sentiment agreement features
        if all(col in df.columns for col in sentiment_cols):
            df['sentiment_agreement'] = df[sentiment_cols].std(axis=1)
            df['sentiment_consensus'] = df[sentiment_cols].mean(axis=1)
            df['sentiment_max_deviation'] = df[sentiment_cols].max(axis=1) - df[sentiment_cols].min(axis=1)
            
            self.feature_columns.extend(['sentiment_agreement', 'sentiment_consensus', 'sentiment_max_deviation'])
        
        # Keyword-based features
        if 'pos_keywords' in df.columns and 'neg_keywords' in df.columns:
            df['pos_keyword_count'] = df['pos_keywords'].str.count(',') + 1
            df['neg_keyword_count'] = df['neg_keywords'].str.count(',') + 1
            df['total_keyword_count'] = df['pos_keyword_count'] + df['neg_keyword_count']
            df['keyword_ratio'] = df['pos_keyword_count'] / (df['neg_keyword_count'] + 1)
            
            self.feature_columns.extend(['pos_keyword_count', 'neg_keyword_count', 'total_keyword_count', 'keyword_ratio'])
        
        # Text-based features
        if 'text' in df.columns:
            df['text_length'] = df['text'].str.len()
            df['text_word_count'] = df['text'].str.split().str.len()
            df['headline_length'] = df['headline'].str.len() if 'headline' in df.columns else 0
            
            self.feature_columns.extend(['text_length', 'text_word_count', 'headline_length'])
        
        # Mention features
        if 'mentions' in df.columns:
            df['mention_count'] = df['mentions'].str.count(',') + 1
            df['has_mentions'] = (df['mentions'].str.len() > 0).astype(int)
            
            self.feature_columns.extend(['mention_count', 'has_mentions'])
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_cols: List[str] = None, 
                          lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lag features for sentiment and other key metrics."""
        df = df.copy()
        if target_cols is None:
            target_cols = ['sentiment_combined', 'sentiment_dynamic', 'sentiment_ml', 'prediction_confidence']
        
        # Sort by ticker and datetime
        df = df.sort_values([self.group_col, self.datetime_col])
        
        for col in target_cols:
            if col in df.columns:
                for lag in lags:
                    lag_col_name = f'{col}_lag{lag}'
                    df[lag_col_name] = df.groupby(self.group_col)[col].shift(lag)
                    self.feature_columns.append(lag_col_name)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_cols: List[str] = None, 
                              windows: List[int] = [3, 5, 10, 20]) -> pd.DataFrame:
        """Create rolling statistics features."""
        df = df.copy()
        if target_cols is None:
            target_cols = ['sentiment_combined', 'sentiment_dynamic', 'prediction_confidence']
        
        for col in target_cols:
            if col in df.columns:
                for window in windows:
                    # Rolling statistics
                    df[f'{col}_rolling_mean_{window}'] = df.groupby(self.group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    df[f'{col}_rolling_std_{window}'] = df.groupby(self.group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
                    df[f'{col}_rolling_min_{window}'] = df.groupby(self.group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).min()
                    )
                    df[f'{col}_rolling_max_{window}'] = df.groupby(self.group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
                    
                    # Rolling trend (current vs rolling mean)
                    df[f'{col}_vs_rolling_mean_{window}'] = df[col] - df[f'{col}_rolling_mean_{window}']
                    
                    # Rolling percentile rank
                    df[f'{col}_rolling_rank_{window}'] = df.groupby(self.group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).rank(pct=True)
                    )
                    
                    rolling_features = [
                        f'{col}_rolling_mean_{window}', f'{col}_rolling_std_{window}',
                        f'{col}_rolling_min_{window}', f'{col}_rolling_max_{window}',
                        f'{col}_vs_rolling_mean_{window}', f'{col}_rolling_rank_{window}'
                    ]
                    self.feature_columns.extend(rolling_features)
        
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features from the main script's price data."""
        df = df.copy()
        
        # Price change features
        price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        
        for col in price_cols:
            if col in df.columns:
                df[f'{col}_abs'] = abs(df[col])
                df[f'{col}_positive'] = (df[col] > 0).astype(int)
                df[f'{col}_negative'] = (df[col] < 0).astype(int)
                df[f'{col}_significant'] = (abs(df[col]) > 2).astype(int)  # >2% change
                
                self.feature_columns.extend([
                    f'{col}_abs', f'{col}_positive', f'{col}_negative', f'{col}_significant'
                ])
        
        # Price momentum features
        if 'pct_change_1h' in df.columns and 'pct_change_4h' in df.columns:
            df['price_momentum_1h_4h'] = df['pct_change_1h'] - df['pct_change_4h']
            self.feature_columns.append('price_momentum_1h_4h')
        
        if 'pct_change_4h' in df.columns and 'pct_change_eod' in df.columns:
            df['price_momentum_4h_eod'] = df['pct_change_4h'] - df['pct_change_eod']
            self.feature_columns.append('price_momentum_4h_eod')
        
        # Volatility features
        for col in price_cols:
            if col in df.columns:
                for window in [3, 5, 10]:
                    vol_col = f'{col}_volatility_{window}d'
                    df[vol_col] = df.groupby(self.group_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
                    self.feature_columns.append(vol_col)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between sentiment and other variables."""
        df = df.copy()
        
        # Sentiment-price interactions
        sentiment_cols = ['sentiment_combined', 'sentiment_dynamic', 'sentiment_ml']
        price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        
        for sent_col in sentiment_cols:
            if sent_col in df.columns:
                for price_col in price_cols:
                    if price_col in df.columns:
                        interaction_col = f'{sent_col}_{price_col}_interaction'
                        df[interaction_col] = df[sent_col] * df[price_col].fillna(0)
                        self.feature_columns.append(interaction_col)
        
        # Sentiment-time interactions
        if 'sentiment_combined' in df.columns:
            df['sentiment_market_hours'] = df['sentiment_combined'] * df['is_market_hours']
            df['sentiment_after_hours'] = df['sentiment_combined'] * df['is_aftermarket']
            df['sentiment_opening_hour'] = df['sentiment_combined'] * df['is_opening_hour']
            df['sentiment_closing_hour'] = df['sentiment_combined'] * df['is_closing_hour']
            
            self.feature_columns.extend([
                'sentiment_market_hours', 'sentiment_after_hours', 
                'sentiment_opening_hour', 'sentiment_closing_hour'
            ])
        
        # Confidence-based interactions
        if 'prediction_confidence' in df.columns and 'sentiment_combined' in df.columns:
            df['weighted_sentiment'] = df['sentiment_combined'] * df['prediction_confidence']
            self.feature_columns.append('weighted_sentiment')
        
        return df
    
    def create_accuracy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on prediction accuracy from main script."""
        df = df.copy()
        
        # Accuracy features for different methods and intervals
        methods = ['dynamic', 'ml', 'keyword', 'combined']
        intervals = ['1h', '4h', 'eod', 'eow']
        
        for method in methods:
            accuracy_cols = [f'accuracy_{method}_{interval}' for interval in intervals]
            valid_cols = [col for col in accuracy_cols if col in df.columns]
            
            if valid_cols:
                # Average accuracy across intervals
                df[f'avg_accuracy_{method}'] = df[valid_cols].mean(axis=1)
                
                # Consistency in accuracy
                df[f'accuracy_consistency_{method}'] = df[valid_cols].std(axis=1)
                
                self.feature_columns.extend([f'avg_accuracy_{method}', f'accuracy_consistency_{method}'])
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame, 
                              price_change_cols: List[str] = None,
                              thresholds: List[float] = [0, 0.01, 0.02, 0.05]) -> pd.DataFrame:
        """Create multiple target variables with different thresholds."""
        df = df.copy()
        if price_change_cols is None:
            price_change_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        
        for col in price_change_cols:
            if col in df.columns:
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
        
        return df
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess features."""
        df = df.copy()
        
        # Drop rows with missing critical features
        critical_cols = ['sentiment_combined']
        existing_critical = [col for col in critical_cols if col in df.columns]
        if existing_critical:
            df = df.dropna(subset=existing_critical)
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Fill categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown')
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def feature_engineering_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        print(f"Starting feature engineering with {len(df)} articles")
        
        # Step 1: Time features
        df = self.create_time_features(df)
        print(f"Created time features: {len(df)} rows")
        
        # Step 2: Sentiment features
        df = self.create_sentiment_features(df)
        print(f"Created sentiment features: {len(df)} rows")
        
        # Step 3: Price features
        df = self.create_price_features(df)
        print(f"Created price features: {len(df)} rows")
        
        # Step 4: Lag features
        df = self.create_lag_features(df)
        print(f"Created lag features: {len(df)} rows")
        
        # Step 5: Rolling features
        df = self.create_rolling_features(df)
        print(f"Created rolling features: {len(df)} rows")
        
        # Step 6: Interaction features
        df = self.create_interaction_features(df)
        print(f"Created interaction features: {len(df)} rows")
        
        # Step 7: Accuracy features
        df = self.create_accuracy_features(df)
        print(f"Created accuracy features: {len(df)} rows")
        
        # Step 8: Target variables
        df = self.create_target_variables(df)
        print(f"Created target variables: {len(df)} rows")
        
        # Step 9: Preprocessing
        df = self.preprocess_features(df)
        print(f"Final preprocessing: {len(df)} rows")
        
        print(f"Feature engineering complete. Created {len(self.feature_columns)} features.")
        return df

# Main execution function
def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Main pipeline function for integration with existing script."""
    engineer = FinancialNewsFeatureEngineer()
    return engineer.feature_engineering_pipeline(df)

if __name__ == "__main__":
    # Example usage
    try:
        df = pd.read_csv("scraped_articles.csv")
        print(f"Original data shape: {df.shape}")
        
        engineer = FinancialNewsFeatureEngineer()
        df_processed = engineer.feature_engineering_pipeline(df)
        
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Created features: {len(engineer.feature_columns)}")
        
        # Save engineered features
        df_processed.to_csv("engineered_features.csv", index=False)
        print("Feature engineering complete. Saved to engineered_features.csv")
        
        # Display feature summary
        print("\nFeature Categories:")
        print(f"- Total features created: {len(engineer.feature_columns)}")
        print(f"- Original columns: {df.shape[1]}")
        print(f"- Final columns: {df_processed.shape[1]}")
        
    except Exception as e:
        print(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()