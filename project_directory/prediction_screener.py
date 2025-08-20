import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, timedelta
import sys
import warnings

# Suppress common pandas and yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="The 'unit' keyword in TimedeltaIndex construction is deprecated")

# === Horizon Mapping ===
TARGET_MAP = {
    "1hr": "pct_change_1h",
    "4hr": "pct_change_4h", 
    "eod": "pct_change_eod"
}


class PredictionScreener:
    """
    A modular tool to compare predicted stock price changes against actual market data
    for different time horizons (1hr, 4hr, EOD).
    """

    def __init__(self, csv_file_path, horizon='eod'):
        """
        Initializes the screener.

        Args:
            csv_file_path (str): Path to the CSV file containing prediction results.
            horizon (str): The prediction horizon to check ('1hr', '4hr', or 'eod').
        """
        if horizon not in TARGET_MAP:
            raise ValueError(f"Horizon must be one of {list(TARGET_MAP.keys())}")

        self.csv_file = csv_file_path
        self.horizon = horizon
        self.predictions_df = None
        self.market_data = {}
        self.results_df = None
        
        # Use TARGET_MAP to get the correct column suffix
        target_suffix = TARGET_MAP[horizon]
        
        # Dynamically set the column names based on TARGET_MAP
        self.prediction_col = f"predicted_{target_suffix}"
        self.actual_col = f"actual_{target_suffix}"

    def load_predictions(self):
        """
        Loads and prepares the prediction data from the CSV file.
        """
        print(f"Loading predictions for the '{self.horizon}' horizon from '{self.csv_file}'...")
        try:
            self.predictions_df = pd.read_csv(self.csv_file)
        except FileNotFoundError:
            print(f"ERROR: The file '{self.csv_file}' was not found.")
            return False

        # Convert datetime columns and ensure they are timezone-aware
        print("Converting datetime columns...")
        self.predictions_df['news_datetime'] = pd.to_datetime(self.predictions_df['news_datetime'])
        
        # Since your timestamps are already in Eastern Time, localize them as ET first
        if self.predictions_df['news_datetime'].dt.tz is None:
            # Timestamps are in Eastern Time (EST/EDT) - localize them properly
            self.predictions_df['news_datetime'] = self.predictions_df['news_datetime'].dt.tz_localize('US/Eastern')
            # Then convert to UTC for internal processing
            self.predictions_df['news_datetime'] = self.predictions_df['news_datetime'].dt.tz_convert('UTC')
        else:
            self.predictions_df['news_datetime'] = self.predictions_df['news_datetime'].dt.tz_convert('UTC')
        
        print(f"Date range: {self.predictions_df['news_datetime'].min()} to {self.predictions_df['news_datetime'].max()}")
        
        # Ensure the prediction column exists and is numeric
        if self.prediction_col not in self.predictions_df.columns:
            print(f"ERROR: Prediction column '{self.prediction_col}' not found in the CSV.")
            return False
        self.predictions_df[self.prediction_col] = pd.to_numeric(self.predictions_df[self.prediction_col], errors='coerce')
        self.predictions_df.dropna(subset=[self.prediction_col], inplace=True)
        
        print(f"Loaded {len(self.predictions_df)} valid predictions.")
        return True

    def is_market_hours(self, dt_et):
        """
        Check if a datetime (in ET) falls during regular market hours.
        Market hours: Monday-Friday, 9:30 AM - 4:00 PM ET
        """
        if dt_et.weekday() >= 5:  # Weekend
            return False
        
        # Convert to decimal hours for easier comparison
        hour_decimal = dt_et.hour + dt_et.minute/60
        
        # Market is open 9:30 AM to 4:00 PM ET (9.5 to 16.0 in decimal)
        return 9.5 <= hour_decimal < 16.0

    def get_target_close_date(self, news_et):
        """
        Helper function to determine which trading day's close a news item should predict.
        This logic is shared between filtering and calculation.
        """
        market_open_hour = 9   # 9:30 AM ET (simplified to 9 AM for safety)
        market_close_hour = 16 # 4:00 PM ET
        
        # If news is during market hours (9 AM - 4 PM ET on a weekday), predict same day's close
        if (news_et.weekday() < 5 and 
            market_open_hour <= news_et.hour < market_close_hour):
            return news_et.date()
        
        # If news is outside market hours or on weekend, predict next trading day's close
        # This includes: before 9 AM, after 4 PM, or weekends
        
        # Start looking from the next day if it's after market close
        if (news_et.weekday() < 5 and news_et.hour >= market_close_hour):
            next_day = news_et.date() + timedelta(days=1)
        # For pre-market or weekend news, target is same day if weekday, otherwise next weekday
        else:
            next_day = news_et.date() if news_et.weekday() < 5 else news_et.date() + timedelta(days=1)
        
        # Find the next weekday (simple proxy for next trading day)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day

    def filter_viable_predictions(self):
        """
        Filters out predictions that cannot be calculated due to missing future market data.
        For intraday predictions, also filters out predictions made outside market hours.
        """
        print("\nFiltering for viable predictions (before averaging)...")
        
        # Get current date and time in UTC and ET
        current_utc = datetime.now(pytz.UTC)
        current_et = current_utc.astimezone(pytz.timezone('US/Eastern'))
        current_date_utc = current_utc.date()
        
        original_count = len(self.predictions_df)
        
        print(f"Current time: {current_et.strftime('%Y-%m-%d %H:%M:%S ET')} ({current_utc.strftime('%H:%M:%S UTC')})")
        print(f"Original predictions count: {original_count}")
        
        # Convert news times to ET for market hours checking
        self.predictions_df['news_et'] = self.predictions_df['news_datetime'].dt.tz_convert('US/Eastern')
        
        if self.horizon == 'eod':
            # EOD predictions filtering with market hours consideration
            market_close_hour = 16
            
            # Initialize counters for detailed reporting
            future_count = 0
            after_hours_count = 0
            weekend_count = 0
            market_still_open_count = 0
            
            viable_indices = []
            
            for idx, row in self.predictions_df.iterrows():
                news_et = row['news_et']
                
                # Use the shared logic to determine target close date
                target_close_date = self.get_target_close_date(news_et)

                # The prediction is viable ONLY if the current date is past the target close date,
                # OR if it's the same day and the market has already closed.
                is_viable = False
                if current_et.date() > target_close_date:
                    is_viable = True
                elif current_et.date() == target_close_date and current_et.hour >= market_close_hour:
                    is_viable = True
                
                if is_viable:
                    viable_indices.append(idx)
                else:
                    # If not viable, categorize why for reporting
                    if news_et.date() > current_et.date():
                        future_count += 1
                    elif news_et.weekday() >= 5:  # Weekend
                        weekend_count += 1
                    elif (news_et.weekday() < 5 and 
                          (news_et.hour >= 16 or news_et.hour < 9)):  # After hours or pre-market
                        after_hours_count += 1
                    else:  # Must be during market hours but market hasn't closed yet
                        market_still_open_count += 1
            
            # Filter the dataframe using the viable indices
            viable_df = self.predictions_df.loc[viable_indices].copy()
            
            # Report filtering results
            total_filtered = original_count - len(viable_df)
            print(f"\nFiltered out {total_filtered} total predictions:")
            if future_count > 0:
                print(f"  - {future_count} from future dates")
            if after_hours_count > 0:
                print(f"  - {after_hours_count} from outside market hours (before 9:00 AM or after 4:00 PM ET)")
            if weekend_count > 0:
                print(f"  - {weekend_count} from weekends")
            if market_still_open_count > 0:
                print(f"  - {market_still_open_count} from today before market close (need to wait for 4:00 PM ET)")
            
            print(f"Keeping {len(viable_df)} viable predictions")
            
            self.predictions_df = viable_df
        
        # For intraday horizons, filter by time elapsed AND market hours
        elif self.horizon in ['1hr', '4hr']:
            time_delta = timedelta(hours=1) if self.horizon == '1hr' else timedelta(hours=4)
            current_time_utc = datetime.now(pytz.UTC)
            
            print(f"Current time (UTC): {current_time_utc}")
            print(f"Time delta for {self.horizon}: {time_delta}")
            
            viable_indices = []
            future_count = 0
            non_market_hours_count = 0
            
            for idx, row in self.predictions_df.iterrows():
                news_time = row['news_datetime']
                news_et = row['news_et']
                target_time = news_time + time_delta
                
                # Check if enough time has passed
                time_elapsed = news_time + time_delta < current_time_utc
                
                # Check if news was during market hours
                during_market_hours = self.is_market_hours(news_et)
                
                # Debug first few predictions
                if idx < 5:
                    print(f"DEBUG {row['ticker']}: {news_time} UTC -> {news_et} ET, market_hours={during_market_hours}, time_elapsed={time_elapsed}")
                
                if time_elapsed and during_market_hours:
                    viable_indices.append(idx)
                elif not time_elapsed:
                    future_count += 1
                elif not during_market_hours:
                    non_market_hours_count += 1
            
            viable_df = self.predictions_df.loc[viable_indices].copy()
            
            filtered_count = original_count - len(viable_df)
            print(f"Filtered out {filtered_count} predictions:")
            if future_count > 0:
                print(f"  - {future_count} where the {self.horizon} horizon has not yet been reached")
            if non_market_hours_count > 0:
                print(f"  - {non_market_hours_count} from outside market hours (9:30 AM - 4:00 PM ET, weekdays)")
            
            self.predictions_df = viable_df
        
        if self.predictions_df.empty:
            print("No viable predictions available for analysis.")
            if self.horizon == 'eod':
                print("All predictions need future market data or market close.")
            else:
                print(f"All predictions are either too recent, outside market hours, or both.")
            return False

        print(f"Proceeding with {len(self.predictions_df)} viable predictions")
        return True

    def average_daily_predictions(self):
        """
        Averages predictions for the same ticker on the same date (EOD only).
        For intraday horizons (1hr, 4hr), keeps individual predictions.
        """
        if self.horizon == 'eod':
            print("\nAveraging multiple EOD predictions per ticker per date...")
            
            # Add a date column for grouping
            self.predictions_df['news_date'] = self.predictions_df['news_datetime'].dt.date
            
            # Show statistics before averaging
            original_count = len(self.predictions_df)
            daily_counts = self.predictions_df.groupby(['ticker', 'news_date']).size()
            multiple_articles = daily_counts[daily_counts > 1]
            
            print(f"Found {len(multiple_articles)} ticker-dates with multiple articles")
            if len(multiple_articles) > 0:
                print(f"Max articles per ticker-date: {daily_counts.max()}")
            
            # Group by ticker and date, average predictions, keep representative datetime
            grouped_data = []
            for (ticker, date), group in self.predictions_df.groupby(['ticker', 'news_date']):
                avg_prediction = group[self.prediction_col].mean()
                representative_datetime = group['news_datetime'].iloc[0]  # Use first article's datetime
                article_count = len(group)
                
                grouped_data.append({
                    'ticker': ticker,
                    'news_date': date,
                    'news_datetime': representative_datetime,
                    self.prediction_col: avg_prediction,
                    'article_count': article_count
                })
            
            # Create new dataframe with averaged predictions
            self.predictions_df = pd.DataFrame(grouped_data)
            
            print(f"Averaged {original_count} individual predictions into {len(self.predictions_df)} daily predictions")
        
        else:
            print(f"\nKeeping individual predictions for {self.horizon} horizon (time-specific predictions)")
            # Add article_count column for consistency in reporting
            self.predictions_df['article_count'] = 1
            original_count = len(self.predictions_df)
            print(f"Processing {original_count} individual {self.horizon} predictions")
        
        return True

    def fetch_market_data(self):
        """
        Fetches the necessary market data from yfinance in batches for reliability.
        For intraday horizons, uses appropriate interval (5m for better reliability than 1m).
        """
        tickers = self.predictions_df['ticker'].unique().tolist()
        # Fetch from 1 day before the earliest news to handle calculations correctly
        min_date = (self.predictions_df['news_datetime'].min() - timedelta(days=3)).date()
        # Add a buffer to capture data
        max_date = (self.predictions_df['news_datetime'].max() + timedelta(days=5)).date()
        
        print(f"Prediction date range: {self.predictions_df['news_datetime'].min()} to {self.predictions_df['news_datetime'].max()}")
        print(f"Fetching market data from {min_date} to {max_date}")

        # Choose interval based on horizon - use 5m for intraday for better reliability
        if self.horizon == 'eod':
            interval = '1d'
        elif self.horizon == '1hr':
            interval = '5m'  # 5-minute data is more reliable than 1m
        else:  # 4hr
            interval = '15m'  # 15-minute data for 4hr horizon
        
        print(f"Using {interval} interval for {self.horizon} horizon")
        print(f"\nFetching market data for {len(tickers)} tickers from {min_date} to {max_date}...")

        # Split tickers into smaller batches to avoid yfinance issues
        batch_size = 50
        ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        
        for i, batch in enumerate(ticker_batches):
            print(f"Fetching batch {i+1}/{len(ticker_batches)} ({len(batch)} tickers)...")
            
            try:
                data = yf.download(batch, start=min_date, end=max_date, interval=interval, progress=False, group_by='ticker')

                if data.empty:
                    print(f"Warning: No data returned for batch {i+1}")
                    continue

                # Process each ticker in the batch
                for ticker in batch:
                    try:
                        if len(batch) > 1:
                            if ticker in data.columns.levels[0]:  # Check if ticker exists
                                ticker_data = data[ticker]
                            else:
                                print(f"Warning: No data for {ticker}")
                                continue
                        else:
                            ticker_data = data
                        
                        if not ticker_data.dropna().empty:
                            # Convert market data index to UTC for consistent comparison
                            if ticker_data.index.tz is None:
                                ticker_data.index = ticker_data.index.tz_localize('US/Eastern')
                            ticker_data.index = ticker_data.index.tz_convert('UTC')
                            self.market_data[ticker] = ticker_data
                    
                    except Exception as e:
                        print(f"Warning: Failed to process {ticker}: {e}")
            
            except Exception as e:
                print(f"Error fetching batch {i+1}: {e}")
        
        print(f"Successfully fetched data for {len(self.market_data)} tickers.")

    def calculate_actual_changes(self):
        """
        Calculates the actual price changes based on the specified horizon.
        Uses appropriate logic for each time horizon.
        """
        print("\nCalculating actual price changes...")
        actual_changes = []
        successful_calcs = 0
        debug_info = []

        for idx, row in self.predictions_df.iterrows():
            ticker = row['ticker']
            news_time = row['news_datetime']
            change = np.nan
            debug_msg = ""
            
            if ticker in self.market_data:
                market_df = self.market_data[ticker]
                
                if self.horizon == 'eod':
                    # EOD Calculation Logic
                    try:
                        # Convert news time to ET for market hours logic
                        news_et = news_time.astimezone(pytz.timezone('US/Eastern'))
                        
                        # Use the same logic as filtering to determine target close date
                        target_close_date = self.get_target_close_date(news_et)
                        
                        # Find all closes available in our market data
                        available_dates = market_df.index.date
                        unique_dates = sorted(set(available_dates))
                        
                        # For EOD predictions, calculate change from previous trading day to target day
                        baseline_close_price = None
                        baseline_date = None
                        target_close_price = None
                        
                        # Find the last trading day before target date for baseline
                        baseline_candidates = [d for d in unique_dates if d < target_close_date]
                        if baseline_candidates:
                            baseline_date = max(baseline_candidates)
                            baseline_data = market_df[market_df.index.date == baseline_date]
                            if not baseline_data.empty:
                                baseline_close_price = baseline_data['Close'].iloc[-1]
                                debug_msg = f"Baseline: {baseline_date}"
                        
                        # Find the target day's close
                        if target_close_date in unique_dates:
                            target_day_data = market_df[market_df.index.date == target_close_date]
                            if not target_day_data.empty:
                                target_close_price = target_day_data['Close'].iloc[-1]
                                debug_msg += f" -> Target: {target_close_date}"
                        
                        # Calculate the change
                        if (pd.notna(baseline_close_price) and pd.notna(target_close_price) and 
                            baseline_close_price != 0):
                            change = (target_close_price - baseline_close_price) / baseline_close_price * 100
                            successful_calcs += 1
                            debug_msg += f" -> SUCCESS: {change:.2f}%"
                        else:
                            debug_msg += " -> FAILED: missing prices"
                            
                    except Exception as e:
                        debug_msg = f"Exception: {e}"

                else:
                    # Intraday (1hr/4hr) Calculation
                    try:
                        time_delta = timedelta(hours=1) if self.horizon == '1hr' else timedelta(hours=4)
                        target_time = news_time + time_delta
                        
                        # Use asof to find closest available prices
                        start_data = market_df.asof(news_time)
                        end_data = market_df.asof(target_time)
                        
                        # Fix the Series ambiguity issue
                        start_price = None
                        end_price = None
                        
                        if start_data is not None and not pd.isna(start_data).all():
                            if hasattr(start_data, 'Close'):
                                start_price = start_data['Close']
                            elif isinstance(start_data, pd.Series) and 'Close' in start_data:
                                start_price = start_data['Close']
                        
                        if end_data is not None and not pd.isna(end_data).all():
                            if hasattr(end_data, 'Close'):
                                end_price = end_data['Close']
                            elif isinstance(end_data, pd.Series) and 'Close' in end_data:
                                end_price = end_data['Close']

                        if (start_price is not None and end_price is not None and 
                            pd.notna(start_price) and pd.notna(end_price) and start_price != 0):
                            change = (end_price - start_price) / start_price * 100
                            successful_calcs += 1
                            debug_msg = f"SUCCESS: {change:.4f}%"
                        else:
                            debug_msg = f"FAILED: start_price={start_price}, end_price={end_price}"
                    except Exception as e:
                        debug_msg = f"Exception: {e}"
            else:
                debug_msg = "No market data"
            
            actual_changes.append(change)
            debug_info.append(debug_msg)

        self.predictions_df[self.actual_col] = actual_changes
        
        # Add debug info temporarily for troubleshooting
        self.predictions_df['debug_info'] = debug_info
        
        self.results_df = self.predictions_df.dropna(subset=[self.actual_col])
        
        # Show how many predictions were dropped for missing market data
        dropped_count = len(self.predictions_df) - len(self.results_df)
        if dropped_count > 0:
            print(f"Dropped {dropped_count} predictions due to missing market data")
            
            # Show some examples of failed calculations for debugging
            failed_df = self.predictions_df[self.predictions_df[self.actual_col].isna()]
            if len(failed_df) > 0:
                print("\nSample of failed calculations:")
                sample_failed = failed_df[['ticker', 'news_datetime', 'debug_info']].head(5)
                for _, row in sample_failed.iterrows():
                    print(f"  {row['ticker']} {row['news_datetime']}: {row['debug_info']}")
        
        # Modified filtering: Only filter out 0% changes if they seem like data issues
        if self.horizon in ['1hr', '4hr']:
            initial_count = len(self.results_df)
            
            if initial_count > 0:  # Avoid division by zero
                # Count how many have exactly 0% change
                zero_changes = (self.results_df[self.actual_col] == 0).sum()
                print(f"Found {zero_changes} predictions with exactly 0% change")
                
                # Only filter if more than 50% are exactly 0% (suggests data issue)
                # Otherwise, keep them as legitimate small movements
                if zero_changes / initial_count > 0.5:
                    print(f"Filtering out 0% changes (likely data quality issue)")
                    self.results_df = self.results_df[self.results_df[self.actual_col] != 0]
                    filtered_count = len(self.results_df)
                    print(f"Filtered from {initial_count} to {filtered_count} predictions")
                else:
                    print(f"Keeping 0% changes as legitimate market data")
            else:
                print("No results to analyze for 0% changes")
        
        print(f"Successfully calculated actuals for {len(self.results_df)} predictions.")
        
        # Clean up debug info
        if 'debug_info' in self.results_df.columns:
            self.results_df = self.results_df.drop('debug_info', axis=1)
        if 'debug_info' in self.predictions_df.columns:
            self.predictions_df = self.predictions_df.drop('debug_info', axis=1)

    def generate_report(self):
        """
        Generates and prints the final comparison report with detailed statistics.
        """
        if self.results_df is None or self.results_df.empty:
            print("\nNo results to report. Please run the calculation steps first.")
            return

        if self.results_df.empty:
            print("No predictions found to report on.")
            return

        # Determine if the predicted direction was correct (handle zero predictions/actuals)
        self.results_df['direction_correct'] = (np.sign(self.results_df[self.prediction_col]) == np.sign(self.results_df[self.actual_col])) | (self.results_df[self.actual_col] == 0)

        # Calculate magnitude accuracy metrics
        self.results_df['abs_error'] = abs(self.results_df[self.prediction_col] - self.results_df[self.actual_col])
        self.results_df['abs_predicted'] = abs(self.results_df[self.prediction_col])
        self.results_df['abs_actual'] = abs(self.results_df[self.actual_col])
        
        # Mean Absolute Error (MAE)
        mae = self.results_df['abs_error'].mean()
        
        # Mean Absolute Percentage Error (MAPE) - only for non-zero actuals
        non_zero_actuals = self.results_df[self.results_df[self.actual_col] != 0]
        if len(non_zero_actuals) > 0:
            mape = (non_zero_actuals['abs_error'] / non_zero_actuals['abs_actual'] * 100).mean()
        else:
            mape = np.nan
        
        # Calculate how often predictions are within certain thresholds of actual
        within_1pct = (self.results_df['abs_error'] <= 1.0).mean() * 100
        within_2pct = (self.results_df['abs_error'] <= 2.0).mean() * 100
        within_5pct = (self.results_df['abs_error'] <= 5.0).mean() * 100

        accuracy = self.results_df['direction_correct'].mean() * 100

        print("\n" + "="*50)
        print(f"   PREDICTION SCREENER REPORT ({self.horizon.upper()})")
        print("="*50)
        print(f"Total {'Daily' if self.horizon == 'eod' else 'Individual'} Predictions Checked: {len(self.results_df)}")
        print(f"Overall Directional Accuracy: {accuracy:.2f}%")
        
        # Magnitude accuracy metrics
        print(f"\nMAGNITUDE ACCURACY:")
        print(f"Mean Absolute Error (MAE): {mae:.4f} percentage points")
        if not np.isnan(mape):
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.1f}%")
        print(f"Predictions within ±1%: {within_1pct:.1f}%")
        print(f"Predictions within ±2%: {within_2pct:.1f}%")
        print(f"Predictions within ±5%: {within_5pct:.1f}%")
        
        # Show statistics about actual changes
        actual_stats = self.results_df[self.actual_col].describe()
        print(f"\nACTUAL CHANGE DISTRIBUTION:")
        print(f"Mean: {actual_stats['mean']:.4f}%")
        print(f"Std: {actual_stats['std']:.4f}%")
        print(f"Range: {actual_stats['min']:.4f}% to {actual_stats['max']:.4f}%")
        
        # Count exactly 0% changes
        zero_count = (self.results_df[self.actual_col] == 0).sum()
        if zero_count > 0:
            print(f"Predictions with exactly 0% actual change: {zero_count} ({zero_count/len(self.results_df)*100:.1f}%)")
        
        # Show article count statistics (only meaningful for EOD)
        if 'article_count' in self.results_df.columns and self.horizon == 'eod':
            total_articles = self.results_df['article_count'].sum()
            avg_articles = self.results_df['article_count'].mean()
            print(f"Total Articles Averaged: {total_articles}")
            print(f"Average Articles per Prediction: {avg_articles:.1f}")
        elif self.horizon in ['1hr', '4hr']:
            print(f"Time-specific {self.horizon} predictions (no averaging)")
        
        print("="*50, "\n")

        # Display a sample of the results with magnitude info
        report_cols = ['ticker', 'news_datetime', self.prediction_col, self.actual_col, 'direction_correct', 'abs_error']
        if 'article_count' in self.results_df.columns:
            report_cols.insert(-2, 'article_count')
            
        display_df = self.results_df[report_cols].copy()
        display_df['news_datetime'] = display_df['news_datetime'].dt.strftime('%Y-%m-%d %H:%M')
        display_df[self.prediction_col] = display_df[self.prediction_col].map('{:,.4f}%'.format)
        display_df[self.actual_col] = display_df[self.actual_col].map('{:,.4f}%'.format)
        display_df['abs_error'] = display_df['abs_error'].map('{:.4f}pp'.format)
        display_df['direction_correct'] = display_df['direction_correct'].map({True: '✅ Yes', False: '❌ No'})
        
        print("Sample of Results:")
        print(display_df.head(20).to_string(index=False))

        # Detailed Statistics
        print(f"\n{'-'*50}")
        print("DETAILED STATISTICS")
        print(f"{'-'*50}")
        
        pred_stats = self.results_df[self.prediction_col].describe()
        
        print(f"\nPrediction Statistics:")
        print(f"  Mean:  {pred_stats['mean']:.4f}%")
        print(f"  Std:   {pred_stats['std']:.4f}%")
        print(f"  Range: {pred_stats['min']:.4f}% to {pred_stats['max']:.4f}%")

    def export_results(self):
        """
        Saves the essential results to a clean CSV file.
        """
        if self.results_df is not None and not self.results_df.empty:
            # Keep only the essential columns including magnitude metrics
            essential_cols = [
                'ticker', 
                'news_datetime', 
                self.prediction_col, 
                self.actual_col, 
                'direction_correct',
                'abs_error'
            ]
            
            # Add article_count if it exists (insert before direction_correct)
            if 'article_count' in self.results_df.columns:
                essential_cols.insert(-2, 'article_count')
            
            # Create clean export dataframe
            export_df = self.results_df[essential_cols].copy()
            
            # Optional: Format the datetime for better readability
            export_df['news_datetime'] = export_df['news_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Export to CSV
            if self.horizon == 'eod':
                filename = f"screener_results_{self.horizon}_averaged.csv"
                suffix_msg = "averaged predictions"
            else:
                filename = f"screener_results_{self.horizon}_individual.csv"
                suffix_msg = "individual predictions"
                
            export_df.to_csv(filename, index=False)
            print(f"\nClean results saved to '{filename}' with {len(export_df)} {suffix_msg}")
            print(f"Columns: {', '.join(essential_cols)}")
        else:
            print("\nNo results to export.")

    def run(self):
        """
        Executes the full screener pipeline with corrected order.
        """
        if self.load_predictions():
            # Filter out predictions that can't be calculated BEFORE averaging
            if not self.filter_viable_predictions():
                return
                
            # Now average the remaining viable predictions
            self.average_daily_predictions()
            self.fetch_market_data()
            self.calculate_actual_changes()
            self.generate_report()
            self.export_results()

# === Main Execution ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prediction_screener.py <path_to_predictions_csv> [horizon]")
        print("Example: python prediction_screener.py continuous_predictions_eod.csv eod")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    prediction_horizon = 'eod'
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in TARGET_MAP:
            prediction_horizon = sys.argv[2].lower()
        else:
            print(f"Warning: Invalid horizon '{sys.argv[2]}'. Must be one of {list(TARGET_MAP.keys())}. Defaulting to 'eod'.")

    screener = PredictionScreener(csv_file_path=csv_path, horizon=prediction_horizon)
    screener.run()