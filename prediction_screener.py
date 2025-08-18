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

        # Convert datetime columns and ensure they are timezone-aware (UTC)
        print("Converting datetime columns...")
        self.predictions_df['news_datetime'] = pd.to_datetime(self.predictions_df['news_datetime'])
        
        if self.predictions_df['news_datetime'].dt.tz is None:
            self.predictions_df['news_datetime'] = self.predictions_df['news_datetime'].dt.tz_localize('UTC')
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

    def filter_viable_predictions(self):
        """
        Filters out predictions that cannot be calculated due to missing future market data.
        For EOD predictions, we need the next trading day's close.
        This now happens BEFORE averaging to avoid losing all data.
        """
        print("\nFiltering for viable predictions (before averaging)...")
        
        # Get current date and time in UTC and ET
        current_utc = datetime.now(pytz.UTC)
        current_et = current_utc.astimezone(pytz.timezone('US/Eastern'))
        current_date_utc = current_utc.date()
        
        original_count = len(self.predictions_df)
        
        print(f"Current time: {current_et.strftime('%Y-%m-%d %H:%M:%S ET')} ({current_utc.strftime('%H:%M:%S UTC')})")
        print(f"Original predictions count: {original_count}")
        
        # EOD predictions filtering with market hours consideration
        if self.horizon == 'eod':
            self.predictions_df['news_date_utc'] = self.predictions_df['news_datetime'].dt.date
            self.predictions_df['news_et'] = self.predictions_df['news_datetime'].dt.tz_convert('US/Eastern')
            
            # Market closes at 4:00 PM ET (16:00)
            market_close_hour = 16
            
            # Initialize counters for detailed reporting
            future_count = 0
            after_hours_count = 0
            weekend_count = 0
            market_still_open_count = 0
            viable_count = 0
            
            viable_indices = []
            
            for idx, row in self.predictions_df.iterrows():
                news_date = row['news_date_utc']
                news_et_time = row['news_et']
                is_viable = False
                
                # Future dates are never viable
                if news_date > current_date_utc:
                    future_count += 1
                    continue
                
                # Past dates (before today) are always viable
                if news_date < current_date_utc:
                    is_viable = True
                    viable_count += 1
                
                # Today's date - check market hours
                elif news_date == current_date_utc:
                    # Check if it's a weekday (Monday=0, Sunday=6)
                    weekday = current_et.weekday()
                    
                    # Weekend articles need next trading day
                    if weekday >= 5:  # Saturday or Sunday
                        weekend_count += 1
                        continue
                    
                    # Weekday - check if article was after market close (4:00 PM ET)
                    if news_et_time.hour >= market_close_hour:
                        # After hours article needs next trading day
                        after_hours_count += 1
                        continue
                    
                    # Article was during market hours today
                    # Check if we're currently past market close
                    if current_et.hour >= market_close_hour:
                        # Market has closed, so we can calculate today's EOD change
                        is_viable = True
                        viable_count += 1
                    else:
                        # Market is still open, need to wait for close
                        market_still_open_count += 1
                        continue
                
                if is_viable:
                    viable_indices.append(idx)
            
            # Filter the dataframe using the viable indices
            viable_df = self.predictions_df.loc[viable_indices].copy()
            
            # Report filtering results
            total_filtered = original_count - len(viable_df)
            print(f"\nFiltered out {total_filtered} total predictions:")
            if future_count > 0:
                print(f"  - {future_count} from future dates")
            if after_hours_count > 0:
                print(f"  - {after_hours_count} from after market close (>4:00 PM ET)")
            if weekend_count > 0:
                print(f"  - {weekend_count} from weekends")
            if market_still_open_count > 0:
                print(f"  - {market_still_open_count} from today before market close (need to wait for 4:00 PM ET)")
            
            print(f"Keeping {len(viable_df)} viable predictions")
            
            self.predictions_df = viable_df
            
            if self.predictions_df.empty:
                print("No viable predictions available for analysis.")
                print("All predictions need future market data or market close.")
                return False
        
        # For intraday horizons, keep the existing time-based logic
        elif self.horizon in ['1hr', '4hr']:
            time_delta = timedelta(hours=1) if self.horizon == '1hr' else timedelta(hours=4)
            current_time_utc = datetime.now(pytz.UTC)
            
            print(f"Current time (UTC): {current_time_utc}")
            print(f"Time delta for {self.horizon}: {time_delta}")
            
            # Filter out predictions where the news time + horizon is in the future
            viable_mask = self.predictions_df['news_datetime'] + time_delta < current_time_utc
            viable_df = self.predictions_df[viable_mask].copy()
            
            filtered_count = original_count - len(viable_df)
            print(f"Filtered out {filtered_count} predictions where the {self.horizon} horizon has not yet been reached.")
            
            self.predictions_df = viable_df
            
            if self.predictions_df.empty:
                print("No viable predictions available for analysis.")
                print(f"All predictions are too recent - need predictions older than {time_delta} to calculate {self.horizon} changes.")
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
        """
        tickers = self.predictions_df['ticker'].unique().tolist()
        # Fetch from 1 day before the earliest news to handle EOD calculations correctly
        min_date = (self.predictions_df['news_datetime'].min() - timedelta(days=3)).date()
        # Add a larger buffer to try to capture next trading day
        max_date = (self.predictions_df['news_datetime'].max() + timedelta(days=5)).date()
        
        print(f"Prediction date range: {self.predictions_df['news_datetime'].min()} to {self.predictions_df['news_datetime'].max()}")
        print(f"Fetching market data from {min_date} to {max_date}")

        print(f"\nFetching market data for {len(tickers)} tickers from {min_date} to {max_date}...")

        # Split tickers into smaller batches to avoid yfinance issues
        batch_size = 50  # Reduced batch size for better reliability
        ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        
        for i, batch in enumerate(ticker_batches):
            print(f"Fetching batch {i+1}/{len(ticker_batches)} ({len(batch)} tickers)...")
            
            try:
                interval = '1d' if self.horizon == 'eod' else '1m'
                data = yf.download(batch, start=min_date, end=max_date, interval=interval, progress=False, group_by='ticker')

                if data.empty:
                    print(f"Warning: No data returned for batch {i+1}")
                    continue

                # Process each ticker in the batch
                for ticker in batch:
                    try:
                        ticker_data = data[ticker] if len(batch) > 1 and ticker in data else data
                        
                        if not ticker_data.dropna().empty:
                            # Convert market data index to UTC for consistent comparison
                            if ticker_data.index.tz is None:
                                ticker_data.index = ticker_data.index.tz_localize('US/Eastern')
                            ticker_data.index = ticker_data.index.tz_convert('UTC')
                            self.market_data[ticker] = ticker_data
                    
                    except Exception as e:
                        # This handles cases where a specific ticker in a batch fails
                        print(f"Warning: Failed to process {ticker}: {e}")
            
            except Exception as e:
                print(f"Error fetching batch {i+1}: {e}")
        
        print(f"Successfully fetched data for {len(self.market_data)} tickers.")

    def calculate_actual_changes(self):
        """
        Calculates the actual price changes based on the specified horizon.
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
                    # --- EOD Calculation Logic ---
                    try:
                        # Get the news date (convert to date for comparison)
                        news_date = news_time.date()
                        
                        # Find all closes available in our market data
                        available_dates = market_df.index.date
                        unique_dates = sorted(set(available_dates))
                        
                        # For EOD predictions, we want to match the actual market change:
                        # Baseline: Previous trading day's close (before news date)
                        # Target: News date's close (to calculate the actual EOD change that occurred)
                        
                        baseline_close_price = None
                        baseline_date = None
                        current_close_price = None
                        
                        # Find the last trading day before news date
                        baseline_candidates = [d for d in unique_dates if d < news_date]
                        if baseline_candidates:
                            baseline_date = max(baseline_candidates)
                            baseline_data = market_df[market_df.index.date == baseline_date]
                            if not baseline_data.empty:
                                baseline_close_price = baseline_data['Close'].iloc[-1]
                                debug_msg = f"Baseline: {baseline_date}"
                        
                        # Find the current day's (news date) close
                        if news_date in unique_dates:
                            current_day_data = market_df[market_df.index.date == news_date]
                            if not current_day_data.empty:
                                current_close_price = current_day_data['Close'].iloc[-1]
                                debug_msg += f" -> Target: {news_date}"
                        
                        # Calculate the change
                        if (pd.notna(baseline_close_price) and pd.notna(current_close_price) and 
                            baseline_close_price != 0):
                            change = (current_close_price - baseline_close_price) / baseline_close_price * 100
                            successful_calcs += 1
                            debug_msg += f" -> SUCCESS: {change:.2f}%"
                        else:
                            debug_msg += " -> FAILED: missing prices"
                            
                    except Exception as e:
                        debug_msg = f"Exception: {e}"

                else:
                    # --- Intraday (1hr/4hr) Calculation ---
                    try:
                        time_delta = timedelta(hours=1) if self.horizon == '1hr' else timedelta(hours=4)
                        target_time = news_time + time_delta
                        
                        start_price = market_df.asof(news_time)['Close']
                        end_price = market_df.asof(target_time)['Close']

                        if pd.notna(start_price) and pd.notna(end_price) and start_price != 0:
                            change = (end_price - start_price) / start_price * 100
                            successful_calcs += 1
                            debug_msg = f"SUCCESS: {change:.2f}%"
                        else:
                            debug_msg = "FAILED: missing prices"
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
        
        # Show how many predictions were dropped for missing future market data
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
        
        # Filter out 0% actual changes for intraday horizons (after-hours/no movement)
        if self.horizon in ['1hr', '4hr']:
            initial_count = len(self.results_df)
            self.results_df = self.results_df[self.results_df[self.actual_col] != 0]
            filtered_count = len(self.results_df)
            if initial_count != filtered_count:
                print(f"Filtered out {initial_count - filtered_count} predictions with 0% actual change (after-hours/no movement)")
        
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
            print("No active market predictions found to report on.")
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
        print(f"Mean Absolute Error (MAE): {mae:.2f} percentage points")
        if not np.isnan(mape):
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.1f}%")
        print(f"Predictions within ±1%: {within_1pct:.1f}%")
        print(f"Predictions within ±2%: {within_2pct:.1f}%")
        print(f"Predictions within ±5%: {within_5pct:.1f}%")
        
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
            report_cols.insert(-2, 'article_count')  # Insert before direction_correct and abs_error
            
        display_df = self.results_df[report_cols].copy()
        display_df['news_datetime'] = display_df['news_datetime'].dt.strftime('%Y-%m-%d %H:%M')
        display_df[self.prediction_col] = display_df[self.prediction_col].map('{:,.2f}%'.format)
        display_df[self.actual_col] = display_df[self.actual_col].map('{:,.2f}%'.format)
        display_df['abs_error'] = display_df['abs_error'].map('{:.2f}pp'.format)  # pp = percentage points
        display_df['direction_correct'] = display_df['direction_correct'].map({True: '✅ Yes', False: '❌ No'})
        
        print("Sample of Results:")
        print(display_df.head(20).to_string(index=False))

        # --- Detailed Statistics ---
        print(f"\n{'-'*50}")
        print("DETAILED STATISTICS")
        print(f"{'-'*50}")
        
        pred_stats = self.results_df[self.prediction_col].describe()
        actual_stats = self.results_df[self.actual_col].describe()
        
        print(f"\nPrediction Statistics:")
        print(f"  Mean:  {pred_stats['mean']:.2f}%")
        print(f"  Std:   {pred_stats['std']:.2f}%")
        print(f"  Range: {pred_stats['min']:.2f}% to {pred_stats['max']:.2f}%")
        
        print(f"\nActual Statistics:")
        print(f"  Mean:  {actual_stats['mean']:.2f}%")
        print(f"  Std:   {actual_stats['std']:.2f}%")
        print(f"  Range: {actual_stats['min']:.2f}% to {actual_stats['max']:.2f}%")

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
            # CRITICAL FIX: Filter out predictions that can't be calculated BEFORE averaging
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