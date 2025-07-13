import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

def get_closest_price(ticker, dt, interval='1m', window=60):  # Increased window to 60 minutes
    """
    Fetch intraday historical data for ticker around datetime dt.
    Returns the closing price closest to dt within a +/- window of minutes.
    """
    try:
        # Ensure datetime is timezone-aware (convert to UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        
        # Download data for date of dt +/- 1 day to ensure coverage
        start_date = (dt - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (dt + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Downloading data for {ticker} from {start_date} to {end_date}")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            print(f"No data found for {ticker} around {dt}")
            return None
        
        # Convert index to datetime if not already and handle timezone
        data.index = pd.to_datetime(data.index)
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        elif data.index.tz != dt.tz:
            data.index = data.index.tz_convert(dt.tz)
        
        # Find closest timestamp
        time_diffs = abs(data.index - dt)
        closest_idx = time_diffs.argmin()
        closest_time = data.index[closest_idx]
        closest_price = data['Close'].iloc[closest_idx]
        
        # Check if closest is within window minutes
        closest_diff = time_diffs[closest_idx]
        if closest_diff > pd.Timedelta(minutes=window):
            print(f"Closest data point for {ticker} at {closest_time} is more than {window} minutes from target {dt}.")
            print(f"Time difference: {closest_diff}")
            print("This might be due to market hours or data availability.")
            return None
        
        return closest_time, closest_price
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def check_price_accuracy(csv_path, ticker, target_datetime_str, price_field='baseline_price'):
    """
    Check if price in CSV matches price from yfinance at target_datetime.
    """
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Parse datetime column
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            print("Error: 'datetime' column not found in CSV")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        # Filter for the specific ticker
        df_ticker = df[df['ticker'] == ticker]
        
        if df_ticker.empty:
            print(f"No data found for ticker {ticker} in CSV")
            available_tickers = df['ticker'].unique()
            print(f"Available tickers: {available_tickers}")
            return
        
        # Parse target datetime
        target_datetime = pd.to_datetime(target_datetime_str)
        
        # Find row(s) matching the datetime exactly (or closest)
        df_ticker = df_ticker.copy()  # Avoid SettingWithCopyWarning
        df_ticker['time_diff'] = (df_ticker['datetime'] - target_datetime).abs()
        df_match = df_ticker[df_ticker['time_diff'] <= pd.Timedelta(minutes=10)]
        
        if df_match.empty:
            print(f"No matching datetime within 10 minutes for {ticker} at {target_datetime}")
            closest_row = df_ticker.loc[df_ticker['time_diff'].idxmin()]
            print(f"Closest datetime available: {closest_row['datetime']} (diff: {closest_row['time_diff']})")
            return
        
        # Get the closest match
        row = df_match.loc[df_match['time_diff'].idxmin()]
        
        # Check if price field exists
        if price_field not in row.index:
            print(f"Error: '{price_field}' column not found in CSV")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        local_price = row[price_field]
        print(f"Local {price_field} from CSV for {ticker} at {row['datetime']}: {local_price}")
        
        # Get Yahoo Finance price
        yf_result = get_closest_price(ticker, target_datetime, window=60)  # Increased window
        if yf_result is None:
            print("No valid Yahoo Finance price found to compare.")
            return
        
        yf_time, yf_price = yf_result
        print(f"Yahoo Finance close price for {ticker} at {yf_time}: {yf_price}")
        
        # Calculate difference
        diff = abs(local_price - yf_price)
        percent_diff = (diff / yf_price) * 100
        print(f"Price difference: ${diff:.4f} ({percent_diff:.2f}%)")
        
        # Set tolerance threshold
        tolerance = 0.5
        if diff <= tolerance:
            print("✓ Prices match within tolerance.")
        else:
            print("✗ Prices differ significantly.")
            
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
    except Exception as e:
        print(f"Error in check_price_accuracy: {str(e)}")

def validate_csv_structure(csv_path):
    """
    Validate the structure of the CSV file and show sample data.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV shape: {df.shape}")
        print(f"Total columns: {len(df.columns)}")
        
        # Check for key columns
        key_columns = ['ticker', 'datetime', 'baseline_price', 'eod_price', 'price_1h', 'price_4h', 'price_eow']
        available_key_columns = [col for col in key_columns if col in df.columns]
        print(f"Key columns found: {available_key_columns}")
        
        if 'ticker' in df.columns:
            unique_tickers = df['ticker'].unique()
            print(f"\nUnique tickers ({len(unique_tickers)}): {unique_tickers}")
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"\nDatetime range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"Total records: {len(df)}")
            
            # Show sample data for each ticker
            if 'ticker' in df.columns:
                print("\nSample data by ticker:")
                for ticker in df['ticker'].unique()[:3]:  # Show first 3 tickers
                    ticker_data = df[df['ticker'] == ticker]
                    print(f"\n{ticker}: {len(ticker_data)} records")
                    if len(ticker_data) > 0:
                        sample = ticker_data[['datetime', 'baseline_price']].head(2)
                        print(sample.to_string(index=False))
            
    except Exception as e:
        print(f"Error validating CSV: {str(e)}")

def check_multiple_price_fields(csv_path, ticker, target_datetime_str):
    """
    Check multiple price fields against Yahoo Finance data.
    """
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Filter for the specific ticker
        df_ticker = df[df['ticker'] == ticker]
        
        if df_ticker.empty:
            print(f"No data found for ticker {ticker} in CSV")
            return
        
        # Parse target datetime
        target_datetime = pd.to_datetime(target_datetime_str)
        
        # Find closest matching row
        df_ticker = df_ticker.copy()
        df_ticker['time_diff'] = (df_ticker['datetime'] - target_datetime).abs()
        df_match = df_ticker[df_ticker['time_diff'] <= pd.Timedelta(minutes=10)]
        
        if df_match.empty:
            print(f"No matching datetime within 10 minutes for {ticker} at {target_datetime}")
            return
        
        row = df_match.loc[df_match['time_diff'].idxmin()]
        
        # Get Yahoo Finance price
        yf_result = get_closest_price(ticker, target_datetime, window=60)  # Increased window
        if yf_result is None:
            print("No valid Yahoo Finance price found to compare.")
            return
        
        yf_time, yf_price = yf_result
        print(f"\nYahoo Finance close price for {ticker} at {yf_time}: ${yf_price:.4f}")
        print(f"CSV datetime: {row['datetime']}")
        print("\nPrice comparisons:")
        print("-" * 50)
        
        # Compare multiple price fields
        price_fields = ['baseline_price', 'eod_price', 'price_1h', 'price_4h', 'price_eow']
        
        for field in price_fields:
            if field in row.index and pd.notna(row[field]):
                local_price = row[field]
                diff = abs(local_price - yf_price)
                percent_diff = (diff / yf_price) * 100
                
                tolerance = 0.5
                status = "✓ MATCH" if diff <= tolerance else "✗ DIFFER"
                
                print(f"{field:15}: ${local_price:8.4f} | Diff: ${diff:6.4f} ({percent_diff:5.2f}%) | {status}")
        
    except Exception as e:
        print(f"Error in check_multiple_price_fields: {str(e)}")

if __name__ == "__main__":
    # Example usage:
    csv_path = 'scraped_articles.csv'
    ticker = 'ABNB'
    
    # Use a historical date that Yahoo Finance will have data for
    target_datetime = '2024-07-03 17:35:00'  # Changed to 2024 for testing
    
    print("=== Validating CSV Structure ===")
    validate_csv_structure(csv_path)
    
    print("\n=== Checking Single Price Field ===")
    check_price_accuracy(csv_path, ticker, target_datetime, price_field='baseline_price')
    
    print("\n=== Checking Multiple Price Fields ===")
    check_multiple_price_fields(csv_path, ticker, target_datetime)
    
    print("\n=== Note ===")
    print("Your CSV contains future dates (2025). Yahoo Finance only has historical data.")
    print("Consider using historical dates for validation or check if your data source is correct.")