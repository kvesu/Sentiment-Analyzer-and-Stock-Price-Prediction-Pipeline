import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from time import sleep
import re
from urllib.parse import urljoin
import logging
import os
from serpapi import GoogleSearch
from dateutil import parser

# For retries
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random # For random sleep

# --- Add this line for testing purposes ONLY ---
os.environ["SERPAPI_API_KEY"] = "4b55007276a5a753bd175962d005d516de2d9f2e2c5623e854434b5bf1c35fb1"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configure requests session with retry logic ---
def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 503, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(['GET']), # Only retry GET requests
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Global session for reusability and efficiency across all requests
global_session = requests_retry_session()
# A more robust User-Agent to mimic a real browser
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}


# === FIXED Time Parsers ===
def parse_time_with_today(time_str):
    """Parse time string and combine with today's date"""
    if not time_str:
        return None
        
    now = datetime.now()
    time_str = time_str.strip()
    
    # Handle different time formats
    formats = [
        "%I:%M%p",      # 12:30PM
        "%I:%M %p",     # 12:30 PM  
        "%H:%M",        # 14:30
        "%I%p",         # 2PM
        "%I %p"         # 2 PM
    ]
    
    for fmt in formats:
        try:
            parsed_time = datetime.strptime(time_str, fmt).time()
            return datetime.combine(now.date(), parsed_time)
        except ValueError:
            continue
    
    logger.debug(f"Could not parse time string: '{time_str}'")
    return None

def parse_finviz_datetime(date_str):
    """Parse Finviz datetime strings with better handling"""
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    
    date_str = date_str.strip()
    now = datetime.now()
    
    logger.debug(f"Parsing datetime string: '{date_str}'")
    
    # Handle "Today" with time
    if date_str.lower().startswith("today"):
        time_part = date_str.replace("Today", "").replace("today", "").strip()
        if time_part:
            result = parse_time_with_today(time_part)
            logger.debug(f"Parsed 'Today {time_part}' as: {result}")
            return result
        else:
            # Just "Today" without time
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Handle "Yesterday" with time
    if date_str.lower().startswith("yesterday"):
        time_part = date_str.replace("Yesterday", "").replace("yesterday", "").strip()
        if time_part:
            dt = parse_time_with_today(time_part)
            if dt:
                result = dt - timedelta(days=1)
                logger.debug(f"Parsed 'Yesterday {time_part}' as: {result}")
                return result
        else:
            # Just "Yesterday" without time
            yesterday = now - timedelta(days=1)
            return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Handle various date formats with time
    formats = [
        "%b-%d-%y %I:%M%p",     # Jan-15-24 2:30PM
        "%b-%d-%y %I:%M %p",    # Jan-15-24 2:30 PM
        "%b-%d-%Y %I:%M%p",     # Jan-15-2024 2:30PM
        "%b-%d-%Y %I:%M %p",    # Jan-15-2024 2:30 PM
        "%Y-%m-%d %I:%M%p",     # 2024-01-15 2:30PM
        "%Y-%m-%d %I:%M %p",    # 2024-01-15 2:30 PM
        "%b %d %I:%M%p",        # Jan 15 2:30PM
        "%b %d %I:%M %p",       # Jan 15 2:30 PM
        "%m/%d/%Y %I:%M%p",     # 01/15/2024 2:30PM
        "%m/%d/%Y %I:%M %p",    # 01/15/2024 2:30 PM
        "%m/%d/%y %I:%M%p",     # 01/15/24 2:30PM
        "%m/%d/%y %I:%M %p",    # 01/15/24 2:30 PM
        "%b-%d-%y",             # Jan-15-24 (date only)
        "%b-%d-%Y",             # Jan-15-2024 (date only)
        "%Y-%m-%d",             # 2024-01-15 (date only)
        "%m/%d/%Y",             # 01/15/2024 (date only)
        "%m/%d/%y",             # 01/15/24 (date only)
    ]
    
    for fmt in formats:
        try:
            result = datetime.strptime(date_str, fmt)
            logger.debug(f"Parsed '{date_str}' using format '{fmt}' as: {result}")
            return result
        except ValueError:
            continue
    
    # Try to extract just time if no date part is found
    time_pattern = r'(\d{1,2}:\d{2}\s*[AP]M)'
    time_match = re.search(time_pattern, date_str, re.I)
    if time_match:
        result = parse_time_with_today(time_match.group(1))
        logger.debug(f"Extracted time '{time_match.group(1)}' from '{date_str}' as: {result}")
        return result
    
    # Try to extract just hour with AM/PM
    hour_pattern = r'(\d{1,2}\s*[AP]M)'
    hour_match = re.search(hour_pattern, date_str, re.I)
    if hour_match:
        result = parse_time_with_today(hour_match.group(1))
        logger.debug(f"Extracted hour '{hour_match.group(1)}' from '{date_str}' as: {result}")
        return result
    
    logger.warning(f"Could not parse datetime string: '{date_str}'")
    return None

def is_recent_article(date_str, days=7):
    """Check if article is recent with better error handling"""
    parsed = parse_finviz_datetime(date_str)
    if not parsed:
        logger.debug(f"Could not determine if article is recent: '{date_str}'")
        return True  # Include articles we can't parse the date for
    
    cutoff_date = datetime.now() - timedelta(days=days)
    is_recent = parsed >= cutoff_date
    logger.debug(f"Article date {parsed} is recent (within {days} days): {is_recent}")
    return is_recent

# === Core Scraper - FIXED ===
def fetch_articles_from_finviz(ticker, days=7):
    """Fetch articles with improved datetime parsing and logging"""
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    try:
        # Use the global session with retry logic
        res = global_session.get(url, headers=headers, timeout=15)
        res.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch Finviz page for {ticker}: {e}")
        return []

    soup = BeautifulSoup(res.text, 'html.parser')
    # Try both common table selectors for news
    table = soup.select_one('table.fullview-news-outer') or soup.select_one('#news-table')

    if not table:
        logger.warning(f"No news table found for {ticker} on Finviz.")
        return []

    articles = []
    current_full_date = None  # Track the full date (not just time)
    last_parsed_date = None   # Track the last successfully parsed date

    for i, row in enumerate(table.find_all('tr')):
        try:
            cols = row.find_all('td')
            if len(cols) < 2: # Ensure row has enough columns for date and headline
                continue

            # Get the date/time cell content
            date_time_cell = cols[0]
            date_time_text = date_time_cell.get_text(strip=True)
            
            # Determine what type of datetime info we have
            if date_time_text:
                # Check if this is just a time (like "04:00AM") or a full date
                is_time_only = bool(re.match(r'^\d{1,2}:\d{2}\s*[AP]M$', date_time_text, re.I))
                
                if is_time_only and last_parsed_date:
                    # This is just a time, use the last known date
                    time_part = date_time_text
                    base_date = last_parsed_date.date()
                    
                    # Parse the time and combine with the base date
                    parsed_time = parse_time_with_today(time_part)
                    if parsed_time:
                        combined_datetime = datetime.combine(base_date, parsed_time.time())
                        current_full_date = date_time_text  # Keep original for display
                        last_parsed_date = combined_datetime
                        logger.debug(f"Row {i+1}: Time-only '{date_time_text}' combined with date {base_date} = {combined_datetime}")
                    else:
                        logger.debug(f"Row {i+1}: Could not parse time-only '{date_time_text}'")
                        continue
                else:
                    # This should be a full date or "Today"/"Yesterday"
                    current_full_date = date_time_text
                    parsed_dt = parse_finviz_datetime(date_time_text)
                    if parsed_dt:
                        last_parsed_date = parsed_dt
                        logger.debug(f"Row {i+1}: Full datetime '{current_full_date}' parsed as {parsed_dt}")
                    else:
                        logger.debug(f"Row {i+1}: Could not parse datetime '{current_full_date}'")
                        continue
            elif not current_full_date:
                logger.debug(f"Row {i+1}: No datetime and no previous date, skipping")
                continue

            # Use the last successfully parsed date for recency check
            if not last_parsed_date:
                logger.debug(f"Row {i+1}: No parsed date available, skipping")
                continue
                
            # Check if article is recent enough
            cutoff_date = datetime.now() - timedelta(days=days)
            is_recent = last_parsed_date >= cutoff_date
            if not is_recent:
                logger.debug(f"Row {i+1}: Article too old ({last_parsed_date}), stopping")
                break # Stop processing if articles are too old (assuming chronological order)

            headline_tag = cols[1].find('a')
            if not headline_tag:
                logger.debug(f"Row {i+1}: No headline link found, skipping")
                continue # Skip if no headline link is found

            headline = headline_tag.text.strip()
            # Construct full URL using urljoin for robustness
            article_url = urljoin("https://finviz.com/", headline_tag.get("href", ""))

            article = {
                'ticker': ticker,
                'datetime': current_full_date,
                'parsed_datetime': last_parsed_date,
                'headline': headline,
                'url': article_url,
                'row_number': i + 1,
                'text': None # Placeholder for article text, to be filled later
            }
            
            articles.append(article)
            logger.debug(f"Row {i+1}: Added article - {headline[:50]}...")
            
        except Exception as e:
            logger.error(f"Row parse error in {ticker} Finviz table (row {i + 1}): {e}")
            continue

    logger.info(f"Fetched {len(articles)} articles for {ticker}")
    return articles

def extract_text_with_serpapi(url):
    try:
        params = {
            "engine": "google",
            "url": url,
            "api_key": os.getenv("SERPAPI_API_KEY")  # Make sure your environment variable is set
        }
        search = GoogleSearch(params)
        result = search.get_dict()

        if "answer_box" in result:
            return clean_text(result["answer_box"].get("snippet", ""))
        elif "organic_results" in result:
            for res in result["organic_results"]:
                if "snippet" in res:
                    return clean_text(res["snippet"])
        logger.warning(f"SerpApi didn't return useful text for: {url}")
        return None
    except Exception as e:
        logger.error(f"SerpApi failed for {url}: {e}")
        return None

# === Function to extract article text from external URLs ===
def extract_article_text(url):
    domain = urljoin(url, '/').replace('https://', '').replace('http://', '').split('/')[0]

    if 'finviz.com' in domain:
        logger.warning(f"Skipping Finviz internal news link as it's not the external article: {url}")
        return None
    elif 'wsj.com' in domain or 'bloomberg.com' in domain or 'nytimes.com' in domain:
        logger.warning(f"Skipping paywalled/subscription article from {domain}: {url}")
        return "PAYWALL_CONTENT"

    try:
        response = global_session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        article_content = soup.find('article') or soup.find('main')
        if article_content:
            return clean_text(article_content.get_text())

        if 'finance.yahoo.com' in domain:
            selectors = [
                'div[data-test-id="article-body"]',
                '.caas-body',
                '.caas-text-body',
            ]
            for selector in selectors:
                main_div = soup.select_one(selector)
                if main_div:
                    for unwanted in main_div.select('.caas-read-more-wrapper, .caas-related-stories, .caas-sdk, .caas-content-wrapper .caas-header'):
                        unwanted.extract()
                    return clean_text(main_div.get_text())

        elif 'businesswire.com' in domain:
            content_div = soup.find('div', id='bw-release-story')
            if content_div:
                return clean_text(content_div.get_text())

        paragraphs = soup.find_all('p')
        if paragraphs:
            full_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
            return clean_text(full_text)

        logger.warning(f"Could not find specific article content for: {url}. Attempting fallback with SerpApi.")
        serpapi_text = extract_text_with_serpapi(url)
        return serpapi_text if serpapi_text else None

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error {e.response.status_code} fetching article from {url}: {e.response.reason}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error fetching article from {url}: {e}")
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout Error fetching article from {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"General Request Error fetching article from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing article from {url}: {e}", exc_info=True)
        return None

def clean_text(text):
    if not text:
        return ""
    # Replace multiple whitespace characters (including newlines) with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Batch and Combine Utilities ===
def batch_process_tickers(csv_path, ticker_col='Ticker', batch_size=500, days=7, output_dir='finviz_batches'):
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        tickers = df[ticker_col].dropna().str.upper().unique()
    except Exception as e:
        logger.error(f"Could not load tickers from {csv_path}: {e}")
        return

    # Simple check for already processed tickers based on filename (might need refinement for complex cases)
    processed_tickers = set()
    for fname in os.listdir(output_dir):
        if fname.startswith('finviz_articles_') and fname.endswith('.csv'):
            # Extract ticker range from filename, e.g., 'finviz_articles_1_to_500.csv'
            # This logic assumes batches save all tickers in that batch.
            # For more granular "processed" tracking, you'd need to read each batch file.
            pass # Currently, this processed tracking is more for individual ticker files if saved that way

    tickers_to_process = list(tickers) # Process all for now, or implement more robust resume logic

    if not tickers_to_process:
        logger.info("No tickers found to process or all appear to have been processed.")
        return

    logger.info(f"Processing {len(tickers_to_process)} tickers...")

    for i in range(0, len(tickers_to_process), batch_size):
        batch = tickers_to_process[i:i + batch_size]
        all_articles_in_batch = [] # To hold articles for this current batch

        for t in batch:
            logger.info(f"Scraping {t} headlines from Finviz...")
            headlines_only = fetch_articles_from_finviz(t, days=days)

            for article in headlines_only:
                if article['url']:
                    logger.info(f"  Attempting to extract text from: {article['url']}")
                    article['text'] = extract_article_text(article['url'])
                    sleep(random.uniform(0.5, 1.5)) # Random delay between individual article fetches
                all_articles_in_batch.append(article)
            sleep(random.uniform(1, 3)) # Random delay between processing different tickers

        if all_articles_in_batch:
            df_batch = pd.DataFrame(all_articles_in_batch)
            # Define output filename for the batch
            batch_start_idx = i + 1
            batch_end_idx = i + len(batch)
            output_filename = os.path.join(output_dir, f'finviz_articles_batch_{batch_start_idx}_to_{batch_end_idx}.csv')
            df_batch.to_csv(output_filename, index=False)
            logger.info(f"Saved batch {batch_start_idx}-{batch_end_idx} to {output_filename}")
        else:
            logger.info(f"No articles fetched for batch starting with ticker {batch[0] if batch else 'N/A'}.")


def combine_batches(output_dir='finviz_batches', output_file='finviz_articles_all.csv'):
    import glob
    files = glob.glob(os.path.join(output_dir, '*.csv'))
    frames = []
    for f in files:
        if os.path.getsize(f) > 0: # Ensure file is not empty
            try:
                df = pd.read_csv(f)
                frames.append(df)
            except pd.errors.EmptyDataError:
                logger.warning(f"Skipping empty file: {f}")
            except Exception as e:
                logger.error(f"Error reading CSV {f}: {e}")

    if frames:
        combined_df = pd.concat(frames, ignore_index=True)
        # Ensure 'text' column exists, even if some articles failed extraction
        if 'text' not in combined_df.columns:
            combined_df['text'] = None
        # Drop duplicates based on URL to avoid redundant articles if running multiple times
        combined_df.drop_duplicates(subset=['url'], inplace=True)
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved combined data to {output_file}")
    else:
        logger.warning("No batch files found or readable to combine.")

def fetch_articles_for_first_n_tickers(csv_path, ticker_col='Ticker', n=5, days=7, output_file='finviz_articles_first_n_with_text.csv'):
    try:
        df = pd.read_csv(csv_path)
        tickers = df[ticker_col].dropna().str.upper().unique()[:n]
    except Exception as e:
        logger.error(f"Failed to read input CSV: {e}")
        return

    articles = []
    if len(tickers) == 0:
        logger.warning("No tickers found to process.")
        return

    for t in tickers:
        logger.info(f"Scraping {t} headlines from Finviz...")
        headlines_only = fetch_articles_from_finviz(t, days=days)

        for article in headlines_only:
            if article['url']:
                logger.info(f"  Attempting to extract text from: {article['url']}")
                article['text'] = extract_article_text(article['url'])
                sleep(random.uniform(0.5, 1.5)) # Random delay between article fetches
            articles.append(article)
        sleep(random.uniform(1, 3)) # Random delay between tickers

    if articles:
        df = pd.DataFrame(articles)

        # Remove rows with missing or unusable article text
        filtered_df = df[
            df['text'].notna() & 
            (df['text'] != "PAYWALL_CONTENT") & 
            (df['text'].str.strip() != "")
        ]

        if not filtered_df.empty:
            filtered_df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(filtered_df)} articles with valid text to {output_file}")
        else:
            logger.warning("All fetched articles had missing or unusable text. No file written.")
    else:
        logger.warning("No articles fetched for the first N tickers.")


# === Main Execution Block ===
if __name__ == "__main__":
    # --- For debugging datetime parsing, uncomment these lines ---
    # logging.getLogger(__name__).setLevel(logging.DEBUG)  # Enable debug logging
    # test_dates = ["Today 2:30PM", "Yesterday 10:15AM", "Jan-15-24 3:45PM", "12:30PM", "2PM"]
    # for date_str in test_dates:
    #     result = parse_finviz_datetime(date_str)
    #     print(f"'{date_str}' -> {result}")
    
    # --- Run for the first N tickers (good for quick testing) ---
    fetch_articles_for_first_n_tickers(
        csv_path='finviz.csv',
        ticker_col='Ticker',
        n=5, # Process the first 5 tickers
        days=7, # Look for articles in the last 7 days
        output_file='finviz_articles_first_5_with_text.csv'
    )