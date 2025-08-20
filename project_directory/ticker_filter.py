import os, requests, random, time, json, logging
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
CONFIG = {
    'CSV_INPUT': "finviz.csv",
    'NO_NEWS_CACHE': "tickers_with_no_news.json",
    'TICKERS_WITH_NEWS': "tickers_with_news.json",
    'PROGRESS_FILE': "filtering_progress.json",  # NEW: Track progress
    'REFRESH_CACHE': False,  # Set to False to resume from where you left off
    'MAX_TEST_TICKERS': None,  # Set to None to process ALL tickers
    'BATCH_SIZE': 100,  # Process in batches
    'TIMEOUT': 10,
    'DELAY_RANGE': (0.5, 1.5),  # Faster delays
    'SAVE_EVERY': 50,  # Save progress every 50 tickers
}

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
]

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ticker_filtering.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TickerNewsChecker:
    def __init__(self):
        self.session = self._create_session()
        self.tickers_with_news = []
        self.tickers_without_news = {}
        self.checked_count = 0
        self.error_count = 0
        
    def _create_session(self):
        """Create a requests session with retry logic"""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session
    
    def load_tickers(self):
        """Load tickers from CSV file"""
        try:
            df = pd.read_csv(CONFIG['CSV_INPUT'])
            if 'Ticker' not in df.columns:
                logger.error("No 'Ticker' column found in CSV")
                return []
            
            tickers = df['Ticker'].dropna().str.upper().unique().tolist()
            logger.info(f"Loaded {len(tickers)} unique tickers from CSV")
            
            # Apply test limit if specified
            if CONFIG['MAX_TEST_TICKERS']:
                tickers = tickers[:CONFIG['MAX_TEST_TICKERS']]
                logger.info(f"Limited to first {CONFIG['MAX_TEST_TICKERS']} tickers for testing")
            
            return sorted(tickers)
            
        except Exception as e:
            logger.error(f"Failed to load tickers: {e}")
            return []
        
    def load_progress(self):
        """Load progress from previous run"""
        try:
            with open(CONFIG['PROGRESS_FILE'], 'r') as f:
                progress = json.load(f)
                self.checked_count = progress.get('checked_count', 0)
                self.error_count = progress.get('error_count', 0)
                logger.info(f"Resumed from previous run: {self.checked_count} tickers checked")
                return progress
        except FileNotFoundError:
            logger.info("No previous progress found, starting fresh")
            return {}
        
    def save_progress(self):
        """Save current progress to resume later"""
        progress = {
            'checked_count': self.checked_count,
            'error_count': self.error_count,
            'last_updated': datetime.now().isoformat(),
            'tickers_with_news_count': len(self.tickers_with_news),
            'tickers_without_news_count': len(self.tickers_without_news)
        }
        
        try:
            with open(CONFIG['PROGRESS_FILE'], 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")

    def load_existing_cache(self):
        """Load existing cache files"""
        # Load no-news cache
        if os.path.exists(CONFIG['NO_NEWS_CACHE']) and not CONFIG['REFRESH_CACHE']:
            try:
                with open(CONFIG['NO_NEWS_CACHE'], 'r') as f:
                    self.tickers_without_news = json.load(f)
                logger.info(f"Loaded {len(self.tickers_without_news)} tickers from no-news cache")
            except Exception as e:
                logger.warning(f"Failed to load no-news cache: {e}")
        
        # Load tickers-with-news cache
        if os.path.exists(CONFIG['TICKERS_WITH_NEWS']) and not CONFIG['REFRESH_CACHE']:
            try:
                with open(CONFIG['TICKERS_WITH_NEWS'], 'r') as f:
                    self.tickers_with_news = json.load(f)
                logger.info(f"Loaded {len(self.tickers_with_news)} tickers from with-news cache")
            except Exception as e:
                logger.warning(f"Failed to load with-news cache: {e}")
    
    def save_caches(self):
        """Save both cache files"""
        try:
            # Save no-news cache
            with open(CONFIG['NO_NEWS_CACHE'], 'w') as f:
                json.dump(self.tickers_without_news, f, indent=2)
            
            # Save with-news cache
            with open(CONFIG['TICKERS_WITH_NEWS'], 'w') as f:
                json.dump(self.tickers_with_news, f, indent=2)
            
            logger.info(f"Saved caches: {len(self.tickers_without_news)} without news, {len(self.tickers_with_news)} with news")
            
        except Exception as e:
            logger.error(f"Failed to save caches: {e}")
    
    def check_ticker_has_news(self, ticker):
        """Check if a ticker has news on Finviz"""
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            response = self.session.get(url, headers=headers, timeout=CONFIG['TIMEOUT'])
            response.raise_for_status()
            
            # Parse the response
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Multiple ways to detect news tables
            news_indicators = [
                # Direct table selectors
                soup.select_one("table.fullview-news-outer"),
                soup.select_one("#news-table"),
                soup.select_one("table[class*='news']"),
                soup.select_one("div[class*='news-table']"),
                
                # Look for any table with news-related content
                soup.find("table", string=lambda text: text and "news" in text.lower() if text else False),
            ]
            
            # Check each indicator
            for indicator in news_indicators:
                if indicator:
                    # Verify it has actual content (not just empty table)
                    rows = indicator.find_all("tr")
                    if len(rows) > 1:  # More than just header
                        # Look for actual news links
                        links = indicator.find_all("a")
                        if links:
                            logger.debug(f"[OK] {ticker}: Found news table with {len(rows)} rows and {len(links)} links")
                            return True, f"news_table_{len(rows)}_rows"
            
            # Check for news-related text patterns in HTML
            news_patterns = [
                'news-table', 'fullview-news', 'latest news', 
                'news content', 'breaking news', 'company news'
            ]
            
            html_lower = response.text.lower()
            for pattern in news_patterns:
                if pattern in html_lower:
                    # Additional verification - look for date patterns near news
                    if any(date_pattern in html_lower for date_pattern in [
                        'today', 'yesterday', 'hours ago', 'minutes ago',
                        'jan-', 'feb-', 'mar-', 'apr-', 'may-', 'jun-',
                        'jul-', 'aug-', 'sep-', 'oct-', 'nov-', 'dec-'
                    ]):
                        logger.debug(f"[OK] {ticker}: Found news pattern '{pattern}' with dates")
                        return True, f"news_pattern_{pattern}"
            
            # If we get here, no news was found
            logger.debug(f"[NO] {ticker}: No news content found")
            return False, "no_news_found"
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error for {ticker}: {e}")
            self.error_count += 1
            return None, f"network_error_{str(e)[:50]}"
            
        except Exception as e:
            logger.warning(f"Parse error for {ticker}: {e}")
            self.error_count += 1
            return None, f"parse_error_{str(e)[:50]}"
    
    def is_already_cached(self, ticker):
        """Check if ticker is already in either cache"""
        if CONFIG['REFRESH_CACHE']:
            return False
            
        # Check if in with-news cache
        if ticker in self.tickers_with_news:
            return True
            
        # Check if in no-news cache and not too old
        if ticker in self.tickers_without_news:
            try:
                last_checked = datetime.fromisoformat(self.tickers_without_news[ticker])
                days_old = (datetime.now() - last_checked).days
                return days_old < 30  # Re-check after 30 days
            except:
                return False
                
        return False
    
    def process_tickers(self, tickers):
        """Process all tickers to determine which have news"""
        logger.info(f"Starting to process {len(tickers)} tickers")
        
        # Load progress
        self.load_progress()
        
        # Skip already processed tickers
        processed_tickers = set(self.tickers_with_news + list(self.tickers_without_news.keys()))
        remaining_tickers = [t for t in tickers if t not in processed_tickers]
        
        logger.info(f"Already processed: {len(processed_tickers)}")
        logger.info(f"Remaining to process: {len(remaining_tickers)}")
        
        if not remaining_tickers:
            logger.info("All tickers already processed!")
            return
        
        # Process in batches
        batch_size = CONFIG['BATCH_SIZE']
        total_batches = (len(remaining_tickers) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(remaining_tickers))
            batch_tickers = remaining_tickers[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_tickers)} tickers)")
            
            for i, ticker in enumerate(batch_tickers, 1):
                logger.info(f"Batch {batch_num + 1} - Checking {i}/{len(batch_tickers)}: {ticker}")
                
                # Check for news
                has_news, reason = self.check_ticker_has_news(ticker)
                self.checked_count += 1
                
                if has_news is True:
                    self.tickers_with_news.append(ticker)
                    logger.info(f"  [+] HAS NEWS: {reason}")
                    
                elif has_news is False:
                    self.tickers_without_news[ticker] = datetime.now().isoformat()
                    logger.info(f"  [-] NO NEWS: {reason}")
                    
                else:  # has_news is None (error)
                    logger.warning(f"  [?] ERROR: {reason}")
                    # Don't cache errors - will retry next time
                
                # Save progress periodically
                if self.checked_count % CONFIG['SAVE_EVERY'] == 0:
                    self.save_caches()
                    self.save_progress()
                    logger.info(f"Progress saved: {self.checked_count} checked, "
                            f"{len(self.tickers_with_news)} with news, "
                            f"{len(self.tickers_without_news)} without news")
                
                # Rate limiting
                if i < len(batch_tickers):
                    delay = random.uniform(*CONFIG['DELAY_RANGE'])
                    time.sleep(delay)
            
            # Save after each batch
            logger.info(f"Batch {batch_num + 1} completed")
            self.save_caches()
            self.save_progress()
            
            # Longer pause between batches to avoid rate limiting
            if batch_num < total_batches - 1:
                logger.info("Pausing 30 seconds between batches...")
                time.sleep(30)
        
        # Final save
        self.save_caches()
        self.save_progress()
    
    def print_summary(self):
        """Print summary of results"""
        total_processed = len(self.tickers_with_news) + len(self.tickers_without_news)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE TICKER FILTERING SUMMARY")
        print("="*60)
        print(f"Total checked this run: {self.checked_count}")
        print(f"Network/Parse errors: {self.error_count}")
        print(f"Tickers WITH news: {len(self.tickers_with_news)}")
        print(f"Tickers WITHOUT news: {len(self.tickers_without_news)}")
        print(f"Total processed: {total_processed}")
        print(f"Completion rate: {total_processed/10045*100:.1f}%")
        
        if self.tickers_with_news:
            print(f"\nSample tickers with news (first 20):")
            for ticker in self.tickers_with_news[:20]:
                print(f"  - {ticker}")
            
            # Show distribution by first letter
            first_letters = {}
            for ticker in self.tickers_with_news:
                letter = ticker[0]
                first_letters[letter] = first_letters.get(letter, 0) + 1
            
            print(f"\nDistribution by first letter:")
            for letter in sorted(first_letters.keys()):
                print(f"  {letter}: {first_letters[letter]} tickers")
        
        print(f"\nCache files:")
        print(f"  - {CONFIG['NO_NEWS_CACHE']} ({len(self.tickers_without_news)} tickers)")
        print(f"  - {CONFIG['TICKERS_WITH_NEWS']} ({len(self.tickers_with_news)} tickers)")
        print(f"  - {CONFIG['PROGRESS_FILE']} (progress tracking)")
        print("="*60)

def main():
    """Main execution function"""
    print("="*60)
    print("COMPREHENSIVE TICKER NEWS FILTERING")
    print("Processing ALL 10,045 tickers...")
    print("="*60)
    
    checker = TickerNewsChecker()
    
    # Load tickers and existing cache
    tickers = checker.load_tickers()
    if not tickers:
        logger.error("No tickers to process")
        return
    
    logger.info(f"Total tickers to process: {len(tickers)}")
    
    checker.load_existing_cache()
    
    # Show current status
    processed = len(checker.tickers_with_news) + len(checker.tickers_without_news)
    remaining = len(tickers) - processed
    
    print(f"Current status:")
    print(f"  Already processed: {processed}/{len(tickers)} ({processed/len(tickers)*100:.1f}%)")
    print(f"  Remaining: {remaining}")
    print(f"  With news so far: {len(checker.tickers_with_news)}")
    print(f"  Without news so far: {len(checker.tickers_without_news)}")
    
    if remaining > 0:
        print(f"\nStarting processing of remaining {remaining} tickers...")
        # Process tickers
        checker.process_tickers(tickers)
    else:
        print("All tickers already processed!")
    
    # Print final summary
    checker.print_summary()
    
    logger.info("Comprehensive ticker filtering completed")

if __name__ == "__main__":
    main()

# Helper function to load results in main script
def get_tickers_with_news():
    """Helper function to load the filtered tickers for use in main script"""
    try:
        with open("tickers_with_news.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: No tickers_with_news.json file found. Run ticker_filter_test.py first.")
        return []
    except Exception as e:
        print(f"Error loading tickers with news: {e}")
        return []