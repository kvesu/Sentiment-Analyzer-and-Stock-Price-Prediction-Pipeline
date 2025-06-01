import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from time import sleep
import re
from urllib.parse import urljoin
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Time Parsers ===
def parse_time_with_today(time_str):
    now = datetime.now()
    formats = ["%I:%M%p", "%I:%M %p", "%H:%M"]
    for fmt in formats:
        try:
            return datetime.combine(now.date(), datetime.strptime(time_str.strip(), fmt).time())
        except ValueError:
            continue
    logger.debug(f"Could not parse time string: '{time_str}'")
    return None

def parse_finviz_datetime(date_str):
    if not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    now = datetime.now()
    
    if "Today" in date_str:
        return parse_time_with_today(date_str.replace("Today", "").strip())
    if "Yesterday" in date_str:
        dt = parse_time_with_today(date_str.replace("Yesterday", "").strip())
        return dt - timedelta(days=1) if dt else None

    formats = [
        "%b-%d-%y %I:%M%p", "%b-%d-%y %I:%M %p", "%Y-%m-%d %I:%M%p",
        "%b %d %I:%M%p", "%m/%d/%Y %I:%M%p", "%b-%d-%Y %I:%M%p"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Fallback to just time string
    match = re.search(r'(\d{1,2}:\d{2}\s*[AP]M)', date_str, re.I)
    return parse_time_with_today(match.group(1)) if match else None

def is_recent_article(date_str, days=7):
    parsed = parse_finviz_datetime(date_str)
    return parsed and parsed >= datetime.now() - timedelta(days=days)

# === Core Scraper ===
def fetch_articles_from_finviz(ticker, days=7):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return []

    soup = BeautifulSoup(res.text, 'html.parser')
    table = soup.select_one('table.fullview-news-outer') or soup.select_one('#news-table')

    if not table:
        logger.warning(f"No news table found for {ticker}")
        return []

    articles, current_date = [], None

    for i, row in enumerate(table.find_all('tr')):
        try:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue

            date_time_text = cols[0].get_text(strip=True)
            if date_time_text:
                current_date = date_time_text
            elif not current_date:
                continue  # skip if date is missing

            if not is_recent_article(current_date, days):
                break

            headline_tag = cols[1].find('a')
            if not headline_tag:
                continue

            headline = headline_tag.text.strip()
            url = urljoin("https://finviz.com/", headline_tag.get("href", ""))

            articles.append({
                'ticker': ticker,
                'datetime': current_date,
                'parsed_datetime': parse_finviz_datetime(current_date),
                'headline': headline,
                'url': url,
                'row_number': i + 1
            })
        except Exception as e:
            logger.debug(f"Row parse error in {ticker} row {i + 1}: {e}")
            continue

    return articles

# === Batch and Combine Utilities ===
def batch_process_tickers(csv_path, ticker_col='Ticker', batch_size=500, days=7, output_dir='finviz_batches'):
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        tickers = df[ticker_col].dropna().str.upper().unique()
    except Exception as e:
        logger.error(f"Could not load tickers: {e}")
        return

    processed = {
        fname.split('.')[0].split('_')[-1]
        for fname in os.listdir(output_dir)
        if fname.endswith('.csv')
    }

    tickers = [t for t in tickers if t not in processed]
    logger.info(f"Processing {len(tickers)} tickers...")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        articles = []

        for t in batch:
            logger.info(f"Scraping {t}")
            articles.extend(fetch_articles_from_finviz(t, days=days))
            sleep(1)

        if articles:
            df_batch = pd.DataFrame(articles)
            df_batch.to_csv(os.path.join(output_dir, f'finviz_articles_{i+1}_to_{i+len(batch)}.csv'), index=False)
            logger.info(f"Saved batch {i+1}-{i+len(batch)}")

def combine_batches(output_dir='finviz_batches', output_file='finviz_articles_all.csv'):
    import glob
    files = glob.glob(os.path.join(output_dir, '*.csv'))
    frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
    if frames:
        pd.concat(frames).to_csv(output_file, index=False)
        logger.info(f"Saved combined data to {output_file}")
    else:
        logger.warning("No batch files found to combine.")

def fetch_articles_for_first_n_tickers(csv_path, ticker_col='Ticker', n=5, days=7, output_file='finviz_articles_first_5.csv'):
    try:
        df = pd.read_csv(csv_path)
        tickers = df[ticker_col].dropna().str.upper().unique()[:n]
    except Exception as e:
        logger.error(f"Failed to read input CSV: {e}")
        return

    articles = []
    for t in tickers:
        logger.info(f"Scraping {t}")
        articles.extend(fetch_articles_from_finviz(t, days=days))
        sleep(1)

    if articles:
        pd.DataFrame(articles).to_csv(output_file, index=False)
        logger.info(f"Saved first {n} tickers to {output_file}")
    else:
        logger.warning("No articles fetched.")

# === Main ===
if __name__ == "__main__":
    fetch_articles_for_first_n_tickers(
        csv_path='finviz.csv',
        ticker_col='Ticker',
        n=5,
        days=7,
        output_file='finviz_articles_first_5.csv'
    )
