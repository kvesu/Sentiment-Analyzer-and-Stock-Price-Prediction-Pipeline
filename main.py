import os
import re
import sqlite3
import logging
import random
import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from time import sleep
from datetime import datetime, timedelta
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from keybert import KeyBERT

# ——— CONFIG ———
API_KEY = "4b55007276a5a753bd175962d005d516de2d9f2e2c5623e854434b5bf1c35fb1"
os.environ["SERPAPI_API_KEY"] = API_KEY

DB_PATH = "articles.db"
CSV_INPUT = "finviz.csv"
CSV_OUTPUT = "finviz_first10.csv"
MAX_TICKERS = 10
DAYS_BACK = 7
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)

POSITIVE_WORDS = set(["gain", "rise", "increase", "surge", "beat", "profit", "up", "growth"])
NEGATIVE_WORDS = set(["drop", "decline", "loss", "miss", "fall", "down", "plunge"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# ——— RETRY SESSION ———
def create_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[403, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = create_session()
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

# ——— DATE PARSING ———
COMMON_FORMATS = [
    "%b-%d-%y %I:%M%p","%b-%d-%y %I:%M %p","%b-%d-%Y %I:%M%p","%b-%d-%Y %I:%M %p",
    "%Y-%m-%d %I:%M%p","%Y-%m-%d %I:%M %p","%b %d %I:%M%p","%b %d %I:%M %p",
    "%m/%d/%Y %I:%M%p","%m/%d/%Y %I:%M %p","%m/%d/%y %I:%M%p","%m/%d/%y %I:%M %p",
    "%b-%d-%y","%b-%d-%Y","%Y-%m-%d","%m/%d/%Y","%m/%d/%y"
]

def parse_datetime(s):
    s = (s or "").strip()
    now = datetime.now()
    if s.lower().startswith("today"):
        return _parse_time(s, now.date())
    if s.lower().startswith("yesterday"):
        return _parse_time(s, (now - timedelta(days=1)).date())
    for fmt in COMMON_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except:
            continue
    match = re.search(r"\d{1,2}:\d{2}\s*[AP]M", s, re.I)
    return _parse_time(match.group(0), now.date()) if match else None

def _parse_time(t, base):
    for fmt in ["%I:%M%p","%I:%M %p","%H:%M","%I%p","%I %p"]:
        try:
            return datetime.combine(base, datetime.strptime(t.strip(), fmt).time())
        except:
            continue
    return None

# ——— TEXT PREPROCESSING ———
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

# ——— ARTICLE TEXT ———
def get_article_text(url):
    try:
        r = SESSION.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        main = soup.find("article") or soup.find("main") or soup
        text = " ".join(p.get_text(strip=True) for p in main.find_all("p"))
        if len(text) < 200:
            raise ValueError("Too short")
        return text
    except Exception as e:
        logger.debug(f"Direct scrape failed for {url}: {e}")
        return serpapi_text(url)

def serpapi_text(url):
    try:
        params = {"engine": "google", "api_key": API_KEY, "q": f"site:{url}"}
        res = GoogleSearch(params).get_dict()
        for item in res.get("organic_results", []):
            if url in item.get("link", ""):
                return item.get("snippet", "").strip()
        return res.get("answer_box", {}).get("snippet", "")
    except Exception as e:
        logger.debug(f"SerpApi scrape failed for {url}: {e}")
        return ""

# ——— SCRAPER ———
def fetch_finviz(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        r = SESSION.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        logger.error(f"Finviz fetch failed for {ticker}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.select_one("table.fullview-news-outer") or soup.select_one("#news-table")
    if not table:
        logger.warning(f"No news table found for {ticker}")
        return []

    articles, last_dt = [], None
    cutoff = datetime.now() - timedelta(days=DAYS_BACK)

    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        dt_txt = cols[0].get_text(strip=True)
        dt = parse_datetime(dt_txt) or last_dt
        if not dt or dt < cutoff:
            # Skip old articles outside the window
            continue

        link = cols[1].find("a")
        if not link:
            continue

        article_url = urljoin("https://finviz.com/", link["href"])
        text = get_article_text(article_url)
        tokens = preprocess_text(text)

        keywords = kw_model.extract_keywords(tokens, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
        pos_keywords = [kw for kw, _ in keywords if kw in POSITIVE_WORDS]
        neg_keywords = [kw for kw, _ in keywords if kw in NEGATIVE_WORDS]

        articles.append({
            "ticker": ticker,
            "datetime": dt_txt,
            "parsed_datetime": dt,
            "headline": link.get_text(strip=True),
            "url": article_url,
            "text": text,
            "tokens": tokens,
            "pos_keywords": ", ".join(pos_keywords),
            "neg_keywords": ", ".join(neg_keywords),
        })
        last_dt = dt

    return articles

# ——— DATABASE ———
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        ticker TEXT,
        datetime TEXT,
        parsed_datetime TIMESTAMP,
        headline TEXT,
        url TEXT UNIQUE,
        text TEXT,
        tokens TEXT,
        pos_keywords TEXT,
        neg_keywords TEXT
    )
    """)
    conn.commit()
    conn.close()

def upsert_articles(rows):
    if not rows:
        return
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame(rows).drop_duplicates(subset="url")
    try:
        existing = pd.read_sql("SELECT url FROM articles", conn)["url"].tolist()
        new_df = df[~df["url"].isin(existing)]
        new_df.to_sql("articles", conn, if_exists="append", index=False)
    except Exception as e:
        logger.error(f"Insert failed: {e}")
    conn.close()

# ——— MAIN ———
if __name__ == "__main__":
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)

    init_db()
    df = pd.read_csv(CSV_INPUT)
    tickers = df["Ticker"].dropna().str.upper().unique()[:MAX_TICKERS]

    all_rows = []
    for t in tickers:
        logger.info(f"Processing {t}")
        all_rows.extend(fetch_finviz(t))
        sleep(random.uniform(1, 3))

    pd.DataFrame(all_rows).to_csv(CSV_OUTPUT, index=False)
    upsert_articles(all_rows)
