import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os
import time
import csv
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import requests
from bs4 import BeautifulSoup

# --- Dummy/Fallback for main.py components if not available ---
# These are simple implementations to allow this script to run standalone
# without needing the full main.py setup for initial testing/scraping.
# For full functionality (advanced sentiment, price data), this script
# should be run *by* main.py or main.py's components properly integrated.
class NewsProcessor:
    def extract_mentions_and_sentiment(self, text: str) -> Tuple[List[str], List[str], List[str], int]:
        # Dummy implementation for sentiment and mentions
        tokens_count = len(text.split())
        return [], [], [], tokens_count

    def scrape_article(self, url: str) -> Optional[str]:
        # Dummy for external article scraping
        return None

def init_database():
    pass

def save_articles(articles: List[Dict]):
    # Dummy save: just log that it would save
    logging.getLogger("FinvizNewsProcessor").info(f"Dummy save_articles called for {len(articles)} articles.")
    if articles:
        # Optionally save to a temporary CSV for inspection during standalone runs
        pd.DataFrame(articles).to_csv("dummy_processed_finviz_articles.csv", index=False)


# --- Enhanced logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler('finviz_news.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FinvizNewsProcessor")

@dataclass
class NewsArticle:
    ticker: str
    company_name: str
    headline: str
    url: str
    source: str
    timestamp: datetime
    minutes_ago: int
    article_text: str
    tokens: int
    sentiment_scores: Dict[str, float]
    mentions: List[str]
    positive_keywords: List[str]
    negative_keywords: List[str]
    market_relevance: float

class FinvizNewsProcessor:
    def __init__(self):
        # self.processor will be a dummy if main.py's NewsProcessor isn't found
        self.processor = NewsProcessor()
        self.processed_urls: Set[str] = set()
        self.ticker_patterns = self._compile_ticker_patterns()
        self.source_reliability = self._get_source_weights()
        self.valid_tickers = self._load_valid_tickers('finviz.csv')
        self.sentiment_keywords = self._load_sentiment_keywords('sentiment_keywords.csv')
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def _load_valid_tickers(self, filepath: str) -> Set[str]:
        try:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                ticker_col = None
                for col in ['Ticker', 'ticker', 'Symbol', 'symbol']:
                    if col in df.columns:
                        ticker_col = col
                        break
                
                if ticker_col:
                    tickers = set(df[ticker_col].astype(str).str.upper().str.strip())
                    logger.info(f"Loaded {len(tickers)} valid tickers from {filepath}")
                    return tickers
                else:
                    logger.warning(f"No ticker column found in {filepath}")
            else:
                logger.warning(f"Ticker file {filepath} not found")
        except Exception as e:
            logger.error(f"Failed to load ticker list from {filepath}: {e}")
        return set()

    def _compile_ticker_patterns(self) -> re.Pattern:
        return re.compile(r'(?:\$|NYSE:|NASDAQ:)?([A-Z]{1,5})(?:\s|$|[^\w])')

    def _get_source_weights(self) -> Dict[str, float]:
        return {
            'Reuters': 0.9, 'Bloomberg': 0.9, 'Wall Street Journal': 0.9,
            'Financial Times': 0.9, 'Yahoo Finance': 0.8, 'MarketWatch': 0.8,
            'CNBC': 0.8, 'Seeking Alpha': 0.7, 'The Motley Fool': 0.6,
            'PR Newswire': 0.5, 'Business Wire': 0.5, 'Zacks': 0.6,
            'Finviz': 0.7, 'default': 0.5
        }

    def _load_sentiment_keywords(self, filepath: str) -> Dict[str, Set[str]]:
        positive = set()
        negative = set()
        
        default_positive = {
            'growth', 'profit', 'revenue', 'beat', 'exceed', 'strong', 'positive',
            'bullish', 'upgrade', 'buy', 'outperform', 'success', 'gain', 'rise',
            'increase', 'breakthrough', 'approval', 'partnership', 'expansion',
            'record', 'high', 'soar', 'surge', 'rally', 'boom', 'optimistic'
        }
        default_negative = {
            'loss', 'decline', 'fall', 'drop', 'weak', 'negative', 'bearish',
            'downgrade', 'sell', 'underperform', 'failure', 'decrease', 'warning',
            'concern', 'risk', 'challenge', 'problem', 'issue', 'reject',
            'plunge', 'crash', 'slump', 'disappointing', 'miss', 'cut'
        }
        
        try:
            if os.path.exists(filepath):
                with open(filepath, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        kw = row.get('keyword', '').strip().lower()
                        sentiment = row.get('sentiment', '').strip().lower()
                        if kw:
                            if sentiment == 'positive':
                                positive.add(kw)
                            elif sentiment == 'negative':
                                negative.add(kw)
                logger.info(f"Loaded {len(positive)} positive and {len(negative)} negative keywords from {filepath}")
            else:
                logger.warning(f"Sentiment keywords file {filepath} not found, using defaults")
                positive = default_positive
                negative = default_negative
        except Exception as e:
            logger.error(f"Failed to load sentiment keywords from {filepath}: {e}")
            logger.info("Using default sentiment keywords")
            positive = default_positive
            negative = default_negative
            
        return {'positive': positive, 'negative': negative}
    
    def _scrape_finviz_main_news(self) -> pd.DataFrame:
        """Directly scrapes the main Finviz news page (news.ashx) and returns a DataFrame."""
        finviz_url = "https://finviz.com/news.ashx?v=3"
        logger.info(f"Attempting direct scrape of Finviz main news from: {finviz_url}")
        
        try:
            response = self.session.get(finviz_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            news_table = soup.find('table', class_='styled-table-new')
            
            if not news_table:
                logger.error("Could not find the main Finviz news table. HTML structure may have changed again or selector is wrong.")
                return pd.DataFrame()

            data = []
            for row in news_table.find_all('tr', class_='news_table-row'):
                time_cell = row.find('td', class_='news_date-cell')
                link_cell_container = row.find('td', class_='news_link-cell')

                if time_cell and link_cell_container:
                    time_str = time_cell.get_text(strip=True)

                    headline_link_tag = link_cell_container.find('a', class_='nn-tab-link')
                    
                    if not headline_link_tag:
                        logger.debug(f"Skipping row as no main link found: {row.get_text()}")
                        continue

                    headline = headline_link_tag.get_text(strip=True)
                    relative_url = headline_link_tag.get('href', '')
                    
                    url = f"https://finviz.com{relative_url}" if relative_url.startswith('/') else relative_url

                    source_span = link_cell_container.find_all('span', class_='news_date-cell')
                    source = source_span[-1].get_text(strip=True) if source_span else "Unknown Source"

                    data.append({
                        'title': f"{time_str} {headline}",
                        'link': url,
                        'source_extracted': source
                    })
            
            logger.info(f"Successfully scraped {len(data)} news items directly from Finviz.")
            return pd.DataFrame(data)

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed while scraping Finviz main news: {e}", exc_info=True)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error scraping Finviz main news: {e}", exc_info=True)
            return pd.DataFrame()

    def parse_finviz_dataframe(self, df: pd.DataFrame) -> List[NewsArticle]:
        articles = []
        
        logger.info(f"Processing {len(df)} Finviz articles")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        for idx, row in df.iterrows():
            try:
                article = self._parse_single_row(row, idx)
                if article:
                    articles.append(article)
                    logger.debug(f"Processed: {article.ticker} - {article.headline[:50]}...")
            except Exception as e:
                logger.error(f"Error parsing row {idx}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(articles)} articles")
        return articles
    
    def _parse_single_row(self, row: pd.Series, idx: int) -> Optional[NewsArticle]:
        full_title = self._extract_field(row, ['title', 'Title', 'headline', 'text'])
        url = self._extract_field(row, ['link', 'Link', 'url', 'URL', 'href'])
        
        if not full_title or not url:
            logger.debug(f"Row {idx}: Missing title or URL")
            return None
            
        if url in self.processed_urls:
            logger.debug(f"Row {idx}: Already processed URL: {url}")
            return None
        
        self.processed_urls.add(url)
        
        headline = re.sub(r'^\d+\s+(min|hour|hours|day|days)\s+', '', full_title, flags=re.IGNORECASE).strip()
        headline_clean = re.sub(r'\s+[A-Z]{2,5}(?:\s+[A-Z]{2,5})*\s*$', '', headline).strip()
        
        source = self._extract_field(row, ['source_extracted']) # Prioritize explicitly extracted source
        if not source:
            source_match = re.search(r'\b(Motley Fool|Bloomberg|Benzinga|TheStreet|Insider Monkey)\s*$', headline, re.IGNORECASE)
            source = source_match.group(1) if source_match else 'Finviz'
            if source_match:
                headline_clean = headline_clean[:source_match.start()].strip()
        
        timestamp, minutes_ago = self._parse_timestamp(row)
        
        ticker, company_name = self._extract_ticker_info(full_title, row)
        
        article_text = self._scrape_article_safely(url)
        if not article_text or len(article_text.strip()) < 50:
            article_text = headline_clean
        
        sentiment_data = self._process_sentiment(headline_clean, article_text)
        market_relevance = self._calculate_market_relevance(headline_clean, article_text, source, ticker, minutes_ago)
        
        return NewsArticle(
            ticker=ticker,
            company_name=company_name,
            headline=headline_clean,
            url=url,
            source=source,
            timestamp=timestamp,
            minutes_ago=minutes_ago,
            article_text=article_text,
            tokens=len(article_text.split()), # Tokens from preprocessed text, or raw text split length
            sentiment_scores=sentiment_data['scores'],
            mentions=sentiment_data['mentions'],
            positive_keywords=sentiment_data['positive_keywords'],
            negative_keywords=sentiment_data['negative_keywords'],
            market_relevance=market_relevance
        )
    
    def _extract_field(self, row: pd.Series, possible_names: List[str]) -> str:
        for name in possible_names:
            if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
                return str(row[name]).strip()
        return ""
    
    def _parse_timestamp(self, row: pd.Series) -> Tuple[datetime, int]:
        timestamp_field = self._extract_field(row, ['time_ago', 'datetime', 'date', 'time', 'timestamp', 'published', 'title'])
        
        if timestamp_field:
            time_patterns = [
                (r'^(\d+)\s*min\b', 1),
                (r'^(\d+)\s*hour\b', 60),
                (r'^(\d+)\s*hours\b', 60),
                (r'^(\d+)\s*day\b', 1440),
                (r'^(\d+)\s*days\b', 1440),
            ]
            
            for pattern, multiplier in time_patterns:
                match = re.search(pattern, timestamp_field.strip(), re.IGNORECASE)
                if match:
                    minutes_ago = int(match.group(1)) * multiplier
                    timestamp = datetime.now() - timedelta(minutes=minutes_ago)
                    return timestamp, minutes_ago
        
        return datetime.now(), 0
    
    def _extract_ticker_info(self, headline: str, row: pd.Series) -> Tuple[str, str]:
        full_text = self._extract_field(row, ['title', 'Title', 'headline', 'text'])
        if not full_text:
            full_text = headline
        
        ticker_matches = re.findall(r'\b([A-Z]{2,5})\b', full_text)
        
        valid_tickers_found = []
        for ticker in ticker_matches:
            if ticker in self.valid_tickers or not self.valid_tickers:
                if ticker not in ['NEWS', 'NYSE', 'ETF', 'IPO', 'CEO', 'CFO', 'USA', 'USD']:
                    valid_tickers_found.append(ticker)
        
        ticker = valid_tickers_found[0] if valid_tickers_found else "UNKNOWN"
        
        company_name = "Unknown Company"
        if ticker != "UNKNOWN":
            clean_text = re.sub(r'^\d+\s+(min|hour|hours|day|days)\s+', '', full_text, flags=re.IGNORECASE)
            words = clean_text.split()
            potential_names = []
            for word in words[:10]:
                if word and word[0].isupper() and len(word) > 2 and word not in valid_tickers_found:
                    if not re.match(r'^[A-Z]{2,5}$', word):
                        potential_names.append(word)
            
            if potential_names:
                company_name = ' '.join(potential_names[:3])
            else:
                company_name = ticker
        
        return ticker, company_name

    def _scrape_article_safely(self, url: str, max_retries: int = 2) -> Optional[str]:
            if url.startswith('/news/'):
                url = f"https://finviz.com{url}"
            elif url.startswith('/'):
                url = f"https://finviz.com{url}"

            skip_domains = [
                'youtube.com', 'youtu.be',
                'twitter.com', 'x.com', 'instagram.com', 'facebook.com', 'linkedin.com'
            ]
            premium_domains = [
                'wsj.com', 'ft.com', 'bloomberg.com', 'reuters.com',
                'barrons.com', 'economist.com'
            ]
            
            domain_lower = url.lower()
            
            if any(skip in domain_lower for skip in skip_domains):
                logger.debug(f"Skipping social media URL: {url}")
                return None
                
            if any(premium in domain_lower for premium in premium_domains):
                logger.debug(f"Skipping premium/paywall URL: {url}")
                return f"Premium content from {url}"

            for attempt in range(max_retries):
                try:
                    # Try external NewsProcessor's scrape_article first if available and not a dummy
                    if hasattr(self.processor, 'scrape_article') and self.processor.__class__.__name__ != 'NewsProcessor':
                        try:
                            article_text = self.processor.scrape_article(url)
                            if article_text and len(article_text.strip()) > 100:
                                return article_text
                        except Exception as e:
                            logger.debug(f"External NewsProcessor.scrape_article failed for {url}: {e}")

                    if attempt > 0:
                        time.sleep(1 + attempt)

                    response = self.session.get(url, timeout=10, allow_redirects=True)
                    
                    if response.status_code in [403, 401, 429]:
                        logger.debug(f"Access denied ({response.status_code}) for {url}")
                        return f"Access restricted: {url}"
                    
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')

                    for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                                        'aside', 'advertisement', 'iframe', 'form']):
                        element.decompose()

                    selectors = [
                        'article', '.article-content', '.content', '.post-content',
                        '.entry-content', '.story-body', '.article-body', 'main',
                        '[role="main"]', '.main-content', '.body-content', '.article-text'
                    ]

                    article_text = None
                    for selector in selectors:
                        elements = soup.select(selector)
                        if elements:
                            article_text = ' '.join([elem.get_text(strip=True) for elem in elements])
                            if len(article_text) > 100:
                                break

                    if not article_text or len(article_text) < 100:
                        paragraphs = soup.find_all('p')
                        article_text = ' '.join([p.get_text(strip=True) for p in paragraphs 
                                                if len(p.get_text(strip=True)) > 20])

                    if article_text and len(article_text) > 100:
                        article_text = re.sub(r'\s+', ' ', article_text).strip()
                        article_text = re.sub(r'(Subscribe|Newsletter|Cookie Policy).*?(?=\.|$)', 
                                            '', article_text, flags=re.IGNORECASE)
                        return article_text

                    # Fallback with newspaper3k
                    try:
                        from newspaper import Article # Import locally for clearer error handling
                        article = Article(url)
                        article.download()
                        article.parse()
                        if len(article.text.strip()) > 100:
                            return article.text.strip()
                    except ImportError:
                        logger.debug("newspaper3k library not installed. Skipping fallback scraping.")
                    except Exception as newspaper_error:
                        logger.debug(f"newspaper3k failed for {url} (attempt {attempt + 1}): {newspaper_error}")

                except requests.exceptions.RequestException as e:
                    logger.debug(f"Request failed for {url} (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                except Exception as e:
                    logger.debug(f"General scraping failed for {url} (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)

            logger.debug(f"Failed to scrape: {url}")
            return None

    def _process_sentiment(self, headline: str, article_text: str) -> Dict:
        try:
            full_text = f"{headline} {article_text}".lower()
            
            # Use self.processor for sentiment extraction, it will be dummy or real
            mentions, pos_keywords_existing, neg_keywords_existing, tokens = \
                self.processor.extract_mentions_and_sentiment(article_text)
            
            pos_keywords_custom = [kw for kw in self.sentiment_keywords['positive'] if kw in full_text]
            neg_keywords_custom = [kw for kw in self.sentiment_keywords['negative'] if kw in full_text]
            
            pos_keywords = list(set(pos_keywords_existing + pos_keywords_custom))
            neg_keywords = list(set(neg_keywords_existing + neg_keywords_custom))
            
            # Calculate sentiment scores - will be 0.0 if self.processor is dummy
            enhanced_sentiment = {'combined': 0.0, 'dynamic_weights': 0.0, 'ml_prediction': 0.0}
            if hasattr(self.processor, 'enhanced_processor'):
                try:
                    enhanced_sentiment = self.processor.enhanced_processor.calculate_enhanced_sentiment(full_text)
                except Exception as e:
                    logger.debug(f"Enhanced sentiment calculation failed: {e}")
            
            keyword_sentiment = (len(pos_keywords_custom) - len(neg_keywords_custom)) * 0.1
            
            scores = enhanced_sentiment.copy()
            scores['keyword_based'] = keyword_sentiment
            scores['combined'] = scores.get('combined', 0.0) + keyword_sentiment # Add keyword to combined

            return {
                'scores': scores,
                'mentions': mentions,
                'positive_keywords': pos_keywords,
                'negative_keywords': neg_keywords,
                'tokens': tokens
            }
            
        except Exception as e:
            logger.error(f"Error processing sentiment: {e}")
            return {
                'scores': {'combined': 0.0, 'dynamic_weights': 0.0, 'ml_prediction': 0.0, 'keyword_based': 0.0},
                'mentions': [],
                'positive_keywords': [],
                'negative_keywords': [],
                'tokens': 0
            }

    def _calculate_market_relevance(self, headline: str, article_text: str, 
                                     source: str, ticker: str, minutes_ago: int) -> float:
        relevance = 0.0
        
        try:
            source_weight = self.source_reliability.get(source, self.source_reliability['default'])
            relevance += source_weight * 0.3
            
            recency_weight = max(0.1, 1 - (minutes_ago / 1440))
            relevance += recency_weight * 0.2
            
            market_keywords = [
                'earnings', 'revenue', 'profit', 'guidance', 'outlook',
                'dividend', 'merger', 'acquisition', 'partnership',
                'fda approval', 'contract', 'upgrade', 'downgrade',
                'target price', 'analyst', 'recommendation', 'results'
            ]
            
            full_text = f"{headline} {article_text}".lower()
            keyword_matches = sum(1 for kw in market_keywords if kw in full_text)
            keyword_score = min(keyword_matches / len(market_keywords), 1.0)
            relevance += keyword_score * 0.3
            
            headline_words = len(headline.split())
            headline_quality = min(headline_words / 15, 1.0)
            relevance += headline_quality * 0.2
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.debug(f"Error calculating market relevance: {e}")
            return 0.5

def analyze_market_sentiment(articles: List[NewsArticle]) -> Dict:
    if not articles:
        return {
            'overall_sentiment': 0.0, 'total_articles': 0,
            'bullish_articles': 0, 'bearish_articles': 0, 'neutral_articles': 0,
            'top_tickers': [], 'average_relevance': 0.0
        }
    
    total_weight = sum(article.market_relevance for article in articles)
    
    if total_weight == 0:
        total_weight = len(articles)
        weighted_sentiment = sum(article.sentiment_scores.get('combined', 0) for article in articles) / total_weight
    else:
        weighted_sentiment = sum(
            article.sentiment_scores.get('combined', 0) * article.market_relevance
            for article in articles
        ) / total_weight
    
    bullish = [a for a in articles if a.sentiment_scores.get('combined', 0) > 0.1]
    bearish = [a for a in articles if a.sentiment_scores.get('combined', 0) < -0.1]
    neutral = [a for a in articles if abs(a.sentiment_scores.get('combined', 0)) <= 0.1]
    
    ticker_counts = defaultdict(int)
    for article in articles:
        if article.ticker != "UNKNOWN":
            ticker_counts[article.ticker] += 1
    
    top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'overall_sentiment': weighted_sentiment,
        'total_articles': len(articles),
        'bullish_articles': len(bullish),
        'bearish_articles': len(bearish),
        'neutral_articles': len(neutral),
        'top_tickers': top_tickers,
        'average_relevance': sum(a.market_relevance for a in articles) / len(articles),
        'time_range': {
            'oldest': min(a.timestamp for a in articles),
            'newest': max(a.timestamp for a in articles)
        } if articles else {}
    }

def write_finviz_articles_to_csv(articles: List[NewsArticle], filename: str = "finviz_articles.csv"):
    if not articles:
        logger.warning("No articles to write to CSV.")
        return
    
    try:
        output_data = []
        for a in articles:
            output_data.append({
                'datetime': a.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': a.ticker,
                'company': a.company_name,
                'headline': a.headline,
                'url': a.url,
                'source': a.source,
                'sentiment_combined': round(a.sentiment_scores.get('combined', 0.0), 4),
                'sentiment_dynamic': round(a.sentiment_scores.get('dynamic_weights', 0.0), 4),
                'sentiment_keyword': round(a.sentiment_scores.get('keyword_based', 0.0), 4),
                'sentiment_ml': round(a.sentiment_scores.get('ml_prediction', 0.0), 4),
                'market_relevance': round(a.market_relevance, 4),
                'mentions': ', '.join(a.mentions),
                'pos_keywords': ', '.join(a.positive_keywords),
                'neg_keywords': ', '.join(a.negative_keywords),
                'tokens': a.tokens,
                'minutes_ago': a.minutes_ago
            })
        
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(filename, index=False)
        logger.info(f"Wrote {len(articles)} articles to {filename}")
        
    except Exception as e:
        logger.error(f"Failed to write CSV file {filename}: {e}")

def main():
    logger.info("Starting Finviz News Processor")
    
    try:
        processor = FinvizNewsProcessor()
        
        init_database() # This will call the dummy if main.py not found
        
        logger.info("Fetching news from Finviz via direct scraping...")
        try:
            df = processor._scrape_finviz_main_news()
        except Exception as e:
            logger.error(f"Failed to fetch Finviz news directly: {e}", exc_info=True)
            return
        
        if df is None or df.empty:
            logger.warning("No news data received from Finviz.")
            return
        
        logger.info(f"Received {len(df)} articles from Finviz (via direct scrape)")
        
        articles = processor.parse_finviz_dataframe(df)
        
        if not articles:
            logger.warning("No articles successfully processed.")
            return
        
        write_finviz_articles_to_csv(articles)

        market_analysis = analyze_market_sentiment(articles)
        
        articles_for_db = []
        for article in articles:
            articles_for_db.append({
                'ticker': article.ticker,
                'datetime': article.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'headline': article.headline,
                'url': article.url,
                'text': article.article_text,
                'tokens': article.tokens,
                'sentiment_dynamic': article.sentiment_scores.get('dynamic_weights', 0),
                'sentiment_ml': article.sentiment_scores.get('ml_prediction', 0),
                'sentiment_keyword': article.sentiment_scores.get('keyword_based', 0),
                'sentiment_combined': article.sentiment_scores.get('combined', 0),
                'prediction_confidence': abs(article.sentiment_scores.get('combined', 0)),
                'mentions': ', '.join(article.mentions),
                'pos_keywords': ', '.join(article.positive_keywords),
                'neg_keywords': ', '.join(article.negative_keywords),
                'total_keywords': len(article.positive_keywords) + len(article.negative_keywords),
                'source': article.source,
                'market_relevance': article.market_relevance,
                'minutes_ago': article.minutes_ago
            })
        
        save_articles(articles_for_db)
        
        logger.info(f"Successfully processed {len(articles)} articles")
        logger.info(f"Overall market sentiment: {market_analysis.get('overall_sentiment', 0):.3f}")
        logger.info(f"Bullish: {market_analysis.get('bullish_articles', 0)}, "
                    f"Bearish: {market_analysis.get('bearish_articles', 0)}, "
                    f"Neutral: {market_analysis.get('neutral_articles', 0)}")
        
        if market_analysis.get('top_tickers'):
            logger.info(f"Top mentioned tickers: {market_analysis['top_tickers'][:5]}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()