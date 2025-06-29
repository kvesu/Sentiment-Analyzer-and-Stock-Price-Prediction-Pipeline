import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
import re
import logging
import json
import time
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAnalysisFramework:
    """Complete news sentiment analysis framework following the flowchart"""
    
    def __init__(self, historical_mode=True, sentiment_keywords_csv: str = "sentiment_keywords.csv"):
        self.historical_mode = historical_mode
        self.sentiment_dictionary = {}
        self.positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'profit', 'growth', 'strong', 'beat', 'positive', 'up']
        self.negative_keywords = ['bearish', 'crash', 'decline', 'loss', 'weak', 'miss', 'negative', 'down', 'fall', 'drop']
        self.load_sentiment_keywords(sentiment_keywords_csv)
        self.results_df = pd.DataFrame()
        
    def load_sentiment_keywords(self, csv_path: str):
        """Load sentiment keywords from a CSV file."""
        try:
            df = pd.read_csv(csv_path)
            new_positive = []
            new_negative = []
            
            if 'keyword' in df.columns and 'sentiment' in df.columns:
                for index, row in df.iterrows():
                    keyword = str(row['keyword']).strip().lower()
                    sentiment = str(row['sentiment']).strip().lower()
                    
                    if sentiment == 'positive':
                        new_positive.append(keyword)
                    elif sentiment == 'negative':
                        new_negative.append(keyword)
                
                if new_positive:
                    self.positive_keywords = list(set(self.positive_keywords + new_positive))
                    logger.info(f"Loaded {len(new_positive)} positive keywords from {csv_path}")
                if new_negative:
                    self.negative_keywords = list(set(self.negative_keywords + new_negative))
                    logger.info(f"Loaded {len(new_negative)} negative keywords from {csv_path}")
            else:
                logger.warning(f"CSV '{csv_path}' must contain 'keyword' and 'sentiment' columns. Using default keywords.")
        except FileNotFoundError:
            logger.warning(f"Sentiment keywords CSV '{csv_path}' not found. Using default keywords.")
        except Exception as e:
            logger.error(f"Error loading sentiment keywords from '{csv_path}': {e}. Using default keywords.")

    def load_ticker_list(self, csv_path: str) -> List[str]:
        """Load ticker list from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            ticker_col = None
            for col in ['ticker', 'symbol', 'stock', 'Ticker', 'Symbol']:
                if col in df.columns:
                    ticker_col = col
                    break
            
            if ticker_col:
                tickers = df[ticker_col].dropna().tolist()
                logger.info(f"Loaded {len(tickers)} tickers from CSV")
                return tickers
            else:
                logger.error("No ticker column found in CSV")
                return []
        except Exception as e:
            logger.error(f"Error loading ticker CSV: {e}")
            return []
    
    def fetch_news_articles(self, ticker: str) -> List[Dict]:
        """Fetch news articles from Finviz (historical mode) or real-time source"""
        articles = []
        try:
            if self.historical_mode:
                url = f"https://finviz.com/quote.ashx?t={ticker}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                news_table = soup.find('table', {'class': 'fullview-news-outer'})
                if news_table:
                    for row in news_table.find_all('tr'):
                        cols = row.find_all('td')
                        if len(cols) >= 2:
                            date_cell = cols[0]
                            news_cell = cols[1]
                            
                            date_text = date_cell.get_text().strip()
                            
                            link = news_cell.find('a')
                            if link:
                                headline = link.get_text().strip()
                                article_url = link.get('href')
                                
                                articles.append({
                                    'headline': headline,
                                    'url': article_url,
                                    'date': date_text,
                                    'ticker': ticker
                                })
            else:
                logger.info(f"Real-time mode not fully implemented for {ticker}")
                
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            
        return articles
    
    def filter_recent_articles(self, articles: List[Dict], days: int = 7) -> List[Dict]:
        """Filter articles to last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered = []
        
        for article in articles:
            filtered.append(article)
            
        logger.info(f"Filtered to {len(filtered)} recent articles (Note: Full date parsing for filtering not implemented)")
        return filtered
    
    def extract_article_text(self, url: str) -> Optional[str]:
        """Extract article text using requests + BeautifulSoup"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.extract()
            
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            
            if len(text) > 100:
                return text
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network or HTTP error extracting text from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {e}")
            return None
    
    def preprocess_text(self, text: str) -> List[str]:
        """Clean, tokenize, and normalize text"""
        if not text:
            return []
        
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        words = [word.strip() for word in text.split() if len(word.strip()) > 2]
        return words
    
    def check_ticker_mention(self, text: str, ticker: str) -> bool:
        """Check if ticker is mentioned in text"""
        if not text or not ticker:
            return False
        patterns = [
            re.escape(ticker.lower()),
            re.escape(ticker.lower()) + r'\s',
            r'\$' + re.escape(ticker.upper()),
            r'\b' + re.escape(ticker.upper()) + r'\b'
        ]
        return any(re.search(pattern, text.lower()) for pattern in patterns)
    
    def get_stock_prices(self, ticker: str, article_date: datetime) -> Dict:
        """Get historical stock prices from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)

            # Fetch a wide enough range to include baseline to EOW
            start_date = article_date - timedelta(days=1)
            end_date = article_date + timedelta(days=7)
            hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

            if hist.empty:
                logger.warning(f"No historical data found for {ticker} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
                return {}

            prices = {
                'baseline': None,
                '1h_after': None,
                '4h_after': None,
                'eod': None,
                'eow': None
            }

            # Get the baseline price (closing price of article date or next available)
            article_day = article_date.date()
            baseline_row = hist[hist.index.date == article_day]
            if not baseline_row.empty:
                prices['baseline'] = baseline_row['Close'].iloc[0]
            else:
                future_data = hist[hist.index.date > article_day]
                if not future_data.empty:
                    prices['baseline'] = future_data['Close'].iloc[0]
                    logger.info(f"Baseline for {ticker} taken from next trading day after {article_day}")
                else:
                    logger.warning(f"Could not establish a baseline price for {ticker} on or after {article_day}")
                    return {}

            # Set EOD price (closing price of the same day if available)
            if not baseline_row.empty:
                prices['eod'] = baseline_row['Close'].iloc[0]
            else:
                prices['eod'] = prices['baseline']

            # Approximate intraday prices: use next day's close for "1h_after" and "4h_after"
            intraday_candidates = hist[hist.index.date > article_day]
            if not intraday_candidates.empty:
                prices['1h_after'] = intraday_candidates['Close'].iloc[0]
                if len(intraday_candidates) >= 2:
                    prices['4h_after'] = intraday_candidates['Close'].iloc[1]
                else:
                    prices['4h_after'] = intraday_candidates['Close'].iloc[0]  # fallback to same as 1h
            else:
                logger.warning(f"No intraday/following data available for {ticker} to estimate 1h/4h after.")

            # Get EOW price (Friday or latest price in 5 trading days)
            hist_dates = hist.index.date
            if article_day in hist_dates:
                baseline_idx = list(hist_dates).index(article_day)
            else:
                baseline_idx = 0  # fallback

            eow_idx = min(baseline_idx + 5, len(hist) - 1)
            prices['eow'] = hist['Close'].iloc[eow_idx]

            return {k: v for k, v in prices.items() if v is not None}

        except Exception as e:
            logger.error(f"Error getting prices for {ticker}: {e}")
            return {}

    def calculate_price_changes(self, prices: Dict) -> Dict:
        """Calculate percentage changes for each interval"""
        changes = {}
        baseline = prices.get('baseline')
        
        if not baseline:
            return changes
        
        for interval, price in prices.items():
            if interval != 'baseline' and price is not None:
                pct_change = ((price - baseline) / baseline) * 100
                changes[f'pct_change_{interval}'] = round(pct_change, 4)
        
        return changes
    
    def scan_keywords(self, words: List[str]) -> Dict:
        """Scan for positive and negative keywords"""
        pos_found = [word for word in words if word in self.positive_keywords]
        neg_found = [word for word in words if word in self.negative_keywords]
        
        return {
            'positive_keywords': pos_found,
            'negative_keywords': neg_found,
            'pos_count': len(pos_found),
            'neg_count': len(neg_found)
        }
    
    def update_sentiment_dictionary(self, words: List[str], price_direction: int):
        """Dynamic learning module - update word sentiment scores"""
        for word in words:
            if word not in self.sentiment_dictionary:
                self.sentiment_dictionary[word] = {
                    'pos_count': 0,
                    'neg_count': 0,
                    'total_count': 0,
                    'sentiment_score': 0.0
                }
                
            self.sentiment_dictionary[word]['total_count'] += 1
            
            if price_direction > 0:
                self.sentiment_dictionary[word]['pos_count'] += 1
            elif price_direction < 0:
                self.sentiment_dictionary[word]['neg_count'] += 1
            
            pos = self.sentiment_dictionary[word]['pos_count']
            neg = self.sentiment_dictionary[word]['neg_count']
            total = self.sentiment_dictionary[word]['total_count']
            
            if total > 0:
                sentiment_score = (pos - neg) / total
                self.sentiment_dictionary[word]['sentiment_score'] = sentiment_score
    
    def calculate_article_sentiment(self, words: List[str]) -> float:
        """Calculate overall article sentiment score"""
        if not words:
            return 0.0
        
        total_score = 0
        scored_words = 0
        
        for word in words:
            if word in self.sentiment_dictionary:
                score = self.sentiment_dictionary[word].get('sentiment_score', 0)
                total_score += score
                scored_words += 1
        
        return total_score / max(scored_words, 1)
    
    def parse_finviz_date(self, date_str: str, current_year: int) -> Optional[datetime]:
        """
        Parses a date string from Finviz.
        Assumes the current year for dates that don't specify a year.
        """
        date_str = date_str.strip()
        now = datetime.now()
        
        if "Today" in date_str:
            time_part = date_str.split(' ')[1]
            try:
                return datetime.strptime(f"{now.strftime('%Y-%m-%d')} {time_part}", "%Y-%m-%d %I:%M%p")
            except ValueError:
                return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif "Yesterday" in date_str:
            yesterday = now - timedelta(days=1)
            time_part = date_str.split(' ')[1]
            try:
                return datetime.strptime(f"{yesterday.strftime('%Y-%m-%d')} {time_part}", "%Y-%m-%d %I:%M%p")
            except ValueError:
                return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            try:
                return datetime.strptime(date_str, "%b-%d-%y %I:%M%p")
            except ValueError:
                try:
                    full_date_str = f"{date_str} {current_year}"
                    return datetime.strptime(full_date_str, "%b %d %I:%M%p %Y")
                except ValueError:
                    logger.warning(f"Could not parse date string: '{date_str}'")
                    return None

    def process_ticker(self, ticker: str) -> pd.DataFrame:
        """Process all articles for a single ticker"""
        logger.info(f"Processing ticker: {ticker}")
        
        articles = self.fetch_news_articles(ticker)
        if not articles:
            logger.warning(f"No articles found for {ticker}")
            return pd.DataFrame()
        
        articles = self.filter_recent_articles(articles)
        
        ticker_results = []
        current_year = datetime.now().year
        
        for article in articles:
            try:
                headline = article.get('headline', '')
                url = article.get('url', '')
                date_str = article.get('date', '')
                
                article_datetime = self.parse_finviz_date(date_str, current_year)
                if not article_datetime:
                    logger.warning(f"Skipping article due to unparsable date: {date_str} - {headline[:50]}")
                    continue

                article_text = self.extract_article_text(url)
                if not article_text:
                    logger.warning(f"Could not extract text from {url} for headline: {headline[:50]}...")
                    continue
                
                words = self.preprocess_text(article_text)
                if not words:
                    logger.warning(f"No meaningful words extracted from {url}")
                    continue
                
                ticker_found = self.check_ticker_mention(article_text, ticker)
                
                prices = self.get_stock_prices(ticker, article_datetime)
                
                price_changes = self.calculate_price_changes(prices)
                
                eod_change = price_changes.get('pct_change_eod', 0)
                price_direction = 1 if eod_change > 0 else -1 if eod_change < 0 else 0
                
                keyword_analysis = self.scan_keywords(words)
                
                self.update_sentiment_dictionary(words, price_direction)
                
                sentiment_score = self.calculate_article_sentiment(words)
                
                result = {
                    'date': article_datetime.strftime('%Y-%m-%d %H:%M'),
                    'ticker': ticker,
                    'headline': headline,
                    'url': url,
                    'text': article_text[:500] + '...' if len(article_text) > 500 else article_text,
                    'sentiment_score': round(sentiment_score, 4),
                    'ticker_found': ticker_found,
                    'pos_keywords_found': keyword_analysis['pos_count'],
                    'neg_keywords_found': keyword_analysis['neg_count'],
                    **price_changes,
                    'price_direction': price_direction
                }
                
                ticker_results.append(result)
                
                logger.info(f"Processed: {headline[:50]}... | Sentiment: {sentiment_score:.3f} | Price Change (EOD): {eod_change:+.2f}%")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing article '{article.get('headline', 'N/A')}' for {ticker}: {e}")
                continue
        
        return pd.DataFrame(ticker_results)
    
    def run_analysis(self) -> pd.DataFrame:
        """Run complete analysis using hardcoded 'finviz.csv' as ticker source"""
        ticker_csv_path = "finviz.csv"  # only finviz.csv is used here
        logger.info(f"Loading tickers from {ticker_csv_path}")
        
        tickers = self.load_ticker_list(ticker_csv_path)
        if not tickers:
            logger.error("No tickers loaded")
            return pd.DataFrame()
        
        all_results = []
        
        for ticker in tickers:
            ticker_df = self.process_ticker(ticker)
            if not ticker_df.empty:
                all_results.append(ticker_df)
        
        if all_results:
            self.results_df = pd.concat(all_results, ignore_index=True)
            logger.info(f"Analysis complete. Processed {len(self.results_df)} articles across {len(tickers)} tickers")
        else:
            logger.warning("No results generated")
            self.results_df = pd.DataFrame()
        
        return self.results_df


    def save_results(self, csv_path: str = "news_analysis_results.csv"):
        """Save combined results to CSV"""
        if not self.results_df.empty:
            self.results_df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
        
        dict_path = "sentiment_dictionary.json"
        with open(dict_path, 'w') as f:
            json.dump(self.sentiment_dictionary, f, indent=2)
        logger.info(f"Sentiment dictionary saved to {dict_path}")
    
    def generate_summary(self) -> Dict:
        """Generate cross-ticker analysis and summary"""
        if self.results_df.empty:
            return {}
        
        summary = {
            'total_articles': len(self.results_df),
            'tickers_analyzed': self.results_df['ticker'].nunique(),
            'avg_sentiment': self.results_df['sentiment_score'].mean(),
            'positive_articles': len(self.results_df[self.results_df['sentiment_score'] > 0]),
            'negative_articles': len(self.results_df[self.results_df['sentiment_score'] < 0]),
            'avg_price_change_eod': self.results_df['pct_change_eod'].mean() if 'pct_change_eod' in self.results_df.columns else 0
        }
        
        logger.info("Summary generated")
        return summary

if __name__ == "__main__":
    # Initialize framework (expects existing finviz.csv and sentiment_keywords.csv files)
    analyzer = NewsAnalysisFramework(historical_mode=True, sentiment_keywords_csv="sentiment_keywords.csv")

    # Run analysis
    results = analyzer.run_analysis()

    if not results.empty:
        analyzer.save_results()

        summary = analyzer.generate_summary()
        print("\nAnalysis Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        print("No results generated")

