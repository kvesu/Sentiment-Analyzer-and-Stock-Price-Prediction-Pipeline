import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import json
import logging
import sqlite3
from datetime import datetime, timedelta
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)

class DynamicSentimentLearner:
    def __init__(self):
        self.word_performance = defaultdict(lambda: {'positive_outcomes': 0, 'negative_outcomes': 0, 'total': 0})
        self.bigram_performance = defaultdict(lambda: {'positive_outcomes': 0, 'negative_outcomes': 0, 'total': 0})
        self.sentiment_model = None
        self.vectorizer = None
        self.logger = logging.getLogger(__name__)
        self.sentiment_weights = {}
    
    def load_sentiment_keywords_from_csv(self, csv_path="sentiment_keywords.csv"):
        """Load basic sentiment keywords from a CSV file and convert them into weighted entries"""
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                keyword = str(row["keyword"]).lower().strip()
                sentiment = str(row["sentiment"]).lower().strip()
                if sentiment == "positive":
                    weight = 1.0
                elif sentiment == "negative":
                    weight = -1.0
                else:
                    continue
                self.sentiment_weights[keyword] = {
                    "weight": weight,
                    "confidence": 1.0,
                    "type": "word",
                    "occurrences": 1
                }
            self.logger.info(f"Loaded {len(self.sentiment_weights)} keywords from CSV")
        except Exception as e:
            self.logger.warning(f"Failed to load sentiment keywords from CSV: {e}")
    
    def analyze_historical_performance(self, articles_df):
        """Analyze historical performance of words and bigrams"""
        results = {}
        
        # Load sentiment weights from CSV first
        self.load_sentiment_keywords_from_csv("sentiment_keywords.csv")

        # Filter articles with valid price data
        valid_articles = articles_df.dropna(subset=['pct_change_eod', 'tokens'])

        if len(valid_articles) < 10:
            self.logger.warning("Insufficient data for sentiment analysis")
            results['sentiment_weights'] = self.sentiment_weights
            return results

        # Analyze performance based on article data
        word_stats = self._analyze_word_performance(valid_articles)
        bigram_stats = self._analyze_bigram_performance(valid_articles)

        results['word_analysis'] = word_stats
        results['bigram_analysis'] = bigram_stats

        # Merge dynamically generated sentiment weights
        dynamic_weights = self._generate_sentiment_weights(word_stats, bigram_stats)

        # Merge CSV weights and dynamic weights (give priority to dynamic if overlap)
        merged_weights = {**self.sentiment_weights, **dynamic_weights}
        self.sentiment_weights = merged_weights
        results['sentiment_weights'] = merged_weights

        # Train ML model
        model_performance = self._train_sentiment_model(valid_articles)
        results['model_performance'] = model_performance

        return results

    def _analyze_word_performance(self, articles_df):
        """Analyze individual word performance"""
        word_stats = {}
        
        for _, article in articles_df.iterrows():
            # Handle different token formats
            if isinstance(article['tokens'], str):
                words = article['tokens'].split()
            elif isinstance(article['tokens'], list):
                words = article['tokens']
            else:
                continue
                
            price_change = article['pct_change_eod']
            
            # Define positive/negative outcomes
            is_positive_outcome = price_change > 0
            
            for word in set(words):  # Use set to avoid double counting
                word = word.lower().strip()
                if len(word) < 3:  # Skip very short words
                    continue
                    
                if word not in word_stats:
                    word_stats[word] = {
                        'positive_outcomes': 0,
                        'negative_outcomes': 0,
                        'total_occurrences': 0,
                        'avg_price_change': 0,
                        'price_changes': []
                    }
                
                word_stats[word]['total_occurrences'] += 1
                word_stats[word]['price_changes'].append(price_change)
                
                if is_positive_outcome:
                    word_stats[word]['positive_outcomes'] += 1
                else:
                    word_stats[word]['negative_outcomes'] += 1
        
        # Calculate statistics
        for word, stats in word_stats.items():
            if stats['total_occurrences'] >= 3:  # Minimum occurrence threshold
                stats['avg_price_change'] = np.mean(stats['price_changes'])
                stats['positive_ratio'] = stats['positive_outcomes'] / stats['total_occurrences']
                stats['predictive_power'] = abs(stats['positive_ratio'] - 0.5) * 2  # 0-1 scale
                stats['confidence'] = min(stats['total_occurrences'] / 10, 1.0)  # Confidence based on frequency
        
        # Filter and sort by predictive power
        filtered_stats = {
            word: stats for word, stats in word_stats.items() 
            if stats['total_occurrences'] >= 3
        }
        
        return dict(sorted(filtered_stats.items(), 
                          key=lambda x: x[1]['predictive_power'], reverse=True))
    
    def _analyze_bigram_performance(self, articles_df):
        """Analyze bigram performance"""
        bigram_stats = {}
        
        for _, article in articles_df.iterrows():
            # Handle different token formats
            if isinstance(article['tokens'], str):
                words = article['tokens'].split()
            elif isinstance(article['tokens'], list):
                words = article['tokens']
            else:
                continue
                
            words = [w.lower().strip() for w in words if len(w) >= 3]
            price_change = article['pct_change_eod']
            is_positive_outcome = price_change > 0
            
            # Generate bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                
                if bigram not in bigram_stats:
                    bigram_stats[bigram] = {
                        'positive_outcomes': 0,
                        'negative_outcomes': 0,
                        'total_occurrences': 0,
                        'price_changes': []
                    }
                
                bigram_stats[bigram]['total_occurrences'] += 1
                bigram_stats[bigram]['price_changes'].append(price_change)
                
                if is_positive_outcome:
                    bigram_stats[bigram]['positive_outcomes'] += 1
                else:
                    bigram_stats[bigram]['negative_outcomes'] += 1
        
        # Calculate statistics for bigrams
        for bigram, stats in bigram_stats.items():
            if stats['total_occurrences'] >= 2:  # Lower threshold for bigrams
                stats['avg_price_change'] = np.mean(stats['price_changes'])
                stats['positive_ratio'] = stats['positive_outcomes'] / stats['total_occurrences']
                stats['predictive_power'] = abs(stats['positive_ratio'] - 0.5) * 2
                stats['confidence'] = min(stats['total_occurrences'] / 5, 1.0)
        
        return {k: v for k, v in bigram_stats.items() if v['total_occurrences'] >= 2}
    
    def _generate_sentiment_weights(self, word_stats, bigram_stats):
        """Generate sentiment weights for words and bigrams"""
        sentiment_weights = {}
        
        # Process words
        for word, stats in word_stats.items():
            if stats['total_occurrences'] >= 3:
                # Weight based on average price change and predictive power
                weight = stats['avg_price_change'] * stats['predictive_power']
                sentiment_weights[word] = {
                    'weight': weight,
                    'confidence': stats['confidence'],
                    'type': 'word',
                    'occurrences': stats['total_occurrences']
                }
        
        # Process bigrams
        for bigram, stats in bigram_stats.items():
            if stats['total_occurrences'] >= 2:
                weight = stats['avg_price_change'] * stats['predictive_power']
                sentiment_weights[bigram] = {
                    'weight': weight,
                    'confidence': stats['confidence'],
                    'type': 'bigram',
                    'occurrences': stats['total_occurrences']
                }
        
        return sentiment_weights
    
    def _train_sentiment_model(self, articles_df):
        """Train a machine learning model for sentiment prediction"""
        try:
            # Prepare data
            X = []
            y = []
            
            for _, article in articles_df.iterrows():
                if isinstance(article['tokens'], str):
                    X.append(article['tokens'])
                elif isinstance(article['tokens'], list):
                    X.append(' '.join(article['tokens']))
                else:
                    continue
                    
                y.append(1 if article['pct_change_eod'] > 0 else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Check for sufficient data
            if len(X) < 10:
                return {'model_trained': False, 'error': 'Insufficient data'}
            
            # Check if we have both classes
            if len(np.unique(y)) < 2:
                return {'model_trained': False, 'error': 'Need both positive and negative examples'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2
            )
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.sentiment_model = LogisticRegression(random_state=42, max_iter=1000)
            self.sentiment_model.fit(X_train_vec, y_train)
            
            # Evaluate
            train_score = self.sentiment_model.score(X_train_vec, y_train)
            test_score = self.sentiment_model.score(X_test_vec, y_test)
            
            return {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'model_trained': True,
                'feature_count': X_train_vec.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {'model_trained': False, 'error': str(e)}
    
    def predict_sentiment(self, text):
        """Predict sentiment using the trained model"""
        if not self.sentiment_model or not self.vectorizer:
            return 0.5  # Neutral
        
        try:
            text_vec = self.vectorizer.transform([text])
            probability = self.sentiment_model.predict_proba(text_vec)[0][1]  # Probability of positive
            return probability
        except Exception as e:
            self.logger.error(f"Sentiment prediction failed: {e}")
            return 0.5
    
    def save_analysis_results(self, results, filename="word_analysis_results.json"):
        """Save analysis results to JSON file"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert all numpy types
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = deep_convert(results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        self.logger.info(f"Analysis results saved to {filename}")


class NewsProcessor:
    """Base NewsProcessor class"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_mentions_and_sentiment(self, article_text, ticker):
        """Basic implementation - replace with your actual method"""
        # This is a placeholder - implement your actual logic
        words = article_text.lower().split()
        mentions = [ticker.lower()]
        
        # Simple positive/negative word lists
        positive_words = ['growth', 'profit', 'gain', 'increase', 'rise', 'bullish', 'positive', 'strong']
        negative_words = ['loss', 'decline', 'fall', 'decrease', 'drop', 'bearish', 'negative', 'weak']
        
        pos_kw = [word for word in words if word in positive_words]
        neg_kw = [word for word in words if word in negative_words]
        
        return mentions, pos_kw, neg_kw, ' '.join(words)
    
    def get_price_data(self, ticker, parsed_dt):
        """Get price data for the ticker around the given datetime"""
        try:
            # This is a simplified version - implement your actual price data logic
            stock = yf.Ticker(ticker)
            
            # Get data for a few days around the article date
            start_date = parsed_dt - timedelta(days=1)
            end_date = parsed_dt + timedelta(days=7)
            
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            # Calculate price changes at different intervals
            base_price = hist['Close'].iloc[0]
            
            price_data = {}
            
            # End of day change
            if len(hist) > 1:
                eod_price = hist['Close'].iloc[1]
                price_data['pct_change_eod'] = (eod_price - base_price) / base_price * 100
            
            # End of week change
            if len(hist) > 5:
                eow_price = hist['Close'].iloc[5]
                price_data['pct_change_eow'] = (eow_price - base_price) / base_price * 100
            
            return price_data
            
        except Exception as e:
            self.logger.error(f"Error getting price data for {ticker}: {e}")
            return None
    
    def calculate_dynamic_sentiment(self, text):
        """Calculate sentiment using dynamic weights - implement your actual method"""
        # Placeholder implementation
        words = text.lower().split()
        positive_words = ['growth', 'profit', 'gain', 'increase', 'rise', 'bullish', 'positive', 'strong']
        negative_words = ['loss', 'decline', 'fall', 'decrease', 'drop', 'bearish', 'negative', 'weak']
        
        pos_score = sum(1 for word in words if word in positive_words)
        neg_score = sum(1 for word in words if word in negative_words)
        
        if pos_score + neg_score == 0:
            return 0
        
        return (pos_score - neg_score) / (pos_score + neg_score)


class EnhancedNewsProcessor(NewsProcessor):
    def __init__(self):
        super().__init__()
        self.sentiment_learner = None
        self.sentiment_weights = {}
        self.load_sentiment_learner()
    
    def load_sentiment_learner(self):
        """Load the trained sentiment learner"""
        try:
            self.sentiment_learner = DynamicSentimentLearner()
            
            # Load existing analysis results if available
            try:
                with open("word_analysis_results.json", 'r') as f:
                    results = json.load(f)
                    self.sentiment_weights = results.get('sentiment_weights', {})
                    self.logger.info("Loaded existing sentiment analysis results")
            except FileNotFoundError:
                self.sentiment_weights = {}
                self.logger.info("No existing sentiment analysis results found")
                
        except Exception as e:
            self.logger.warning(f"Sentiment learner not available: {e}")
            self.sentiment_learner = None
    
    def calculate_enhanced_sentiment(self, text):
        """Calculate sentiment using multiple methods"""
        sentiment_scores = {}
        
        # Method 1: Dynamic weights (existing)
        sentiment_scores['dynamic_weights'] = self.calculate_dynamic_sentiment(text)
        
        # Method 2: ML model prediction
        if self.sentiment_learner:
            sentiment_scores['ml_prediction'] = self.sentiment_learner.predict_sentiment(text)
        else:
            sentiment_scores['ml_prediction'] = None
        
        # Method 3: Keyword-based scoring
        sentiment_scores['keyword_based'] = self.calculate_keyword_sentiment(text)
        
        # Method 4: Weighted combination
        sentiment_scores['combined'] = self.combine_sentiment_scores(sentiment_scores)
        
        return sentiment_scores
    
    def calculate_keyword_sentiment(self, text):
        """Calculate sentiment based on keyword matching"""
        if not self.sentiment_weights:
            return 0
        
        words = text.lower().split()
        positive_score = 0
        negative_score = 0
        total_weight = 0
        
        # Check individual words
        for word in words:
            if word in self.sentiment_weights:
                weight = self.sentiment_weights[word]['weight']
                confidence = self.sentiment_weights[word]['confidence']
                
                if weight > 0:
                    positive_score += weight * confidence
                else:
                    negative_score += abs(weight) * confidence
                
                total_weight += confidence
        
        # Check bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in self.sentiment_weights:
                weight = self.sentiment_weights[bigram]['weight']
                confidence = self.sentiment_weights[bigram]['confidence']
                
                if weight > 0:
                    positive_score += weight * confidence * 1.2  # Boost bigrams
                else:
                    negative_score += abs(weight) * confidence * 1.2
                
                total_weight += confidence
        
        if total_weight == 0:
            return 0
        
        # Return normalized score (-1 to 1)
        net_score = (positive_score - negative_score) / total_weight
        return max(-1, min(1, net_score))
    
    def combine_sentiment_scores(self, sentiment_scores):
        """Combine multiple sentiment scores with weights"""
        weights = {
            'dynamic_weights': 0.3,
            'ml_prediction': 0.4,
            'keyword_based': 0.3
        }
        
        combined_score = 0
        total_weight = 0
        
        for method, score in sentiment_scores.items():
            if method in weights and score is not None:
                weight = weights[method]
                combined_score += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return combined_score / total_weight
    
    def enhanced_article_processing(self, article_text, headline, ticker, parsed_dt):
        """Process article with enhanced sentiment analysis"""
        # Extract mentions and basic sentiment
        mentions, pos_kw, neg_kw, tokens = self.extract_mentions_and_sentiment(article_text, ticker)
        
        # Calculate enhanced sentiment
        full_text = f"{headline} {article_text}"
        sentiment_scores = self.calculate_enhanced_sentiment(full_text)
        
        # Get price data
        price_data = self.get_price_data(ticker, parsed_dt)
        
        # Create enhanced article entry
        article_entry = {
            "ticker": ticker,
            "headline": headline,
            "text": article_text,
            "tokens": tokens,
            "mentions": ", ".join(mentions),
            "pos_keywords": ", ".join(pos_kw),
            "neg_keywords": ", ".join(neg_kw),
            "total_keywords": len(pos_kw) + len(neg_kw),
            
            # Enhanced sentiment scores
            "sentiment_dynamic": sentiment_scores.get('dynamic_weights', 0),
            "sentiment_ml": sentiment_scores.get('ml_prediction', 0),
            "sentiment_keyword": sentiment_scores.get('keyword_based', 0),
            "sentiment_combined": sentiment_scores.get('combined', 0),
            
            # Traditional sentiment score for comparison
            "sentiment_score_traditional": len(pos_kw) - len(neg_kw),
            
            # Prediction based on combined sentiment
            "predicted_direction": "Positive" if sentiment_scores.get('combined', 0) > 0 else "Negative",
            "prediction_confidence": abs(sentiment_scores.get('combined', 0))
        }
        
        # Add price data if available
        if price_data:
            article_entry.update(price_data)
            
            # Calculate prediction accuracy for different sentiment methods
            intervals = ['eod', 'eow']  # Removed 1h, 4h as they're not implemented
            for interval in intervals:
                pct_change_key = f'pct_change_{interval}'
                if pct_change_key in price_data and price_data[pct_change_key] is not None:
                    actual_positive = price_data[pct_change_key] > 0
                    
                    # Check accuracy for each sentiment method
                    for method in ['dynamic', 'ml', 'keyword', 'combined']:
                        sentiment_key = f'sentiment_{method}'
                        if sentiment_key in article_entry and article_entry[sentiment_key] is not None:
                            predicted_positive = article_entry[sentiment_key] > 0
                            article_entry[f'accuracy_{method}_{interval}'] = (
                                predicted_positive == actual_positive
                            )
        
        return article_entry


# Usage example
def run_sentiment_analysis():
    """Run the complete sentiment analysis pipeline"""
    # Load existing data
    try:
        conn = sqlite3.connect("articles.db")
        articles_df = pd.read_sql("SELECT * FROM articles", conn)
        conn.close()
        
        if articles_df.empty:
            print("No articles found in database")
            return
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize learner
    learner = DynamicSentimentLearner()
    
    # Run analysis
    print("Starting sentiment analysis...")
    results = learner.analyze_historical_performance(articles_df)
    
    # Save results
    learner.save_analysis_results(results)
    
    # Print summary
    print(f"Analysis complete:")
    print(f"- Words analyzed: {len(results['word_analysis'])}")
    print(f"- Bigrams analyzed: {len(results['bigram_analysis'])}")
    print(f"- Sentiment weights generated: {len(results['sentiment_weights'])}")
    
    if results['model_performance'].get('model_trained'):
        print(f"- Model accuracy: {results['model_performance']['test_accuracy']:.3f}")
    else:
        print(f"- Model training failed: {results['model_performance'].get('error', 'Unknown error')}")


def process_articles_with_enhanced_sentiment():
    """Process articles using the enhanced sentiment framework"""
    processor = EnhancedNewsProcessor()
    
    # Example usage:
    # article_entry = processor.enhanced_article_processing(
    #     article_text="Company reports strong quarterly earnings with 20% growth",
    #     headline="Strong Q4 Results",
    #     ticker="AAPL",
    #     parsed_dt=datetime.now()
    # )
    
    return processor

if __name__ == "__main__":
    run_sentiment_analysis()