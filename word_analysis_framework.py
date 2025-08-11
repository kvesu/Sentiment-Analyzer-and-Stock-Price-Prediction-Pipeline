import pandas as pd
import numpy as np
import os
import joblib
import logging
import json
import re
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

class DynamicSentimentLearner:
    """
    A class to dynamically learn and analyze sentiment from text data,
    specifically financial news articles, and their correlation with
    stock price movements.
    """
    def __init__(self, min_word_frequency: int = 3, min_bigram_frequency: int = 2, max_features: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.vectorizer = None
        self.sentiment_model = None
        self.sentiment_weights = {}
        self.feature_selector = None
        self.scaler = None
        self.optimal_threshold = 0.5

        # New attributes to fix the TypeError
        self.min_word_frequency = min_word_frequency
        self.min_bigram_frequency = min_bigram_frequency
        self.max_features = max_features
        
        self._load_existing_models()

    def _load_existing_models(self):
        """Load pre-trained model and vectorizer if they exist"""
        try:
            if os.path.exists("sentiment_model_enhanced.pkl") and os.path.exists("vectorizer_enhanced.pkl"):
                self.sentiment_model = joblib.load("sentiment_model_enhanced.pkl")
                self.vectorizer = joblib.load("vectorizer_enhanced.pkl")
                if os.path.exists("feature_selector_enhanced.pkl"):
                    self.feature_selector = joblib.load("feature_selector_enhanced.pkl")
                if os.path.exists("scaler_enhanced.pkl"):
                    self.scaler = joblib.load("scaler_enhanced.pkl")
                # Load optimal threshold
                try:
                    with open("optimal_threshold.json", "r") as f:
                        self.optimal_threshold = json.load(f)["optimal_threshold"]
                        self.logger.info(f"Loaded optimal threshold: {self.optimal_threshold:.3f}")
                except (FileNotFoundError, json.JSONDecodeError):
                    self.logger.info("Optimal threshold file not found or invalid, defaulting to 0.5")

                self.logger.info("Loaded existing enhanced sentiment model and vectorizer")
        except Exception as e:
            self.logger.warning(f"Failed to load existing models: {e}")
            self.sentiment_model = None
            self.vectorizer = None
            self.feature_selector = None
            self.scaler = None

    def load_sentiment_keywords_from_csv(self, csv_path="sentiment_keywords.csv"):
        """
        Load and validate sentiment keywords from a CSV file.
        Assigns a strength multiplier to each keyword based on its intensity,
        and also checks for a 'strength' column in the CSV for manual weighting.
        """
        try:
            if not os.path.exists(csv_path):
                self.logger.warning(f"CSV file {csv_path} not found, skipping keyword loading")
                return

            df = pd.read_csv(csv_path)

            # Validate required columns
            if 'keyword' not in df.columns or 'sentiment' not in df.columns:
                self.logger.error("CSV must contain 'keyword' and 'sentiment' columns")
                return

            has_strength_column = 'strength' in df.columns
            if has_strength_column:
                self.logger.info("Found 'strength' column in CSV. Using custom weights.")

            keywords_loaded = 0
            duplicates = set()
            invalid_sentiments = set()

            for _, row in df.iterrows():
                keyword = str(row["keyword"]).lower().strip()
                sentiment = str(row["sentiment"]).lower().strip()

                # Skip empty or invalid keywords
                if not keyword or keyword == 'nan' or len(keyword) < 2:
                    continue

                # Check for duplicates
                if keyword in self.sentiment_weights:
                    duplicates.add(keyword)
                    continue

                # Validate sentiment value
                if sentiment == "positive":
                    weight = 1.0
                elif sentiment == "negative":
                    weight = -1.0
                else:
                    invalid_sentiments.add(f"{keyword}: {sentiment}")
                    continue

                # Apply strength from code logic and optional CSV column
                strength_multiplier = self._get_keyword_strength(keyword)
                if has_strength_column:
                    # Use a default of 1.0 if the CSV value is missing or invalid
                    try:
                        strength_multiplier *= float(row["strength"])
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid strength value for '{keyword}', defaulting to 1.0.")

                self.sentiment_weights[keyword] = {
                    "weight": weight * strength_multiplier,
                    "confidence": 1.0,
                    "type": "phrase" if " " in keyword else "word",
                    "occurrences": 1,
                    "source": "csv"
                }
                keywords_loaded += 1

            self.logger.info(f"Loaded {keywords_loaded} keywords from CSV")
            if duplicates:
                self.logger.warning(f"Found {len(duplicates)} duplicate keywords")
            if invalid_sentiments:
                self.logger.warning(f"Found {len(invalid_sentiments)} invalid sentiments")

        except Exception as e:
            self.logger.error(f"Failed to load sentiment keywords from CSV: {e}")

    def _get_keyword_strength(self, keyword: str) -> float:
        """Assign strength multipliers based on keyword intensity"""
        # Strong positive indicators
        strong_positive = {'surge', 'soar', 'rocket', 'breakout', 'breakthrough', 'massive',
                             'exceptional', 'outstanding', 'stellar', 'explosive'}

        # Strong negative indicators
        strong_negative = {'crash', 'plunge', 'collapse', 'disaster', 'catastrophic',
                             'devastating', 'plummet', 'nosedive', 'bankruptcy'}

        # Moderate indicators
        moderate_words = {'beat', 'miss', 'upgrade', 'downgrade', 'rally', 'decline'}

        # Check for strength indicators
        if any(word in keyword for word in strong_positive | strong_negative):
            return 1.5
        elif any(word in keyword for word in moderate_words):
            return 1.2
        else:
            return 1.0

    def analyze_historical_performance(self, articles_df):
        """Enhanced analysis with better feature filtering"""
        results = {}

        # Load sentiment weights from CSV first
        self.load_sentiment_keywords_from_csv("sentiment_keywords.csv")

        if articles_df.empty:
            self.logger.warning("Empty articles dataframe provided")
            results['sentiment_weights'] = self.sentiment_weights
            return results

        # Enhanced data validation
        required_columns = ['pct_change_eod', 'tokens']
        missing_columns = [col for col in required_columns if col not in articles_df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            results['sentiment_weights'] = self.sentiment_weights
            return results

        # Enhanced data cleaning
        articles_df = articles_df.copy()
        articles_df['tokens'] = articles_df['tokens'].fillna('')
        articles_df['pct_change_eod'] = pd.to_numeric(articles_df['pct_change_eod'], errors='coerce')

        # Filter articles with valid data
        valid_articles = articles_df.dropna(subset=['pct_change_eod'])
        valid_articles = valid_articles[valid_articles['tokens'].str.len() > 0]
        
        # Remove extreme outliers (beyond 3 standard deviations)
        price_std = valid_articles['pct_change_eod'].std()
        price_mean = valid_articles['pct_change_eod'].mean()
        valid_articles = valid_articles[
            abs(valid_articles['pct_change_eod'] - price_mean) <= 3 * price_std
        ]

        if len(valid_articles) < 30:
            self.logger.warning(f"Insufficient data for analysis: {len(valid_articles)} valid articles")
            results['sentiment_weights'] = self.sentiment_weights
            return results

        self.logger.info(f"Analyzing {len(valid_articles)} valid articles")

        # Analyze with improved thresholds
        word_stats = self._analyze_word_performance_enhanced(valid_articles)
        bigram_stats = self._analyze_bigram_performance(valid_articles)

        results['word_analysis'] = word_stats
        results['bigram_analysis'] = bigram_stats

        dynamic_weights = self._generate_sentiment_weights(word_stats, bigram_stats)

        # Merge weights with priority to high-confidence dynamic weights
        merged_weights = self.sentiment_weights.copy()
        for term, weight_info in dynamic_weights.items():
            if weight_info['confidence'] > 0.7:
                merged_weights[term] = weight_info
            elif term not in merged_weights:
                merged_weights[term] = weight_info

        self.sentiment_weights = merged_weights
        results['sentiment_weights'] = merged_weights

        # Train enhanced model
        model_performance = self._train_sentiment_model(valid_articles)
        results['model_performance'] = model_performance

        return results

    def _analyze_word_performance_enhanced(self, articles_df):
        """Enhanced word performance analysis with statistical significance"""
        word_stats = defaultdict(lambda: {
            'positive_outcomes': 0, 'negative_outcomes': 0, 'total_occurrences': 0,
            'price_changes': [], 'articles': []
        })

        for idx, article in articles_df.iterrows():
            try:
                if isinstance(article['tokens'], str):
                    words = article['tokens'].lower().split()
                elif isinstance(article['tokens'], list):
                    words = [w.lower() for w in article['tokens']]
                else:
                    continue

                # Filter words by length and remove common stop words
                stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'may', 'she', 'use', 'way', 'will'}
                words = [w for w in words if len(w) >= 3 and w not in stop_words]

                price_change = float(article['pct_change_eod'])
                is_positive_outcome = price_change > 0

                for word in set(words):
                    word_stats[word]['total_occurrences'] += 1
                    word_stats[word]['price_changes'].append(price_change)
                    word_stats[word]['articles'].append(idx)

                    if is_positive_outcome:
                        word_stats[word]['positive_outcomes'] += 1
                    else:
                        word_stats[word]['negative_outcomes'] += 1

            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error processing article: {e}")
                continue

        # Enhanced statistical analysis
        filtered_stats = {}
        for word, stats in word_stats.items():
            if stats['total_occurrences'] >= self.min_word_frequency:
                # Calculate enhanced metrics
                avg_change = np.mean(stats['price_changes'])
                std_change = np.std(stats['price_changes'])
                positive_ratio = stats['positive_outcomes'] / stats['total_occurrences']
                
                # Statistical significance test (simple t-test against zero)
                if std_change > 0:
                    t_stat = avg_change / (std_change / np.sqrt(stats['total_occurrences']))
                    p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + stats['total_occurrences'] - 1))
                else:
                    t_stat = 0
                    p_value = 1.0

                predictive_power = abs(positive_ratio - 0.5) * 2
                confidence = min(stats['total_occurrences'] / 20, 1.0) * (1 - p_value)

                filtered_stats[word] = {
                    **stats,
                    'avg_price_change': avg_change,
                    'std_price_change': std_change,
                    'positive_ratio': positive_ratio,
                    'predictive_power': predictive_power,
                    'confidence': confidence,
                    't_statistic': t_stat,
                    'p_value': p_value
                }

        # Return top predictive words
        return dict(sorted(filtered_stats.items(),
                            key=lambda x: x[1]['confidence'] * x[1]['predictive_power'], 
                            reverse=True)[:self.max_features])

    def _analyze_bigram_performance(self, articles_df):
        """Analyze bigram performance"""
        bigram_stats = {}

        for _, article in articles_df.iterrows():
            try:
                # Handle different token formats
                if isinstance(article['tokens'], str):
                    words = article['tokens'].split()
                elif isinstance(article['tokens'], list):
                    words = article['tokens']
                else:
                    continue

                words = [w.lower().strip() for w in words if len(w) >= 3]
                price_change = float(article['pct_change_eod'])
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
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error processing article for bigrams: {e}")
                continue

        # Calculate statistics for bigrams
        for bigram, stats in bigram_stats.items():
            if stats['total_occurrences'] >= self.min_bigram_frequency:
                stats['avg_price_change'] = np.mean(stats['price_changes'])
                stats['positive_ratio'] = stats['positive_outcomes'] / stats['total_occurrences']
                stats['predictive_power'] = abs(stats['positive_ratio'] - 0.5) * 2
                stats['confidence'] = min(stats['total_occurrences'] / 5, 1.0)

        return {k: v for k, v in bigram_stats.items() if v.get('total_occurrences', 0) >= self.min_bigram_frequency}

    def _generate_sentiment_weights(self, word_stats, bigram_stats):
        """Generate sentiment weights for words and bigrams"""
        sentiment_weights = {}

        # Process words
        for word, stats in word_stats.items():
            if stats['total_occurrences'] >= self.min_word_frequency:
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
            if stats['total_occurrences'] >= self.min_bigram_frequency:
                weight = stats['avg_price_change'] * stats['predictive_power']
                sentiment_weights[bigram] = {
                    'weight': weight,
                    'confidence': stats['confidence'],
                    'type': 'bigram',
                    'occurrences': stats['total_occurrences']
                }

        return sentiment_weights
    
    def _train_sentiment_model(self, articles_df):
        """Train an improved model with balanced splits and better feature selection"""
        try:
            X_text = []
            y = []
            metadata = []
            
            for _, article in articles_df.iterrows():
                try:
                    tokens = article['tokens']
                    if isinstance(tokens, list):
                        text = ' '.join(tokens)
                    elif isinstance(tokens, str):
                        text = tokens
                    else:
                        continue

                    if not text.strip() or len(text.split()) < 3:
                        continue
                    
                    label = 1 if float(article['pct_change_eod']) > 0 else 0
                    y.append(label)
                    X_text.append(text)

                    pos_count = len(str(article.get("pos_keywords", "")).split(",")) if str(article.get("pos_keywords", "")) != "nan" else 0
                    neg_count = len(str(article.get("neg_keywords", "")).split(",")) if str(article.get("neg_keywords", "")) != "nan" else 0
                    article_len = len(text.split())
                    total_kw = pos_count + neg_count
                    sentiment_dyn = float(article.get("sentiment_dynamic", 0))
                    sentiment_kw = float(article.get("sentiment_keyword", 0))
                    sentiment_ml = float(article.get("sentiment_ml", 0.5))
                    
                    kw_density = total_kw / article_len if article_len > 0 else 0
                    pos_neg_ratio = pos_count / (neg_count + 1)
                    text_complexity = len(set(text.split())) / article_len if article_len > 0 else 0

                    metadata.append([
                        pos_count, neg_count, total_kw, article_len,
                        sentiment_dyn, sentiment_kw, sentiment_ml,
                        kw_density, pos_neg_ratio, text_complexity
                    ])

                except Exception as e:
                    self.logger.warning(f"Skipping article due to error: {e}")
                    continue

            if len(X_text) < 50:
                return {'model_trained': False, 'error': f'Not enough valid samples: {len(X_text)}'}

            y = np.array(y)
            metadata = np.array(metadata)
            
            pos_ratio = np.mean(y)
            self.logger.info(f"Class balance: {pos_ratio:.3f} positive, {1-pos_ratio:.3f} negative")

            tfidf = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english', 
                ngram_range=(1, 2), 
                min_df=max(2, len(X_text) // 100),
                max_df=0.8,
                sublinear_tf=True,
                use_idf=True
            )
            
            X_tfidf = tfidf.fit_transform(X_text).toarray()
            
            if X_tfidf.shape[1] > 500:
                selector = SelectKBest(mutual_info_classif, k=min(500, X_tfidf.shape[1] // 2))
                X_tfidf_selected = selector.fit_transform(X_tfidf, y)
            else:
                selector = None
                X_tfidf_selected = X_tfidf

            X_combined = np.hstack([X_tfidf_selected, metadata])
            scaler = StandardScaler()
            X_combined = scaler.fit_transform(X_combined)

            # Handle class imbalance with SMOTE
            if pos_ratio < 0.4:
                self.logger.info("Applying SMOTE to balance classes...")
                try:
                    smote = SMOTE(random_state=42, k_neighbors=min(5, int(np.sum(y)) - 1))
                    X_combined_balanced, y_balanced = smote.fit_resample(X_combined, y)
                    self.logger.info(f"SMOTE applied: {len(y)} -> {len(y_balanced)} samples")
                    X_combined = X_combined_balanced
                    y = y_balanced
                    pos_ratio_new = np.mean(y)
                    self.logger.info(f"New class balance: {pos_ratio_new:.3f} positive, {1-pos_ratio_new:.3f} negative")
                except Exception as e:
                    self.logger.warning(f"SMOTE failed: {e}, continuing without balancing")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=0.2, random_state=42, stratify=y
            )

            models = {
                'XGBoost_Balanced': xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    scale_pos_weight= (1-pos_ratio) / pos_ratio if pos_ratio > 0 else 1
                ),
                'LightGBM_Balanced': lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    class_weight='balanced'
                ),
                'RandomForest_Balanced': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'LogisticRegression_Balanced': LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                )
            }

            best_model = None
            best_score = 0
            best_name = ""
            model_results = {}

            for name, model in models.items():
                try:
                    # Cross-validation with better scoring
                    tscv = TimeSeriesSplit(n_splits=3)
                    cv_scores = []
                    cv_precisions = []
                    cv_recalls = []
                    cv_f1s = []
                    cv_aucs = []

                    for train_idx, val_idx in tscv.split(X_train):
                        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                        
                        model.fit(X_cv_train, y_cv_train)
                        y_pred = model.predict(X_cv_val)
                        y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
                        
                        cv_scores.append(accuracy_score(y_cv_val, y_pred))
                        cv_precisions.append(precision_score(y_cv_val, y_pred, zero_division=0))
                        cv_recalls.append(recall_score(y_cv_val, y_pred, zero_division=0))
                        cv_f1s.append(f1_score(y_cv_val, y_pred, zero_division=0))
                        
                        try:
                            cv_aucs.append(roc_auc_score(y_cv_val, y_pred_proba))
                        except ValueError:
                            cv_aucs.append(0.5)

                    avg_f1 = np.mean(cv_f1s)
                    avg_score = np.mean(cv_scores)

                    model_results[name] = {
                        'cv_accuracy': avg_score,
                        'cv_precision': np.mean(cv_precisions),
                        'cv_recall': np.mean(cv_recalls),
                        'cv_f1': avg_f1,
                        'cv_auc': np.mean(cv_aucs)
                    }

                    if avg_f1 > best_score:
                        best_score = avg_f1
                        best_model = model
                        best_name = name
                        
                    self.logger.info(f"{name}: F1={avg_f1:.4f}, Acc={avg_score:.4f}, Recall={np.mean(cv_recalls):.4f}")

                except Exception as e:
                    self.logger.warning(f"Model {name} failed: {e}")
                    continue

            if best_model is None:
                return {'model_trained': False, 'error': 'All models failed'}

            # Train best model on full training set
            best_model.fit(X_train, y_train)

            # Add threshold optimization
            optimal_threshold, optimal_f1 = self._optimize_threshold(best_model, X_test, y_test)
            self.optimal_threshold = optimal_threshold
            self.logger.info(f"Optimal threshold: {optimal_threshold:.3f} (F1: {optimal_f1:.4f})")
            
            # Use optimal threshold for final predictions
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)

            # Original metrics with default threshold (0.5) for comparison
            y_pred = (y_pred_proba >= 0.5).astype(int)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)

            # Calculate metrics with optimized threshold
            test_accuracy_opt = accuracy_score(y_test, y_pred_optimized)
            test_precision_opt = precision_score(y_test, y_pred_optimized, zero_division=0)
            test_recall_opt = recall_score(y_test, y_pred_optimized, zero_division=0)
            test_f1_opt = f1_score(y_test, y_pred_optimized, zero_division=0)

            self.logger.info(f"Optimized metrics: Acc={test_accuracy_opt:.4f}, Prec={test_precision_opt:.4f}, Rec={test_recall_opt:.4f}, F1={test_f1_opt:.4f}")
            
            conf_matrix = confusion_matrix(y_test, y_pred_optimized)

            # Save models and threshold
            joblib.dump(best_model, "sentiment_model_enhanced.pkl")
            joblib.dump(tfidf, "vectorizer_enhanced.pkl")
            if selector:
                joblib.dump(selector, "feature_selector_enhanced.pkl")
            joblib.dump(scaler, "scaler_enhanced.pkl")
            
            with open("optimal_threshold.json", "w") as f:
                json.dump({"optimal_threshold": self.optimal_threshold}, f)

            self.sentiment_model = best_model
            self.vectorizer = tfidf
            self.feature_selector = selector
            self.scaler = scaler

            self.logger.info(f"Saved enhanced {best_name} sentiment model")

            return {
                'model_trained': True,
                'best_model': best_name,
                'optimal_threshold': optimal_threshold,
                'test_accuracy': test_accuracy,
                'test_accuracy_optimized': test_accuracy_opt,
                'test_precision': test_precision,
                'test_precision_optimized': test_precision_opt,
                'test_recall': test_recall,
                'test_recall_optimized': test_recall_opt,
                'test_f1': test_f1,
                'test_f1_optimized': test_f1_opt,
                'cv_results': model_results,
                'confusion_matrix': conf_matrix.tolist(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X_combined.shape[1],
                'text_features': X_tfidf_selected.shape[1],
                'metadata_features': metadata.shape[1],
                'class_balance': {'positive': float(pos_ratio), 'negative': float(1-pos_ratio)}
            }
        except Exception as e:
            self.logger.error(f"Failed to train enhanced model: {e}")
            return {'model_trained': False, 'error': str(e)}

    def _optimize_threshold(self, model, X_val, y_val):
        """Find optimal probability threshold for better precision/recall balance"""
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0
        
        thresholds = np.arange(0.3, 0.8, 0.05)
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        return best_threshold, best_f1

    def predict_sentiment(self, text):
        """Enhanced sentiment prediction with optimal threshold"""
        if not self.sentiment_model or not self.vectorizer:
            return 0.5
        try:
            if not text or not text.strip():
                return 0.5
            
            # Enhanced text preprocessing
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text.split()) < 2:
                return 0.5
            
            # Transform text
            text_vec = self.vectorizer.transform([text])
            
            # Apply feature selection if available
            if self.feature_selector:
                text_vec = self.feature_selector.transform(text_vec.toarray())
            else:
                text_vec = text_vec.toarray()
            
            # Add dummy metadata features
            metadata_features = np.zeros((1, 10))
            combined_features = np.hstack([text_vec, metadata_features])
            
            # Scale features
            if self.scaler:
                combined_features = self.scaler.transform(combined_features)
            
            # Get probability
            probability = self.sentiment_model.predict_proba(combined_features)[0][1]
            
            # Apply optimal threshold if available
            if hasattr(self, 'optimal_threshold'):
                # Convert to binary prediction then back to probability-like score
                binary_pred = int(probability >= self.optimal_threshold)
                # Adjust probability based on threshold
                if binary_pred == 1:
                    # Boost positive predictions, but not above 1.0
                    probability = max(probability, self.optimal_threshold)
                else:
                    # Reduce negative predictions, but not below 0.0
                    probability = min(probability, self.optimal_threshold)
            
            return float(probability)
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

        try:
            converted_results = deep_convert(results)

            with open(filename, 'w') as f:
                json.dump(converted_results, f, indent=2)

            self.logger.info(f"Analysis results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {e}")


class NewsProcessor:
    """Base NewsProcessor class"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_mentions_and_sentiment(self, article_text, ticker):
        """Basic implementation for extracting mentions and sentiment"""
        if not article_text:
            return [], [], [], ""

        words = article_text.lower().split()
        mentions = [ticker.lower()] if ticker else []

        # Simple positive/negative word lists
        positive_words = {'growth', 'profit', 'gain', 'increase', 'rise', 'bullish', 'positive', 'strong',
                          'buy', 'upgrade', 'beat', 'exceed', 'outperform', 'rally', 'surge', 'jump'}
        negative_words = {'loss', 'decline', 'fall', 'decrease', 'drop', 'bearish', 'negative', 'weak',
                          'sell', 'downgrade', 'miss', 'underperform', 'crash', 'plunge', 'dive'}

        pos_kw = [word for word in words if word in positive_words]
        neg_kw = [word for word in words if word in negative_words]

        return mentions, pos_kw, neg_kw, ' '.join(words)

    def get_price_data(self, ticker, parsed_dt, max_retries=3):
        """Get price data for the ticker around the given datetime"""
        if not ticker or not parsed_dt:
            return None

        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)

                # Get data for a few days around the article date
                start_date = parsed_dt - timedelta(days=1)
                end_date = parsed_dt + timedelta(days=7)

                hist = stock.history(start=start_date, end=end_date)

                if hist.empty:
                    self.logger.warning(f"No price data found for {ticker} around {parsed_dt}")
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
                self.logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to get price data for {ticker} after {max_retries} attempts")
                    return None

    def calculate_dynamic_sentiment(self, text):
        """Calculate sentiment using basic dynamic weights"""
        if not text:
            return 0

        words = text.lower().split()
        positive_words = {'growth', 'profit', 'gain', 'increase', 'rise', 'bullish', 'positive', 'strong'}
        negative_words = {'loss', 'decline', 'fall', 'decrease', 'drop', 'bearish', 'negative', 'weak'}

        pos_score = sum(1 for word in words if word in positive_words)
        neg_score = sum(1 for word in words if word in negative_words)

        if pos_score + neg_score == 0:
            return 0

        return (pos_score - neg_score) / (pos_score + neg_score)


class EnhancedNewsProcessor(NewsProcessor):
    """An enhanced NewsProcessor using DynamicSentimentLearner"""
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
                with open("enhanced_analysis_results.json", 'r') as f:
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
        if not text:
            return {
                'dynamic_weights': 0,
                'ml_prediction': 0.5,
                'keyword_based': 0,
                'combined': 0
            }

        sentiment_scores = {}

        # Method 1: Dynamic weights (existing)
        sentiment_scores['dynamic_weights'] = self.calculate_dynamic_sentiment(text)

        # Method 2: ML model prediction
        if self.sentiment_learner and self.sentiment_learner.sentiment_model and self.sentiment_learner.vectorizer:
            sentiment_scores['ml_prediction'] = self.sentiment_learner.predict_sentiment(text)
        else:
            sentiment_scores['ml_prediction'] = 0.5

        # Method 3: Keyword-based scoring
        sentiment_scores['keyword_based'] = self.calculate_keyword_sentiment(text)

        # Method 4: Weighted combination
        sentiment_scores['combined'] = self.combine_sentiment_scores(sentiment_scores)

        return sentiment_scores

    def calculate_keyword_sentiment(self, text):
        """Calculate sentiment using keyword weights, including negation"""
        negation_words = {'not', 'no', 'never', "n't", 'none', 'cannot'}
        words = re.findall(r'\b\w+\b', text.lower())
        positive_score = 0
        negative_score = 0
        total_weight = 0

        # Check for words and bigrams from the sentiment_weights dictionary
        i = 0
        while i < len(words):
            word = words[i]
            # Check for negation in the preceding 3 words
            negated = any(w in negation_words for w in words[max(0, i-3):i])

            # Check for bigram first to give it priority
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in self.sentiment_weights:
                    weight = self.sentiment_weights[bigram]['weight']
                    confidence = self.sentiment_weights[bigram]['confidence']
                    score = weight * confidence
                    if negated:
                        score *= -1
                    if score > 0:
                        positive_score += score
                    else:
                        negative_score += abs(score)
                    total_weight += confidence
                    i += 2
                    continue

            # Check for single word
            if word in self.sentiment_weights:
                weight = self.sentiment_weights[word]['weight']
                confidence = self.sentiment_weights[word]['confidence']
                score = weight * confidence
                if negated:
                    score *= -1
                if score > 0:
                    positive_score += score
                else:
                    negative_score += abs(score)
                total_weight += confidence
            i += 1

        if total_weight == 0:
            return 0

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
                # Convert ML prediction from 0-1 to -1-1 scale
                if method == 'ml_prediction':
                    score = (score - 0.5) * 2
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
        full_text = f"{headline} " * 1.5 + article_text if headline and article_text else (headline or article_text or "")
        sentiment_scores = self.calculate_enhanced_sentiment(full_text)

        # Get price data
        price_data = self.get_price_data(ticker, parsed_dt)

        # Create enhanced article entry
        article_entry = {
            "ticker": ticker or "",
            "headline": headline or "",
            "text": article_text or "",
            "tokens": tokens,
            "mentions": ", ".join(mentions),
            "pos_keywords": ", ".join(pos_kw),
            "neg_keywords": ", ".join(neg_kw),
            "total_keywords": len(pos_kw) + len(neg_kw),

            # Enhanced sentiment scores
            "sentiment_dynamic": sentiment_scores.get('dynamic_weights', 0),
            "sentiment_ml": sentiment_scores.get('ml_prediction', 0.5),
            "sentiment_keyword": sentiment_scores.get('keyword_based', 0),
            "sentiment_combined": sentiment_scores.get('combined', 0),

            # Traditional sentiment score for comparison
            "sentiment_score_traditional": len(pos_kw) - len(neg_kw),

            # Prediction based on combined sentiment
            "predicted_direction": "Positive" if sentiment_scores.get('combined', 0) > 0 else "Negative",
        }

        # Normalize combined sentiment score
        combined_score = sentiment_scores.get('combined', 0)
        if combined_score > 0.3:
            sentiment_category = "Bullish"
        elif combined_score < -0.3:
            sentiment_category = "Bearish"
        else:
            sentiment_category = "Neutral"
        article_entry["sentiment_category"] = sentiment_category
        article_entry["prediction_confidence"] = abs(combined_score)

        # Add price data if available
        if price_data:
            article_entry.update(price_data)

            # Calculate prediction accuracy for different sentiment methods
            intervals = ['eod', 'eow']
            for interval in intervals:
                pct_change_key = f'pct_change_{interval}'
                if pct_change_key in price_data and price_data[pct_change_key] is not None:
                    actual_positive = price_data[pct_change_key] > 0

                    # Check accuracy for each sentiment method
                    for method in ['dynamic', 'ml', 'keyword', 'combined']:
                        sentiment_key = f'sentiment_{method}'
                        if sentiment_key in article_entry and article_entry[sentiment_key] is not None:
                            if method == 'ml':
                                predicted_positive = article_entry[sentiment_key] > 0.5
                            else:
                                predicted_positive = article_entry[sentiment_key] > 0
                            article_entry[f'accuracy_{method}_{interval}'] = (
                                predicted_positive == actual_positive
                            )
        return article_entry


def compute_keyword_weights(df, keyword_column, target_column):
    """Compute keyword weights based on their performance"""
    if df.empty or keyword_column not in df.columns or target_column not in df.columns:
        return {}

    keyword_weights = {}

    for idx, row in df.iterrows():
        try:
            keywords_str = str(row[keyword_column])
            if keywords_str == 'nan' or not keywords_str.strip():
                continue

            keywords = keywords_str.split(',')
            target = float(row[target_column])

            for kw in keywords:
                kw = kw.strip()
                if kw:
                    if kw not in keyword_weights:
                        keyword_weights[kw] = {'count': 0, 'positive': 0}
                    keyword_weights[kw]['count'] += 1
                    if target > 0:
                        keyword_weights[kw]['positive'] += 1
        except (ValueError, TypeError):
            continue

    # Calculate scores
    for kw in keyword_weights:
        count = keyword_weights[kw]['count']
        pos = keyword_weights[kw]['positive']
        if count > 0:
            keyword_weights[kw]['score'] = pos / count
        else:
            keyword_weights[kw]['score'] = 0.5

    return keyword_weights


# Usage functions
def run_sentiment_analysis():
    """Run the enhanced sentiment analysis pipeline"""
    try:
        conn = sqlite3.connect("articles.db")
        articles_df = pd.read_sql("SELECT * FROM articles", conn)
        conn.close()

        if articles_df.empty:
            print("No articles found in database")
            return

        print(f"Loaded {len(articles_df)} articles from database")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize enhanced learner
    learner = DynamicSentimentLearner(
        min_word_frequency=3,
        min_bigram_frequency=2,
        max_features=1000
    )

    print("Starting enhanced sentiment analysis...")
    results = learner.analyze_historical_performance(articles_df)

    # Save results
    learner.save_analysis_results(results, "enhanced_analysis_results.json")

    # Print detailed summary
    print(f"\n=== ENHANCED ANALYSIS RESULTS ===")
    print(f"Words analyzed: {len(results.get('word_analysis', {}))}")
    print(f"Bigrams analyzed: {len(results.get('bigram_analysis', {}))}")
    print(f"Sentiment weights generated: {len(results.get('sentiment_weights', {}))}")

    model_perf = results.get('model_performance', {})
    if model_perf.get('model_trained'):
        print(f"\n=== MODEL PERFORMANCE ===")
        print(f"Best model: {model_perf.get('best_model', 'Unknown')}")
        print(f"Optimal Threshold: {model_perf.get('optimal_threshold', 0.5):.4f}")
        print("\n--- Metrics with Default Threshold (0.5) ---")
        print(f"Test accuracy: {model_perf.get('test_accuracy', 0.0):.4f}")
        print(f"Test precision: {model_perf.get('test_precision', 0.0):.4f}")
        print(f"Test recall: {model_perf.get('test_recall', 0.0):.4f}")
        print(f"Test F1-score: {model_perf.get('test_f1', 0.0):.4f}")
        print("\n--- Metrics with Optimized Threshold ---")
        print(f"Optimized Test accuracy: {model_perf.get('test_accuracy_optimized', 0.0):.4f}")
        print(f"Optimized Test precision: {model_perf.get('test_precision_optimized', 0.0):.4f}")
        print(f"Optimized Test recall: {model_perf.get('test_recall_optimized', 0.0):.4f}")
        print(f"Optimized Test F1-score: {model_perf.get('test_f1_optimized', 0.0):.4f}")
        print(f"\nTraining samples: {model_perf.get('training_samples', 'N/A')}")
        print(f"Test samples: {model_perf.get('test_samples', 'N/A')}")
        print(f"Feature count: {model_perf.get('feature_count', 'N/A')}")
        
        class_balance = model_perf.get('class_balance', {})
        print(f"Class balance: {class_balance.get('positive', 0.0):.3f} positive, {class_balance.get('negative', 0.0):.3f} negative")
        
        # Show CV results for all models
        cv_results = model_perf.get('cv_results', {})
        if cv_results:
            print(f"\n=== CROSS-VALIDATION RESULTS ===")
            for model_name, metrics in cv_results.items():
                print(f"{model_name}: F1={metrics.get('cv_f1', 0.0):.4f}, Accuracy={metrics.get('cv_accuracy', 0.0):.4f}, AUC={metrics.get('cv_auc', 0.0):.4f}")
    else:
        error_msg = model_perf.get('error', 'Unknown error')
        print(f"Model training failed: {error_msg}")

    return results

def process_articles_with_enhanced_sentiment():
    """Process articles using the enhanced sentiment framework"""
    processor = EnhancedNewsProcessor()
    return processor

if __name__ == "__main__":
    run_sentiment_analysis()