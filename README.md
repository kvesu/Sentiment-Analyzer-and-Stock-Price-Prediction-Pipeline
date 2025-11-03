# Sentiment Analyzer and Stock Price Prediction Pipeline

An advanced machine learning pipeline that automatically scrapes financial news, performs sentiment analysis, and predicts stock price movements using engineered features from news sentiment, technical indicators, and market data.

## Features

- **Automated News Collection**: Intelligent ticker filtering and news scraping across multiple sources
- **Advanced NLP Processing**: Sentiment analysis using BERT-based models, keyword extraction, and text classification
- **Technical Analysis Integration**: Incorporates price data and technical indicators via `pandas_ta`
- **Two-Stage ML Pipeline**: 
  - Gatekeeper classifier to identify news correlated with significant price moves
  - Regression model for actual price change prediction
- **Real-Time Predictions**: Live news monitoring with continuous price movement forecasting
- **Multiple Time Horizons**: Support for end-of-day (`eod`), 1-hour (`1hr`), and 4-hour (`4hr`) predictions
- **Performance Evaluation**: Automated backtesting and prediction accuracy analysis

## Critical Environment Requirements

**Before installation, ensure your environment meets these exact version requirements:**

### Python Version
- **Required**: Python 3.12.X
- **Not supported**: Python 3.13+ (incompatible with pandas-ta, yfinance, and other ML libraries)

### NumPy Version
- **Required**: numpy>=1.24.3
- **Critical**: Newer NumPy versions (1.25+) cause breaking import errors with pandas-ta, yfinance, and pandas dependencies

### Environment Setup
```bash
# Create virtual environment with Python 3.12 or lower
python -m venv stock_prediction_env
source stock_prediction_env/bin/activate  # Linux/Mac
# stock_prediction_env\Scripts\activate  # Windows

# Install NumPy 1.24.3 FIRST
pip install numpy>=1.24.3

# Then install other dependencies
pip install -r .\requirements.txt
```

## Installation

### 1. Download Project Files
1. Go to https://github.com/kvesu/Sentiment-Analyzer-and-Stock-Price-Prediction-Pipeline
2. Download the `requirements.txt` and `project_directory` folder containing all the scripts and data files
3. Navigate to the downloaded `project_directory` folder

### 2. Set Up Environment
Follow the [Critical Environment Requirements](#critical-environment-requirements) above.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Required File Structure
Ensure these files are in the same directory as the main scripts:
- `feature_engineering.py` and `word_analysis_framework.py` (required dependencies for `main.py`)
- `sentiment_keywords.csv` (financial sentiment lexicon - required for sentiment analysis)
- `finviz.csv` (stock ticker enumeration - required for ticker filtering)

## Data Files

### Core Data Assets
| File | Purpose | Format |
|------|---------|--------|
| `sentiment_keywords.csv` | Curated financial sentiment lexicon with keywords/phrases mapped to sentiment scores and strength values | CSV with columns: keyword, sentiment (positive/negative), strength |
| `finviz.csv` | Enumeration of stock tickers available for Finviz news scraping | CSV with ticker symbols |

### Sentiment Analysis Integration
The project uses a sophisticated sentiment analysis approach combining:
- **Rule-based sentiment** from the curated `sentiment_keywords.csv` lexicon
- **ML-enhanced sentiment** using BERT-based models
- **Financial context awareness** through domain-specific keywords

Keywords like "bullish", "earnings beat", "recession", "profit warning" are matched against news articles and weighted by their predetermined sentiment strength to generate explainable sentiment scores.

## Dependencies

### Core Required Packages
| Package | Purpose | Notes |
|---------|---------|-------|
| `pandas` | Data manipulation and analysis | Essential throughout |
| `numpy==1.24.3` | Numerical operations | **Exact version required** |
| `yfinance` | Stock data downloading | Essential for price data |
| `pandas_ta` | Technical analysis indicators | Feature engineering |
| `pytz` | Timezone management | Used everywhere |
| `python-dateutil` | Date/time parsing | Essential |
| `requests` | HTTP requests | Web scraping |
| `beautifulsoup4` | HTML parsing | Web scraping |
| `lxml` | HTML parser backend | Speed optimization |

### Machine Learning & NLP
| Package | Purpose |
|---------|---------|
| `scikit-learn` | ML models, scaling, evaluation |
| `lightgbm` | Gradient boosting models |
| `xgboost` | Alternative ML models |
| `optuna` | Hyperparameter optimization |
| `imbalanced-learn` | SMOTE oversampling |
| `joblib` | Model serialization |
| `nltk` | Text processing, stopwords |
| `sentence-transformers` | BERT-based embeddings |
| `keybert` | Keyword extraction |
| `transformers` | Deep learning NLP |

### Visualization & Utilities
| Package | Purpose |
|---------|---------|
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical plotting (optional) |
| `pyarrow` | Fast parquet I/O |
| `tqdm` | Progress bars |
| `sqlite3` | Database (standard library) |

### Web Scraping Enhancement (Optional)
| Package | Purpose |
|---------|---------|
| `cloudscraper` | Anti-block scraping |
| `fake-useragent` | Header rotation |
| `requests-html` | JS rendering |
| `googlesearch-python` | Google search fallback |

## Usage

### Complete Pipeline Execution

Follow this exact order for a full pipeline run:

#### 1. Ticker Filtering
```bash
python ticker_filter.py
```
**Purpose**: Filters stock tickers to find those with available news  
**Output**: `tickers_with_news.json`

#### 2. Data Collection & Feature Engineering
```bash
python main.py
```
**Purpose**: Scrapes news, performs sentiment analysis, and creates engineered features  
**Dependencies**: Requires `feature_engineering.py` and `word_analysis_framework.py` in same directory  
**Output**: 
- `cleaned_engineered_features.csv`
- `scraped_articles.csv`
- `articles.db` (SQLite database)

#### 3. Train Gatekeeper Classifier
```bash
python train_classifier.py [horizon]
```
**Purpose**: Trains model to identify news correlated with significant price moves 
**Options**: Replace `[horizon]` with `eod`, `1hr`, or `4hr` (defaults to `eod` if not specified)  
**Output**: 
- `models/stock_move_classifier_[horizon].pkl`
- `classified_features_[horizon].csv`

#### 4. Train Price Regression Model
```bash
python train_regressor.py [horizon]
```
**Purpose**: Trains regression model for actual price change prediction 
**Options**: Replace `[horizon]` with `eod`, `1hr`, or `4hr` (defaults to `eod` if not specified)  
**Output**: `models/stock_price_regressor_[horizon].pkl`

#### 5. Real-Time Prediction
```bash
python predict_stock_price.py [horizon]
```
**Purpose**: Live news monitoring and price movement prediction  
**Options**: Replace `[horizon]` with `eod`, `1hr`, or `4hr` (defaults to `eod` if not specified)  
**Output**: `continuous_predictions_[horizon].csv`

#### 6. Performance Evaluation
```bash
python prediction_screener.py continuous_predictions_[horizon].csv [horizon]
```
**Purpose**: Compares predictions with actual market data  
**Output**: Performance metrics and accuracy reports

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ticker        │    │   News Scraping  │    │   Feature       │
│   Filtering     │───▶│   & Collection   │───▶│   Engineering   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Performance   │    │   Real-time      │    │   Model         │
│   Evaluation    │◀───│   Prediction     │◀───│   Training      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **Data Pipeline**: 
   - Web scraping with anti-detection measures
   - Multi-source news aggregation
   - Technical indicator calculation

2. **NLP Processing**:
   - BERT-based sentiment analysis
   - Keyword extraction with KeyBERT
   - Text classification and feature engineering

3. **ML Pipeline**:
   - Two-stage prediction system
   - Hyperparameter optimization with Optuna
   - Class imbalance handling with SMOTE

4. **Real-time System**:
   - Continuous news monitoring
   - Live feature computation
   - Automated prediction generation

## Input and Output Files

### Input Files (Required)
| File | Description |
|------|-------------|
| `sentiment_keywords.csv` | Financial sentiment lexicon with keyword mappings |
| `finviz.csv` | Stock ticker enumeration for news filtering |

### Output Files (Generated by Pipeline)
| Script | Output Files | Description |
|--------|--------------|-------------|
| `ticker_filter.py` | `tickers_with_news.json`<br>`tickers_with_no_news.json`<br>`filtering_progress.json` | Filtered ticker lists and progress tracking |
| `feature_engineering.py` | `processed_financial_news_features.csv` | Raw feature-engineered dataset with technical indicators, market context, and sentiment features |
| `main.py` | `scraped_articles.csv`<br>`articles.db` (SQLite)<br>`cleaned_engineered_features.csv` | Raw scraped data, database storage, and cleaned features ready for model training |
| `word_analysis_framework.py` | `enhanced_analysis_results.json`<br>`word_analysis_results.json` | Sentiment analysis results and trained sentiment model weights |
| `train_classifier.py` | `models/stock_move_classifier_[horizon].pkl`<br>`classified_features_[horizon].csv` | Trained gatekeeper models and classified datasets |
| `train_regressor.py` | `models/stock_price_regressor_[horizon].pkl`<br>`regression_test_predictions_[horizon].csv` | Trained regression models and test predictions |
| `predict_stock_price.py` | `continuous_predictions_[horizon].csv` | Real-time prediction results for specified time horizons |
| `prediction_screener.py` | `screener_results_eod_averaged.csv` (for eod)<br>`screener_results_[horizon]_individual.csv` (for 1hr/4hr) | Performance evaluation comparing predictions vs actual market data |

### Data Processing Pipeline
1. **Raw Data**: `processed_financial_news_features.csv` (direct feature engineering output)
2. **Training Ready**: `cleaned_engineered_features.csv` (cleaned and filtered for model training)
3. **Model Outputs**: Classified features and trained model files
4. **Predictions**: Continuous predictions and performance evaluations

**Note on EOD Averaging**: For end-of-day predictions, multiple article-based predictions per ticker per day are averaged into single daily predictions, hence the "_averaged" suffix. Intraday horizons (1hr/4hr) maintain individual predictions.

## Quick Start

```bash
# 1. Set up environment with correct Python/NumPy versions
python3.12 -m venv stock_env
source stock_env/bin/activate
pip install numpy==1.24.3

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run complete pipeline
python ticker_filter.py
python main.py
python train_classifier.py eod
python train_regressor.py eod
python predict_stock_price.py eod
```

## Troubleshooting

### Common Issues

**ImportError with pandas-ta or yfinance**:
- Ensure NumPy 1.24.3 is installed
- Verify Python version is 3.12 or lower

**Missing feature_engineering.py or word_analysis_framework.py**:
- These files must be in the same directory as `main.py`

**Web scraping errors**:
- Check internet connection
- Verify anti-detection packages are installed

## Acknowledgments

- Built with robust ML libraries: scikit-learn, LightGBM, XGBoost
- NLP powered by Transformers and sentence-transformers
- Financial data from yfinance
- Technical analysis via pandas-ta
