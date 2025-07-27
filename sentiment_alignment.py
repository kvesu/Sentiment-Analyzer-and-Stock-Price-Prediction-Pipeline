import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SentimentPriceAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.alignment_issues = {}
        
    def analyze_data_quality(self) -> Dict:
        """Analyze data quality issues that might cause misalignment."""
        print("=== DATA QUALITY ANALYSIS ===")
        
        issues = {}
        
        # 1. Check for missing values
        sentiment_missing = self.df['sentiment_combined'].isna().sum()
        price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        
        print(f"Missing sentiment values: {sentiment_missing}/{len(self.df)} ({sentiment_missing/len(self.df)*100:.1f}%)")
        
        for col in price_cols:
            if col in self.df.columns:
                missing = self.df[col].isna().sum()
                print(f"Missing {col}: {missing}/{len(self.df)} ({missing/len(self.df)*100:.1f}%)")
        
        # 2. Check sentiment distribution
        print(f"\nSentiment distribution:")
        print(f"  Min: {self.df['sentiment_combined'].min():.3f}")
        print(f"  Max: {self.df['sentiment_combined'].max():.3f}")
        print(f"  Mean: {self.df['sentiment_combined'].mean():.3f}")
        print(f"  Std: {self.df['sentiment_combined'].std():.3f}")
        
        # Count sentiment categories
        positive_count = (self.df['sentiment_combined'] > 0).sum()
        negative_count = (self.df['sentiment_combined'] < 0).sum()
        neutral_count = (self.df['sentiment_combined'] == 0).sum()
        
        print(f"  Positive: {positive_count} ({positive_count/len(self.df)*100:.1f}%)")
        print(f"  Negative: {negative_count} ({negative_count/len(self.df)*100:.1f}%)")
        print(f"  Neutral: {neutral_count} ({neutral_count/len(self.df)*100:.1f}%)")
        
        # 3. Check price change distributions
        print(f"\nPrice change distributions:")
        for col in price_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                print(f"  {col}: Mean={data.mean():.3f}, Std={data.std():.3f}, Range=[{data.min():.3f}, {data.max():.3f}]")
        
        # 4. Check time alignment issues
        if 'datetime' in self.df.columns:
            print(f"\nTemporal analysis:")
            try:
                # Convert datetime column if it's not already datetime
                if self.df['datetime'].dtype == 'object':
                    self.df['datetime'] = pd.to_datetime(self.df['datetime'])
                
                date_min = self.df['datetime'].min()
                date_max = self.df['datetime'].max()
                print(f"  Date range: {date_min} to {date_max}")
                print(f"  Total time span: {(date_max - date_min).days} days")
            except Exception as e:
                print(f"  Could not parse datetime column: {e}")
                print(f"  Sample values: {self.df['datetime'].head(3).tolist()}")
        
        return issues
    
    def analyze_alignment_patterns(self) -> pd.DataFrame:
        """Analyze detailed alignment patterns."""
        print("\n=== ALIGNMENT PATTERN ANALYSIS ===")
        
        price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        
        # Calculate alignment for each ticker and time horizon
        alignment_data = []
        
        for ticker in self.df['ticker'].unique():
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            
            if len(ticker_data) == 0:
                continue
                
            avg_sentiment = ticker_data['sentiment_combined'].mean()
            
            ticker_stats = {
                'ticker': ticker,
                'article_count': len(ticker_data),
                'avg_sentiment': avg_sentiment
            }
            
            for col in price_cols:
                if col in ticker_data.columns:
                    avg_price_change = ticker_data[col].mean()
                    ticker_stats[f'avg_{col}'] = avg_price_change
                    
                    # Calculate alignment
                    if pd.isna(avg_sentiment) or pd.isna(avg_price_change):
                        alignment = 'Missing'
                    elif avg_sentiment * avg_price_change > 0:
                        alignment = 'Aligned'
                    elif avg_sentiment == 0 or avg_price_change == 0:
                        alignment = 'Neutral'
                    else:
                        alignment = 'Not Aligned'
                    
                    ticker_stats[f'alignment_{col}'] = alignment
                    
                    # Calculate correlation strength
                    if alignment == 'Aligned':
                        strength = min(abs(avg_sentiment), abs(avg_price_change))
                        ticker_stats[f'strength_{col}'] = strength
                    else:
                        ticker_stats[f'strength_{col}'] = 0
            
            alignment_data.append(ticker_stats)
        
        alignment_df = pd.DataFrame(alignment_data)
        
        # Summary statistics
        print(f"Total tickers analyzed: {len(alignment_df)}")
        
        for col in price_cols:
            alignment_col = f'alignment_{col}'
            if alignment_col in alignment_df.columns:
                alignment_counts = alignment_df[alignment_col].value_counts()
                print(f"\n{col} alignment:")
                for status, count in alignment_counts.items():
                    pct = count / len(alignment_df) * 100
                    print(f"  {status}: {count} ({pct:.1f}%)")
        
        return alignment_df
    
    def identify_misalignment_causes(self, alignment_df: pd.DataFrame) -> Dict:
        """Identify potential causes of misalignment."""
        print("\n=== MISALIGNMENT CAUSE ANALYSIS ===")
        
        causes = {}
        
        # 1. Check for systematic bias
        overall_sentiment = self.df['sentiment_combined'].mean()
        print(f"Overall sentiment bias: {overall_sentiment:.3f}")
        
        if abs(overall_sentiment) > 0.1:
            causes['sentiment_bias'] = f"Strong {'positive' if overall_sentiment > 0 else 'negative'} sentiment bias detected"
        
        # 2. Check for weak sentiment signals
        weak_sentiment_tickers = alignment_df[abs(alignment_df['avg_sentiment']) < 0.05]['ticker'].tolist()
        if weak_sentiment_tickers:
            causes['weak_sentiment'] = f"{len(weak_sentiment_tickers)} tickers have very weak sentiment signals (< 0.05)"
            print(f"Tickers with weak sentiment: {weak_sentiment_tickers[:10]}...")  # Show first 10
        
        # 3. Check for small price movements
        price_cols = ['avg_pct_change_1h', 'avg_pct_change_4h', 'avg_pct_change_eod', 'avg_pct_change_eow']
        for col in price_cols:
            if col in alignment_df.columns:
                small_movement_tickers = alignment_df[abs(alignment_df[col]) < 0.5]['ticker'].tolist()
                if small_movement_tickers:
                    causes[f'small_movements_{col}'] = f"{len(small_movement_tickers)} tickers have small movements (< 0.5%)"
        
        # 4. Check for insufficient data per ticker
        small_sample_tickers = alignment_df[alignment_df['article_count'] < 5]['ticker'].tolist()
        if small_sample_tickers:
            causes['insufficient_data'] = f"{len(small_sample_tickers)} tickers have < 5 articles"
        
        # 5. Check temporal issues
        if 'datetime' in self.df.columns:
            try:
                # Ensure datetime is properly parsed
                if self.df['datetime'].dtype == 'object':
                    self.df['datetime'] = pd.to_datetime(self.df['datetime'])
                
                # Check if sentiment and price changes are from different time periods
                date_range_days = (self.df['datetime'].max() - self.df['datetime'].min()).days
                if date_range_days > 30:
                    causes['temporal_mismatch'] = f"Data spans {date_range_days} days - sentiment and price timing may be mismatched"
            except Exception as e:
                causes['datetime_parsing_error'] = f"Could not parse datetime column: {e}"
        
        print("\nIdentified potential causes:")
        for cause, description in causes.items():
            print(f"  - {cause}: {description}")
        
        return causes
    
    def suggest_fixes(self, causes: Dict) -> List[str]:
        """Suggest fixes based on identified causes."""
        print("\n=== SUGGESTED FIXES ===")
        
        fixes = []
        
        if 'sentiment_bias' in causes:
            fixes.append("1. Normalize sentiment scores by removing overall bias")
            fixes.append("   df['sentiment_combined'] = df['sentiment_combined'] - df['sentiment_combined'].mean()")
        
        if 'weak_sentiment' in causes:
            fixes.append("2. Filter out articles with very weak sentiment signals")
            fixes.append("   df = df[abs(df['sentiment_combined']) >= 0.05]")
        
        if any('small_movements' in cause for cause in causes):
            fixes.append("3. Focus on more significant price movements")
            fixes.append("   # Filter for tickers with meaningful average price changes")
        
        if 'insufficient_data' in causes:
            fixes.append("4. Filter out tickers with insufficient data")
            fixes.append("   ticker_counts = df['ticker'].value_counts()")
            fixes.append("   valid_tickers = ticker_counts[ticker_counts >= 5].index")
            fixes.append("   df = df[df['ticker'].isin(valid_tickers)]")
        
        if 'temporal_mismatch' in causes:
            fixes.append("5. Implement proper temporal alignment")
            fixes.append("   # Ensure sentiment and price changes are from appropriate time windows")
        
        # Always suggest these general improvements
        fixes.extend([
            "6. Consider using different sentiment thresholds for classification",
            "7. Implement rolling averages for more stable sentiment signals",
            "8. Use correlation analysis instead of simple directional alignment",
            "9. Consider market conditions and volatility when evaluating alignment"
        ])
        
        for fix in fixes:
            print(fix)
        
        return fixes
    
    def apply_basic_fixes(self) -> pd.DataFrame:
        """Apply basic fixes to improve alignment."""
        print("\n=== APPLYING BASIC FIXES ===")
        
        df_fixed = self.df.copy()
        initial_rows = len(df_fixed)
        
        # 1. Remove sentiment bias
        sentiment_bias = df_fixed['sentiment_combined'].mean()
        if abs(sentiment_bias) > 0.05:
            df_fixed['sentiment_combined_debiased'] = df_fixed['sentiment_combined'] - sentiment_bias
            print(f"Removed sentiment bias of {sentiment_bias:.3f}")
        else:
            df_fixed['sentiment_combined_debiased'] = df_fixed['sentiment_combined']
        
        # 2. Filter weak sentiment signals
        strong_sentiment_mask = abs(df_fixed['sentiment_combined_debiased']) >= 0.03
        df_fixed = df_fixed[strong_sentiment_mask]
        print(f"Filtered weak sentiment: {len(df_fixed)}/{initial_rows} rows remaining")
        
        # 3. Filter tickers with insufficient data
        ticker_counts = df_fixed['ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= 3].index
        df_fixed = df_fixed[df_fixed['ticker'].isin(valid_tickers)]
        print(f"Filtered low-count tickers: {len(df_fixed)} rows, {len(valid_tickers)} tickers remaining")
        
        # 4. Create smoothed sentiment using rolling average if we have datetime
        if 'datetime' in df_fixed.columns:
            df_fixed = df_fixed.sort_values(['ticker', 'datetime'])
            df_fixed['sentiment_smoothed'] = df_fixed.groupby('ticker')['sentiment_combined_debiased'].transform(
                lambda x: x.rolling(window=min(3, len(x)), min_periods=1).mean()
            )
            print("Created smoothed sentiment using rolling average")
        else:
            df_fixed['sentiment_smoothed'] = df_fixed['sentiment_combined_debiased']
        
        return df_fixed
    
    def evaluate_improved_alignment(self, df_fixed: pd.DataFrame) -> pd.DataFrame:
        """Evaluate alignment after applying fixes."""
        print("\n=== EVALUATING IMPROVED ALIGNMENT ===")
        
        price_cols = ['pct_change_1h', 'pct_change_4h', 'pct_change_eod', 'pct_change_eow']
        
        improved_alignment = []
        
        for ticker in df_fixed['ticker'].unique():
            ticker_data = df_fixed[df_fixed['ticker'] == ticker].copy()
            
            # Use smoothed sentiment
            avg_sentiment = ticker_data['sentiment_smoothed'].mean()
            
            ticker_stats = {
                'ticker': ticker,
                'article_count': len(ticker_data),
                'original_sentiment': ticker_data['sentiment_combined'].mean(),
                'improved_sentiment': avg_sentiment
            }
            
            for col in price_cols:
                if col in ticker_data.columns:
                    avg_price_change = ticker_data[col].mean()
                    
                    # Calculate improved alignment
                    if pd.isna(avg_sentiment) or pd.isna(avg_price_change):
                        alignment = 'Missing'
                    elif avg_sentiment * avg_price_change > 0:
                        alignment = 'Aligned'
                    elif avg_sentiment == 0 or avg_price_change == 0:
                        alignment = 'Neutral'
                    else:
                        alignment = 'Not Aligned'
                    
                    ticker_stats[f'improved_alignment_{col}'] = alignment
            
            improved_alignment.append(ticker_stats)
        
        improved_df = pd.DataFrame(improved_alignment)
        
        # Compare improvement
        print("Alignment improvement summary:")
        for col in price_cols:
            alignment_col = f'improved_alignment_{col}'
            if alignment_col in improved_df.columns:
                aligned_count = (improved_df[alignment_col] == 'Aligned').sum()
                total_count = len(improved_df)
                pct = aligned_count / total_count * 100
                print(f"  {col}: {aligned_count}/{total_count} aligned ({pct:.1f}%)")
        
        return improved_df

def analyze_sentiment_price_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to analyze and fix sentiment-price alignment."""
    analyzer = SentimentPriceAnalyzer(df)
    
    # Step 1: Analyze data quality
    analyzer.analyze_data_quality()
    
    # Step 2: Analyze alignment patterns
    alignment_df = analyzer.analyze_alignment_patterns()
    
    # Step 3: Identify causes of misalignment
    causes = analyzer.identify_misalignment_causes(alignment_df)
    
    # Step 4: Suggest fixes
    fixes = analyzer.suggest_fixes(causes)
    
    # Step 5: Apply basic fixes
    df_improved = analyzer.apply_basic_fixes()
    
    # Step 6: Evaluate improvement
    improved_alignment = analyzer.evaluate_improved_alignment(df_improved)
    
    return df_improved, improved_alignment

# Usage example
if __name__ == "__main__":
    # Load your data
    try:
        df = pd.read_csv("scraped_articles.csv")  # Replace with your CSV file
        df_improved, alignment_results = analyze_sentiment_price_alignment(df)
        
        # Save improved dataset
        df_improved.to_csv("improved_sentiment_price_data.csv", index=False)
        alignment_results.to_csv("alignment_analysis.csv", index=False)
        
        print(f"\nImproved dataset saved with {len(df_improved)} rows")
        print("Check 'improved_sentiment_price_data.csv' and 'alignment_analysis.csv' for results")
        
    except FileNotFoundError:
        print("Please replace 'your_file.csv' with your actual CSV filename")
    except Exception as e:
        print(f"Error: {e}")
        print("Please share your CSV file for specific analysis")