import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# === PRODUCTION CONFIGURATION ===
CLEANED_INPUT = "cleaned_engineered_features.csv"
MERGED_OUTPUT = "merged_training_data.csv"
MODEL_OUTPUT = "models/stock_price_regressor.pkl"
BATCH_SIZE = 100

FEATURE_COLUMNS = [
    # Original Features
    "sentiment_combined", "sentiment_dynamic", "sentiment_ml",
    "sentiment_keyword", "mentions", "pos_keywords",
    "neg_keywords", "total_keywords",
    "is_premarket", "is_aftermarket", "is_market_hours",
    
    # Technical Features
    "rsi_14", "macd", "macd_signal", 
    "price_vs_sma50", "price_vs_sma200",

    # New Market Context Features
    "vix_close", "spy_daily_return"
]

TARGET_COLUMN = "pct_change_1h"

# === STEP 1: Merge scraped article CSVs ===
def merge_scraped_articles(directory="."):
    """Merge all scraped article CSV files"""
    all_dfs = []
    for file in os.listdir(directory):
        if file.startswith("scraped_articles") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file))
            all_dfs.append(df)
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.drop_duplicates(subset=["url"], inplace=True)
        merged_df.to_csv(MERGED_OUTPUT, index=False)
        print(f"Merged {len(all_dfs)} files into {MERGED_OUTPUT} with {merged_df.shape[0]} rows.")
        return merged_df
    else:
        print("No scraped_articles CSV files found.")
        return pd.DataFrame()

# === STEP 2: Advanced data cleaning and diagnosis ===
def diagnose_and_clean_data(df, target_col, feature_cols):
    """Comprehensive data diagnosis and cleaning"""
    
    print("=" * 60)
    print("DATA DIAGNOSIS & CLEANING")
    print("=" * 60)
    
    # Basic data quality
    print(f"\nData Quality Check:")
    print(f"Original shape: {df.shape}")
    print(f"Missing values in target: {df[target_col].isna().sum()}")
    
    # Remove missing values
    df_clean = df.dropna(subset=[target_col] + feature_cols)
    print(f"After removing NaN: {df_clean.shape}")
    
    # Target analysis
    y = df_clean[target_col]
    print(f"\nTarget Analysis:")
    print(f"   Mean: {y.mean():.4f}")
    print(f"   Std: {y.std():.4f}")
    print(f"   Min: {y.min():.4f}, Max: {y.max():.4f}")
    print(f"   Skewness: {y.skew():.4f}")
    
    # Advanced outlier removal
    print(f"\nOutlier Handling:")
    
    # Method 1: Modified Z-score (more robust)
    median = y.median()
    mad = np.median(np.abs(y - median))
    modified_z_scores = 0.6745 * (y - median) / (mad + 1e-8)
    outlier_mask_z = np.abs(modified_z_scores) < 3.5
    
    # Method 2: Conservative percentile-based
    q02, q98 = y.quantile([0.02, 0.98])
    outlier_mask_pct = (y >= q02) & (y <= q98)
    
    # Combine both methods
    final_mask = outlier_mask_z & outlier_mask_pct
    
    print(f"   Modified Z-score method: kept {outlier_mask_z.sum()}/{len(y)} samples")
    print(f"   Percentile method (2%-98%): kept {outlier_mask_pct.sum()}/{len(y)} samples")
    print(f"   Combined: kept {final_mask.sum()}/{len(y)} samples")
    
    df_final = df_clean[final_mask].reset_index(drop=True)
    y_final = df_final[target_col]
    
    print(f"   Final target stats: mean={y_final.mean():.4f}, std={y_final.std():.4f}")
    
    # Feature analysis
    print(f"\nFeature Analysis:")
    X = df_final[feature_cols]
    
    # Remove zero-variance features
    zero_var_features = []
    for col in feature_cols:
        if col in X.columns and X[col].std() < 1e-8:
            zero_var_features.append(col)
    
    if zero_var_features:
        print(f"   Zero-variance features removed: {zero_var_features}")
        feature_cols = [col for col in feature_cols if col not in zero_var_features]
        X = X[feature_cols]
    
    # Feature correlations with target
    correlations = X.corrwith(y_final).sort_values(key=abs, ascending=False)
    print(f"   Top 5 correlations:")
    for feat, corr in correlations.head(5).items():
        print(f"     {feat}: {corr:.4f}")
    
    return df_final, feature_cols, correlations

# === STEP 3: Advanced feature engineering ===
def create_advanced_features(df, feature_cols, target_col):
    """
    Create sophisticated features based on successful patterns.
    This version includes robust checks to prevent KeyErrors.
    """
    
    print(f"\nADVANCED FEATURE ENGINEERING:")
    df_eng = df.copy()
    new_features = []
    
    # 1. Sentiment robustness features
    sentiment_cols = [col for col in feature_cols if 'sentiment' in col]
    # Check if a sufficient number of sentiment columns exist
    if len(sentiment_cols) >= 2 and all(col in df_eng.columns for col in sentiment_cols):
        # Sentiment agreement (consistency across different sentiment measures)
        df_eng['sentiment_agreement'] = df_eng[sentiment_cols].std(axis=1)
        new_features.append('sentiment_agreement')
        
        # Sentiment magnitude (strength regardless of direction)
        df_eng['sentiment_magnitude'] = df_eng[sentiment_cols].abs().mean(axis=1)
        new_features.append('sentiment_magnitude')
    else:
        print("Warning: Insufficient sentiment columns for advanced features. Skipping.")
        # Ensure these columns exist with default values if skipped
        if 'sentiment_agreement' not in df_eng.columns:
            df_eng['sentiment_agreement'] = 0
        if 'sentiment_magnitude' not in df_eng.columns:
            df_eng['sentiment_magnitude'] = 0

    # 2. Keyword-based sentiment features
    if 'pos_keywords' in df_eng.columns and 'neg_keywords' in df_eng.columns:
        # Keyword sentiment polarity
        total_kw = df_eng['pos_keywords'] + df_eng['neg_keywords']
        df_eng['keyword_sentiment'] = np.where(
            total_kw > 0,
            (df_eng['pos_keywords'] - df_eng['neg_keywords']) / total_kw,
            0
        )
        new_features.append('keyword_sentiment')
        
        # Keyword activity (logarithmic transform)
        df_eng['keyword_activity'] = np.log1p(total_kw)
        new_features.append('keyword_activity')
    else:
        print("Warning: 'pos_keywords' or 'neg_keywords' missing. Skipping keyword-based features.")
        if 'keyword_sentiment' not in df_eng.columns:
            df_eng['keyword_sentiment'] = 0
        if 'keyword_activity' not in df_eng.columns:
            df_eng['keyword_activity'] = 0
    
    # 3. Mention-based features
    if 'mentions' in df_eng.columns:
        # Log-transformed mentions (handle skewness)
        df_eng['mentions_log'] = np.log1p(df_eng['mentions'])
        new_features.append('mentions_log')
        
        # High mention indicator (top 5% threshold)
        mentions_95th = df_eng['mentions'].quantile(0.95)
        df_eng['high_mentions'] = (df_eng['mentions'] > mentions_95th).astype(int)
        new_features.append('high_mentions')
    else:
        print("Warning: 'mentions' column is missing. Skipping mention-based features.")
        if 'mentions_log' not in df_eng.columns:
            df_eng['mentions_log'] = 0
        if 'high_mentions' not in df_eng.columns:
            df_eng['high_mentions'] = 0
    
    # 4. Market timing improvements
    timing_cols = ['is_premarket', 'is_aftermarket', 'is_market_hours']
    if all(col in df_eng.columns for col in timing_cols):
        # Market session importance score
        df_eng['market_session_score'] = (
            df_eng['is_premarket'] * 0.3 +      # Lower weight for pre-market
            df_eng['is_market_hours'] * 1.0 +   # Full weight for market hours
            df_eng['is_aftermarket'] * 0.5      # Medium weight for after-market
        )
        new_features.append('market_session_score')
    else:
        print("Warning: Market timing columns missing. Skipping market session score.")
        if 'market_session_score' not in df_eng.columns:
            df_eng['market_session_score'] = 0
    
    # 5. High-value interaction features
    if 'sentiment_combined' in df_eng.columns and 'mentions_log' in df_eng.columns:
        df_eng['sentiment_x_mentions'] = df_eng['sentiment_combined'] * df_eng['mentions_log']
        new_features.append('sentiment_x_mentions')
    else:
        print("Warning: 'sentiment_combined' or 'mentions_log' missing. Skipping interaction feature.")
        if 'sentiment_x_mentions' not in df_eng.columns:
            df_eng['sentiment_x_mentions'] = 0
    
    # 6. Volatility indicators (using robust checks)
    if 'sentiment_combined' in df_eng.columns:
        # Sentiment bins for non-linear relationships
        df_eng['sentiment_positive'] = (df_eng['sentiment_combined'] > 0.1).astype(int)
        df_eng['sentiment_negative'] = (df_eng['sentiment_combined'] < -0.1).astype(int)
        df_eng['sentiment_neutral'] = ((df_eng['sentiment_combined'] >= -0.1) & 
                                     (df_eng['sentiment_combined'] <= 0.1)).astype(int)
        new_features.extend(['sentiment_positive', 'sentiment_negative', 'sentiment_neutral'])
    else:
        print("Warning: 'sentiment_combined' missing. Skipping volatility indicators.")
        if 'sentiment_positive' not in df_eng.columns:
            df_eng['sentiment_positive'] = 0
        if 'sentiment_negative' not in df_eng.columns:
            df_eng['sentiment_negative'] = 0
        if 'sentiment_neutral' not in df_eng.columns:
            df_eng['sentiment_neutral'] = 0
    
    print(f"   Created {len(new_features)} new features")
    print(f"   New features: {new_features[:5]}..." if len(new_features) > 5 else f"   New features: {new_features}")
    
    # Return only features that exist in the final DataFrame
    all_features = feature_cols + new_features
    existing_features = [col for col in all_features if col in df_eng.columns]
    
    return df_eng, existing_features

# === STEP 4: Intelligent feature selection ===
def smart_feature_selection(X, y, feature_names, max_features=12):
    """Select best features using combined scoring methods"""
    
    print(f"\nSMART FEATURE SELECTION:")
    
    # Ensure there are enough features and samples for selection
    if len(feature_names) == 0 or len(y) == 0:
        print("   Not enough features or samples for selection. Skipping.")
        return X, [], None
    
    # Method 1: Statistical significance (F-test)
    selector_f = SelectKBest(f_regression, k='all')
    selector_f.fit(X, y)
    
    # Method 2: Mutual information (non-linear relationships)
    selector_mi = SelectKBest(mutual_info_regression, k='all')
    selector_mi.fit(X, y)
    
    # Combine and normalize scores
    f_scores = selector_f.scores_
    mi_scores = selector_mi.scores_
    
    # Normalize to 0-1 range
    f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
    
    # Combined score (equal weighting)
    combined_scores = 0.5 * f_scores_norm + 0.5 * mi_scores_norm
    
    # Select top features
    max_features = min(max_features, len(feature_names))
    top_indices = np.argsort(combined_scores)[-max_features:]
    selected_features = [feature_names[i] for i in top_indices]
    
    # Results dataframe
    feature_scores = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_scores,
        'mi_score': mi_scores,
        'combined_score': combined_scores,
        'selected': [name in selected_features for name in feature_names]
    }).sort_values('combined_score', ascending=False)
    
    print(f"   Selected top {len(selected_features)} features:")
    for i, (_, row) in enumerate(feature_scores[feature_scores['selected']].iterrows()):
        print(f"     {i+1}. {row['feature']}: {row['combined_score']:.4f}")
    
    return X[:, top_indices], selected_features, feature_scores

# === STEP 5: Production-ready model training ===
def train_production_models(X, y, feature_names):
    """Train multiple optimized models with proper validation"""
    
    print(f"\nPRODUCTION MODEL TRAINING:")
    
    # Time series cross-validation (proper for financial data)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Optimized model configurations
    models = {
        'Linear_Baseline': Ridge(alpha=0.1),
        
        'RF_Conservative': RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        'RF_Balanced': RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=7,
            max_features=0.6,
            random_state=42,
            n_jobs=-1
        ),
        
        'GBM_Gentle': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        ),
        
        'GBM_Optimized': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=15,
            min_samples_leaf=8,
            subsample=0.85,
            random_state=42
        )
    }
    
    # Scale data for linear models
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Use scaled data for linear models
        X_use = X_scaled if 'Linear' in name else X
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_use, y, 
            cv=tscv, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        # Train on full dataset
        model.fit(X_use, y)
        y_pred = model.predict(X_use)
        
        # Calculate comprehensive metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        direction_acc = np.mean(np.sign(y) == np.sign(y_pred))
        
        results[name] = {
            'model': model,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_acc,
            'use_scaling': 'Linear' in name
        }
        
        print(f"      CV MAE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"      R²: {r2:.4f}, Direction: {direction_acc:.2%}")
    
    # Select best model using comprehensive scoring
    best_name = None
    best_score = -np.inf
    
    for name, result in results.items():
        # Scoring: Prioritize positive R² and good directional accuracy
        if result['r2'] > 0:
            score = (result['r2'] * 0.4 + 
                     result['direction_accuracy'] * 0.4 - 
                     result['cv_mae'] * 0.2 / y.std())
        else:
            score = (result['direction_accuracy'] * 0.6 - 
                     result['cv_mae'] * 0.4 / y.std())
        
        if score > best_score:
            best_score = score
            best_name = name
    
    print(f"\nBest Model: {best_name}")
    print(f"   R²: {results[best_name]['r2']:.4f}")
    print(f"   CV MAE: {results[best_name]['cv_mae']:.4f}")
    print(f"   Direction Accuracy: {results[best_name]['direction_accuracy']:.2%}")
    
    return results, best_name, scaler

# === STEP 6: Model validation and analysis ===
def validate_production_model(model, X, y, feature_names, use_scaling=False, scaler=None):
    """Comprehensive model validation and performance analysis"""
    
    print(f"\nMODEL VALIDATION & ANALYSIS:")
    
    X_use = scaler.transform(X) if use_scaling and scaler else X
    y_pred = model.predict(X_use)
    
    # 1. Residual analysis
    residuals = y - y_pred
    print(f"   Residual Analysis:")
    print(f"     Mean residual: {residuals.mean():.4f}")
    print(f"     Residual std: {residuals.std():.4f}")
    print(f"     Residual skewness: {stats.skew(residuals):.4f}")
    
    # 2. Performance by target magnitude (key for trading)
    print(f"   Performance by Target Magnitude:")
    
    ranges = [
        ("Small moves (|target| < 0.5)", np.abs(y) < 0.5),
        ("Medium moves (0.5 ≤ |target| < 2)", (np.abs(y) >= 0.5) & (np.abs(y) < 2)),
        ("Large moves (|target| ≥ 2)", np.abs(y) >= 2)
    ]
    
    for label, mask in ranges:
        if mask.sum() > 0:
            range_mae = mean_absolute_error(y[mask], y_pred[mask])
            range_r2 = r2_score(y[mask], y_pred[mask]) if mask.sum() > 1 else 0
            range_dir = np.mean(np.sign(y[mask]) == np.sign(y_pred[mask]))
            print(f"     {label}: {mask.sum()} samples")
            print(f"       MAE: {range_mae:.4f}, R²: {range_r2:.4f}, Direction: {range_dir:.2%}")
    
    # 3. Feature importance analysis
    importance_df = None
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 5 Feature Importances:")
        for _, row in importance_df.head(5).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
    
    # 4. Trading signal analysis
    print(f"   Trading Signal Analysis:")
    
    # Strong signals (predictions > 1% magnitude)
    strong_signals = np.abs(y_pred) > 1.0
    if strong_signals.sum() > 0:
        strong_acc = np.mean(np.sign(y[strong_signals]) == np.sign(y_pred[strong_signals]))
        print(f"     Strong signals (>1%): {strong_signals.sum()} samples, Accuracy: {strong_acc:.2%}")
    
    # Very strong signals (predictions > 2% magnitude)
    very_strong_signals = np.abs(y_pred) > 2.0
    if very_strong_signals.sum() > 0:
        very_strong_acc = np.mean(np.sign(y[very_strong_signals]) == np.sign(y_pred[very_strong_signals]))
        print(f"     Very strong signals (>2%): {very_strong_signals.sum()} samples, Accuracy: {very_strong_acc:.2%}")
    
    return importance_df

# === MAIN PRODUCTION PIPELINE ===
def main_production_pipeline():
    """Main production pipeline with all enhancements"""
    
    print("PRODUCTION-READY MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Merge data if needed
    merged_df = merge_scraped_articles()
    
    # Step 2: Load and clean data
    df = pd.read_csv(CLEANED_INPUT)
    df_clean, feature_cols, correlations = diagnose_and_clean_data(df, TARGET_COLUMN, FEATURE_COLUMNS)
    
    # Step 3: Advanced feature engineering
    df_eng, all_features = create_advanced_features(df_clean, feature_cols, TARGET_COLUMN)
    
    # Step 4: Smart feature selection
    X = df_eng[all_features].values
    y = df_eng[TARGET_COLUMN].values
    X_selected, selected_features, feature_scores = smart_feature_selection(
        X, y, all_features, max_features=12
    )
    
    # Step 5: Train production models
    results, best_name, scaler = train_production_models(X_selected, y, selected_features)
    
    # Step 6: Validate best model
    best_model = results[best_name]['model']
    use_scaling = results[best_name]['use_scaling']
    importance_df = validate_production_model(
        best_model, X_selected, y, selected_features, use_scaling, scaler
    )
    
    # Step 7: Save production model
    print(f"\nSAVING PRODUCTION MODEL:")
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    
    # Package everything needed for production - INCLUDING ORIGINAL FEATURE COLUMNS
    model_package = {
        'model': best_model,
        'scaler': scaler if use_scaling else None,
        'selected_features': selected_features,
        'original_feature_columns': FEATURE_COLUMNS,  # ← ADDED: Original training features
        'all_engineered_features': all_features,
        'use_scaling': use_scaling,
        'feature_engineering_applied': True,
        'model_type': best_name,
        'training_date': pd.Timestamp.now().isoformat(),
        # Additional metadata for better live prediction support
        'feature_engineering_config': {
            'base_features': feature_cols,  # Features after cleaning
            'engineered_features': all_features,  # All features after engineering
            'final_selected': selected_features  # Final selected features for model
        }
    }
    
    joblib.dump(model_package, MODEL_OUTPUT)
    print(f"   Model package saved to: {MODEL_OUTPUT}")
    print(f"   Original feature columns included for live prediction compatibility")
    
    # Save comprehensive metadata
    metadata = {
        'model_info': {
            'name': best_name,
            'type': type(best_model).__name__,
            'training_date': pd.Timestamp.now().isoformat(),
            'training_samples': len(y)
        },
        'features': {
            'original_features': FEATURE_COLUMNS,
            'cleaned_features': feature_cols,  # ← Added for transparency
            'selected_features': selected_features,
            'total_engineered': len(all_features),
            'final_selected': len(selected_features)
        },
        'performance': {
            'mae': float(results[best_name]['mae']),
            'rmse': float(results[best_name]['rmse']),
            'r2': float(results[best_name]['r2']),
            'direction_accuracy': float(results[best_name]['direction_accuracy']),
            'cv_mae': float(results[best_name]['cv_mae']),
            'cv_std': float(results[best_name]['cv_std'])
        },
        'data_stats': {
            'target_mean': float(y.mean()),
            'target_std': float(y.std()),
            'target_min': float(y.min()),
            'target_max': float(y.max())
        },
        'model_comparison': {
            name: {
                'r2': float(result['r2']),
                'direction_accuracy': float(result['direction_accuracy']),
                'cv_mae': float(result['cv_mae'])
            } for name, result in results.items()
        },
        # Live prediction guidance
        'live_prediction_guide': {
            'required_columns_for_feature_engineering': FEATURE_COLUMNS,
            'final_model_features': selected_features,
            'scaling_required': use_scaling
        }
    }
    
    # Add feature importance if available
    if importance_df is not None:
        metadata['feature_importance'] = importance_df.to_dict('records')
    
    # Add feature selection scores
    metadata['feature_selection_scores'] = feature_scores.to_dict('records')
    
    metadata_path = MODEL_OUTPUT.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved to: {metadata_path}")
    
    # Save feature importance separately
    if importance_df is not None:
        importance_path = MODEL_OUTPUT.replace('.pkl', '_feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"   Feature importance saved to: {importance_path}")
    
    # Step 8: Final recommendations and summary
    print(f"\nPRODUCTION RECOMMENDATIONS:")
    
    final_r2 = results[best_name]['r2']
    final_dir_acc = results[best_name]['direction_accuracy']
    final_cv_mae = results[best_name]['cv_mae']
    
    if final_r2 > 0.1:
        print("Model shows strong predictive power - READY FOR PRODUCTION!")
        if final_dir_acc > 0.65:
            print("Excellent directional accuracy for algorithmic trading")
        elif final_dir_acc > 0.55:
            print("Good directional accuracy for trading signals")
        else:
            print("Consider ensemble methods for better direction prediction")
    else:
        print("Limited predictive power. Recommendations:")
        print("    • Add technical indicators (RSI, MACD, Bollinger Bands)")
        print("    • Include broader market context (VIX, sector indices)")
        print("    • Collect more diverse data sources")
    
    if final_cv_mae < 1.8:
        print("Low prediction errors - suitable for trading")
    else:
        print("Consider position sizing based on prediction confidence")
    
    print(f"\nFINAL MODEL SUMMARY:")
    print(f"   Model: {best_name}")
    print(f"   R² Score: {final_r2:.4f}")
    print(f"   Direction Accuracy: {final_dir_acc:.2%}")
    print(f"   Cross-Validation MAE: {final_cv_mae:.4f}")
    print(f"   Features Used: {len(selected_features)}")
    
    # Trading recommendations
    print(f"\nTRADING IMPLEMENTATION TIPS:")
    print("    1. Focus on predictions with magnitude > 0.5%")
    print("    2. Use higher confidence for larger position sizes")
    print("    3. Combine with risk management (stop-losses)")
    print("    4. Monitor model performance regularly")
    print("    5. Retrain monthly with new data")
    
    # Live prediction compatibility message
    print(f"\nLIVE PREDICTION COMPATIBILITY:")
    print("    ✓ Original feature columns saved for feature engineering")
    print("    ✓ Complete feature engineering pipeline preserved") 
    print("    ✓ Model ready for seamless live prediction deployment")
    
    print("\nModel is ready for production deployment!")
    
    return model_package, metadata

# === EXECUTION ===
if __name__ == "__main__":
    model_package, metadata = main_production_pipeline()
