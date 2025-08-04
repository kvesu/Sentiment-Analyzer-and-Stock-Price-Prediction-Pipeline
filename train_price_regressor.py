import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

# === CONFIGURATION ===
CLEANED_INPUT = "cleaned_engineered_features.csv"
MERGED_OUTPUT = "merged_training_data.csv"
MODEL_OUTPUT = "models/stock_price_regressor.pkl"
BATCH_SIZE = 100

FEATURE_COLUMNS = [
    "sentiment_combined", "sentiment_dynamic", "sentiment_ml",
    "sentiment_keyword", "mentions", "pos_keywords",
    "neg_keywords", "total_keywords",
    "is_premarket", "is_aftermarket", "is_market_hours"
]
TARGET_COLUMN = "pct_change_1h"

# === STEP 1: Merge scraped article CSVs ===
def merge_scraped_articles(directory="."):
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

# === STEP 2: Train model ===
def train_model(input_csv):
    global FEATURE_COLUMNS  # Move this to the top, before any usage
    
    print("Loading data...")
    df = pd.read_csv(input_csv)
    print(f"Original data shape: {df.shape}")

    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        print(f"Available features: {available_features}")
    
    if not available_features:
        print("Error: No valid feature columns found in the dataset!")
        exit(1)

    FEATURE_COLUMNS = available_features

    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found!")
        exit(1)

    # Clean data
    initial_rows = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    final_rows = len(df)
    print(f"Dropped {initial_rows - final_rows} rows with missing values")
    print(f"Final data shape: {df.shape}")

    if len(df) == 0:
        print("Error: No data remaining after cleaning!")
        exit(1)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n=== MODEL EVALUATION RESULTS ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")

    # Residual analysis
    residuals = y_test - y_pred
    outlier_threshold = 2 * rmse
    outliers = np.abs(residuals) > outlier_threshold

    print("\n=== RESIDUAL ANALYSIS ===")
    print(f"Mean residual: {residuals.mean():.4f}")
    print(f"Outliers (>2*RMSE): {outliers.sum()} ({100 * outliers.mean():.2f}%)")

    # Range analysis
    print("\n=== PREDICTION RANGE MAE ===")
    pred_ranges = [
        ("Very Negative", y_pred < -2),
        ("Negative", (y_pred >= -2) & (y_pred < 0)),
        ("Positive", (y_pred >= 0) & (y_pred < 2)),
        ("Very Positive", y_pred >= 2)
    ]
    for label, mask in pred_ranges:
        if mask.sum():
            range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            print(f"{label}: {mask.sum()} samples, MAE = {range_mae:.4f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    print(f"Model saved to {MODEL_OUTPUT}")

    # Save feature importance
    importance_path = MODEL_OUTPUT.replace('.pkl', '_feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

    # Save metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'features': FEATURE_COLUMNS,
        'target': TARGET_COLUMN,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_estimators': 100
    }
    metadata_path = MODEL_OUTPUT.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    print("\n Training complete. Model is ready for prediction.")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    merged_df = merge_scraped_articles()
    if not merged_df.empty:
        # Assume merged_df is already cleaned/engineered if using it directly
        train_model(CLEANED_INPUT)
    else:
        print("Skipped model training due to no new data.")