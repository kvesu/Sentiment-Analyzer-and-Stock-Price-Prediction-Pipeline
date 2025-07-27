import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Configuration ===
INPUT_CSV = "cleaned_engineered_features.csv"
MODEL_OUTPUT_PATH = "models/stock_price_regressor.pkl"
FEATURE_COLUMNS = [
    "sentiment_combined", "sentiment_dynamic", "sentiment_ml",
    "sentiment_keyword", "mentions", "pos_keywords",
    "neg_keywords", "total_keywords",
    "is_premarket", "is_aftermarket", "is_market_hours"
]

TARGET_COLUMN = "pct_change_1h"

# === Load Data ===
print("Loading data...")
df = pd.read_csv(INPUT_CSV)
print(f"Original data shape: {df.shape}")

# Check which feature columns actually exist in the data
available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]

if missing_features:
    print(f"Warning: Missing features: {missing_features}")
    print(f"Available features: {available_features}")
    
if not available_features:
    print("Error: No feature columns found in the dataset!")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

# Update feature columns to only use available ones
FEATURE_COLUMNS = available_features
print(f"Using {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}")

# Check if target column exists
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found!")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

# Drop rows with missing values in features or target
print("Cleaning data...")
initial_rows = len(df)
df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
final_rows = len(df)
print(f"Dropped {initial_rows - final_rows} rows with missing values")
print(f"Final data shape: {df.shape}")

if len(df) == 0:
    print("Error: No data remaining after cleaning!")
    exit(1)

# === Prepare Features and Target ===
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Target statistics:")
print(f"  Mean: {y.mean():.4f}")
print(f"  Std:  {y.std():.4f}")
print(f"  Min:  {y.min():.4f}")
print(f"  Max:  {y.max():.4f}")

# === Split Data ===
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# === Train Model ===
print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42,
    n_jobs=-1,  # Use all available cores
    verbose=1   # Show progress
)
model.fit(X_train, y_train)

# === Evaluate Model ===
print("Evaluating model...")
y_pred = model.predict(X_test)

# Calculate metrics (compatible with older scikit-learn versions)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculate RMSE manually
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"Mean Absolute Error (MAE):  {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print("="*50)

# === Feature Importance ===
print("\nFEATURE IMPORTANCE:")
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# === Model Performance Analysis ===
print("\nMODEL PERFORMANCE ANALYSIS:")
# Calculate residuals
residuals = y_test - y_pred
print(f"Residual statistics:")
print(f"  Mean residual: {residuals.mean():.4f}")
print(f"  Std residual:  {residuals.std():.4f}")

# Check for outliers in predictions
outlier_threshold = 2 * rmse
outliers = np.abs(residuals) > outlier_threshold
print(f"  Outliers (>2*RMSE): {outliers.sum()} ({100*outliers.mean():.1f}%)")

# Performance by prediction range
pred_ranges = [
    ("Very Negative", y_pred < -2),
    ("Negative", (y_pred >= -2) & (y_pred < 0)),
    ("Positive", (y_pred >= 0) & (y_pred < 2)),
    ("Very Positive", y_pred >= 2)
]

print(f"\nPerformance by prediction range:")
for range_name, mask in pred_ranges:
    if mask.sum() > 0:
        range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
        print(f"  {range_name}: {mask.sum()} samples, MAE = {range_mae:.4f}")

# === Save Model ===
print(f"\nSaving model...")
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"Trained model saved to {MODEL_OUTPUT_PATH}")

# === Save Feature Importance ===
feature_importance_path = MODEL_OUTPUT_PATH.replace('.pkl', '_feature_importance.csv')
feature_importance.to_csv(feature_importance_path, index=False)
print(f"Feature importance saved to {feature_importance_path}")

# === Save Model Metadata ===
metadata = {
    'model_type': 'RandomForestRegressor',
    'n_estimators': 100,
    'features': FEATURE_COLUMNS,
    'target': TARGET_COLUMN,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'feature_count': len(FEATURE_COLUMNS)
}

metadata_path = MODEL_OUTPUT_PATH.replace('.pkl', '_metadata.json')
import json
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Model metadata saved to {metadata_path}")

print(f"\n Training complete! Model ready for prediction.")
print(f"Model files saved in: {os.path.dirname(MODEL_OUTPUT_PATH)}")