import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test_csv_and_train():
    """Test if the CSV can be read and used for training"""
    
    # Configuration
    CLEANED_INPUT = "cleaned_engineered_features.csv"
    FEATURE_COLUMNS = [
        "sentiment_combined", "sentiment_dynamic", "sentiment_ml",
        "sentiment_keyword", "mentions", "pos_keywords",
        "neg_keywords", "total_keywords",
        "is_premarket", "is_aftermarket", "is_market_hours"
    ]
    TARGET_COLUMN = "pct_change_1h"
    
    print("=== TESTING CSV READING AND TRAINING ===\n")
    
    # Step 1: Try to read the CSV
    try:
        print("1. Reading CSV file...")
        df = pd.read_csv(CLEANED_INPUT)
        print(f"   ✓ Success! Shape: {df.shape}")
        print(f"   ✓ Columns: {len(df.columns)}")
    except Exception as e:
        print(f"   ✗ Failed to read CSV: {e}")
        return False
    
    # Step 2: Check available columns
    print("\n2. Checking feature columns...")
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    
    print(f"   Available features ({len(available_features)}): {available_features}")
    if missing_features:
        print(f"   Missing features ({len(missing_features)}): {missing_features}")
    
    if not available_features:
        print("   ✗ No feature columns found!")
        return False
    
    # Step 3: Check target column
    print(f"\n3. Checking target column '{TARGET_COLUMN}'...")
    if TARGET_COLUMN not in df.columns:
        print(f"   ✗ Target column '{TARGET_COLUMN}' not found!")
        print(f"   Available columns with 'pct_change': {[col for col in df.columns if 'pct_change' in col]}")
        return False
    else:
        print(f"   ✓ Target column found")
    
    # Step 4: Check data quality
    print("\n4. Analyzing data quality...")
    
    # Check for missing values
    missing_in_features = df[available_features].isnull().sum().sum()
    missing_in_target = df[TARGET_COLUMN].isnull().sum()
    
    print(f"   Missing values in features: {missing_in_features}")
    print(f"   Missing values in target: {missing_in_target}")
    
    # Clean data
    initial_rows = len(df)
    df_clean = df.dropna(subset=available_features + [TARGET_COLUMN])
    final_rows = len(df_clean)
    
    print(f"   Rows before cleaning: {initial_rows}")
    print(f"   Rows after cleaning: {final_rows}")
    print(f"   Rows dropped: {initial_rows - final_rows}")
    
    if final_rows == 0:
        print("   ✗ No data remaining after cleaning!")
        return False
    
    # Step 5: Prepare data for training
    print("\n5. Preparing data for training...")
    
    X = df_clean[available_features]
    y = df_clean[TARGET_COLUMN]
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target vector shape: {y.shape}")
    
    # Check if we have numeric data
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_features = [col for col in available_features if col not in numeric_features]
    
    print(f"   Numeric features: {len(numeric_features)}")
    if non_numeric_features:
        print(f"   Non-numeric features: {non_numeric_features}")
        
        # Convert non-numeric features
        for col in non_numeric_features:
            if col in X.columns:
                print(f"   Converting {col} to numeric...")
                # Try to convert string features to numeric
                if X[col].dtype == 'object':
                    # For string columns, count non-empty values or use length
                    X[col] = X[col].fillna('').astype(str).str.len()
    
    # Check target statistics
    print(f"\n   Target statistics:")
    print(f"   Mean: {y.mean():.4f}")
    print(f"   Std: {y.std():.4f}")
    print(f"   Min: {y.min():.4f}")
    print(f"   Max: {y.max():.4f}")
    print(f"   Non-zero values: {(y != 0).sum()} ({100 * (y != 0).mean():.1f}%)")
    
    # Step 6: Train a simple model
    print("\n6. Training test model...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=50,  # Smaller for quick test
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n   ✓ Training successful!")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n   Top 5 important features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        print(f"\n=== TEST SUCCESSFUL ===")
        print(f"Your CSV file is working correctly!")
        print(f"You can proceed with training using the fixed script.")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        import traceback
        print(f"   Full error: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_csv_and_train()
    
    if success:
        print("\n" + "="*50)
        print("CONCLUSION: Your CSV file is ready for training!")
        print("The original error was likely just a syntax issue in the")
        print("training script, not a problem with the CSV file itself.")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("There are still issues to resolve before training.")
        print("="*50)