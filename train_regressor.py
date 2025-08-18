import os
import sys
import pandas as pd
import numpy as np
import joblib
import pandas_ta as ta  # Import pandas-ta for ATR calculation
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso  # Import Lasso Regressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
# Map horizon to column/file
TARGET_MAP = {
    "1hr": "pct_change_1h",
    "4hr": "pct_change_4h",
    "eod": "pct_change_eod"
}

SIGNIFICANT_COL = "significant_move"
CLASSIFIER_MODEL_PATH_TEMPLATE = "models/stock_move_classifier_{horizon}.pkl"
CLASSIFIED_INPUT_TEMPLATE      = "classified_features_{horizon}.csv"
MODEL_OUTPUT_TEMPLATE          = "models/stock_price_regressor_{horizon}.pkl"

FEATURE_COLUMNS = [
    "sentiment_combined", "sentiment_dynamic", "sentiment_ml",
    "sentiment_keyword", "mentions", "pos_keywords",
    "neg_keywords", "total_keywords", "is_premarket",
    "is_aftermarket", "is_market_hours",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "price_vs_sma50", "price_vs_sma200",
    "vix_close", "spy_daily_return"
]

def add_gatekeeper_feature(df, classifier_model_path):
    if not os.path.exists(classifier_model_path):
        raise FileNotFoundError(f"Classifier model not found at {classifier_model_path}")
    model_package = joblib.load(classifier_model_path)
    clf_model = model_package["model"]
    clf_scaler = model_package["scaler"]
    clf_features = model_package["selected_features"]
    avail_feats = [f for f in clf_features if f in df.columns]
    X_clf = df[avail_feats].copy()
    X_clf = X_clf.fillna(X_clf.median(numeric_only=True))
    X_scaled = clf_scaler.transform(X_clf)
    df["gatekeeper_confidence"] = clf_model.predict_proba(X_scaled)[:, 1]
    return df

def diagnose_and_clean(df, target_col, feature_cols, datetime_col='datetime'):
    if datetime_col in df.columns:
        df = df.sort_values(datetime_col).reset_index(drop=True)
    else:
        raise ValueError(f"{datetime_col} column not found.")
    df_clean = df.dropna(subset=[target_col])
    available = [c for c in feature_cols if c in df_clean.columns]
    zero_var = [c for c in available if df_clean[c].std() < 1e-8 or df_clean[c].isna().all()]
    available = [c for c in available if c not in zero_var]
    return df_clean, available

def add_volatility_features(df):
    """
    Engineers volatility features like Historical Volatility and ATR.
    NOTE: This requires OHLCV columns (Open, High, Low, Close, Volume)
    and a 'ticker' column in your input CSV.
    """
    required_cols = ['High', 'Low', 'Close', 'ticker', 'spy_daily_return']
    if not all(col in df.columns for col in required_cols):
        print("Warning: OHLCV or ticker columns not found. Skipping volatility features.")
        return df, []

    df = df.sort_values(by=['ticker', 'datetime'])
    
    # Calculate 30-day historical volatility for each ticker
    df['hist_vol_30d'] = df.groupby('ticker')['spy_daily_return'].transform(lambda x: x.rolling(window=30).std())
    
    # Calculate ATR for each ticker
    # pandas-ta requires lowercase column names
    df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close', 'Open':'open'}, inplace=True)
    df['atr_14d'] = df.groupby('ticker').ta.atr(length=14)
    # Rename back to original case if needed
    df.rename(columns={'high': 'High', 'low': 'Low', 'close': 'Close', 'open':'Open'}, inplace=True)

    new_feats = ['hist_vol_30d', 'atr_14d']
    print(f"Added volatility features: {new_feats}")
    return df, new_feats

def engineer_features(df, feature_cols):
    df_eng = df.copy()
    new_feats = []
    sent_cols = [c for c in feature_cols if 'sentiment' in c and c in df_eng.columns]
    if len(sent_cols) >= 2:
        df_eng['sentiment_agreement'] = df_eng[sent_cols].std(axis=1)
        new_feats.append('sentiment_agreement')
        df_eng['sentiment_magnitude'] = df_eng[sent_cols].abs().mean(axis=1)
        new_feats.append('sentiment_magnitude')
    if 'pos_keywords' in df_eng.columns and 'neg_keywords' in df_eng.columns:
        tot_kw = df_eng['pos_keywords'] + df_eng['neg_keywords']
        df_eng['keyword_sentiment'] = np.where(tot_kw > 0,
                                               (df_eng['pos_keywords'] - df_eng['neg_keywords']) / tot_kw, 0)
        new_feats.append('keyword_sentiment')
        df_eng['keyword_activity'] = np.log1p(tot_kw)
        new_feats.append('keyword_activity')
    if 'mentions' in df_eng.columns:
        df_eng['mentions_log'] = np.log1p(df_eng['mentions'])
        new_feats.append('mentions_log')
    if 'sentiment_combined' in df_eng.columns and 'mentions_log' in df_eng.columns:
        df_eng['sentiment_x_mentions'] = df_eng['sentiment_combined'] * df_eng['mentions_log']
        new_feats.append('sentiment_x_mentions')
    all_feats = feature_cols + new_feats
    return df_eng, [c for c in all_feats if c in df_eng.columns]

def select_features(X, y, names, max_feats=10):
    sel_f = SelectKBest(f_regression, k='all').fit(X, y)
    sel_mi = SelectKBest(mutual_info_regression, k='all').fit(X, y)
    f_norm = (sel_f.scores_ - np.nanmin(sel_f.scores_)) / (np.nanmax(sel_f.scores_) - np.nanmin(sel_f.scores_) + 1e-8)
    mi_norm = (sel_mi.scores_ - np.nanmin(sel_mi.scores_)) / (np.nanmax(sel_mi.scores_) - np.nanmin(sel_mi.scores_) + 1e-8)
    combined = 0.5 * f_norm + 0.5 * mi_norm
    idxs = np.argsort(combined)[-min(max_feats, len(names)):]
    selected = [names[i] for i in idxs]
    print(f"Selected features: {selected}")
    return selected

def train_models(df, target_col, features):
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # --- 1. Transform Target Variable ---
    # Clip extreme outliers to make the target more stable and predictable
    y_clipped = df[target_col].clip(-10, 10).values
    X = df[features].values

    tscv = TimeSeriesSplit(n_splits=5, gap=24)
    models = {
        'RF_Conservative': RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_split=50,
            min_samples_leaf=25, max_features=0.5, random_state=42, n_jobs=-1
        ),
        'GBM_Conservative': GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            min_samples_split=50, min_samples_leaf=25,
            subsample=0.7, random_state=42
        ),
        'Lasso': Lasso(alpha=0.1, random_state=42) # --- 3. Add Simpler, Regularized Model ---
    }
    results = {}
    for name, model in models.items():
        cv_mae, cv_dir = [], []
        for tr, te in tscv.split(X):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y_clipped[tr], y_clipped[te] # Use the clipped target
            
            mdl = type(model)(**model.get_params())
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            cv_mae.append(mean_absolute_error(yte, pred))
            cv_dir.append(np.mean(np.sign(yte) == np.sign(pred)))
            
        model.fit(X, y_clipped) # Fit final model on all clipped data
        results[name] = {
            'model': model,
            'cv_mae': np.mean(cv_mae),
            'cv_dir': np.mean(cv_dir),
            'r2_full': r2_score(y_clipped, model.predict(X))
        }
    best = max(results, key=lambda k: results[k]['cv_dir'])
    return results, best

def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("1hr", "4hr", "eod"):
        target_horizon = sys.argv[1].lower()
    else:
        target_horizon = "eod"

    target_col = TARGET_MAP[target_horizon]
    classified_input = CLASSIFIED_INPUT_TEMPLATE.format(horizon=target_horizon)
    classifier_model_path = CLASSIFIER_MODEL_PATH_TEMPLATE.format(horizon=target_horizon)
    model_output = MODEL_OUTPUT_TEMPLATE.format(horizon=target_horizon)

    print(f"=== Leakage-Free Regression for {target_horizon.upper()}: {target_col} ===")
    
    df = pd.read_csv(classified_input)
    if SIGNIFICANT_COL not in df.columns:
        raise ValueError(f"'{SIGNIFICANT_COL}' column missing. Run Gatekeeper first.")
    
    df = df[df[SIGNIFICANT_COL] == 1].copy()
    df = add_gatekeeper_feature(df, classifier_model_path)
    feature_list = FEATURE_COLUMNS + ["gatekeeper_confidence"]

    df_clean, avail_feats = diagnose_and_clean(df, target_col, feature_list)
    
    # --- 2. Add Volatility Features ---
    df_vol, vol_feats = add_volatility_features(df_clean)
    all_feats = avail_feats + vol_feats

    df_eng, feats = engineer_features(df_vol, all_feats)

    split = int(0.8 * len(df_eng))
    train_df, test_df = df_eng.iloc[:split], df_eng.iloc[:split]

    non_all_nan_cols = [c for c in feats if train_df[c].notna().any()]
    train_df = train_df[non_all_nan_cols + [target_col, 'datetime']]
    test_df  = test_df[non_all_nan_cols + [target_col, 'datetime']]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(train_df[non_all_nan_cols]),
                               columns=non_all_nan_cols, index=train_df.index)
    X_test_imp = pd.DataFrame(imputer.transform(test_df[non_all_nan_cols]),
                              columns=non_all_nan_cols, index=test_df.index)
    
    sel_feats = select_features(X_train_imp.values, train_df[target_col].values, non_all_nan_cols, max_feats=15)
    
    train_df_sel = pd.concat([X_train_imp[sel_feats], train_df[[target_col, 'datetime']]], axis=1)
    test_df_sel  = pd.concat([X_test_imp[sel_feats],  test_df[[target_col, 'datetime']]],  axis=1)

    results, best_name = train_models(train_df_sel, target_col, sel_feats)
    print(f"\nBest model: {best_name}")
    best_model = results[best_name]['model']

    X_test_sel = test_df_sel[sel_feats].values
    y_test = test_df_sel[target_col].values
    pred_test = best_model.predict(X_test_sel)
    
    print("\nHeld-out Test Metrics:")
    print(f"MAE={mean_absolute_error(y_test, pred_test):.4f}")
    print(f"RMSE={np.sqrt(mean_squared_error(y_test, pred_test)):.4f}")
    print(f"R²={r2_score(y_test, pred_test):.4f}")
    print(f"Direction Accuracy={np.mean(np.sign(y_test) == np.sign(pred_test)):.2%}")

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump({
        'model': best_model,
        'selected_features': sel_feats,
        'imputer': imputer, # Save the imputer
        'target_horizon': target_horizon,
        'target_column': target_col,
        'training_date': pd.Timestamp.now().isoformat()
    }, model_output)
    print(f"\n✅ Regression model saved to {model_output}")

    pd.DataFrame({
        'datetime': test_df_sel['datetime'],
        'y_true': y_test,
        'y_pred': pred_test
    }).to_csv(f"regression_test_predictions_{target_horizon}.csv", index=False)
    print(f"📄 Saved test predictions to regression_test_predictions_{target_horizon}.csv")

if __name__ == "__main__":
    main()