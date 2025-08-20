import pandas as pd
import numpy as np
import joblib
import os
import sys
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
CLEANED_INPUT = "cleaned_engineered_features.csv"
SIGNIFICANCE_THRESHOLD = 1.5
TARGET_COLUMN = "significant_move"

# Map horizon name to target column in engineered features CSV
TARGET_MAP = {
    "1hr": "pct_change_1h",
    "4hr": "pct_change_4h",
    "eod": "pct_change_eod"
}

FEATURE_COLUMNS = [
    "sentiment_combined", "sentiment_dynamic", "sentiment_ml",
    "sentiment_keyword", "prediction_confidence", "mentions", "pos_keywords",
    "neg_keywords", "total_keywords", "headline_sentiment", "keyword_density",
    "day_of_week", "hour_of_day", "is_market_hours", "is_premarket", "is_aftermarket",
    "hour_sin", "hour_cos", "rsi_14", "macd", "macd_signal", "macd_hist",
    "price_vs_sma50", "price_vs_sma200", "vix_close", "spy_daily_return"
]

def select_best_features_time_series(X, y, feature_names, n_splits=5, max_features=30):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    f_scores = np.zeros(X.shape[1])
    mi_scores = np.zeros(X.shape[1])
    for train_idx, test_idx in tscv.split(X):
        X_cv, y_cv = X[train_idx], y[train_idx]
        sel_f = SelectKBest(f_classif, k='all').fit(X_cv, y_cv)
        sel_mi = SelectKBest(mutual_info_classif, k='all').fit(X_cv, y_cv)
        f_scores += np.nan_to_num(sel_f.scores_)
        mi_scores += np.nan_to_num(sel_mi.scores_)
    f_scores /= n_splits
    mi_scores /= n_splits
    f_norm = f_scores / (f_scores.max() + 1e-8)
    mi_norm = mi_scores / (mi_scores.max() + 1e-8)
    combined = 0.5 * f_norm + 0.5 * mi_norm
    top_idx = np.argsort(combined)[-max_features:]
    selected = [feature_names[i] for i in top_idx]
    print(f"Selected top {max_features} features using TimeSeriesSplit CV")
    print(f"Top 5 features: {selected[-5:]}")
    return selected

def find_best_threshold_macroF1(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_macro = []
    for thr in thresholds:
        preds = (y_probs >= thr).astype(int)
        f1_macro.append(f1_score(y_true, preds, average="macro"))
    best_idx = np.argmax(f1_macro)
    return thresholds[best_idx], f1_macro[best_idx]

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2)
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_val)[:, 1]
    _, macroF1 = find_best_threshold_macroF1(y_val, y_probs)
    return macroF1

def main():
    # Horizon argument
    if len(sys.argv) > 1:
        target_horizon = sys.argv[1].lower()
        if target_horizon not in TARGET_MAP:
            print(f"Invalid horizon: {target_horizon}. Choose from {list(TARGET_MAP.keys())}")
            return
    else:
        target_horizon = "eod"  # default

    target_price_col = TARGET_MAP[target_horizon]
    model_output = f"models/stock_move_classifier_{target_horizon}.pkl"
    classified_output = f"classified_features_{target_horizon}.csv"

    print(f"--- Gatekeeper Model Training for {target_horizon.upper()} ---")

    df = pd.read_csv(CLEANED_INPUT)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if target_price_col not in df.columns:
        raise ValueError(f"Target column '{target_price_col}' not found in data.")

    df[TARGET_COLUMN] = (df[target_price_col].abs() >= SIGNIFICANCE_THRESHOLD).astype(int)
    df = df.dropna(subset=[TARGET_COLUMN])

    if len(df) < 200:
        print(f"ERROR: Not enough samples for training. Found {len(df)}.")
        return

    print(f"Dataset ready: {len(df)} samples. Class balance:")
    print(df[TARGET_COLUMN].value_counts(normalize=True).rename("proportion"))

    split_idx = int(0.8 * len(df))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    available_features = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    # Remove all-NaN features based on training set
    usable_cols = [c for c in available_features if train_df[c].notna().any()]

    X_train, y_train = train_df[usable_cols], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[usable_cols], test_df[TARGET_COLUMN]

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=usable_cols, index=train_df.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=usable_cols, index=test_df.index)

    selected_features = select_best_features_time_series(X_train.values, y_train.values, X_train.columns.tolist(), max_features=30)
    X_train, X_test = X_train[selected_features], X_test[selected_features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    val_split = int(0.8 * len(X_train_scaled))
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial,
                                           X_train_scaled[:val_split], y_train[:val_split],
                                           X_train_scaled[val_split:], y_train[val_split:]),
                   n_trials=25)

    print("Best hyperparameters:", study.best_params)

    lgbm_classifier = lgb.LGBMClassifier(**study.best_params, objective="binary",
                                         metric="binary_logloss", class_weight="balanced",
                                         random_state=42)
    lgbm_classifier.fit(X_train_scaled, y_train)

    y_probs = lgbm_classifier.predict_proba(X_test_scaled)[:, 1]
    best_thresh, best_macroF1 = find_best_threshold_macroF1(y_test, y_probs)
    y_pred = (y_probs >= best_thresh).astype(int)

    print(f"\nOptimal classification threshold: {best_thresh:.3f} | Macro-F1: {best_macroF1:.4f}")
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # Optional Plot
    plt.hist(y_probs[y_test == 0], bins=50, alpha=0.6, label="Class 0")
    plt.hist(y_probs[y_test == 1], bins=50, alpha=0.6, label="Class 1")
    plt.axvline(best_thresh, color="red", linestyle="--", label="Optimal Threshold")
    plt.legend()
    plt.title(f"Probability Distribution - {target_horizon.upper()}")
    plt.show()

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    model_package = {
        "model": lgbm_classifier,
        "scaler": scaler,
        "selected_features": selected_features,
        "optimal_threshold": best_thresh,
        "model_type": "classifier",
        "target_horizon": target_horizon,
        "target_column": target_price_col,
        "training_date": pd.Timestamp.now().isoformat(),
        "macro_f1_score_test": best_macroF1
    }
    joblib.dump(model_package, model_output)
    print(f"✅ Saved Gatekeeper to {model_output}")

    df.to_csv(classified_output, index=False)
    print(f"✅ Saved classified dataset to: {classified_output}")

if __name__ == "__main__":
    main()
