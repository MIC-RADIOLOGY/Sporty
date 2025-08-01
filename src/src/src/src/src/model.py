import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

def train_model(feature_df, save_path: str):
    X = feature_df[["elo_diff", "home_implied", "draw_implied", "away_implied"]]
    y = feature_df["target_home_win"]
    tscv = TimeSeriesSplit(n_splits=5)
    base = GradientBoostingClassifier()
    calibrated = CalibratedClassifierCV(base, cv=tscv, method="isotonic")
    calibrated.fit(X, y)
    joblib.dump(calibrated, save_path)
    return calibrated

def load_model(path: str):
    return joblib.load(path)
