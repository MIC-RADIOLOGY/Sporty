import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import os

from src.features import compute_elo_ratings, engineer_match_features
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

MODEL_PATH = "models/ensemble_latest.pkl"

@st.cache_resource(show_spinner=True)
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            st.warning("Failed to load existing model; retraining fallback.")
    sample_path = "data/sample_matches.csv"
    if not os.path.exists(sample_path):
        st.error("No model and no sample data to train fallback.")
        st.stop()

    df_raw = pd.read_csv(sample_path, parse_dates=["date"])
    elo_df = compute_elo_ratings(df_raw)
    feat_df = engineer_match_features(elo_df)
    X = feat_df[["elo_diff", "home_implied", "draw_implied", "away_implied"]]
    y = feat_df["target_home_win"]

    tscv = TimeSeriesSplit(n_splits=2)
    base = GradientBoostingClassifier()
    calibrated = CalibratedClassifierCV(base, cv=tscv, method="isotonic")
    calibrated.fit(X, y)

    try:
        os.makedirs("models", exist_ok=True)
        joblib.dump(calibrated, MODEL_PATH)
    except Exception:
        pass

    return calibrated

def compute_implied_probs(home_odds, draw_odds, away_odds):
    home_implied = 1 / home_odds
    draw_implied = 1 / draw_odds
    away_implied = 1 / away_odds
    total = home_implied + draw_implied + away_implied
    return home_implied / total, draw_implied / total, away_implied / total

def make_feature_vector(home_elo, away_elo, home_odds, draw_odds, away_odds):
    elo_diff = home_elo - away_elo
    home_implied, draw_implied, away_implied = compute_implied_probs(home_odds, draw_odds, away_odds)
    return np.array([[elo_diff, home_implied, draw_implied, away_implied]]), {
        "elo_diff": elo_diff,
        "home_implied": home_implied,
        "draw_implied": draw_implied,
        "away_implied": away_implied,
    }

def main():
    st.set_page_config(page_title="Sports Predictor", layout="centered")
    st.title("🏟️ Sports Match Outcome Predictor")

    st.sidebar.header("Inputs")
    home_elo = st.sidebar.number_input("Home team Elo", value=1500.0)
    away_elo = st.sidebar.number_input("Away team Elo", value=1500.0)
    home_odds = st.sidebar.number_input("Home odds", value=2.0)
    draw_odds = st.sidebar.number_input("Draw odds", value=3.5)
    away_odds = st.sidebar.number_input("Away odds", value=4.0)

    model = load_or_train_model()
    features_array, raw_feats = make_feature_vector(home_elo, away_elo, home_odds, draw_odds, away_odds)
    prob_home_win = model.predict_proba(features_array)[0][1]

    st.subheader("Prediction")
    st.metric("Home win probability", f"{prob_home_win*100:.1f}%")

    st.subheader("Inputs / Baselines")
    st.write(f"Elo diff: {raw_feats['elo_diff']}")
    st.write(f"Market implied home win: {raw_feats['home_implied']:.3f}")
    st.write(f"Edge vs market: {(prob_home_win - raw_feats['home_implied'])*100:.1f}%")

    st.subheader("Feature Attribution (SHAP)")
    try:
        explainer = shap.Explainer(model, feature_names=["elo_diff", "home_implied", "draw_implied", "away_implied"])
        shap_values = explainer(features_array)
        st.pyplot(shap.plots.bar(shap_values[0], show=False).figure)
    except Exception as e:
        st.warning(f"SHAP failed: {e}")

    st.subheader("Raw feature vector")
    df_in = pd.DataFrame(features_array, columns=["elo_diff", "home_implied", "draw_implied", "away_implied"])
    st.dataframe(df_in.T)

if __name__ == "__main__":
    main()
