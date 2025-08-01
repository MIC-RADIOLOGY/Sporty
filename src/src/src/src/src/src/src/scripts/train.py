import os
from src.ingestion import load_matches
from src.features import compute_elo_ratings, engineer_match_features
from src.model import train_model

def main():
    os.makedirs("models", exist_ok=True)
    df = load_matches("data/sample_matches.csv")
    elo_df = compute_elo_ratings(df)
    feat_df = engineer_match_features(elo_df)
    train_model(feat_df, "models/ensemble_latest.pkl")
    print("Training complete. Model saved to models/ensemble_latest.pkl")

if __name__ == "__main__":
    main()
