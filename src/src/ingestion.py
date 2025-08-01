import pandas as pd

def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {
        "date", "home_team", "away_team", "home_score",
        "away_score", "home_odds", "draw_odds", "away_odds"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    return df
