import pandas as pd
from collections import defaultdict
from .utils import elo_expected, update_elo

def compute_elo_ratings(matches: pd.DataFrame, base: int = 1500) -> pd.DataFrame:
    teams = defaultdict(lambda: base)
    records = []

    for _, row in matches.sort_values("date").iterrows():
        home, away = row.home_team, row.away_team
        expected_home = elo_expected(teams[home], teams[away])
        expected_away = 1 - expected_home

        if row.home_score > row.away_score:
            s_home, s_away = 1, 0
        elif row.home_score < row.away_score:
            s_home, s_away = 0, 1
        else:
            s_home = s_away = 0.5

        records.append({
            "date": row.date,
            "home_team": home,
            "away_team": away,
            "home_score": row.home_score,
            "away_score": row.away_score,
            "home_odds": row.home_odds,
            "draw_odds": row.draw_odds,
            "away_odds": row.away_odds,
            "home_elo": teams[home],
            "away_elo": teams[away],
            "expected_home_elo": expected_home,
            "expected_away_elo": expected_away,
            "result_home_win": int(s_home == 1),
            "result_draw": int(s_home == 0.5),
            "result_away_win": int(s_away == 1),
        })

        teams[home] = update_elo(teams[home], expected_home, s_home)
        teams[away] = update_elo(teams[away], expected_away, s_away)

    return pd.DataFrame(records)

def engineer_match_features(elo_df: pd.DataFrame) -> pd.DataFrame:
    df = elo_df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["home_implied"] = 1 / df["home_odds"]
    df["draw_implied"] = 1 / df["draw_odds"]
    df["away_implied"] = 1 / df["away_odds"]
    total = df["home_implied"] + df["draw_implied"] + df["away_implied"]
    df["home_implied"] /= total
    df["draw_implied"] /= total
    df["away_implied"] /= total
    df["target_home_win"] = df["result_home_win"]
    return df
