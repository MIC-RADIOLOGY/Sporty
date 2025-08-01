import numpy as np

def predict_match(model, home_elo, away_elo, home_odds, draw_odds, away_odds):
    elo_diff = home_elo - away_elo
    home_implied = 1 / home_odds
    draw_implied = 1 / draw_odds
    away_implied = 1 / away_odds
    total = home_implied + draw_implied + away_implied
    home_implied /= total
    draw_implied /= total
    away_implied /= total

    X = np.array([[elo_diff, home_implied, draw_implied, away_implied]])
    prob_home_win = model.predict_proba(X)[0][1]

    explanation = {
        "elo_diff": elo_diff,
        "market_baseline_home_implied": home_implied,
        "market_baseline_draw_implied": draw_implied,
        "market_baseline_away_implied": away_implied,
        "predicted_home_win_probability": float(prob_home_win)
    }
    return explanation
