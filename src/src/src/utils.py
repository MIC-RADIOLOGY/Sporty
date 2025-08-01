def elo_expected(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1 + 10 ** ((away_elo - home_elo) / 400))

def update_elo(old_elo: float, expected: float, score: float, k=20) -> float:
    return old_elo + k * (score - expected)
