from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from .predict import predict_match

app = FastAPI(title="Sports Outcome Predictor")
MODEL_PATH = "models/ensemble_latest.pkl"
model = joblib.load(MODEL_PATH)

class MatchInput(BaseModel):
    home_elo: float
    away_elo: float
    home_odds: float
    draw_odds: float
    away_odds: float

@app.post("/predict")
def predict(data: MatchInput):
    explanation = predict_match(
        model,
        data.home_elo,
        data.away_elo,
        data.home_odds,
        data.draw_odds
    )
    return {
        "prediction": explanation["predicted_home_win_probability"],
        "explanation": explanation
    }
