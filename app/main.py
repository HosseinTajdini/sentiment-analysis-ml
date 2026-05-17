from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent))

from app.models.inference import SentimentPredictor

app = FastAPI(title="Multi-Model Sentiment Analysis API", 
              description="API for sentiment analysis supporting LSTM, CNN, and GRU models",
              version="2.0.0")

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load model based on config
MODEL_PATH = f"data/models/sentiment_model_{config['model']['type']}.pth"
CONFIG_PATH = "config.yaml"

try:
    predictor = SentimentPredictor(MODEL_PATH, CONFIG_PATH)
    print(f"Model loaded successfully! Type: {config['model']['type']}")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

class ReviewRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    label_id: int
    confidence: float
    model_type: str

@app.get("/")
async def root():
    return {
        "message": "Multi-Model Sentiment Analysis API",
        "available_models": ["lstm", "cnn", "gru"],
        "current_model": config['model']['type']
    }

@app.get("/health")
async def health_check():
    if predictor is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {
        "status": "healthy", 
        "device": predictor.device,
        "model_type": config['model']['type']
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    result = predictor.predict(request.text)
    return SentimentResponse(**result)

@app.post("/predict_batch")
async def predict_batch(reviews: list[ReviewRequest]):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    results = [predictor.predict(review.text) for review in reviews]
    return {"predictions": results}