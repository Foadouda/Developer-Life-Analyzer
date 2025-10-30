"""
FastAPI application for predicting task success based on developer productivity metrics.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="AI Developer Productivity Predictor",
    description="Predict task success based on developer productivity metrics",
    version="1.0.0"
)

# Load the trained model
try:
    model = joblib.load("task_success_model.pkl")
    feature_names = joblib.load("feature_names.pkl")['features']
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Warning: Model files not found. Please run train_model.py first.")
    model = None
    feature_names = []

# Define request/response models
class ProductivityData(BaseModel):
    """Input data for prediction"""
    hours_coding: float = Field(..., ge=0, le=24, description="Hours spent coding")
    coffee_intake_mg: int = Field(..., ge=0, le=600, description="Coffee intake in mg")
    distractions: int = Field(..., ge=0, le=10, description="Number of distractions")
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours of sleep")
    commits: int = Field(..., ge=0, description="Number of commits")
    bugs_reported: int = Field(..., ge=0, description="Number of bugs reported")
    ai_usage_hours: float = Field(..., ge=0, le=24, description="Hours using AI tools")
    cognitive_load: float = Field(..., ge=0, le=10, description="Cognitive load (1-10)")

class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: int = Field(..., description="Predicted task success (0 or 1)")
    probability: float = Field(..., description="Probability of success")
    confidence: str = Field(..., description="Confidence level")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    data: List[ProductivityData]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[dict]

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Developer Productivity Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/predict-batch": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_features": len(feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(data: ProductivityData):
    """
    Predict task success for a single developer day.
    
    Returns:
    - prediction: 0 (failed) or 1 (succeeded)
    - probability: Probability of success (0-1)
    - confidence: Low/Medium/High based on probability
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Get probability of success
        success_prob = probabilities[1]
        
        # Determine confidence level
        if success_prob >= 0.7:
            confidence = "High"
        elif success_prob >= 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(success_prob),
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict task success for multiple developer days (batch processing).
    
    Returns list of predictions with probabilities and confidence levels.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert to DataFrame
        data_list = [item.dict() for item in request.data]
        df = pd.DataFrame(data_list)
        
        # Reorder columns
        df = df[feature_names]
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        # Format results
        results = []
        for pred, prob in zip(predictions, probabilities):
            success_prob = float(prob[1])
            if success_prob >= 0.7:
                confidence = "High"
            elif success_prob >= 0.5:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            results.append({
                "prediction": int(pred),
                "probability": success_prob,
                "confidence": confidence
            })
        
        return BatchPredictionResponse(predictions=results)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/features")
async def get_features():
    """Get list of required features for prediction"""
    return {
        "features": feature_names,
        "description": {
            "hours_coding": "Hours spent coding (0-24)",
            "coffee_intake_mg": "Coffee intake in mg (0-600)",
            "distractions": "Number of distractions (0-10)",
            "sleep_hours": "Hours of sleep (0-24)",
            "commits": "Number of commits",
            "bugs_reported": "Number of bugs reported",
            "ai_usage_hours": "Hours using AI tools (0-24)",
            "cognitive_load": "Cognitive load rating (0-10)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

