# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict

class ReviewRequest(BaseModel):
    """Request model for single review prediction"""
    text: str = Field(
        ..., 
        description="Review text to analyze", 
        min_length=10,
        examples=["This product is amazing! Best purchase ever!"]
    )

class ReviewResponse(BaseModel):
    """Response model for single review prediction"""
    text: str
    prediction: str  # "FAKE" or "REAL"
    confidence: float
    probabilities: Dict[str, float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Great product!",
                "prediction": "REAL",
                "confidence": 0.8534,
                "probabilities": {
                    "REAL": 0.8534,
                    "FAKE": 0.1466
                }
            }
        }

class BatchReviewRequest(BaseModel):
    """Request model for batch predictions"""
    reviews: List[str] = Field(
        ..., 
        description="List of review texts",
        max_length=100
    )

class BatchReviewResponse(BaseModel):
    """Response model for batch predictions"""
    results: List[ReviewResponse]
    summary: Dict[str, float]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    message: str = ""
