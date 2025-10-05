# app/main.py
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from app.model import detector
from app.schemas import (
    ReviewRequest, 
    ReviewResponse, 
    BatchReviewRequest, 
    BatchReviewResponse,
    HealthResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Fake Review Detector API...")
    try:
        detector.load_model()
        logger.info("✓ Model loaded successfully!")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Fake Review Detector API...")

# Initialize FastAPI app
app = FastAPI(
    title="Fake Review Detection API",
    description="Detect AI-generated and fake product reviews using DistilBERT",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for Chrome Extension
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="chrome-extension://.*",  # Allow all Chrome extensions
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Alternative CORS for development (allows all origins)
# Use this for testing, but restrict in production
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fake Review Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict_single": "/predict",
            "predict_batch": "/batch-predict"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and model health status"""
    model_loaded = detector.model is not None and detector.tokenizer is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        message="Model is loaded and ready" if model_loaded else "Model not loaded"
    )

@app.post("/predict", response_model=ReviewResponse, tags=["Prediction"])
async def predict_review(request: ReviewRequest):
    """
    Predict if a single review is fake or real
    
    - **text**: The review text to analyze (minimum 10 characters)
    
    Returns prediction with confidence score and probabilities
    """
    try:
        if len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Review text must be at least 10 characters long"
            )
        
        result = detector.predict(request.text)
        return ReviewResponse(**result)
        
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict", response_model=BatchReviewResponse, tags=["Prediction"])
async def batch_predict_reviews(request: BatchReviewRequest):
    """
    Predict multiple reviews at once
    
    - **reviews**: List of review texts (maximum 100 reviews per request)
    
    Returns predictions for all reviews plus summary statistics
    """
    try:
        # Validate batch size
        if len(request.reviews) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one review is required"
            )
        
        if len(request.reviews) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 reviews per batch request"
            )
        
        # Get predictions
        results = detector.predict_batch(request.reviews)
        
        # Calculate summary statistics
        fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
        real_count = len(results) - fake_count
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        summary = {
            "total_reviews": len(results),
            "fake_reviews": fake_count,
            "real_reviews": real_count,
            "fake_percentage": round((fake_count / len(results)) * 100, 2),
            "average_confidence": round(avg_confidence, 4)
        }
        
        return BatchReviewResponse(
            results=[ReviewResponse(**r) for r in results],
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# For running locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
