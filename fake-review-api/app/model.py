# app/model.py
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeReviewDetector:
    """Handler for fake review detection model"""
    
    def __init__(self, model_path: str = "./models/fake_review_detector_model"):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the saved model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_map = {0: "REAL", 1: "FAKE"}
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_path
            )
            
            # Move to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✓ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"✗ Error loading model: {str(e)}")
            raise
    
    def predict(self, text: str) -> Dict:
        """
        Predict if a review is fake or real
        
        Args:
            text: Review text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Format results
            result = {
                "text": text,
                "prediction": self.label_map[predicted_class],
                "confidence": round(confidence, 4),
                "probabilities": {
                    "REAL": round(probabilities[0][0].item(), 4),
                    "FAKE": round(probabilities[0][1].item(), 4)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict multiple reviews at once
        
        Args:
            texts: List of review texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting text: {text[:50]}... Error: {str(e)}")
                # Add error result
                results.append({
                    "text": text,
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "probabilities": {"REAL": 0.0, "FAKE": 0.0}
                })
        
        return results

# Global model instance
detector = FakeReviewDetector()
