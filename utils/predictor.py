"""
Prediction utility for FoodViT
Handles model inference and prediction logic
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from config import CLASS_CONFIG
from utils.model_loader import model_loader
from utils.image_processor import image_processor

class FoodPredictor:
    """Class to handle food classification predictions"""
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.device = None
        self.class_names = CLASS_CONFIG["class_names"]
        self.id2label = CLASS_CONFIG["id2label"]
        
    def initialize(self):
        """Initialize the predictor with loaded model and feature extractor"""
        try:
            # Load model and feature extractor
            if not model_loader.load_model():
                return False
            if not model_loader.load_feature_extractor():
                return False
            
            self.model = model_loader.get_model()
            self.feature_extractor = model_loader.get_feature_extractor()
            self.device = model_loader.get_device()
            
            print("Predictor initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            return False
    
    def predict(self, image_input) -> Dict:
        """
        Predict food class for given image
        
        Args:
            image_input: Image path, PIL Image, or numpy array
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        try:
            if self.model is None:
                return {"error": "Model not initialized"}
            
            # Preprocess image
            processed_image = image_processor.preprocess_image(image_input)
            if processed_image is None:
                return {"error": "Failed to preprocess image"}
            
            # Move to device
            processed_image = processed_image.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(processed_image)
                logits = outputs.logits
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Get all class probabilities
                all_probabilities = probabilities[0].cpu().numpy()
                
                # Create result dictionary
                result = {
                    "class": self.id2label[predicted_class],
                    "class_id": predicted_class,
                    "confidence": confidence,
                    "probabilities": {
                        self.id2label[i]: float(all_probabilities[i])
                        for i in range(len(self.class_names))
                    },
                    "success": True
                }
                
                return result
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": str(e), "success": False}
    
    def predict_batch(self, image_inputs) -> list:
        """
        Predict food classes for multiple images
        
        Args:
            image_inputs: List of image inputs
            
        Returns:
            list: List of prediction results
        """
        results = []
        for image_input in image_inputs:
            result = self.predict(image_input)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "device": str(self.device),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "num_classes": len(self.class_names),
                "class_names": self.class_names
            }
        except Exception as e:
            return {"error": str(e)}

# Global predictor instance
predictor = FoodPredictor() 