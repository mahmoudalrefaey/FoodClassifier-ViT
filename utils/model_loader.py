"""
Model loader utility for FoodViT
Handles loading the trained PyTorch model and feature extractor
"""

import torch
import os
from transformers import ViTForImageClassification, ViTFeatureExtractor
from config import MODEL_CONFIG, CLASS_CONFIG
from huggingface_hub import hf_hub_download

class ModelLoader:
    """Class to handle model loading and initialization"""
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.device = MODEL_CONFIG["device"]
        
    def load_model(self):
        """Load the trained PyTorch model from Hugging Face Hub"""
        try:
            # Download the model from the Hugging Face Hub
            model_path = hf_hub_download(
                repo_id="mahmoudalrefaey/FoodViT-weights",
                filename="bestViT_PT.pth"
            )
            from transformers import ViTForImageClassification
            self.model = ViTForImageClassification.from_pretrained(
                MODEL_CONFIG["feature_extractor_name"],
                num_labels=MODEL_CONFIG["num_labels"],
                ignore_mismatched_sizes=True
            )
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_feature_extractor(self):
        """Load the ViT feature extractor"""
        try:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(
                MODEL_CONFIG["feature_extractor_name"]
            )
            print("Feature extractor loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading feature extractor: {e}")
            return False
    
    def get_model(self):
        """Get the loaded model"""
        return self.model
    
    def get_feature_extractor(self):
        """Get the loaded feature extractor"""
        return self.feature_extractor
    
    def get_device(self):
        """Get the current device"""
        return self.device

# Global model loader instance
model_loader = ModelLoader() 