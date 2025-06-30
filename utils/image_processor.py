"""
Image processing utility for FoodViT
Handles image preprocessing and transformation for model inference
"""

import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as A
from config import IMAGE_CONFIG

class ImageProcessor:
    """Class to handle image preprocessing and transformation"""
    
    def __init__(self):
        self.target_size = IMAGE_CONFIG["target_size"]
        self.normalize_mean = IMAGE_CONFIG["normalize_mean"]
        self.normalize_std = IMAGE_CONFIG["normalize_std"]
        
        # Initialize transformations
        self.normalize = A.Normalize(
            mean=self.normalize_mean, 
            std=self.normalize_std
        )
        
        self.val_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.CenterCrop(self.target_size[0], self.target_size[1]),
            self.normalize
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model inference
        
        Args:
            image_path: Path to the image file or PIL Image object
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load image
            if isinstance(image_path, str):
                # Load from file path
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_path, Image.Image):
                # Convert PIL Image to numpy array
                image = np.array(image_path)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Already RGB
                    pass
                else:
                    # Convert to RGB if needed
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Unsupported image format")
            
            # Apply transformations
            transformed = self.val_transform(image=image)
            processed_image = transformed['image']
            
            # Convert to tensor and change format
            tensor_image = torch.tensor(processed_image, dtype=torch.float32)
            tensor_image = tensor_image.permute(2, 0, 1)  # HWC to CHW
            
            # Add batch dimension
            tensor_image = tensor_image.unsqueeze(0)
            
            return tensor_image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def preprocess_pil_image(self, pil_image):
        """
        Preprocess PIL Image for model inference
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        return self.preprocess_image(pil_image)
    
    def get_image_info(self, image_path):
        """
        Get basic information about an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Image information
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            return {
                "height": image.shape[0],
                "width": image.shape[1],
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "dtype": str(image.dtype)
            }
        except Exception as e:
            print(f"Error getting image info: {e}")
            return None

# Global image processor instance
image_processor = ImageProcessor() 