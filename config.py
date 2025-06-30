"""
Configuration file for FoodViT project
Contains all model and application settings
"""

import os
import torch

# Model Configuration
MODEL_CONFIG = {
    "model_path": "model/bestViT_PT.pth",
    "feature_extractor_name": "google/vit-base-patch16-224",
    "num_labels": 3,
    "image_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Class Configuration
CLASS_CONFIG = {
    "class_names": ["pizza", "steak", "sushi"],
    "id2label": {0: "pizza", 1: "steak", 2: "sushi"},
    "label2id": {"pizza": 0, "steak": 1, "sushi": 2}
}

# Image Processing Configuration
IMAGE_CONFIG = {
    "target_size": (224, 224),
    "normalize_mean": [0.5, 0.5, 0.5],
    "normalize_std": [0.5, 0.5, 0.5]
}

# Gradio Interface Configuration
GRADIO_CONFIG = {
    "title": "FoodViT - Food Classification",
    "description": "Upload an image to classify it as pizza, steak, or sushi",
    "examples": [
        ["assets/example_pizza.jpg"],
        ["assets/example_steak.jpg"],
        ["assets/example_sushi.jpg"]
    ],
    "theme": "default"
}

# Application Configuration
APP_CONFIG = {
    "debug": False,
    "host": "127.0.0.1",
    "port": 7860,
    "share": False
} 