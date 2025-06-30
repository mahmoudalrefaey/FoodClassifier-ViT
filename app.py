"""
Main application file for FoodViT
Entry point for the food classification application
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import APP_CONFIG
from interface.gradio_app import launch_interface
from utils.predictor import predictor

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'torch',
        'transformers',
        'gradio',
        'PIL',
        'cv2',
        'albumentations',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_model_file():
    """Check if the model file exists"""
    model_path = Path("model/bestViT_PT.pth")
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print("Please ensure the trained model file is in the model/ directory")
        return False
    return True

def main():
    """Main function to run the application"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FoodViT - Food Classification Application")
    parser.add_argument(
        "--port", 
        type=int, 
        default=APP_CONFIG["port"],
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default=APP_CONFIG["host"],
        help="Host to run the server on"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create a public link for the interface"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("FoodViT - Food Classification Application")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ All dependencies available")
    
    # Check model file
    print("Checking model file...")
    if not check_model_file():
        sys.exit(1)
    print("✓ Model file found")
    
    # Initialize predictor
    print("Initializing model...")
    if not predictor.initialize():
        print("✗ Failed to initialize model")
        sys.exit(1)
    print("✓ Model initialized successfully")
    
    # Get model info
    model_info = predictor.get_model_info()
    if "error" not in model_info:
        print(f"✓ Model loaded on {model_info['device']}")
        print(f"✓ Total parameters: {model_info['total_parameters']:,}")
    
    print("\nStarting Gradio interface...")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    
    try:
        # Launch the interface
        launch_interface()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 