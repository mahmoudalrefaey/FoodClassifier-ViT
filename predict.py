"""
Command-line prediction script for FoodViT
Allows batch prediction and testing of the model
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.predictor import predictor
from config import CLASS_CONFIG

def predict_single_image(image_path):
    """
    Predict food class for a single image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Prediction results
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        # Load image
        image = Image.open(image_path)
        
        # Make prediction
        result = predictor.predict(image)
        
        return result
        
    except Exception as e:
        return {"error": f"Error processing {image_path}: {str(e)}"}

def predict_batch_images(image_dir):
    """
    Predict food classes for all images in a directory
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        list: List of prediction results
    """
    results = []
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    try:
        # Get all image files in directory
        image_files = [
            f for f in os.listdir(image_dir)
            if Path(f).suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return results
        
        print(f"Found {len(image_files)} image files")
        
        # Process each image
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing {i}/{len(image_files)}: {filename}")
            
            result = predict_single_image(image_path)
            result['filename'] = filename
            results.append(result)
            
        return results
        
    except Exception as e:
        print(f"Error processing directory {image_dir}: {str(e)}")
        return results

def print_results(results, detailed=False):
    """
    Print prediction results in a formatted way
    
    Args:
        results: Single result dict or list of results
        detailed: Whether to print detailed information
    """
    if isinstance(results, dict):
        results = [results]
    
    for result in results:
        if "error" in result:
            filename = result.get('filename', 'Unknown')
            print(f"❌ {filename}: {result['error']}")
            continue
        
        if not result.get("success", False):
            filename = result.get('filename', 'Unknown')
            print(f"❌ {filename}: Prediction failed")
            continue
        
        # Extract information
        filename = result.get('filename', 'Image')
        predicted_class = result["class"]
        confidence = result["confidence"]
        
        # Print basic result
        print(f"✅ {filename}: {predicted_class.title()} ({confidence:.2%})")
        
        # Print detailed information if requested
        if detailed:
            print(f"   Class ID: {result['class_id']}")
            print("   All probabilities:")
            for class_name, prob in result["probabilities"].items():
                print(f"     - {class_name.title()}: {prob:.2%}")
            print()

def main():
    """Main function for command-line prediction"""
    
    parser = argparse.ArgumentParser(description="FoodViT - Command Line Prediction")
    parser.add_argument(
        "input",
        help="Image file path or directory containing images"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed prediction information"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save results (JSON format)"
    )
    
    args = parser.parse_args()
    
    print("FoodViT - Command Line Prediction")
    print("=" * 40)
    
    # Initialize predictor
    print("Initializing model...")
    if not predictor.initialize():
        print("Failed to initialize model")
        sys.exit(1)
    print("✓ Model initialized successfully")
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image prediction
        print(f"Predicting single image: {args.input}")
        result = predict_single_image(args.input)
        print_results([result], args.detailed)
        results = [result]
        
    elif input_path.is_dir():
        # Batch prediction
        print(f"Predicting images in directory: {args.input}")
        results = predict_batch_images(args.input)
        print_results(results, args.detailed)
        
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)
    
    # Save results if output file specified
    if args.output and results:
        try:
            import json
            # Convert numpy types to native Python types for JSON serialization
            json_results = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if key == 'probabilities':
                        json_result[key] = {k: float(v) for k, v in value.items()}
                    elif isinstance(value, (int, float, str, bool)):
                        json_result[key] = value
                    else:
                        json_result[key] = str(value)
                json_results.append(json_result)
            
            with open(args.output, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Results saved to: {args.output}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    # Print summary
    successful_predictions = [r for r in results if r.get("success", False)]
    failed_predictions = len(results) - len(successful_predictions)
    
    print(f"\nSummary:")
    print(f"Total images: {len(results)}")
    print(f"Successful predictions: {len(successful_predictions)}")
    print(f"Failed predictions: {failed_predictions}")

if __name__ == "__main__":
    main() 