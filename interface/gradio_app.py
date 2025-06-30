"""
Gradio interface for FoodViT
Provides a web interface for food classification
"""

import gradio as gr
import sys
import os
from PIL import Image
import numpy as np
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GRADIO_CONFIG, CLASS_CONFIG
from utils.predictor import predictor

SAMPLES_DIR = "assets/samples"
def get_random_examples(n=3):
    files = [os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
    return [[f] for f in random.sample(files, min(n, len(files)))] if files else []

def classify_food(image):
    """
    Classify food in the uploaded image
    
    Args:
        image: PIL Image object from Gradio
        
    Returns:
        tuple: (predicted_class, confidence, detailed_results)
    """
    if image is None:
        return "No image uploaded", 0.0, "Please upload an image to classify."
    
    try:
        # Make prediction
        result = predictor.predict(image)
        
        if not result.get("success", False):
            return "Error", 0.0, f"Prediction failed: {result.get('error', 'Unknown error')}"
        
        # Extract results
        predicted_class = result["class"]
        confidence = result["confidence"]
        
        # Create detailed results string
        detailed_results = f"**Predicted Class:** {predicted_class.title()}\n\n"
        detailed_results += f"**Confidence:** {confidence:.2%}\n\n"
        detailed_results += "**All Class Probabilities:**\n"
        
        for class_name, prob in result["probabilities"].items():
            detailed_results += f"- {class_name.title()}: {prob:.2%}\n"
        
        return predicted_class.title(), confidence, detailed_results
        
    except Exception as e:
        return "Error", 0.0, f"An error occurred: {str(e)}"

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Initialize predictor
    if not predictor.initialize():
        raise RuntimeError("Failed to initialize predictor")
    
    # Create interface
    with gr.Blocks(
        title=GRADIO_CONFIG["title"],
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown(f"# {GRADIO_CONFIG['title']}")
        gr.Markdown(GRADIO_CONFIG["description"])
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## Upload Image")
                input_image = gr.Image(
                    type="pil",
                    label="Upload a food image",
                    height=300
                )
                
                classify_btn = gr.Button(
                    "Classify Food",
                    variant="primary",
                    size="lg"
                )
                
                # Example images
                gr.Markdown("## Example Images")
                gr.Examples(
                    examples=get_random_examples(3),
                    inputs=input_image,
                    label="Try these examples"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("## Results")
                
                predicted_class = gr.Textbox(
                    label="Predicted Food Class",
                    interactive=False
                )
                
                confidence_score = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    label="Confidence Score",
                    interactive=False
                )
                
                detailed_results = gr.Markdown(
                    label="Detailed Results",
                    value="Upload an image and click 'Classify Food' to see results."
                )
        
        # Model information
        with gr.Accordion("Model Information", open=False):
            model_info = predictor.get_model_info()
            if "error" not in model_info:
                info_text = f"""
                **Device:** {model_info['device']}
                **Total Parameters:** {model_info['total_parameters']:,}
                **Number of Classes:** {model_info['num_classes']}
                **Classes:** {', '.join(model_info['class_names'])}
                """
            else:
                info_text = f"Error loading model info: {model_info['error']}"
            
            gr.Markdown(info_text)
        
        # Connect button to function
        classify_btn.click(
            fn=classify_food,
            inputs=input_image,
            outputs=[predicted_class, confidence_score, detailed_results]
        )
        
        # Auto-classify when image is uploaded
        input_image.change(
            fn=classify_food,
            inputs=input_image,
            outputs=[predicted_class, confidence_score, detailed_results]
        )
    
    return interface

def launch_interface():
    """Launch the Gradio interface"""
    interface = create_interface()
    
    # Launch with configuration
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    launch_interface() 