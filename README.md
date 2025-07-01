---
title: FoodClassifier-ViT
emoji: 🍕
colorFrom: indigo
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# FoodViT - Food Classification Application

A production-ready food classification application using Vision Transformer (ViT) that can classify images into three categories: **pizza**, **steak**, and **sushi**.

## 🍕 Features

- **Web Interface**: Beautiful Gradio web interface for easy image upload and classification
- **Command Line Tool**: Batch prediction capabilities for processing multiple images
- **High Accuracy**: Trained Vision Transformer model with excellent performance
- **Production Ready**: Modular, well-structured codebase with proper error handling
- **Dynamic Example Images**: Example images are randomly selected from `assets/samples/` at each app launch
- **Easy Deployment**: Simple setup and configuration
- **Model weights hosted on Hugging Face Hub**: The model file is not included in this repository; it is automatically downloaded from the Hugging Face Model Hub at runtime.

## 📁 Project Structure

```
FoodViT/
├── app.py                 # Main application entry point
├── predict.py            # Command-line prediction script
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── INSTALLATION.md      # Installation and troubleshooting guide
├── utils/
│   ├── model_loader.py  # Model loading utilities
│   ├── image_processor.py # Image preprocessing
│   └── predictor.py     # Prediction logic
├── interface/
│   └── gradio_app.py    # Gradio web interface
└── assets/
    └── samples/         # Example images for Gradio interface
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd FoodViT

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Interface

```bash
# Start the Gradio web interface
python app.py
```

The interface will be available at `http://localhost:7860`

### 3. Command Line Usage

```bash
# Predict a single image
python predict.py path/to/image.jpg

# Predict all images in a directory
python predict.py path/to/image/directory

# Get detailed prediction information
python predict.py path/to/image.jpg --detailed

# Save results to JSON file
python predict.py path/to/image/directory --output results.json
```

## 🎯 Usage Examples

### Web Interface

1. Open your browser and go to `http://localhost:7860`
2. Upload an image of pizza, steak, or sushi
3. View the prediction results with confidence scores
4. Try the example images provided (randomly selected from `assets/samples/`)

### Command Line

```bash
# Single image prediction
python predict.py pizza.jpg
# Output: ✅ pizza.jpg: Pizza (95.23%)

# Batch prediction with details
python predict.py test_images/ --detailed --output results.json
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Model settings**: Model path, device, image size
- **Class configuration**: Class names and mappings
- **Gradio interface**: Title, description, theme
- **Application settings**: Host, port, debug mode

## 🔧 Advanced Usage

### Custom Model Loading

```python
from utils.model_loader import ModelLoader

# Load custom model
loader = ModelLoader()
loader.load_model()
model = loader.get_model()
```

### Image Preprocessing

```python
from utils.image_processor import ImageProcessor

# Preprocess custom image
processor = ImageProcessor()
tensor = processor.preprocess_image("path/to/image.jpg")
```

### Direct Prediction

```python
from utils.predictor import FoodPredictor

# Initialize and predict
predictor = FoodPredictor()
predictor.initialize()
result = predictor.predict("path/to/image.jpg")
print(f"Predicted: {result['class']} ({result['confidence']:.2%})")
```

## 📊 Model Information

- **Architecture**: Vision Transformer (ViT-Base)
- **Input Size**: 224x224 pixels
- **Classes**: 3 (pizza, steak, sushi)
- **Training Data**: Pizza-Steak-Sushi dataset
- **Framework**: PyTorch with Transformers
- **Model weights**: Downloaded automatically from the Hugging Face Model Hub ([see model repo](https://huggingface.co/mahmoudalrefaey/FoodViT-weights))

## 🛠️ Development

### Project Structure

- **`utils/`**: Core utilities for model loading, image processing, and prediction
- **`interface/`**: Web interface components
- **`assets/samples/`**: Example images and static assets

### Adding New Features

1. **New Model**: Update `config.py` and `utils/model_loader.py`
2. **New Classes**: Modify `config.py` CLASS_CONFIG
3. **New Interface**: Create new files in `interface/`
4. **New Utilities**: Add to `utils/` directory

## 🧹 Project Cleanliness & GitHub Readiness

- All unnecessary files and caches have been removed
- Example images are dynamically loaded
- No test or debug files in the repo
- Model weights are not included in the repo (downloaded from the Hub)
- Ready for production and version control

## 🐛 Troubleshooting

See `INSTALLATION.md` for detailed troubleshooting, dependency, and environment tips.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

---

**Enjoy classifying your food images! 🍕🥩🍣** 