# Installation Guide for FoodViT

## Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended)
- GPU support optional but recommended for faster inference

## Installation Steps

### 1. Clone or Download the Project

Make sure you have all the project files in your directory:
- `app.py` - Main application
- `predict.py` - Command line tool
- `config.py` - Configuration
- `requirements.txt` - Dependencies
- `model/bestViT_PT.pth` - Trained model
- All utility and interface files

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv foodvit_env

# Activate virtual environment
# On Windows:
foodvit_env\Scripts\activate
# On macOS/Linux:
source foodvit_env/bin/activate
```

### 3. Install Dependencies

```bash
# Install PyTorch first (choose appropriate version for your system)
# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA (if you have NVIDIA GPU):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 4. Troubleshooting Dependency Issues

If you encounter dependency conflicts, try this step-by-step approach:

```bash
# 1. Install core dependencies first
pip install torch torchvision
pip install transformers==4.28.0
pip install huggingface-hub==0.15.1
pip install accelerate==0.20.3

# 2. Install image processing libraries
pip install Pillow opencv-python albumentations

# 3. Install Gradio
pip install gradio==3.35.2

# 4. Install other utilities
pip install numpy scikit-learn datasets
```

### 5. Alternative: Use Conda

If you prefer conda:

```bash
# Create conda environment
conda create -n foodvit python=3.9
conda activate foodvit

# Install PyTorch
conda install pytorch torchvision -c pytorch

# Install other packages
pip install transformers==4.28.0 huggingface-hub==0.15.1
pip install gradio==3.35.2
pip install -r requirements.txt
```

## Testing the Installation

### 1. Run Basic Tests

```bash
python simple_test.py
```

This should show all tests passing.

### 2. Test the Web Interface

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

### 3. Test Command Line Tool

```bash
# Test help
python predict.py --help

# Test with a sample image (if you have one)
python predict.py path/to/your/image.jpg
```

## Common Issues and Solutions

### Issue: "cannot import name 'split_torch_state_dict_into_shards'"

**Solution**: This is a version compatibility issue. Try:

```bash
pip uninstall huggingface-hub transformers accelerate
pip install huggingface-hub==0.15.1 transformers==4.28.0 accelerate==0.20.3
```

### Issue: CUDA/GPU not working

**Solution**: 
1. Check if you have NVIDIA GPU
2. Install appropriate CUDA version
3. Install PyTorch with CUDA support
4. Or set device to 'cpu' in `config.py`

### Issue: Model file not found

**Solution**: Ensure `model/bestViT_PT.pth` exists in the project directory.

### Issue: Memory errors

**Solution**: 
1. Close other applications
2. Use CPU instead of GPU
3. Reduce batch size in configuration

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 500MB disk space

### Recommended Requirements
- Python 3.9+
- 8GB RAM
- NVIDIA GPU with CUDA support
- 1GB disk space

## Verification

After successful installation, you should be able to:

1. ✅ Run `python simple_test.py` without errors
2. ✅ Start the web interface with `python app.py`
3. ✅ Use command line tool with `python predict.py --help`
4. ✅ Upload images and get predictions in the web interface

## Getting Help

If you encounter issues:

1. Check the error messages carefully
2. Ensure all dependencies are installed correctly
3. Try the troubleshooting steps above
4. Check if your Python version is compatible
5. Verify the model file exists and is not corrupted

## Next Steps

Once installation is complete:

1. **Web Interface**: Run `python app.py` and visit `http://localhost:7860`
2. **Command Line**: Use `python predict.py` for batch processing
3. **Customization**: Edit `config.py` to modify settings
4. **Development**: Use the modular structure for extending functionality 