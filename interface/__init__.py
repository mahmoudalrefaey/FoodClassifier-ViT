"""
Interface package for FoodViT
Contains web interface components
"""

from .gradio_app import create_interface, launch_interface

__all__ = [
    'create_interface',
    'launch_interface'
] 