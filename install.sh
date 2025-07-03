#!/bin/bash

# Create virtual environment
python -m venv security_env

# Activate virtual environment
source security_env/bin/activate  # On Windows: security_env\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
# mkdir -p images
mkdir -p logs

echo "Setup complete! Add known face images to the 'images' directory."
