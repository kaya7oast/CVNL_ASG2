"""
Configuration and Hyperparameters for Changi AeroVision
"""

import os
import torch

# Random seed for reproducibility
SEED = 42

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset paths - Google Colab structure
# Note: FGVC dataset will be reorganized into this structure
FGVC_ROOT = '/content/drive/MyDrive/Colab Notebooks/aircraft_dataset/fgvc-aircraft-2013b/data'
DATA_ROOT = '/content/drive/MyDrive/Colab Notebooks/aircraft_dataset'  # Organized dataset output
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

# Aircraft classes representing 92% of Changi's traffic
AIRCRAFT_CLASSES = [
    'B737',      # Boeing 737 - 32% of movements
    'A320',      # Airbus A320 - 28% of movements
    'B777',      # Boeing 777 - 15% of movements
    'A330',      # Airbus A330 - 8% of movements
    'A380',      # Airbus A380 - 3% of movements (special handling)
    'B787',      # Boeing 787 - 2% of movements
    'ATR72',     # ATR 72 - 2% of movements (regional turboprop)
    'CRJ900',    # Bombardier CRJ - 1% of movements
    'E190',      # Embraer E-Jet - 1% of movements
    'MD11'       # MD-11 - rare but unique (cargo operations)
]

# NOTE: FGVC dataset only contains 8 of these 10 classes (B787 and E190 missing)
NUM_CLASSES = 8  # Updated to match actual FGVC dataset availability

# Training hyperparameters
BATCH_SIZE = 32
IMG_SIZE = 224
LEARNING_RATE_PHASE1 = 0.001  # Phase 1: Train classifier head only
LEARNING_RATE_PHASE2 = 0.0001  # Phase 2: Fine-tune entire network
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20
CONFIDENCE_THRESHOLD = 0.85  # Operational requirement

# Operational parameters
ANNUAL_MOVEMENTS = 400000
AVG_DELAY_REDUCTION = 2.5  # minutes
COST_PER_DELAY_MINUTE = 180  # SGD

# Output directories
OUTPUT_DIR = 'outputs'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Variant to family mapping for the 10 Changi classes
# FGVC uses specific variants (e.g., "737-300", "737-700"), we map to families
VARIANT_TO_FAMILY = {
    # Boeing 737 family
    '737-300': 'B737', '737-400': 'B737', '737-500': 'B737',
    '737-600': 'B737', '737-700': 'B737', '737-800': 'B737', '737-900': 'B737',

    # Airbus A320 family
    'A320': 'A320', 'A321': 'A320', 'A319': 'A320', 'A318': 'A320',

    # Boeing 777
    '777-200': 'B777', '777-300': 'B777',

    # Airbus A330
    'A330-200': 'A330', 'A330-300': 'A330',

    # Airbus A380
    'A380': 'A380',

    # Boeing 787
    '787-8': 'B787', '787-9': 'B787',

    # ATR 72
    'ATR-72': 'ATR72',

    # Bombardier CRJ900
    'CRJ-900': 'CRJ900',

    # Embraer E190
    'ERJ 190': 'E190', 'E190': 'E190',

    # MD-11
    'MD-11': 'MD11'
}

# Operational parameters for each aircraft class
OPERATIONAL_PARAMS = {
    'B737': {'traffic': 32, 'handling': 'Standard narrow-body', 'risk': 'Low'},
    'A320': {'traffic': 28, 'handling': 'Standard narrow-body', 'risk': 'Low'},
    'B777': {'traffic': 15, 'handling': 'Long-haul intensive', 'risk': 'Medium'},
    'A330': {'traffic': 8, 'handling': 'Medium-haul', 'risk': 'Medium'},
    'A380': {'traffic': 3, 'handling': 'Special double-deck', 'risk': 'Critical'},
    'B787': {'traffic': 2, 'handling': 'Modern composite', 'risk': 'Medium'},
    'ATR72': {'traffic': 2, 'handling': 'Regional turboprop', 'risk': 'Low'},
    'CRJ900': {'traffic': 1, 'handling': 'Regional jet', 'risk': 'Low'},
    'E190': {'traffic': 1, 'handling': 'Regional alternative', 'risk': 'Low'},
    'MD11': {'traffic': 1, 'handling': 'Cargo special', 'risk': 'Medium'}
}

def print_config():
    """Print configuration summary"""
    print("="*60)
    print("CHANGI AEROVISION (CNN) - CONFIGURATION")
    print("="*60)
    print(f"FGVC source: {FGVC_ROOT}")
    print(f"Organized dataset: {DATA_ROOT}")
    print(f"Target aircraft classes: {len(AIRCRAFT_CLASSES)}")
    print(f"Available in FGVC dataset: {NUM_CLASSES} classes")
    print(f"Missing from FGVC: B787, E190")
    print(f"Classes: {', '.join(AIRCRAFT_CLASSES)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Phase 1 epochs: {EPOCHS_PHASE1} (frozen backbone)")
    print(f"Phase 2 epochs: {EPOCHS_PHASE2} (fine-tuning)")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Device: {DEVICE}")
    print("="*60)
