"""
Setup and utility functions
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import time


def setup_environment(seed=42):
    """
    Setup environment for reproducibility
    
    Args:
        seed: Random seed
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)

    return device


def create_dataloaders(train_dir, val_dir, test_dir, train_transform, val_transform, batch_size=32):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory
        train_transform: Training transformations
        val_transform: Validation transformations
        batch_size: Batch size
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset)
    """
    print("="*80)
    print("LOADING ORGANIZED DATASET")
    print("="*80)

    try:
        # Create datasets
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

        print("✓ Dataset loaded successfully!")
        print(f"  Training samples: {len(train_dataset):,}")
        print(f"  Validation samples: {len(val_dataset):,}")
        print(f"  Test samples: {len(test_dataset):,}")
        print(f"  Training batches: {len(train_loader):,}")
        print(f"  Classes: {train_dataset.classes}")

        # Verify class mapping
        class_to_idx = train_dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"\nClass mapping:")
        for idx, class_name in idx_to_class.items():
            print(f"  {idx}: {class_name}")

        print("="*80)

        return train_loader, val_loader, test_loader, train_dataset

    except FileNotFoundError as e:
        print(f"Dataset not found")
        print(f"   Error: {e}")
        print(f"\n   Please run dataset organization first:")
        print(f"   python -m src.data.dataset_organizer")
        raise


def save_model_checkpoint(model, filepath, **kwargs):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        filepath: Path to save checkpoint
        **kwargs: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Model checkpoint saved: {filepath}")


def load_model_checkpoint(model, filepath, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        filepath: Path to checkpoint
        device: Device to load on
        
    Returns:
        dict: Checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model checkpoint loaded: {filepath}")
    
    # Return metadata
    metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    return metadata
