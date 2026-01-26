"""
Main Training Script for Changi AeroVision
Two-phase training: frozen backbone + fine-tuning
"""

import os
import sys
import argparse
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CNN_config as config
from src.models import ChangiAeroVisionModel
from src.data import get_train_transform, get_val_transform, organize_fgvc_dataset
from src.training import train_two_phase
from src.utils import setup_environment, create_dataloaders


def main(args):
    """Main training function"""
    
    # Print configuration
    config.print_config()
    
    # Setup environment
    device = setup_environment(seed=config.SEED)
    
    # Organize dataset if needed
    if args.organize_dataset:
        print("\n" + "="*80)
        print("ORGANIZING FGVC DATASET")
        print("="*80)
        organize_fgvc_dataset()
    
    # Get data transforms
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        train_transform, val_transform, config.BATCH_SIZE
    )
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    model = ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    model.print_model_info(config.NUM_CLASSES)
    
    # Two-phase training
    results = train_two_phase(model, train_loader, val_loader, config, device)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Phase 1 Best Accuracy: {results['best_val_acc_phase1']:.2f}%")
    print(f"Phase 2 Best Accuracy: {results['best_val_acc_phase2']:.2f}%")
    print(f"Models saved in: {config.MODELS_DIR}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Changi AeroVision Model')
    parser.add_argument('--organize-dataset', action='store_true',
                       help='Organize FGVC dataset before training')
    
    args = parser.parse_args()
    main(args)
