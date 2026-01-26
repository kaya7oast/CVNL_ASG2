"""
Evaluation Script for Changi AeroVision
Comprehensive model evaluation with visualizations
"""

import os
import sys
import argparse
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CNN_config as config
from src.models import CNN_ChangiAeroVisionModel
from src.data import get_val_transform
from src.training import validate
from src.evaluation import (
    evaluate_model, 
    calculate_confidence_stats,
    analyze_misclassifications,
    plot_confusion_matrix,
    plot_sample_predictions
)
from src.utils import setup_environment, create_dataloaders
from sklearn.metrics import confusion_matrix


def main(args):
    """Main evaluation function"""
    
    # Setup environment
    device = setup_environment(seed=config.SEED)
    
    # Get data transforms
    val_transform = get_val_transform()
    
    # Create data loaders
    _, _, test_loader, train_dataset = create_dataloaders(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        val_transform, val_transform, config.BATCH_SIZE
    )
    
    # Get class mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Create model and load weights
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=False)
    model = model.to(device)
    
    # Load best model from phase 2
    model_path = args.model_path or f'{config.MODELS_DIR}/best_model_phase2.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Loaded model from: {model_path}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device, phase_name="Test Set Evaluation"
    )
    
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*80}")
    
    # Confidence statistics
    conf_stats = calculate_confidence_stats(test_probs, config.CONFIDENCE_THRESHOLD)
    
    # Operational readiness assessment
    high_conf_rate = conf_stats['high_confidence'] / len(test_probs)
    if test_acc >= 85 and high_conf_rate >= 0.80:
        print(f"\n✓ PRODUCTION READY - Meets Changi operational requirements")
        print(f"  - Accuracy ≥85% (ICAO Standard): ✓ {test_acc:.2f}%")
        print(f"  - High confidence rate ≥80%: ✓ {100*high_conf_rate:.1f}%")
    elif test_acc >= 76.5:
        print(f"\nLIMITED DEPLOYMENT - Suitable for training scenarios")
        print(f"  - Accuracy: {test_acc:.2f}% (Target: ≥85%)")
    else:
        print(f"\n✗ NOT SUITABLE - Additional training required")
        print(f"  - Accuracy: {test_acc:.2f}% (Target: ≥85%)")
    
    # Detailed evaluation
    results = evaluate_model(test_preds, test_labels, test_probs, 
                            class_names, config.CONFIDENCE_THRESHOLD)
    
    # Confusion matrix and misclassification analysis
    cm, cm_normalized = plot_confusion_matrix(test_labels, test_preds, class_names)
    analyze_misclassifications(test_preds, test_labels, class_names, cm)
    
    # Sample predictions
    if args.visualize:
        plot_sample_predictions(model, test_loader, idx_to_class, device)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Changi AeroVision Model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    main(args)
