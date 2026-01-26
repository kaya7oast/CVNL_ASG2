"""
Quick Start Script for Changi AeroVision
Complete pipeline: organize dataset → train → evaluate
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import CNN_config as config
from src.models import CNN_ChangiAeroVisionModel
from src.training import CNN_train_two_phase
from src.evaluation import (
    evaluate_model, 
    calculate_confidence_stats,
    plot_confusion_matrix,
    plot_training_history,
    plot_business_impact
)
from src.deployment import CNN_benchmark_inference_time
from src.utils import setup_environment, create_dataloaders, save_model_checkpoint
from src.training import validate
import torch
from torchvision import transforms


def main():
    """Complete pipeline execution"""
    
    print("="*80)
    print("CHANGI AEROVISION - COMPLETE PIPELINE")
    print("="*80)
    
    # Step 1: Setup
    print("\n[1/5] Setting up environment...")
    device = setup_environment(seed=config.SEED)
    config.print_config()
    
    # Step 2: Create data transforms
    print("\n[2/5] Creating data transforms...")
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    # Step 3: Create data loaders
    print("\n[3/5] Creating data loaders...")
    
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        train_transform, val_transform, config.BATCH_SIZE
    )
    
    # Get class mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Step 4: Train model
    print("\n[4/5] Training model (two-phase)...")
    model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    model.print_model_info(config.NUM_CLASSES)
    
    results = CNN_train_two_phase(model, train_loader, val_loader, config, device)
    
    # Step 5: Evaluate
    print("\n[5/5] Evaluating on test set...")
    model.load_state_dict(torch.load(f'{config.MODELS_DIR}/best_model_phase2.pth'))
    
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, results['criterion'], device, 
        phase_name="Final Test Evaluation"
    )
    
    # Detailed evaluation
    eval_results = evaluate_model(
        test_preds, test_labels, test_probs,
        class_names, config.CONFIDENCE_THRESHOLD
    )
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(test_labels, test_preds, class_names)
    plot_training_history(
        results['history_phase1'], 
        results['history_phase2'],
        results['best_val_acc_phase2']
    )
    plot_business_impact(
        config.ANNUAL_MOVEMENTS,
        config.AVG_DELAY_REDUCTION,
        config.COST_PER_DELAY_MINUTE
    )
    
    # Step 6: Benchmark inference
    print("\n[6/6] Benchmarking inference time...")
    inference_stats = CNN_benchmark_inference_time(model, device)
    
    # Final summary
    print("\n" + "="*80)
    print("CHANGI AEROVISION - FINAL SUMMARY")
    print("="*80)
    print(f"\nTraining Complete")
    print(f"   - Phase 1 Best Accuracy: {results['best_val_acc_phase1']:.2f}%")
    print(f"   - Phase 2 Best Accuracy: {results['best_val_acc_phase2']:.2f}%")
    
    print(f"\nEvaluation Complete")
    print(f"   - Test Accuracy: {test_acc:.2f}%")
    print(f"   - Avg Confidence: {eval_results['avg_confidence']:.3f}")
    print(f"   - High Confidence Rate: {100*eval_results['high_confidence_count']/len(test_probs):.1f}%")
    
    print(f"\nPerformance Benchmarking Complete")
    print(f"   - Avg Inference Time: {inference_stats['avg_inference_time']:.2f}ms")
    
    # Operational readiness
    high_conf_rate = eval_results['high_confidence_count'] / len(test_probs)
    if test_acc >= 85 and high_conf_rate >= 0.80:
        print(f"\n🎉 PRODUCTION READY - Meets all operational requirements!")
    elif test_acc >= 76.5:
        print(f"\nLIMITED DEPLOYMENT - Suitable for training scenarios")
    else:
        print(f"\nNOT SUITABLE - Additional training required")
    
    print(f"\nOutputs saved to:")
    print(f"   - Models: {config.MODELS_DIR}/")
    print(f"   - Plots: {config.PLOTS_DIR}/")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
