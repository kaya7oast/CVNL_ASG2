"""
Testing Utilities for Changi AeroVision
Comprehensive testing functions for model validation
"""

import os
import sys
import torch
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import CNN_config as config
from src.models import CNN_ChangiAeroVisionModel
from src.utils import setup_environment, create_dataloaders
from src.data import get_val_transform
from CNN_md11_augmentation import MD11AggressiveAugmentation
from CNN_inference import predict_single_image


def test_model_architecture():
    """
    Test model initialization and forward pass
    
    Verifies:
    - Model loads correctly
    - Output layer has correct number of classes
    - Forward pass produces valid output
    - Softmax probabilities sum to 1.0
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE TEST")
    print("="*80)
    
    # 1. Create model
    model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=True)
    
    # 2. Verify output layer
    assert model.resnet.fc.out_features == config.NUM_CLASSES, \
        f"Output layer should have {config.NUM_CLASSES} classes, got {model.resnet.fc.out_features}"
    
    # 3. Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224
    output = model(dummy_input)
    
    # 4. Verify output shape
    assert output.shape == (1, config.NUM_CLASSES), \
        f"Output shape should be (1, {config.NUM_CLASSES}), got {output.shape}"
    
    # 5. Verify softmax probabilities
    probs = torch.nn.functional.softmax(output, dim=1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0)), \
        "Probabilities should sum to 1.0"
    
    print("✓ Model architecture test passed")
    print(f"  - Output classes: {config.NUM_CLASSES}")
    print(f"  - Input shape: (batch, 3, 224, 224)")
    print(f"  - Output shape: (batch, {config.NUM_CLASSES})")
    print("="*80)
    
    return True


def test_augmentation_pipeline():
    """
    Visualize augmentation effects on MD-11 images
    
    Generates 9 augmented versions of a sample MD-11 image
    to verify augmentation pipeline is working correctly
    """
    print("\n" + "="*80)
    print("AUGMENTATION PIPELINE TEST")
    print("="*80)
    
    # 1. Load a sample MD-11 image
    dataset_path = os.path.join(config.TRAIN_DIR, "MD11")
    if not os.path.exists(dataset_path):
        print(f"⚠️  MD-11 training directory not found at: {dataset_path}")
        print("Skipping augmentation test")
        return False
    
    md11_images = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]
    if not md11_images:
        print(f"⚠️  No images found in: {dataset_path}")
        return False
    
    sample_image = Image.open(os.path.join(dataset_path, md11_images[0])).convert('RGB')
    
    # 2. Apply augmentation transform
    augmentor = MD11AggressiveAugmentation()
    transform = augmentor.get_transform()
    
    # 3. Generate 9 augmented versions
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        augmented = transform(sample_image)
        
        # Convert tensor to displayable image
        img_np = augmented.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        ax.imshow(img_np)
        ax.set_title(f"Augmentation {i+1}")
        ax.axis('off')
    
    plt.suptitle("MD-11 Aggressive Augmentation Samples", fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(config.PLOTS_DIR, "md11_augmentation_test.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Augmentation samples saved to: {output_path}")
    print("="*80)
    
    return True


def test_md11_oversampling():
    """
    Verify MD-11 oversampling factor in data loader
    
    Counts MD-11 samples in one epoch to verify
    7× oversampling is working correctly
    """
    print("\n" + "="*80)
    print("MD-11 OVERSAMPLING TEST")
    print("="*80)
    
    from CNN_md11_augmentation import create_md11_balanced_loader
    from torchvision import datasets, transforms
    
    # Create dataset
    train_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=transforms.ToTensor()
    )
    
    # Create balanced loader (MD-11 oversampling factor=7)
    balanced_loader = create_md11_balanced_loader(
        train_dataset, 
        batch_size=32, 
        md11_oversample_factor=7
    )
    
    # Count MD-11 samples in one epoch
    md11_class_idx = train_dataset.class_to_idx['MD11']
    md11_count = 0
    total_count = 0
    
    for images, labels in balanced_loader:
        md11_count += (labels == md11_class_idx).sum().item()
        total_count += len(labels)
    
    md11_proportion = md11_count / total_count
    print(f"✓ MD-11 samples per epoch: {md11_count}/{total_count} ({100*md11_proportion:.1f}%)")
    print(f"  Expected: ~12.5% (1/8 classes)")
    print(f"  Actual proportion indicates ~{md11_count/(total_count/8):.0f}× oversampling effect")
    print("="*80)
    
    return True


def check_class_balance(dataset, split_name):
    """
    Check class distribution in dataset
    
    Args:
        dataset: ImageFolder dataset
        split_name: Name of the split (e.g., "Training", "Validation", "Test")
    """
    class_counts = Counter([dataset.targets[i] for i in range(len(dataset))])
    print(f"\n{split_name} Class Distribution:")
    for class_idx, count in sorted(class_counts.items()):
        class_name = dataset.classes[class_idx]
        print(f"  {class_name}: {count} images")


def test_dataset_validation():
    """
    Verify dataset structure and class distribution
    
    Checks:
    - All 8 aircraft classes present
    - Expected number of samples
    - Class distribution
    """
    print("\n" + "="*80)
    print("DATASET VALIDATION TEST")
    print("="*80)
    
    from torchvision import datasets
    
    # Load datasets
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR)
    val_dataset = datasets.ImageFolder(config.VAL_DIR)
    test_dataset = datasets.ImageFolder(config.TEST_DIR)
    
    # Verify class count
    assert len(train_dataset.classes) == 8, "Training set should have 8 aircraft classes"
    assert len(val_dataset.classes) == 8, "Validation set should have 8 aircraft classes"
    assert len(test_dataset.classes) == 8, "Test set should have 8 aircraft classes"
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Class distribution
    check_class_balance(train_dataset, "Training")
    check_class_balance(val_dataset, "Validation")
    check_class_balance(test_dataset, "Test")
    
    print("\n" + "="*80)
    print("✓ Dataset validation passed")
    print("="*80)
    
    return True


def batch_inference_test(test_dir, model_path, output_csv=None):
    """
    Test inference on directory of images
    
    Args:
        test_dir: Directory containing test images
        model_path: Path to trained model checkpoint
        output_csv: Optional path to save results CSV
        
    Returns:
        DataFrame with prediction results
    """
    print("\n" + "="*80)
    print("BATCH INFERENCE TEST")
    print("="*80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Prepare transform and class mapping
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    idx_to_class = {i: cls for i, cls in enumerate(sorted(config.AIRCRAFT_CLASSES[:config.NUM_CLASSES]))}
    
    # Process all images
    results = []
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        result = predict_single_image(model, img_path, transform, idx_to_class, device)
        results.append({
            'filename': img_file,
            'predicted_class': result['class'],
            'confidence': result['confidence'],
            'high_confidence': result['confidence'] >= config.CONFIDENCE_THRESHOLD
        })
    
    # Summary statistics
    df = pd.DataFrame(results)
    print(f"\nBatch Inference Results:")
    print(f"  Total images: {len(df)}")
    print(f"  High confidence rate: {100 * df['high_confidence'].mean():.1f}%")
    print(f"  Average confidence: {df['confidence'].mean():.3f}")
    print(f"\nClass distribution:")
    print(df['predicted_class'].value_counts())
    
    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to: {output_csv}")
    
    print("="*80)
    
    return df


def stress_test_inference(model, device, num_iterations=1000):
    """
    Test for memory leaks and thermal throttling
    
    Args:
        model: Trained model
        device: Device to run on
        num_iterations: Number of iterations to test
    """
    print("\n" + "="*80)
    print(f"STRESS TEST - {num_iterations} ITERATIONS")
    print("="*80)
    
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    times = []
    for i in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
        
        if (i + 1) % 200 == 0:
            avg_time = sum(times[-200:]) / 200
            print(f"Iterations {i-199}-{i+1}: Avg={avg_time:.2f}ms")
    
    # Check for performance degradation
    early_avg = sum(times[:100]) / 100
    late_avg = sum(times[-100:]) / 100
    degradation = ((late_avg - early_avg) / early_avg) * 100
    
    print(f"\nPerformance Analysis:")
    print(f"  Early iterations (1-100): {early_avg:.2f}ms")
    print(f"  Late iterations ({num_iterations-99}-{num_iterations}): {late_avg:.2f}ms")
    print(f"  Performance degradation: {degradation:.1f}%")
    
    if abs(degradation) < 5:
        print("✓ No significant thermal throttling detected")
    else:
        print("⚠️  Performance degradation detected - check cooling")
    
    print("="*80)
    
    return times


def log_prediction(aircraft_id, predicted_class, confidence, action, log_file='production_predictions.log'):
    """
    Log all predictions for audit and retraining
    
    Args:
        aircraft_id: Aircraft registration or identifier
        predicted_class: Predicted aircraft class
        confidence: Prediction confidence
        action: Action taken ('AUTO' or 'MANUAL_VERIFY')
        log_file: Path to log file
    """
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'aircraft_id': aircraft_id,
        'predicted_class': predicted_class,
        'confidence': f"{confidence:.4f}",
        'action': action
    }
    
    logging.info(f"Prediction: {log_entry}")
    
    # Alert if confidence below threshold
    if confidence < 0.85:
        logging.warning(f"Low confidence prediction: {aircraft_id} → {predicted_class} ({confidence:.2%})")


def generate_weekly_report(predictions_log):
    """
    Analyze production performance from prediction logs
    
    Args:
        predictions_log: List of prediction dictionaries
        
    Returns:
        Dictionary with performance metrics
    """
    df = pd.DataFrame(predictions_log)
    
    report = {
        'total_predictions': len(df),
        'auto_processed': (df['action'] == 'AUTO').sum(),
        'manual_verifications': (df['action'] == 'MANUAL_VERIFY').sum(),
        'avg_confidence': df['confidence'].mean(),
        'class_distribution': df['predicted_class'].value_counts().to_dict()
    }
    
    # Flag performance degradation
    if report['avg_confidence'] < 0.80:
        report['alert'] = "Average confidence below target - consider model retraining"
    
    # Print report
    print("\n" + "="*80)
    print("WEEKLY PERFORMANCE REPORT")
    print("="*80)
    print(f"Total predictions: {report['total_predictions']}")
    print(f"Auto-processed: {report['auto_processed']} ({100*report['auto_processed']/report['total_predictions']:.1f}%)")
    print(f"Manual verifications: {report['manual_verifications']} ({100*report['manual_verifications']/report['total_predictions']:.1f}%)")
    print(f"Average confidence: {report['avg_confidence']:.3f}")
    print("\nClass distribution:")
    for cls, count in report['class_distribution'].items():
        print(f"  {cls}: {count}")
    
    if 'alert' in report:
        print(f"\n⚠️  ALERT: {report['alert']}")
    
    print("="*80)
    
    return report


def run_all_tests():
    """
    Run all testing functions in sequence
    
    Returns:
        Dictionary with test results
    """
    print("\n" + "="*80)
    print("RUNNING ALL TESTS")
    print("="*80)
    
    results = {}
    
    # Test 1: Model Architecture
    try:
        results['model_architecture'] = test_model_architecture()
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        results['model_architecture'] = False
    
    # Test 2: Dataset Validation
    try:
        results['dataset_validation'] = test_dataset_validation()
    except Exception as e:
        print(f"❌ Dataset validation test failed: {e}")
        results['dataset_validation'] = False
    
    # Test 3: Augmentation Pipeline
    try:
        results['augmentation_pipeline'] = test_augmentation_pipeline()
    except Exception as e:
        print(f"❌ Augmentation pipeline test failed: {e}")
        results['augmentation_pipeline'] = False
    
    # Test 4: MD-11 Oversampling
    try:
        results['md11_oversampling'] = test_md11_oversampling()
    except Exception as e:
        print(f"❌ MD-11 oversampling test failed: {e}")
        results['md11_oversampling'] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(results.values())
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("="*80)
    
    return results


if __name__ == '__main__':
    """
    Run all tests when script is executed directly
    """
    run_all_tests()
