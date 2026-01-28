"""
Comprehensive Test Runner for Changi AeroVision
Execute all tests from documentation Section 7

Usage:
    python run_tests.py --all                    # Run all tests
    python run_tests.py --model-arch             # Test model architecture only
    python run_tests.py --dataset                # Test dataset validation only
    python run_tests.py --augmentation           # Test augmentation pipeline only
    python run_tests.py --batch-inference DIR    # Test batch inference on directory
    python run_tests.py --stress-test            # Run inference stress test
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_utils import (
    test_model_architecture,
    test_dataset_validation,
    test_augmentation_pipeline,
    test_md11_oversampling,
    batch_inference_test,
    stress_test_inference,
    run_all_tests
)
from src.models import CNN_ChangiAeroVisionModel
from src.utils import setup_environment
from src import CNN_config as config
import torch


def test_inference_speed(model_path='outputs/models/best_model_phase2.pth'):
    """
    Test inference speed benchmark
    
    Args:
        model_path: Path to trained model
    """
    print("\n" + "="*80)
    print("INFERENCE SPEED TEST")
    print("="*80)
    
    # Setup
    device = setup_environment(seed=config.SEED)
    model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=False)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first using: python main.py")
        return False
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Run stress test
    times = stress_test_inference(model, device, num_iterations=100)
    
    import numpy as np
    avg_time = np.mean(times)
    median_time = np.median(times)
    p95_time = np.percentile(times, 95)
    
    print(f"\nInference Time Statistics (100 iterations):")
    print(f"  Mean: {avg_time:.2f}ms")
    print(f"  Median: {median_time:.2f}ms")
    print(f"  95th percentile: {p95_time:.2f}ms")
    print(f"  Max: {max(times):.2f}ms")
    
    # Check requirement
    requirement = 100  # ms
    if avg_time < requirement:
        print(f"\n✓ MEETS REAL-TIME REQUIREMENT")
        print(f"  Target: <{requirement}ms")
        print(f"  Achieved: {avg_time:.2f}ms")
        print(f"  Margin: {requirement - avg_time:.2f}ms")
        return True
    else:
        print(f"\n❌ DOES NOT MEET REAL-TIME REQUIREMENT")
        print(f"  Target: <{requirement}ms")
        print(f"  Current: {avg_time:.2f}ms")
        return False


def test_deployment_simulation(model_path='outputs/models/best_model_phase2.pth'):
    """
    Test multi-angle deployment simulation
    
    Args:
        model_path: Path to trained model
    """
    print("\n" + "="*80)
    print("DEPLOYMENT SIMULATION TEST")
    print("="*80)
    
    from src.deployment import test_deployment_simulation
    from src.utils import create_dataloaders
    from src.data import get_val_transform
    
    # Setup
    device = setup_environment(seed=config.SEED)
    model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=False)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return False
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Load test data
    _, _, test_loader, train_dataset = create_dataloaders(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        get_val_transform(), get_val_transform(), batch_size=32
    )
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    val_transform = get_val_transform()
    
    # Run simulation
    test_deployment_simulation(
        model, test_loader, val_transform, idx_to_class,
        conf_threshold=0.85, device=device, num_test_samples=10
    )
    
    return True


def test_gradcam_analysis(model_path='outputs/models/best_model_phase2.pth', 
                         image_path=None):
    """
    Test Grad-CAM visualization
    
    Args:
        model_path: Path to trained model
        image_path: Path to MD-11 image (optional)
    """
    print("\n" + "="*80)
    print("GRAD-CAM VISUALIZATION TEST")
    print("="*80)
    
    from src.evaluation.CNN_gradcam import analyze_md11_confusion
    
    # Setup
    device = setup_environment(seed=config.SEED)
    model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=False)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return False
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Find an MD-11 test image if not provided
    if image_path is None:
        md11_test_dir = os.path.join(config.TEST_DIR, 'MD11')
        if os.path.exists(md11_test_dir):
            md11_images = [f for f in os.listdir(md11_test_dir) if f.endswith(('.jpg', '.png'))]
            if md11_images:
                image_path = os.path.join(md11_test_dir, md11_images[0])
                print(f"Using test image: {image_path}")
    
    if image_path is None or not os.path.exists(image_path):
        print("⚠️  No MD-11 test image found. Skipping Grad-CAM test.")
        print("You can provide an image path with --gradcam-image flag")
        return False
    
    # Run analysis
    analyze_md11_confusion(model, image_path)
    
    return True


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Changi AeroVision Test Suite')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--model-arch', action='store_true',
                       help='Test model architecture')
    parser.add_argument('--dataset', action='store_true',
                       help='Test dataset validation')
    parser.add_argument('--augmentation', action='store_true',
                       help='Test augmentation pipeline')
    parser.add_argument('--md11-oversample', action='store_true',
                       help='Test MD-11 oversampling')
    parser.add_argument('--inference-speed', action='store_true',
                       help='Test inference speed benchmark')
    parser.add_argument('--batch-inference', type=str, metavar='DIR',
                       help='Test batch inference on directory')
    parser.add_argument('--stress-test', action='store_true',
                       help='Run inference stress test (1000 iterations)')
    parser.add_argument('--deployment-sim', action='store_true',
                       help='Test deployment simulation')
    parser.add_argument('--gradcam', action='store_true',
                       help='Test Grad-CAM visualization')
    parser.add_argument('--gradcam-image', type=str,
                       help='Path to image for Grad-CAM analysis')
    parser.add_argument('--model-path', type=str, 
                       default='outputs/models/best_model_phase2.pth',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Run tests based on arguments
    results = {}
    
    if args.all:
        print("\n" + "="*80)
        print("RUNNING ALL TESTS")
        print("="*80)
        
        results['basic_tests'] = run_all_tests()
        results['inference_speed'] = test_inference_speed(args.model_path)
        results['deployment_sim'] = test_deployment_simulation(args.model_path)
        results['gradcam'] = test_gradcam_analysis(args.model_path, args.gradcam_image)
    else:
        if args.model_arch:
            results['model_architecture'] = test_model_architecture()
        
        if args.dataset:
            results['dataset_validation'] = test_dataset_validation()
        
        if args.augmentation:
            results['augmentation_pipeline'] = test_augmentation_pipeline()
        
        if args.md11_oversample:
            results['md11_oversampling'] = test_md11_oversampling()
        
        if args.inference_speed:
            results['inference_speed'] = test_inference_speed(args.model_path)
        
        if args.batch_inference:
            if not os.path.exists(args.batch_inference):
                print(f"❌ Directory not found: {args.batch_inference}")
            else:
                results['batch_inference'] = batch_inference_test(
                    args.batch_inference, 
                    args.model_path,
                    output_csv='batch_inference_results.csv'
                )
        
        if args.stress_test:
            device = setup_environment(seed=config.SEED)
            model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=False)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model = model.to(device)
            results['stress_test'] = stress_test_inference(model, device, num_iterations=1000)
        
        if args.deployment_sim:
            results['deployment_sim'] = test_deployment_simulation(args.model_path)
        
        if args.gradcam:
            results['gradcam'] = test_gradcam_analysis(args.model_path, args.gradcam_image)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    if results:
        for test_name, result in results.items():
            if isinstance(result, bool):
                status = "✓ PASS" if result else "❌ FAIL"
                print(f"{test_name}: {status}")
            elif isinstance(result, dict):
                # For run_all_tests result
                passed = sum(1 for v in result.values() if v)
                total = len(result)
                print(f"{test_name}: {passed}/{total} tests passed")
    
    print("="*80)


if __name__ == '__main__':
    main()
