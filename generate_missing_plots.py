"""
Generate Missing Visualization Plots
- Figure 4: Inference Time Benchmark
- Figure 5: Sample Predictions
"""

import os
import sys
import torch
import time
from torchvision import transforms

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CNN_config as config
from src.models import CNN_ChangiAeroVisionModel
from src.utils import setup_environment, create_dataloaders
from src.evaluation import plot_inference_time, plot_sample_predictions


def benchmark_inference_time(model, device, img_size=224, num_iterations=100):
    """
    Benchmark model inference speed
    
    Args:
        model: Model to benchmark
        device: Device to run on
        img_size: Input image size
        num_iterations: Number of iterations
        
    Returns:
        list: Inference times in milliseconds
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warm-up GPU
    print("\nWarming up GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {num_iterations} inference iterations...")
    inference_times = []
    
    for i in range(num_iterations):
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        inference_times.append(inference_time_ms)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_iterations} iterations")
    
    return inference_times


def main():
    """Generate missing plots"""
    
    print("="*80)
    print("GENERATING MISSING VISUALIZATION PLOTS")
    print("="*80)
    
    # Setup environment
    device = setup_environment(seed=config.SEED)
    
    # Create model
    print("\nLoading model...")
    model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=False)
    model = model.to(device)
    
    # Load trained weights
    model_path = f'{config.MODELS_DIR}/best_model_phase2.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run CNN_train.py first to train the model.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Model loaded from: {model_path}")
    
    # ========================================
    # FIGURE 4: Inference Time Benchmark
    # ========================================
    print("\n" + "="*80)
    print("GENERATING FIGURE 4: INFERENCE TIME BENCHMARK")
    print("="*80)
    
    inference_times = benchmark_inference_time(
        model, device, 
        img_size=config.IMG_SIZE, 
        num_iterations=100
    )
    
    # Calculate statistics
    import numpy as np
    mean_time = np.mean(inference_times)
    median_time = np.median(inference_times)
    p95_time = np.percentile(inference_times, 95)
    max_time = np.max(inference_times)
    
    print(f"\n{'='*80}")
    print("INFERENCE TIME STATISTICS")
    print(f"{'='*80}")
    print(f"Mean:        {mean_time:.2f} ms")
    print(f"Median:      {median_time:.2f} ms")
    print(f"95th %ile:   {p95_time:.2f} ms")
    print(f"Max:         {max_time:.2f} ms")
    print(f"Requirement: <100 ms")
    print(f"Safety Margin: {100 - mean_time:.2f} ms")
    print(f"{'='*80}")
    
    # Plot inference time
    plot_inference_time(
        inference_times,
        operational_requirement=100,
        save_path='outputs/plots/inference_time_benchmark.png'
    )
    
    print("\n✓ Figure 4 saved to: outputs/plots/inference_time_benchmark.png")
    
    # ========================================
    # FIGURE 5: Sample Predictions
    # ========================================
    print("\n" + "="*80)
    print("GENERATING FIGURE 5: SAMPLE PREDICTIONS")
    print("="*80)
    
    # Create validation transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    # Create data loaders
    _, _, test_loader, train_dataset = create_dataloaders(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        val_transform, val_transform, config.BATCH_SIZE
    )
    
    # Get class mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Plot sample predictions
    plot_sample_predictions(
        model, test_loader, idx_to_class, device,
        num_samples=12,
        save_path='outputs/plots/prediction_examples.png'
    )
    
    print("\n✓ Figure 5 saved to: outputs/plots/prediction_examples.png")
    
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  - outputs/plots/inference_time_benchmark.png")
    print("  - outputs/plots/prediction_examples.png")
    print("\nYou can now view these in your documentation!")


if __name__ == '__main__':
    main()
