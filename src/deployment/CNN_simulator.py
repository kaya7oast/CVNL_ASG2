"""
Real-World Deployment Simulation
Multi-angle verification and inference benchmarking
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from ..CNN_config import IMG_SIZE


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def CNN_simulate_changi_deployment(model, image, val_transform, num_angles=3, 
                               conf_threshold=0.85, device='cuda'):
    """
    Simulate real-world CNN deployment with multi-angle verification
    Following the pseudocode from the deployment protocol
    
    Args:
        model: Trained model
        image: Input image tensor
        val_transform: Validation transform
        num_angles: Number of camera angles to simulate
        conf_threshold: Confidence threshold
        device: Device to run on
        
    Returns:
        dict: Deployment decision with prediction, confidence, status, and action
    """
    model.eval()
    predictions = []
    confidences = []

    # Simulate different camera angles with slight transformations
    angles = ['side', 'front', 'three_quarter']

    with torch.no_grad():
        for angle_idx in range(num_angles):
            # Apply slight transformation to simulate different angles
            angle_transform = transforms.Compose([
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
            ])

            # Get transformed view
            transformed = angle_transform(TF.to_pil_image(denormalize_image(image.cpu())))
            transformed = val_transform(transformed).unsqueeze(0).to(device)

            # Make prediction
            output = model(transformed)
            prob = F.softmax(output, dim=1)
            confidence, prediction = torch.max(prob, 1)

            # Check if confidence meets threshold
            if confidence.item() >= conf_threshold:
                predictions.append(prediction.item())
                confidences.append(confidence.item())

    # Consensus decision
    if len(predictions) > 0 and len(set(predictions)) == 1:
        # All angles agree
        final_prediction = predictions[0]
        avg_confidence = np.mean(confidences)
        status = "CONSENSUS"
        action = "Update_Ground_System"
    elif len(predictions) > 0:
        # Disagreement between angles
        final_prediction = max(set(predictions), key=predictions.count)
        avg_confidence = np.mean([c for p, c in zip(predictions, confidences)
                                 if p == final_prediction])
        status = "PARTIAL_AGREEMENT"
        action = "Flag_For_Manual_Verification"
    else:
        # No high confidence predictions
        final_prediction = None
        avg_confidence = 0.0
        status = "LOW_CONFIDENCE"
        action = "Flag_For_Manual_Verification"

    return {
        'prediction': final_prediction,
        'confidence': avg_confidence,
        'status': status,
        'action': action,
        'num_angles': len(predictions)
    }


def print_operational_action(pred_label):
    """
    Print operational instructions based on aircraft type
    
    Args:
        pred_label: Predicted aircraft class
    """
    print(f"\nOPERATIONAL ACTION:")
    if pred_label == 'A380':
        print(f"  - Auto-assign dual-deck boarding bridges (Type F)")
        print(f"  - Deploy specialized A380 cargo loaders")
        print(f"  - Allocate 120-minute turnaround slot")
    elif pred_label in ['B737', 'A320']:
        print(f"  - Assign standard boarding bridge")
        print(f"  - Deploy standard ground equipment")
        print(f"  - Allocate 45-minute quick-turn window")
    elif pred_label in ['ATR72', 'CRJ900']:
        print(f"  - Dispatch regional tow bars")
        print(f"  - Assign ground power unit Type 3")
        print(f"  - Route to regional terminal gates")
    else:
        print(f"  - Deploy standard equipment for {pred_label}")
        print(f"  - Follow standard operating procedures")


def test_deployment_simulation(model, test_loader, val_transform, idx_to_class, 
                               conf_threshold=0.85, device='cuda', num_test_samples=5):
    """
    Test deployment simulation on sample images
    
    Args:
        model: Trained model
        test_loader: Test data loader
        val_transform: Validation transform
        idx_to_class: Index to class name mapping
        conf_threshold: Confidence threshold
        device: Device to run on
        num_test_samples: Number of samples to test
    """
    print("="*80)
    print("REAL-WORLD DEPLOYMENT SIMULATION - Multi-Angle Verification")
    print("="*80)

    # Get test samples
    dataiter = iter(test_loader)
    test_images, test_labels = next(dataiter)

    # Test on samples
    num_test_samples = min(num_test_samples, len(test_images))
    deployment_results = []

    for idx in range(num_test_samples):
        image = test_images[idx].to(device)
        true_label = idx_to_class[test_labels[idx].item()]

        # Simulate deployment
        result = CNN_simulate_changi_deployment(model, image, val_transform, num_angles=3,
                                           conf_threshold=conf_threshold, device=device)

        print(f"\n{'='*80}")
        print(f"Aircraft {idx+1}: Ground Truth = {true_label}")
        print(f"{'='*80}")

        if result['prediction'] is not None:
            pred_label = idx_to_class[result['prediction']]
            is_correct = (result['prediction'] == test_labels[idx].item())

            print(f"Predicted: {pred_label}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Status: {result['status']}")
            print(f"Angles agreeing: {result['num_angles']}/3")
            print(f"Action: {result['action']}")
            print(f"Accuracy: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

            # Operational instructions
            if result['action'] == "Update_Ground_System":
                print_operational_action(pred_label)
            else:
                print(f"\nMANUAL VERIFICATION REQUIRED")
                print(f"  - Ground staff to visually confirm aircraft type")
                print(f"  - Cross-reference with flight plan")
        else:
            print(f"Status: {result['status']}")
            print(f"Action: {result['action']}")
            print(f"\nLOW CONFIDENCE - Manual identification required")

        deployment_results.append(result)

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"DEPLOYMENT SIMULATION SUMMARY")
    print(f"{'='*80}")

    consensus_count = sum(1 for r in deployment_results if r['status'] == 'CONSENSUS')
    partial_count = sum(1 for r in deployment_results if r['status'] == 'PARTIAL_AGREEMENT')
    low_conf_count = sum(1 for r in deployment_results if r['status'] == 'LOW_CONFIDENCE')

    print(f"Total samples tested: {num_test_samples}")
    print(f"Consensus (all angles agree): {consensus_count} ({100*consensus_count/num_test_samples:.1f}%)")
    print(f"Partial agreement: {partial_count} ({100*partial_count/num_test_samples:.1f}%)")
    print(f"Low confidence: {low_conf_count} ({100*low_conf_count/num_test_samples:.1f}%)")
    print(f"\nAutomatic processing rate: {100*(consensus_count)/num_test_samples:.1f}%")
    print(f"Manual verification required: {100*(partial_count + low_conf_count)/num_test_samples:.1f}%")
    print(f"{'='*80}")


def CNN_benchmark_inference_time(model, device, img_size=224, num_iterations=100):
    """
    Benchmark CNN inference time
    
    Args:
        model: Trained model
        device: Device to benchmark on
        img_size: Input image size
        num_iterations: Number of iterations to run
        
    Returns:
        dict: Inference time statistics
    """
    print("="*80)
    print("INFERENCE TIME BENCHMARK - Real-Time Performance")
    print("="*80)

    model.eval()

    # Warm-up
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark inference time
    inference_times = []

    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc="Benchmarking inference"):
            start_time = time.time()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms

    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)

    print(f"\nInference Time Statistics ({num_iterations} iterations):")
    print(f"  Average: {avg_inference_time:.2f} ms")
    print(f"  Std Dev: {std_inference_time:.2f} ms")
    print(f"  Min: {min_inference_time:.2f} ms")
    print(f"  Max: {max_inference_time:.2f} ms")

    # Operational requirement check
    operational_requirement = 100  # ms
    if avg_inference_time < operational_requirement:
        print(f"\n✓ MEETS REAL-TIME REQUIREMENT")
        print(f"  Target: <{operational_requirement}ms for moving aircraft")
        print(f"  Achieved: {avg_inference_time:.2f}ms")
        print(f"  Margin: {operational_requirement - avg_inference_time:.2f}ms")
    else:
        print(f"\n✗ DOES NOT MEET REAL-TIME REQUIREMENT")
        print(f"  Target: <{operational_requirement}ms")
        print(f"  Current: {avg_inference_time:.2f}ms")

    print(f"{'='*80}")

    return {
        'inference_times': inference_times,
        'avg_inference_time': avg_inference_time,
        'std_inference_time': std_inference_time,
        'min_inference_time': min_inference_time,
        'max_inference_time': max_inference_time
    }
