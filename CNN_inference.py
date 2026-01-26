"""
Inference Script for Changi AeroVision
Real-time aircraft classification
"""

import os
import sys
import argparse
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CNN_config as config
from src.models import ChangiAeroVisionModel
from src.data import get_val_transform


def predict_single_image(model, image_path, transform, idx_to_class, device):
    """
    Predict aircraft class for a single image
    
    Args:
        model: Trained model
        image_path: Path to image
        transform: Image transform
        idx_to_class: Index to class mapping
        device: Device to run on
        
    Returns:
        dict: Prediction results
    """
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)
    
    pred_class = idx_to_class[prediction.item()]
    conf_value = confidence.item()
    
    return {
        'class': pred_class,
        'confidence': conf_value,
        'all_probs': {idx_to_class[i]: probs[0][i].item() for i in range(len(idx_to_class))}
    }


def main(args):
    """Main inference function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=False)
    model = model.to(device)
    
    model_path = args.model_path or f'{config.MODELS_DIR}/best_model_phase2.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Model loaded from: {model_path}")
    
    # Create class mapping (simplified for standalone use)
    # In production, this should be loaded from checkpoint metadata
    idx_to_class = {i: cls for i, cls in enumerate(sorted(config.AIRCRAFT_CLASSES[:config.NUM_CLASSES]))}
    
    # Get transform
    transform = get_val_transform()
    
    # Make prediction
    print(f"\nProcessing image: {args.image}")
    result = predict_single_image(model, args.image, transform, idx_to_class, device)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nAll Class Probabilities:")
    for cls, prob in sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {prob:.2%}")
    
    # Operational decision
    if result['confidence'] >= config.CONFIDENCE_THRESHOLD:
        print(f"\n✓ HIGH CONFIDENCE - Auto-assignment approved")
        print(f"  Action: Update ground system with {result['class']}")
    else:
        print(f"\n⚠️  LOW CONFIDENCE - Manual verification required")
        print(f"  Action: Flag for ground staff verification")
    
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on aircraft image')
    parser.add_argument('image', type=str, help='Path to aircraft image')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    main(args)
