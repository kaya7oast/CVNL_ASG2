"""
Inference Script for Changi AeroVision
Real-time aircraft classification
"""

import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CNN_config as config
from src.models import CNN_ChangiAeroVisionModel


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


def main():
    """Main inference function with interactive menu"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model once
    print("\nLoading model...")
    model = CNN_ChangiAeroVisionModel(num_classes=config.NUM_CLASSES, pretrained=False)
    model = model.to(device)
    
    model_path = f'{config.MODELS_DIR}/best_model_phase2.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Model loaded from: {model_path}")
    
    # Create class mapping
    idx_to_class = {i: cls for i, cls in enumerate(sorted(config.AIRCRAFT_CLASSES[:config.NUM_CLASSES]))}
    
    # Create validation transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    # Interactive loop
    while True:
        print("\n" + "="*60)
        print("CHANGI AEROVISION - AIRCRAFT CLASSIFICATION")
        print("="*60)
        print("1. Predict aircraft class")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            # Get image path from user
            image_path = input("\nEnter the path to the aircraft image: ").strip()
            
            # Remove quotes if user added them
            image_path = image_path.strip('"').strip("'")
            
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"\n❌ Error: Image not found at: {image_path}")
                print("Please check the path and try again.")
                continue
            
            try:
                # Make prediction
                print(f"\nProcessing image: {image_path}")
                result = predict_single_image(model, image_path, transform, idx_to_class, device)
                
                # Display results
                print("\n")
                print("PREDICTION RESULTS")
                print(f"\nPredicted Class: {result['class']}")
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
                
            except Exception as e:
                print(f"\n❌ Error processing image: {e}")
                print("Please try another image.")
        
        elif choice == '2':
            print("\n👋 Exiting Changi AeroVision. Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1 or 2.")


if __name__ == '__main__':
    main()
