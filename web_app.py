"""
Changi AeroVision - Web Application
Simple Flask web interface for aircraft classification
"""

from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import base64
from torchvision import transforms
from src.models import CNN_ChangiAeroVisionModel
from src.CNN_config import NUM_CLASSES, CLASS_NAMES
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_ChangiAeroVisionModel(num_classes=NUM_CLASSES, pretrained=False)
model_path = 'outputs/models/best_model_phase2.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from {model_path}")
else:
    print(f"⚠ Model not found at {model_path}")

# Image preprocessing
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess image
        image = Image.open(file.stream).convert('RGB')
        image_tensor = val_transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
            
        # Prepare response
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Create probability distribution
        class_probs = [
            {'class': CLASS_NAMES[i], 'probability': float(all_probs[i] * 100)}
            for i in range(len(CLASS_NAMES))
        ]
        # Sort by probability
        class_probs = sorted(class_probs, key=lambda x: x['probability'], reverse=True)
        
        # Determine confidence level and action
        if confidence_score >= 85:
            confidence_level = 'HIGH'
            action = 'Auto-assignment approved'
            color = 'success'
        elif confidence_score >= 70:
            confidence_level = 'MEDIUM'
            action = 'Manual verification recommended'
            color = 'warning'
        else:
            confidence_level = 'LOW'
            action = 'Manual verification required'
            color = 'danger'
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence_score, 2),
            'confidence_level': confidence_level,
            'action': action,
            'color': color,
            'class_probabilities': class_probs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info')
def info():
    """Model information endpoint"""
    return jsonify({
        'model': 'ResNet50 Transfer Learning',
        'accuracy': '87.36%',
        'classes': CLASS_NAMES,
        'device': str(device),
        'inference_time': '6.73ms (avg)',
        'confidence_threshold': '85%'
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("CHANGI AEROVISION - WEB APPLICATION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model loaded: {os.path.exists(model_path)}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print("="*70 + "\n")
    app.run(debug=True, port=5000)
