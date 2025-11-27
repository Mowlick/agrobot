"""
PyTorch model loading module for AgroBot
Based on the CNN model architecture from cnn_model.py
"""

import os
import torch
import torch.nn as nn

class PlantCNN(nn.Module):
    """CNN architecture for plant disease detection"""
    
    def __init__(self, num_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),  nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(model_path, num_classes=38):
    """
    Load the trained PyTorch model
    
    Args:
        model_path (str): Path to the .pth model file
        num_classes (int): Number of output classes (default: 38)
    
    Returns:
        tuple: (model, device) - Loaded model and device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = PlantCNN(num_classes=num_classes)
    
    # Load state dict
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"✅ Model loaded from {model_path}")
            print(f"📍 Using device: {device}")
            return model, device
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def get_model_info():
    """Get model architecture information"""
    model = PlantCNN()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'architecture': 'PlantCNN',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_size': (3, 224, 224),  # Standard ImageNet size
        'num_classes': 38
    }
    
    return info

if __name__ == '__main__':
    # Test model loading
    try:
        model, device = load_model('best_model.pth')
        print("✅ Model loaded successfully!")
        print(f"📍 Device: {device}")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"🎯 Output shape: {output.shape}")
        
        # Print model info
        info = get_model_info()
        print("\n📊 Model Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
