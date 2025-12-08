"""
Enhanced Flask App with Custom NLP and Multilingual Support
No pretrained NLP models - Pure custom implementation
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
from werkzeug.utils import secure_filename

# Import configuration
from config import MODEL_PATH, CLASS_NAMES_PATH, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, IMG_SIZE, NUM_CLASSES

# Demo user configuration
DEMO_USER = {
    'id': 1,
    'name': 'Demo User',
    'email': 'demo@example.com'
}

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from pytorch_model import load_model

# Import custom NLP modules
from multilingual_nlp_processor import multilingual_processor, process_message, process_image, reset

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'static')
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-for-sessions'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- MODEL LOADING ---
print(" Loading PyTorch model...")
model = None
device = None
class_names = []

try:
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, '..', 'model', 'best_model.pth')
    class_names_path = os.path.join(model_dir, '..', 'model', 'class_names.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file not found at {class_names_path}")
    
    model, device = load_model(model_path, num_classes=NUM_CLASSES)
    model.eval()
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    print(f" Model loaded successfully! Device: {device}")
    print(f"Number of classes: {len(class_names)}")
    
except Exception as e:
    print(f" Error loading model: {str(e)}")
    class_names = []

# Login required decorator
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render main page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Initialize session
    if 'user_language' not in session:
        session['user_language'] = 'en'
    if 'conversation_context' not in session:
        session['conversation_context'] = {'disease': None}
    
    return render_template('index.html')

@app.route('/demo-login', methods=['POST'])
def demo_login():
    """Handle demo user login"""
    # Set demo user data in session
    session['user_id'] = DEMO_USER['id']
    session['user_name'] = 'Demo User'
    
    return jsonify({
        'success': True,
        'message': 'Demo login successful',
        'user': {'name': 'Demo User'}
    })

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login (demo mode)"""
    if request.method == 'GET':
        return render_template('login.html')
        
    # In demo mode, any non-empty credentials will work
    email = request.json.get('email', '').strip()
    if not email:
        return jsonify({'success': False, 'error': 'Email is required'}), 400
        
    # Set demo user data in session
    session['user_id'] = DEMO_USER['id']
    session['user_name'] = email.split('@')[0]  # Use the part before @ as username
    
    return jsonify({
        'success': True, 
        'message': 'Login successful',
        'user': {'name': session['user_name']}
    })

@app.route('/register', methods=['POST'])
def register():
    """Handle new user registration (demo mode)"""
    name = request.json.get('name', '').strip()
    email = request.json.get('email', '').strip()
    
    if not email:
        return jsonify({'success': False, 'error': 'Email is required'}), 400
    
    # In demo mode, just set the username from email if name not provided
    username = name if name else email.split('@')[0]
    
    # Set demo user data in session
    session['user_id'] = DEMO_USER['id']
    session['user_name'] = username
    
    return jsonify({
        'success': True, 
        'message': 'Registration successful!',
        'user': {'name': username}
    })

@app.route('/logout')
def logout():
    """Handle user logout"""
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """
    Handle chat messages with multilingual NLP
    Automatic translation and intent processing
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Invalid request data',
                'status': 'error'
            }), 400
        
        user_message = str(data.get('message', '')).strip()
        explicit_lang = data.get('lang')
        
        if not user_message:
            return jsonify({
                'error': 'Empty message',
                'status': 'error',
                'response': 'Please enter a message.'
            }), 400
        
        print(f"[CHAT] Message: {user_message}")
        print(f"[CHAT] Language: {explicit_lang}")
        
        # Get current context
        context = session.get('conversation_context', {})
        context_disease = context.get('disease')
        
        print(f"[CHAT] Session context disease: {context_disease}")
        
        # Process the message with context
        result = process_message(
            user_message=user_message,
            lang=explicit_lang,
            image_prediction=None,  # Never pass image prediction for text queries
            context_disease=context_disease  # Pass the current disease context
        )
        
        if result.get('status') == 'error':
            return jsonify({
                'error': result.get('error', 'Processing error'),
                'status': 'error',
                'response': 'Sorry, I encountered an error. Please try again.'
            }), 500
        
        # Update session
        session['user_language'] = result.get('language', 'en')
        if 'disease' in result.get('context', {}):
            session['conversation_context']['disease'] = result['context']['disease']
        
        response = {
            'response': result.get('response', 'No response generated'),
            'language': result.get('language', 'en'),
            'status': 'success'
        }
        
        print(f"[CHAT] Response: {response['response'][:100]}...")
        
        return jsonify(response)
    
    except Exception as e:
        error_msg = str(e)
        print(f"Chat error: {error_msg}")
        return jsonify({
            'error': 'Error processing message',
            'status': 'error',
            'response': 'Sorry, I encountered an error. Please try again.'
        }), 500

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Handle image prediction with multilingual response
    """
    try:
        print("[PREDICT] Received prediction request")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {ALLOWED_EXTENSIONS}'}), 400
        
        lang = request.form.get('lang', 'en')
        
        # Save file
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"[PREDICT] Saving file to: {filepath}")
        file.save(filepath)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(filepath).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        disease_name = class_names[pred_idx]
        
        print(f"[PREDICT] Disease: {disease_name}, Confidence: {confidence:.2%}")
        
        # Normalize disease name
        normalized_disease = disease_name.lower().replace('___', '_')
        
        # Process with multilingual NLP
        result = process_image(
            disease_key=normalized_disease,
            confidence=confidence * 100,
            lang=lang
        )
        
        if result.get('status') == 'error':
            return jsonify({'error': result.get('error', 'Processing error')}), 500
        
        # Update session context
        session['conversation_context'] = {
            'disease': normalized_disease,
            'confidence': confidence * 100
        }
        session['user_language'] = lang
        
        response = {
            'disease': disease_name.replace('___', ' ').title(),
            'original_disease': disease_name,
            'confidence_text': f"{confidence * 100:.1f}%",
            'treatment': result.get('response', 'No information available'),
            'status': 'success',
            'language': lang
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"[PREDICT] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
@login_required
def reset_conversation():
    """Reset conversation context"""
    try:
        reset()
        session['conversation_context'] = {'disease': None}
        return jsonify({
            'status': 'success',
            'message': 'Conversation reset successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/set-language', methods=['POST'])
@login_required
def set_language():
    """Set user's preferred language"""
    try:
        data = request.get_json()
        lang = data.get('language', 'en')
        
        if lang not in ['en', 'hi', 'ta', 'te', 'ml']:
            return jsonify({
                'status': 'error',
                'error': 'Unsupported language'
            }), 400
        
        session['user_language'] = lang
        multilingual_processor.set_language(lang)
        
        return jsonify({
            'status': 'success',
            'language': lang
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health')
@login_required
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes_loaded': len(class_names) > 0,
        'num_classes': len(class_names),
        'device': str(device) if device else None,
        'supported_languages': ['en', 'hi', 'ta', 'te', 'ml'],
        'nlp_engine': 'custom',
        'translation': 'enabled',
        'features': {
            'image_detection': True,
            'text_detection': True,
            'multilingual': True,
            'conversation_memory': True
        }
    })

@app.route('/languages')
@login_required
def supported_languages():
    """Return list of supported languages"""
    return jsonify({
        'languages': {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam'
        },
        'status': 'success'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting AgroBot with Custom NLP System")
    print("="*60)
    print("✅ Features:")
    print("   - Custom NLP (No pretrained models)")
    print("   - Image-based disease detection")
    print("   - Text-based disease identification")
    print("   - Multilingual support (5 languages)")
    print("   - Conversation memory")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)