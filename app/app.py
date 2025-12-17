"""
Enhanced Flask App with User Authentication
Voice input handled by browser Web Speech API
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import json
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename

# Import configuration
from config import MODEL_PATH, CLASS_NAMES_PATH, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, IMG_SIZE, NUM_CLASSES

# Import database
from database import db, create_user, authenticate_user, get_user

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- MODEL LOADING ---
print("⏳ Loading PyTorch model...")
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
    
    print(f"✅ Model loaded successfully! Device: {device}")
    print(f"Number of classes: {len(class_names)}")
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
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
    
    # Get user info
    user_id = session.get('user_id')
    user_info = get_user(user_id)
    
    if 'user_language' not in session:
        session['user_language'] = user_info.get('preferred_language', 'en')
    if 'conversation_context' not in session:
        session['conversation_context'] = {'disease': None}
    
    return render_template('index.html', user=user_info)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'GET':
        return render_template('login.html')
    
    data = request.get_json()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    
    if not email or not password:
        return jsonify({
            'success': False,
            'error': 'Email and password are required'
        }), 400
    
    # Authenticate user
    auth_result = authenticate_user(email, password)
    
    if auth_result['success']:
        session['user_id'] = auth_result['user_id']
        session['user_name'] = auth_result['name']
        session['user_email'] = auth_result['email']
        session['user_language'] = auth_result.get('preferred_language', 'en')
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'name': auth_result['name'],
                'email': auth_result['email']
            }
        })
    else:
        return jsonify({
            'success': False,
            'error': auth_result.get('error', 'Authentication failed')
        }), 401

@app.route('/register', methods=['POST'])
def register():
    """Handle new user registration"""
    data = request.get_json()
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    
    if not name or not email or not password:
        return jsonify({
            'success': False,
            'error': 'All fields are required'
        }), 400
    
    if len(password) < 6:
        return jsonify({
            'success': False,
            'error': 'Password must be at least 6 characters'
        }), 400
    
    # Create user
    result = create_user(name, email, password)
    
    if result['success']:
        session['user_id'] = result['user_id']
        session['user_name'] = result['name']
        session['user_email'] = result['email']
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': {
                'name': result['name'],
                'email': result['email']
            }
        })
    else:
        return jsonify({
            'success': False,
            'error': result.get('error', 'Registration failed')
        }), 400

@app.route('/logout')
def logout():
    """Handle user logout"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat messages with multilingual NLP"""
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
        
        # Process the message with context
        result = process_message(
            user_message=user_message,
            lang=explicit_lang,
            image_prediction=None,
            context_disease=context_disease
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
        
        # Update user language preference in database
        user_id = session.get('user_id')
        db.update_user_language(user_id, result.get('language', 'en'))
        
        response = {
            'response': result.get('response', 'No response generated'),
            'language': result.get('language', 'en'),
            'status': 'success'
        }
        
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
    """Handle image prediction with multilingual response"""
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
        file.save(filepath)
        
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
        
        # Update in database
        user_id = session.get('user_id')
        db.update_user_language(user_id, lang)
        
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
        'voice_available': True,  # Web Speech API in browser
        'nlp_engine': 'custom',
        'translation': 'enabled',
        'features': {
            'image_detection': True,
            'text_detection': True,
            'voice_detection': True,  # Handled by browser Web Speech API
            'multilingual': True,
            'conversation_memory': True,
            'user_authentication': True
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting AgroBot with Voice Support")
    print("="*60)
    print("✅ Features:")
    print("   - Custom NLP (No pretrained models)")
    print("   - Image-based disease detection")
    print("   - Text-based disease identification")
    print("   - Voice-based disease classification 🎤")
    print("   - Multilingual support (5 languages)")
    print("   - User authentication & credential storage")
    print("   - Conversation memory")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)