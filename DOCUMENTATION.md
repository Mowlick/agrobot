# Infosys Internship Project Documentation

AgroBot Universal — AI-based Agricultural Assistant

Review date: 07-Nov-2025


## 1. Executive Summary
AgroBot Universal is a web-based assistant that helps farmers detect plant diseases from leaf images and receive actionable advisories in English or Hindi. The solution combines:
- A custom Convolutional Neural Network (CNN) trained from scratch (no pre-trained models)
- A rule-based, multilingual chatbot that provides general tips and symptom-based guidance
- A simple Flask web app for image upload, prediction, and chat


## 2. Problem Statement & Objectives
- Farmers often lack timely, accessible guidance for crop health, pest control, and best practices.
- Language barriers (English/Hindi) limit adoption of existing solutions.

**Objective**: Build a multi-class image classifier for plant diseases and integrate it into an accessible, multilingual assistant that:
- Classifies disease from an uploaded leaf image
- Returns the most probable disease and top-3 predictions with confidence
- Provides language-aware treatment guidance and farming tips

**Constraints**: No pre-trained models are allowed. The model is defined and trained from scratch.


## 3. Scope
- Frontend: Single-page interface for uploads, results, and chat
- Backend: Flask server exposing `/predict` and `/chat` endpoints
- ML: Keras/TensorFlow custom CNN, trained on PlantVillage
- Languages: English and Hindi (auto-detected), with optional translation utilities


## 4. System Architecture
- UI (`templates/index.html`, `static/css/style.css`, `static/js/script.js`)
- API (`app/app.py`):
  - `GET /` — serves UI
  - `POST /predict` — image -> preprocessing -> CNN -> class -> treatment
  - `POST /chat` — user text -> language detection -> rule-based response
  - `GET /health` — runtime status
- Chatbot (`app/chatbot_logic.py`): greetings, general tips, symptom advisories, disease treatment lookup
- Config/Data (`app/config.py`): paths, advisory database, treatment texts, constants
- Language (`app/translator.py`): language detection (langdetect), best-effort translation (googletrans)
- Model (`model/model_architecture.py`): custom CNN definition
- Training (`model/train_model.py`): dataset pipeline, augmentation, callbacks, saving artifacts


## 5. Dataset
- Recommended: PlantVillage (public, multi-class plant disease dataset)
- Expected layout (directory-per-class):
```
data/plantvillage/
  ├─ Tomato___Early_blight/
  ├─ Tomato___Late_blight/
  └─ ...
```
- The training script uses Keras `ImageDataGenerator` with an 80/20 train/validation split via `validation_split`.


## 6. Model Design (No Pre-trained Models)
- Architecture: Keras Sequential CNN (from scratch)
  - 5 convolutional blocks with BatchNorm, MaxPooling, and Dropout
  - Flatten → Dense(512) → Dense(256) → Output Dense(num_classes, softmax)
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Metrics: `accuracy`, `TopKCategoricalAccuracy(k=3)`
- Input size: 224×224×3


## 7. Training Methodology
- Augmentation: rotation, shifts, shear, zoom, horizontal/vertical flips
- Hyperparameters (default):
  - `IMG_SIZE=224`, `BATCH_SIZE=32`, `EPOCHS=50`, `learning_rate=1e-3`
- Callbacks:
  - EarlyStopping (patience=10, restore best)
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - ModelCheckpoint (save best by val_accuracy)
- Outputs:
  - `model/plant_disease_model.h5`
  - `model/class_names.json` (class order used at inference)
  - `model/training_history.png` (accuracy/loss plots)


## 8. Evaluation & Results
- Primary metrics: validation accuracy and loss; top-3 accuracy for robustness
- Post-training artifacts include plots in `training_history.png`
- For review, demonstrate:
  - Health check (`/health`) with model/classes loaded
  - Several test images showing correct predictions and confidences


## 9. Inference Pipeline
1. User uploads image (JPG/PNG)
2. Server saves to `static/uploads/`
3. Preprocessing (Pillow): RGB convert → resize to 224 → normalize → batch dimension
4. Model predicts probabilities → argmax for top-1, argsort for top-3
5. Map indices to names via `class_names.json`
6. Lookup treatment text from `app/config.py` in selected language
7. Return JSON response and show results in UI


## 10. Chatbot Logic
- Language detection (English/Hindi)
- Rules:
  - Greetings → greeting message
  - General keywords (fertilizer, soil, irrigation, etc.) → random tip
  - Symptom keywords (yellowing, brown spots, wilting, pests) → crop-aware advisory
  - Default fallback if no rule applies
- Disease-specific treatment lookup by normalized disease key


## 11. API Specification
- `POST /predict`
  - Request: multipart form with `file` (image), optional `lang` (`en`|`hi`)
  - Response:
    ```json
    {
      "disease": "Tomato___Early_blight",
      "confidence": 0.92,
      "treatment": "...",
      "top_predictions": [ {"disease": "...", "confidence": 0.92}, ...],
      "status": "success"
    }
    ```
- `POST /chat`
  - Request: `{ "message": "user text" }`
  - Response: `{ "response": "...", "status": "success" }`
- `GET /health`
  - Response: `{ status, model_loaded, classes_loaded }`


## 12. Runbook
Prerequisites:
- Python 3.10+, `pip install -r requirements.txt`
- Trained artifacts in `model/plant_disease_model.h5` and `model/class_names.json`

Run locally:
```
python app/app.py
```
Open `http://localhost:5000`.

Common issues:
- Model not loaded → train and place `.h5` file in `model/`
- Unknown classes → ensure `class_names.json` matches training order
- Translation errors → app falls back to original text


## 13. Risks & Limitations
- Dataset bias: model performance tied to PlantVillage distribution
- Real-world variance (lighting, device cameras) may reduce accuracy
- Network dependence of `googletrans` can add latency or fail; handled gracefully


## 14. Future Enhancements
- Add test suite and CI for linting and basic route tests
- Add confidence calibration/thresholding and abstain behavior
- Add more languages and offline translation options
- Export ONNX/TFLite for mobile deployment


## 15. Review Checklist
- Code walk-through: data flow, routes, and model
- Live demo: upload → predict → treatment
- Health check shows model/classes loaded
- Documentation: README + this document
- PPT summarizing problem, approach, results, and demo plan


## 16. References
- PlantVillage dataset: Kaggle / GitHub mirrors
- TensorFlow/Keras documentation

