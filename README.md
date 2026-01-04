# 🌱 AgroBot  — AI-based Agricultural Assistant

An intelligent web application that helps farmers and gardeners identify plant diseases from leaf images using a custom PyTorch CNN model and provides interactive, multilingual agricultural advice through an advanced chatbot interface.

## 🌟 Features

- **Plant Disease Detection**: Upload images of plant leaves to identify potential diseases
- **Multilingual Support**: Chatbot supports multiple languages (English/Hindi) with automatic language detection
- **Custom CNN Model**: Built from scratch using PyTorch (no pre-trained models)
- **Interactive Chat**: Get personalized farming advice and disease information
- **Responsive Web Interface**: Modern, user-friendly interface accessible on any device
- **Secure Image Processing**: All processing happens locally - your images stay private


## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AgroBot-Universal.git
   cd AgroBot-Universal
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate   # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env` and update any necessary configurations

5. **Run the application**
   ```bash
   python app/app.py
   ```
   - Open your browser and navigate to `http://localhost:5000`


## 🏗️ Project Structure

```
AgroBot-Universal/
├── .env                    # Environment variables
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── app/
│   ├── __init__.py
│   ├── app.py              # Main Flask application
│   ├── chatbot_logic.py    # Chatbot responses and logic
│   ├── config.py           # Configuration and constants
│   ├── custom_nlp.py       # Custom NLP implementation
│   ├── custom_nlp_system.py # Advanced NLP processing
│   ├── enhanced_translator.py # Translation services
│   ├── multilingual_nlp_processor.py # Multilingual support
│   └── translator.py       # Basic translation utilities
├── model/
│   ├── best_model.pth      # Trained PyTorch model
│   ├── class_names.json    # Class labels for predictions
│   ├── cnn_model.py        # Custom CNN architecture
│   └── pytorch_model.py    # Model loading and prediction logic
├── plantvillagedata/       # Dataset directory (for training)
├── static/                 # Static files (CSS, JS, uploads)
│   ├── css/
│   ├── js/
│   └── uploads/
└── templates/              # HTML templates
    └── index.html          # Main web interface
```


## 🛠️ Development

### Training the Model

To train the model with your own dataset:

1. **Prepare your dataset**
   - Organize images in the `plantvillagedata/` directory with subdirectories for each class
   - Example structure:
     ```
     plantvillagedata/
     ├── Tomato___Bacterial_spot/
     ├── Tomato___Early_blight/
     └── Tomato___healthy/
     ```

2. **Start training**
   ```bash
   python model/cnn_model.py
   ```
   - The trained model will be saved as `model/best_model.pth`
   - Training metrics and model architecture will be logged

### Adding New Languages

To add support for additional languages:
1. Update the language configuration in `app/multilingual_nlp_processor.py`
2. Add translations in `app/chatbot_logic.py`
3. Test the new language support using the language selection in the web interface


## Training the Model (from scratch)
Important: This project must not use any pre-trained models. The provided architecture is a custom CNN.

1) Download PlantVillage dataset
   - Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
   - GitHub mirror: https://github.com/spMohanty/PlantVillage-Dataset

2) Place the dataset in:
```
data/plantvillage
```
The directory inside should have one subfolder per class (category), e.g.:
```
data/plantvillage/
  ├─ Tomato___Early_blight/
  ├─ Tomato___Late_blight/
  ├─ ...
```

3) Train
Change directory to the model folder (or run with that as working directory), then:
```
python model/train_model.py
```
This will:
- Build the custom CNN in `model/model_architecture.py`
- Use Keras `ImageDataGenerator` with augmentation and validation split
- Save the trained model to `model/plant_disease_model.h5`
- Save the class names to `model/class_names.json`
- Save training plots to `model/training_history.png`

Notes:
- Training time depends on hardware and dataset size
- You can adjust hyperparameters in `model/train_model.py` (epochs, batch size, learning rate)


## Running the Web App
Ensure you have a trained model and `class_names.json` in `model/`.

From project root:
```
python app/app.py
```
Then open:
```
http://localhost:5000
```


## Usage
- Upload a leaf image (JPG/PNG) in the left panel and click "Analyze Image"
- View top prediction and top-3 probabilities, plus treatment advisory
- Use the chat panel to ask general farming questions or describe symptoms
- Toggle English/Hindi via header buttons


## API Endpoints
- `GET /` — Returns the web UI
- `POST /chat` — JSON: `{ "message": "..." }`
  - Response: `{ response: string, status: "success" }`
- `POST /predict` — multipart form with `file` (image), optional `lang` (`en`/`hi`)
  - Response: `{ disease, confidence, treatment, top_predictions[], status }`
- `GET /health` — Health/status JSON


## Architecture Overview
- A Flask server (`app/app.py`) serves the UI and APIs
- Inference:
  - Uploaded image is preprocessed with Pillow (RGB, resize to 224, normalize)
  - Custom CNN predicts class probabilities
  - Top class name is mapped from `class_names.json`
  - Treatment text is looked up from `DISEASE_TREATMENT` in `app/config.py`
- Chatbot:
  - Language detection via `langdetect`
  - Rule-based responses for greetings, general tips, and symptom advisories
  - Optional translation via `googletrans` (best-effort)


## Constraints and Compliance
- No pre-trained models are used. The CNN is defined and trained from scratch.
- Ensure `class_names.json` length matches the model output units; both are produced by the same training run.


## Troubleshooting
- "Model not loaded" on `/predict`:
  - Train the model and place `model/plant_disease_model.h5` in `model/`
- Predictions return "Unknown":
  - `class_names.json` is empty or mismatched; re-train or provide the correct class list
- Translation errors or delays:
  - The app falls back gracefully; you can disable translation or ignore
- File upload issues on Windows paths:
  - The app writes to `static/uploads`. Ensure the process has write permissions


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PlantVillage dataset for providing the training data
- PyTorch and Flask communities for their excellent documentation
- All the open-source libraries that made this project possible

## Contact

For any questions or feedback, please open an issue on GitHub or contact the maintainers.
