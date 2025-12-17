// Enhanced script.js with improved language dropdown (no chat input translation)

// UI Text Translations
const translations = {
    en: {
        mainTitle: "🚜 AgroBot Universal",
        mainSubtitle: "AI-Based Agricultural Assistant",
        detectionTitle: "🔍 Plant Disease Detection",
        uploadText: "Click or drag to upload plant image",
        uploadHint: "Supports: JPG, PNG, JPEG",
        analyzeText: "Analyze Image",
        clearBtn: "Clear",
        resultTitle: "📊 Detection Results",
        diseaseLabel: "Disease:",
        confidenceLabel: "Confidence:",
        treatmentLabel: "💊 Treatment:",
        chatTitle: "💬 Ask Me Anything",
        chatPlaceholder: "Type your question here...",
        welcomeMessage: "Hello! I'm your agricultural assistant. Ask me about crop diseases, pests, or farming practices. You can also upload a plant image for disease detection!",
        footerText: "🌾 AgroBot Universal | Powered by Custom CNN & NLP | Multilingual Support"
    },
    hi: {
        mainTitle: "🚜 एग्रोबोट यूनिवर्सल",
        mainSubtitle: "AI-आधारित कृषि सहायक",
        detectionTitle: "🔍 पौधे की बीमारी का पता लगाना",
        uploadText: "पौधे की छवि अपलोड करने के लिए क्लिक करें या खींचें",
        uploadHint: "समर्थन: JPG, PNG, JPEG",
        analyzeText: "छवि का विश्लेषण करें",
        clearBtn: "साफ़ करें",
        resultTitle: "📊 पहचान परिणाम",
        diseaseLabel: "रोग:",
        confidenceLabel: "विश्वास:",
        treatmentLabel: "💊 उपचार:",
        chatTitle: "💬 मुझसे कुछ भी पूछें",
        chatPlaceholder: "अपना प्रश्न यहाँ टाइप करें...",
        welcomeMessage: "नमस्ते! मैं आपका कृषि सहायक हूं। मुझसे फसल रोगों, कीटों या खेती की प्रथाओं के बारे में पूछें। आप रोग का पता लगाने के लिए पौधे की छवि भी अपलोड कर सकते हैं!",
        footerText: "🌾 एग्रोबोट यूनिवर्सल | कस्टम CNN और NLP द्वारा संचालित | बहुभाषी समर्थन"
    },
    ta: {
        mainTitle: "🚜 அக்ரோபாட் யூனிவர்சல்",
        mainSubtitle: "AI-அடிப்படையிலான விவசாய உதவியாளர்",
        detectionTitle: "🔍 தாவர நோய் கண்டறிதல்",
        uploadText: "தாவர படத்தை பதிவேற்ற கிளிக் செய்யவும் அல்லது இழுக்கவும்",
        uploadHint: "ஆதரவு: JPG, PNG, JPEG",
        analyzeText: "படத்தை பகுப்பாய்வு செய்யவும்",
        clearBtn: "துடைக்கவும்",
        resultTitle: "📊 கண்டறிதல் முடிவுகள்",
        diseaseLabel: "நோய்:",
        confidenceLabel: "நம்பிக்கை:",
        treatmentLabel: "💊 சிகிச்சை:",
        chatTitle: "💬 என்னிடம் எதுவும் கேளுங்கள்",
        chatPlaceholder: "உங்கள் கேள்வியை இங்கே டைப் செய்யவும்...",
        welcomeMessage: "வணக்கம்! நான் உங்கள் விவசாய உதவியாளர். பயிர் நோய்கள், பூச்சிகள் அல்லது விவசாய நடைமுறைகள் பற்றி என்னிடம் கேளுங்கள். நீங்கள் தாவர படத்தை நோய் கண்டறிதலுக்கு பதிவேற்றலாம்!",
        footerText: "🌾 அக்ரோபாட் யூனிவர்சல் | தனிப்பயன் CNN & NLP ஆல் இயக்கப்படுகிறது | பலமொழி ஆதரவு"
    },
    te: {
        mainTitle: "🚜 అగ్రోబాట్ యూనివర్సల్",
        mainSubtitle: "AI-ఆధారిత అగ్రికల్చరల్ అసిస్టెంట్",
        detectionTitle: "🔍 ప్లాంట్ డిసీజ్ డిటెక్షన్",
        uploadText: "ప్లాంట్ ఇమేజ్ అప్‌లోడ్ చేయడానికి క్లిక్ చేయండి లేదా డ్రాగ్ చేయండి",
        uploadHint: "సపోర్ట్స్: JPG, PNG, JPEG",
        analyzeText: "ఇమేజ్ అనాలైజ్ చేయండి",
        clearBtn: "క్లియర్",
        resultTitle: "📊 డిటెక్షన్ రిజల్ట్స్",
        diseaseLabel: "డిసీజ్:",
        confidenceLabel: "కాన్ఫిడెన్స్:",
        treatmentLabel: "💊 ట్రీట్‌మెంట్:",
        chatTitle: "💬 నన్ను ఏమైనా అడగండి",
        chatPlaceholder: "మీ క్వెస్టన్ ఇక్కడ టైప్ చేయండి...",
        welcomeMessage: "హలో! నేను మీ అగ్రికల్చరల్ అసిస్టెంట్. క్రాప్ డిసీజ్‌లు, పెస్ట్స్ లేదా ఫార్మింగ్ ప్రాక్టీస్‌ల గురించి నన్ను అడగండి. మీరు ప్లాంట్ ఇమేజ్ అప్‌లోడ్ చేసి డిసీజ్ డిటెక్షన్ చేయవచ్చు!",
        footerText: "🌾 అగ్రోబాట్ యూనివర్సల్ | కస్టమ్ CNN & NLP ద్వారా పవర్డ్ | మల్టీలింగ్వల్ సపోర్ట్"
    },
    ml: {
        mainTitle: "🚜 അഗ്രോബോട്ട് യൂണിവേഴ്സൽ",
        mainSubtitle: "AI-അടിസ്ഥാനമാക്കിയ അഗ്രികൾച്ചറൽ അസിസ്റ്റന്റ്",
        detectionTitle: "🔍 പ്ലാന്റ് ഡിസീസ് ഡിടെക്ഷൻ",
        uploadText: "പ്ലാന്റ് ഇമേജ് അപ്‌ലോഡ് ചെയ്യാൻ ക്ലിക്ക് ചെയ്യുക അല്ലെങ്കിൽ ഡ്രാഗ് ചെയ്യുക",
        uploadHint: "സപ്പോർട്ട്സ്: JPG, PNG, JPEG",
        analyzeText: "ഇമേജ് അനാലൈസ് ചെയ്യുക",
        clearBtn: "ക്ലിയർ",
        resultTitle: "📊 ഡിടെക്ഷൻ റിസൾട്ട്സ്",
        diseaseLabel: "ഡിസീസ്:",
        confidenceLabel: "കോൺഫിഡൻസ്:",
        treatmentLabel: "💊 ട്രീറ്റ്മെന്റ്:",
        chatTitle: "💬 എന്നോട് എന്തും ചോദിക്കൂ",
        chatPlaceholder: "നിങ്ങളുടെ ക്വസ്റ്റൻ ഇവിടെ ടൈപ്പ് ചെയ്യുക...",
        welcomeMessage: "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ അഗ്രികൾച്ചറൽ അസിസ്റ്റന്റാണ്. ക്രോപ്പ് ഡിസീസുകൾ, പെസ്റ്റ്സ് അല്ലെങ്കിൽ ഫാമിങ് പ്രാക്ടീസുകൾ ഗുരിന്ചി എന്നോട് ചോദിക്കൂ. നിങ്ങൾ പ്ലാന്റ് ഇമേജ് അപ്‌ലോഡ് ചെയ്ത് ഡിസീസ് ഡിടെക്ഷൻ ചെയ്യാം!",
        footerText: "🌾 അഗ്രോബോട്ട് യൂണിവേഴ്സൽ | കസ്റ്റം CNN & NLP ദ്വാരാ പവർഡ് | മൾട്ടിലിങ്ക്വൽ സപ്പോർട്ട്"
    }
};

// Language display names
const languageDisplayNames = {
    'en': 'English',
    'hi': 'हिन्दी',
    'ta': 'தமிழ்',
    'te': 'తెలుగు',
    'ml': 'മലയാളം'
};

// Variables
let selectedFile = null;
let lastPredictionContext = null;
let currentLanguage = 'en';

// Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImage = document.getElementById('preview-image');
const resultsSection = document.getElementById('results-section');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');

// Language Dropdown Functions
function toggleLanguageDropdown() {
    const dropdown = document.getElementById('lang-dropdown');
    const button = document.getElementById('lang-dropdown-btn');

    dropdown.classList.toggle('show');
    button.classList.toggle('open');
}

function changeLanguage(langCode, langName) {
    currentLanguage = langCode;

    // Update display button
    document.getElementById('current-lang-display').textContent = languageDisplayNames[langCode];

    // Remove active class from all options
    document.querySelectorAll('.lang-option').forEach(opt => {
        opt.classList.remove('active');
    });

    // Add active class to selected option
    const selectedOption = document.querySelector(`.lang-option[data-lang="${langCode}"]`);
    if (selectedOption) {
        selectedOption.classList.add('active');
    }

    // Close dropdown
    document.getElementById('lang-dropdown').classList.remove('show');
    document.getElementById('lang-dropdown-btn').classList.remove('open');

    // Update UI language
    updateUILanguage();

    // Send to backend
    fetch('/set_language', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ language: langCode })
    })
        .catch(error => console.error('Error setting language:', error));
}

// Close dropdown when clicking outside
document.addEventListener('click', function (event) {
    const dropdown = document.getElementById('lang-dropdown');
    const button = document.getElementById('lang-dropdown-btn');

    if (dropdown && button &&
        !dropdown.contains(event.target) &&
        !button.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('open');
    }
});

// User Profile Menu (Placeholder)
function showProfileMenu() {
    alert('Profile menu coming soon! This will include:\n- Account settings\n- Logout\n- Preferences');
}

// Update UI language
function updateUILanguage() {
    const t = translations[currentLanguage] || translations.en;

    document.getElementById('main-title').textContent = t.mainTitle;
    document.getElementById('main-subtitle').textContent = t.mainSubtitle;
    document.getElementById('detection-title').textContent = t.detectionTitle;
    document.getElementById('upload-text').textContent = t.uploadText;
    document.getElementById('upload-hint').textContent = t.uploadHint;
    document.getElementById('analyze-text').textContent = t.analyzeText;
    document.getElementById('clear-btn').textContent = t.clearBtn;
    document.getElementById('result-title').textContent = t.resultTitle;
    document.getElementById('disease-label').textContent = t.diseaseLabel;
    document.getElementById('confidence-label').textContent = t.confidenceLabel;
    document.getElementById('treatment-label').textContent = t.treatmentLabel;
    document.getElementById('chat-title').textContent = t.chatTitle;
    chatInput.placeholder = t.chatPlaceholder;
    document.getElementById('welcome-message').textContent = t.welcomeMessage;
    document.getElementById('footer-text').textContent = t.footerText;
}

// Upload handlers
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle file selection
function handleFileSelect(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];

    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image (JPG, PNG, JPEG)');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        document.querySelector('.upload-section').style.display = 'none';
        previewSection.style.display = 'flex';
        resultsSection.style.display = 'none';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Clear button
clearBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    analyzeBtn.disabled = true;
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const t = translations[currentLanguage] || translations.en;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing...';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('lang', currentLanguage);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Display results
        displayResults(data);

        // Store context for chat
        lastPredictionContext = `disease=${data.original_disease}, confidence=${data.confidence_text}`;

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze image. Please try again.');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `<span>${t.analyzeText}</span>`;
    }
});

// Simple markdown parser for bold text
function parseMarkdown(text) {
    // Replace **bold** with <strong>bold</strong>
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// Display prediction results
function displayResults(data) {
    const t = translations[currentLanguage] || translations.en;

    document.getElementById('result-title').textContent = t.resultTitle;
    document.getElementById('disease-label').textContent = t.diseaseLabel;
    document.getElementById('confidence-label').textContent = t.confidenceLabel;
    document.getElementById('treatment-label').textContent = t.treatmentLabel;

    document.getElementById('disease-value').textContent = data.disease;
    document.getElementById('confidence-value').textContent = data.confidence_text;

    // Parse markdown in treatment text and set as HTML
    const treatmentElement = document.getElementById('treatment-value');
    treatmentElement.innerHTML = parseMarkdown(data.treatment || 'No treatment information available');

    resultsSection.style.display = 'block';
}

// Chat functionality (Simplified - no input translation dropdown)
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message
    addMessage(message, 'user');
    chatInput.value = '';

    // Show typing indicator
    const typingDiv = addMessage('Typing...', 'bot', true);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                lang: currentLanguage,
                context: lastPredictionContext
            })
        });

        const data = await response.json();

        // Remove typing indicator
        typingDiv.remove();

        if (data.error) {
            addMessage(data.error, 'bot');
        } else {
            addMessage(data.response, 'bot');
        }

    } catch (error) {
        console.error('Chat error:', error);
        typingDiv.remove();
        addMessage('Failed to get response. Please try again.', 'bot');
    }
}

// Simple markdown parser for bold text
function parseMarkdown(text) {
    // Replace **bold** with <strong>bold</strong>
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

function addMessage(text, sender, isTyping = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (isTyping) {
        contentDiv.innerHTML = '<span class="loading"></span> ' + text;
    } else {
        // Parse markdown and set as HTML
        contentDiv.innerHTML = parseMarkdown(text);
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageDiv;
}

// Initialize UI on load
document.addEventListener('DOMContentLoaded', () => {
    updateUILanguage();
});