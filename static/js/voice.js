// Voice Input Functionality for AgroBot
// Uses Web Speech API (webkitSpeechRecognition) for FREE speech recognition

// Voice recording variables
let recognition = null;
let isRecording = false;

// Voice button element
const voiceBtn = document.getElementById('voice-btn');
const voiceIndicator = document.getElementById('voice-indicator');

// Check if Web Speech API is supported
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
    console.warn('[VOICE] Web Speech API not supported in this browser');
    if (voiceBtn) {
        voiceBtn.disabled = true;
        voiceBtn.title = 'Voice input not supported in this browser. Please use Chrome or Edge.';
    }
} else {
    // Initialize Speech Recognition
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    // Set language based on current app language
    function updateRecognitionLanguage() {
        const langMap = {
            'en': 'en-US',
            'hi': 'hi-IN',
            'ta': 'ta-IN',
            'te': 'te-IN',
            'ml': 'ml-IN'
        };
        recognition.lang = langMap[currentLanguage] || 'en-US';
    }

    // Event handlers
    recognition.onstart = () => {
        console.log('[VOICE] Recognition started');
        isRecording = true;
        voiceBtn.classList.add('recording');
        voiceBtn.innerHTML = '<span>🛑</span>';
        voiceBtn.title = 'Stop Recording';

        if (voiceIndicator) {
            voiceIndicator.style.display = 'block';
            voiceIndicator.textContent = '🎤 Listening...';
        }
    };

    recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }

        // Update indicator with interim results
        if (voiceIndicator && (interimTranscript || finalTranscript)) {
            voiceIndicator.textContent = `🎤 "${interimTranscript || finalTranscript}"`;
        }

        // Process final transcript immediately
        if (finalTranscript && finalTranscript.trim()) {
            console.log('[VOICE] Final transcript:', finalTranscript);
            recognition.stop(); // Stop to prevent duplicate processing
            processVoiceTranscript(finalTranscript.trim());
        }
    };

    recognition.onerror = (event) => {
        console.error('[VOICE] Recognition error:', event.error);
        stopRecording();

        let errorMessage = 'Voice recognition failed. ';
        switch (event.error) {
            case 'no-speech':
                errorMessage += 'No speech detected. Please try again.';
                break;
            case 'audio-capture':
                errorMessage += 'No microphone found.';
                break;
            case 'not-allowed':
                errorMessage += 'Microphone permission denied.';
                break;
            case 'network':
                errorMessage += 'Network error. Check your connection.';
                break;
            default:
                errorMessage += 'Please try again.';
        }

        addMessage(`❌ ${errorMessage}`, 'bot');
    };

    // Track if we got any results
    let gotResults = false;

    const originalOnResult = recognition.onresult;
    recognition.onresult = (event) => {
        gotResults = true;
        originalOnResult(event);
    };

    recognition.onend = () => {
        console.log('[VOICE] Recognition ended, gotResults:', gotResults);

        // If recognition ended without any results, show helpful message
        if (!gotResults && isRecording) {
            addMessage('🎤 No speech detected. Please try speaking louder and longer, or check your microphone.', 'bot');
        }

        gotResults = false; // Reset for next time
        stopRecording();
    };
}

// Initialize voice button
if (voiceBtn && SpeechRecognition) {
    voiceBtn.addEventListener('click', toggleVoiceRecording);
}

/**
 * Toggle voice recording on/off
 */
function toggleVoiceRecording() {
    if (!recognition) {
        addMessage('❌ Voice recognition not supported in this browser. Please use Chrome or Edge.', 'bot');
        return;
    }

    if (isRecording) {
        recognition.stop();
    } else {
        startRecording();
    }
}

/**
 * Start voice recording
 */
function startRecording() {
    try {
        updateRecognitionLanguage();
        recognition.start();
    } catch (error) {
        console.error('[VOICE] Error starting recognition:', error);
        addMessage('❌ Failed to start voice recognition. Please try again.', 'bot');
    }
}

/**
 * Stop voice recording
 */
function stopRecording() {
    isRecording = false;

    voiceBtn.classList.remove('recording');
    voiceBtn.innerHTML = '<span>🎤</span>';
    voiceBtn.title = 'Voice Input';

    if (voiceIndicator) {
        voiceIndicator.style.display = 'none';
    }
}

/**
 * Process the transcribed voice text
 */
async function processVoiceTranscript(transcript) {
    // Show what was heard
    addMessage(`🎤 "${transcript}"`, 'user');

    // Show processing indicator
    const processingDiv = addMessage('🔄 Processing...', 'bot', true);

    try {
        // Send to chat endpoint for processing
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: transcript,
                lang: currentLanguage
            })
        });

        const data = await response.json();

        // Remove processing message
        processingDiv.remove();

        if (data.error) {
            addMessage(`❌ Error: ${data.error}`, 'bot');
            return;
        }

        // Display response
        addMessage(data.response, 'bot');

        console.log('[VOICE] Processing successful:', data);

    } catch (error) {
        console.error('[VOICE] Error processing transcript:', error);
        processingDiv.remove();
        addMessage('❌ Failed to process voice input. Please try again.', 'bot');
    }
}

/**
 * Voice command shortcuts (optional enhancement)
 * Allows users to trigger voice recording with keyboard shortcuts
 */
document.addEventListener('keydown', (e) => {
    // Press and hold Space to record (when chat input is not focused)
    if (e.code === 'Space' && !isRecording && document.activeElement.id !== 'chat-input') {
        e.preventDefault();
        if (recognition) {
            startRecording();
        }
    }
});

document.addEventListener('keyup', (e) => {
    // Release Space to stop recording (not needed for Web Speech API but kept for UX)
    if (e.code === 'Space' && isRecording) {
        e.preventDefault();
        // Web Speech API auto-stops, but we can trigger stop if still recording
        if (recognition) {
            recognition.stop();
        }
    }
});

/**
 * Show voice recording tutorial (first time users)
 */
function showVoiceTutorial() {
    const hasSeenTutorial = localStorage.getItem('voiceTutorialSeen');

    if (!hasSeenTutorial && voiceBtn && SpeechRecognition) {
        setTimeout(() => {
            const tutorial = document.createElement('div');
            tutorial.className = 'voice-tutorial';
            tutorial.innerHTML = `
                <div class="tutorial-content">
                    <h4>🎤 Voice Input</h4>
                    <p>Click the microphone button to speak your question about plant diseases!</p>
                    <p><small>Tip: You can also press and hold Space to record</small></p>
                    <button onclick="this.parentElement.parentElement.remove(); localStorage.setItem('voiceTutorialSeen', 'true');">Got it!</button>
                </div>
            `;
            tutorial.style.cssText = `
                position: fixed;
                bottom: 80px;
                right: 20px;
                background: white;
                border: 2px solid #25a05a;
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                z-index: 1000;
                max-width: 300px;
            `;
            document.body.appendChild(tutorial);

            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (tutorial.parentElement) {
                    tutorial.remove();
                    localStorage.setItem('voiceTutorialSeen', 'true');
                }
            }, 10000);
        }, 2000);
    }
}

// Show tutorial on first visit
showVoiceTutorial();

console.log('[VOICE] Web Speech API initialized successfully');
