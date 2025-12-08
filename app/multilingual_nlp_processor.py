"""
Multilingual NLP Processor
Integrates custom NLP engine with bidirectional translation
All internal processing happens in English
FIXED: Better symptom preservation during translation
"""

from typing import Dict, Optional
import sys
import os

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Initialize modules as None
bidirectional_translator = None
to_english = None
from_english = None
detect_lang = None
nlp_engine = None

# Import custom modules
try:
    # Import enhanced_translator
    from enhanced_translator import (
        bidirectional_translator,
        to_english as et_to_english,
        from_english as et_from_english,
        detect_lang as et_detect_lang
    )
    to_english = et_to_english
    from_english = et_from_english
    detect_lang = et_detect_lang
    
    # Import custom_nlp_system
    from custom_nlp_system import nlp_engine
    
    print("Successfully imported all custom modules")
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    import traceback
    traceback.print_exc()
    print(f"Current sys.path: {sys.path}")
    print(f"Current directory: {os.path.abspath(os.curdir)}")
    print("Available files in directory:", os.listdir(current_dir))


class MultilingualNLPProcessor:
    """
    Coordinates translation and NLP processing
    
    Workflow:
    1. User input (any language) → English
    2. NLP processing in English
    3. Response generation in English
    4. English → User's language
    
    FIXED: Better handling of symptom keywords during translation
    """
    
    def __init__(self):
        self.translator = bidirectional_translator
        self.nlp = nlp_engine
        self.current_user_language = 'en'
        self.conversation_context = {
            'disease': None,
            'language': 'en'
        }
        
        # Symptom keyword mapping for better translation
        self.symptom_keywords_mapping = {
            # Hindi
            'पीला': 'yellow', 'पीली': 'yellow', 'भूरा': 'brown', 'काला': 'black',
            'सफेद': 'white', 'धब्बे': 'spots', 'मुरझाना': 'wilting', 'सड़न': 'rot',
            'फफूंद': 'mold', 'जंग': 'rust', 'पाउडर': 'powdery',
            # Tamil
            'மஞ்சள்': 'yellow', 'பழுப்பு': 'brown', 'கருப்பு': 'black', 'வெள்ளை': 'white',
            'புள்ளிகள்': 'spots', 'வாடுதல்': 'wilting', 'அழுகல்': 'rot', 'பூஞ்சை': 'mold',
            'துரு': 'rust', 'தூள்': 'powdery', 'இலை': 'leaf',
            # Telugu
            'పసుపు': 'yellow', 'గోధుమ': 'brown', 'నలుపు': 'black', 'తెలుపు': 'white',
            'మచ్చలు': 'spots', 'విల్టింగ్': 'wilting', 'కుళ్ళు': 'rot', 'ఫంగస్': 'mold',
            'తుప్పు': 'rust', 'పొడి': 'powdery', 'ఆకు': 'leaf',
            # Malayalam
            'മഞ്ഞ': 'yellow', 'തവിട്ട്': 'brown', 'കറുപ്പ്': 'black', 'വെള്ള': 'white',
            'പാടുകൾ': 'spots', 'വാടൽ': 'wilting', 'ചീയൽ': 'rot', 'പൂപ്പൽ': 'mold',
            'തുരുമ്പ്': 'rust', 'പൊടി': 'powdery', 'ഇല': 'leaf'
        }
    
    def enhance_translation_for_symptoms(self, text, translated_text, user_lang):
        """
        Enhance translated text by ensuring symptom keywords are preserved
        
        Args:
            text: Original text in user's language
            translated_text: Machine-translated English text
            user_lang: User's language code
        
        Returns:
            Enhanced English text with symptom keywords
        """
        if user_lang == 'en':
            return translated_text
        
        # Add explicit symptom keywords if found in original
        enhanced = translated_text
        found_symptoms = []
        
        for native_word, english_word in self.symptom_keywords_mapping.items():
            if native_word in text:
                # Check if English equivalent is missing
                if english_word not in enhanced.lower():
                    found_symptoms.append(english_word)
        
        # Append found symptoms to enhance translation
        if found_symptoms:
            enhanced = f"{enhanced} (showing: {', '.join(found_symptoms)})"
            print(f"[TRANSLATION ENHANCEMENT] Added symptom keywords: {found_symptoms}")
        
        return enhanced
    
    def process_user_message(self, user_message, explicit_lang=None, image_prediction=None, context_disease=None):
        """
        Complete multilingual processing pipeline
        
        Args:
            user_message: User's input text (any supported language)
            explicit_lang: Explicitly specified language (optional)
            image_prediction: Dict with disease_key and confidence (if image was uploaded)
            context_disease: Current disease context from session (optional)
        
        Returns:
            Dict with response, language, and context
        """
        try:
            # Step 1: Detect or use explicit language
            if explicit_lang:
                user_lang = explicit_lang
            else:
                user_lang = detect_lang(user_message)
            
            self.current_user_language = user_lang
            self.conversation_context['language'] = user_lang
            
            # Step 2: Translate to English
            english_text = to_english(user_message, user_lang)
            
            # Step 2.5: Enhance translation with symptom keywords (FIXED)
            english_text = self.enhance_translation_for_symptoms(user_message, english_text, user_lang)
            
            print(f"[NLP] Original ({user_lang}): {user_message}")
            print(f"[NLP] Enhanced English: {english_text}")
            print(f"[NLP] Current context disease: {self.conversation_context.get('disease')}")
            
            # Step 3: Process with NLP engine
            if image_prediction:
                # Image-based prediction - NEW disease context
                disease_key = image_prediction.get('disease_key', '')
                confidence = image_prediction.get('confidence', 0.0)
                
                print(f"[NLP] Processing IMAGE prediction: {disease_key}")
                
                english_response = self.nlp.process_image_prediction(disease_key, confidence)
                self.conversation_context['disease'] = disease_key
            
            else:
                # Use provided context_disease or fall back to conversation context
                context_disease = context_disease or self.conversation_context.get('disease')
                self.conversation_context['disease'] = context_disease  # Update context
                
                print(f"[NLP] Processing TEXT query with context: {context_disease}")
                
                english_response = self.nlp.process_text_query(english_text, context_disease)
                
                # Update context with new disease if detected
                new_disease = self.nlp.memory.last_disease
                if new_disease:
                    print(f"[NLP] New disease detected: {new_disease}")
                    self.conversation_context['disease'] = new_disease
            
            print(f"[NLP] English response: {english_response[:100]}...")
            
            # Step 4: Translate response back to user's language
            translated_response = from_english(english_response, user_lang)
            
            return {
                'response': translated_response,
                'language': user_lang,
                'english_text': english_text,
                'english_response': english_response,
                'context': self.conversation_context,
                'status': 'success'
            }
        
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            return {
                'response': "Sorry, I encountered an error processing your request.",
                'language': self.current_user_language or 'en',
                'error': error_msg,
                'status': 'error'
            }
    
    def process_image_result(self, disease_key, confidence, user_lang='en'):
        """
        Process image prediction and generate multilingual response
        
        Args:
            disease_key: Predicted disease key
            confidence: Prediction confidence
            user_lang: User's language
        
        Returns:
            Translated response dict
        """
        try:
            self.current_user_language = user_lang
            self.conversation_context['disease'] = disease_key
            self.conversation_context['language'] = user_lang
            
            # Generate response in English
            english_response = self.nlp.process_image_prediction(disease_key, confidence)
            
            # Translate to user's language
            translated_response = from_english(english_response, user_lang)
            
            return {
                'response': translated_response,
                'language': user_lang,
                'disease': disease_key,
                'confidence': confidence,
                'status': 'success'
            }
        
        except Exception as e:
            error_msg = f"Error processing image result: {str(e)}"
            print(error_msg)
            
            return {
                'response': "Sorry, I encountered an error.",
                'language': user_lang,
                'error': error_msg,
                'status': 'error'
            }
    
    def reset_conversation(self):
        """Reset conversation context"""
        self.nlp.reset_conversation()
        self.conversation_context = {
            'disease': None,
            'language': self.current_user_language
        }
    
    def set_language(self, lang_code):
        """Set user's preferred language"""
        self.current_user_language = lang_code
        self.conversation_context['language'] = lang_code
    
    def get_context(self):
        """Get current conversation context"""
        return self.conversation_context


# Global processor instance
multilingual_processor = MultilingualNLPProcessor()


# Convenience functions
def process_message(user_message, lang=None, image_prediction=None, context_disease=None):
    """Process user message with automatic translation"""
    return multilingual_processor.process_user_message(
        user_message=user_message,
        explicit_lang=lang,
        image_prediction=image_prediction,
        context_disease=context_disease
    )


def process_image(disease_key, confidence, lang='en'):
    """Process image prediction result"""
    return multilingual_processor.process_image_result(disease_key, confidence, lang)


def reset():
    """Reset conversation"""
    multilingual_processor.reset_conversation()


# Test function
def test_multilingual_nlp():
    """Test the multilingual NLP system"""
    
    print("=" * 60)
    print("TESTING MULTILINGUAL NLP SYSTEM")
    print("=" * 60)
    
    test_cases = [
        {
            'message': 'Hello, how can you help me?',
            'lang': 'en',
            'description': 'English greeting'
        },
        {
            'message': 'My tomato plant has yellow leaves with brown spots',
            'lang': 'en',
            'description': 'English symptom description'
        },
        {
            'message': 'टमाटर के पत्तों पर भूरे धब्बे हैं और पत्तियाँ पीली हो रही हैं',
            'lang': 'hi',
            'description': 'Hindi symptom description'
        },
        {
            'message': 'தக்காளி இலைகளில் பழுப்பு புள்ளிகள் மற்றும் மஞ்சள் நிறம்',
            'lang': 'ta',
            'description': 'Tamil symptom description'
        },
        {
            'message': 'టమాటో ఆకులపై గోధుమ మచ్చలు మరియు పసుపు రంగు',
            'lang': 'te',
            'description': 'Telugu symptom description'
        },
        {
            'message': 'തക്കാളി ഇലകളിൽ തവിട്ട് പാടുകളും മഞ്ഞ നിറവും',
            'lang': 'ml',
            'description': 'Malayalam symptom description'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['description']} ---")
        print(f"Input ({test['lang']}): {test['message']}")
        
        result = process_message(test['message'], test['lang'])
        
        print(f"Detected Language: {result.get('language', 'N/A')}")
        print(f"English Translation: {result.get('english_text', 'N/A')}")
        print(f"Response ({result.get('language', 'N/A')}): {result.get('response', 'N/A')[:100]}...")
        print(f"Status: {result.get('status', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Testing image prediction integration")
    print("=" * 60)
    
    # Test image prediction
    image_result = process_image('tomato_early_blight', 95.5, 'hi')
    print(f"\nImage Prediction (Hindi):")
    print(f"Response: {image_result.get('response', 'N/A')[:100]}...")
    
    # Test follow-up question in Hindi
    followup_result = process_message('इसका इलाज क्या है?', 'hi')
    print(f"\nFollow-up Question (Hindi - 'What is the treatment?'):")
    print(f"Response: {followup_result.get('response', 'N/A')[:100]}...")
    
    # Test severity question in English
    severity_result = process_message('Is this serious?', 'en')
    print(f"\nSeverity Question (English):")
    print(f"Response: {severity_result.get('response', 'N/A')[:100]}...")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_multilingual_nlp()