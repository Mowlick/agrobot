"""
Multilingual NLP Processor
Integrates custom NLP engine with bidirectional translation
All internal processing happens in English
"""

from typing import Dict, Optional
import sys
import os

# Import custom modules
sys.path.append(os.path.dirname(__file__))

try:
    from enhanced_translator import bidirectional_translator, to_english, from_english, detect_lang
    from custom_nlp_system import nlp_engine
except ImportError:
    print("Warning: Could not import custom modules. Using fallback mode.")
    bidirectional_translator = None
    nlp_engine = None


class MultilingualNLPProcessor:
    """
    Coordinates translation and NLP processing
    
    Workflow:
    1. User input (any language) → English
    2. NLP processing in English
    3. Response generation in English
    4. English → User's language
    """
    
    def __init__(self):
        self.translator = bidirectional_translator
        self.nlp = nlp_engine
        self.current_user_language = 'en'
        self.conversation_context = {
            'disease': None,
            'language': 'en'
        }
    
    def process_user_message(self, user_message, explicit_lang=None, image_prediction=None):
        """
        Complete multilingual processing pipeline
        
        Args:
            user_message: User's input text (any supported language)
            explicit_lang: Explicitly specified language (optional)
            image_prediction: Dict with disease_key and confidence (if image was uploaded)
        
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
            
            print(f"[NLP] Original: {user_message}")
            print(f"[NLP] English: {english_text}")
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
                # Text-based query - might override context with new symptoms
                context_disease = self.conversation_context.get('disease')
                
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
def process_message(user_message, lang=None, image_prediction=None):
    """Process user message with automatic translation"""
    return multilingual_processor.process_user_message(user_message, lang, image_prediction)


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
            'message': 'नमस्ते, मेरी मदद करें',
            'lang': 'hi',
            'description': 'Hindi greeting'
        },
        {
            'message': 'टमाटर के पत्तों पर भूरे धब्बे हैं',
            'lang': 'hi',
            'description': 'Hindi symptom description'
        },
        {
            'message': 'வணக்கம், எனக்கு உதவி செய்யுங்கள்',
            'lang': 'ta',
            'description': 'Tamil greeting'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['description']} ---")
        print(f"Input ({test['lang']}): {test['message']}")
        
        result = process_message(test['message'], test['lang'])
        
        print(f"Detected Language: {result.get('language', 'N/A')}")
        print(f"English Translation: {result.get('english_text', 'N/A')}")
        print(f"Response ({result.get('language', 'N/A')}): {result.get('response', 'N/A')}")
        print(f"Status: {result.get('status', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Testing image prediction integration")
    print("=" * 60)
    
    # Test image prediction
    image_result = process_image('tomato_early_blight', 95.5, 'hi')
    print(f"\nImage Prediction (Hindi):")
    print(f"Response: {image_result.get('response', 'N/A')}")
    
    # Test follow-up question
    followup_result = process_message('इसका इलाज क्या है?', 'hi')
    print(f"\nFollow-up Question (Hindi):")
    print(f"Response: {followup_result.get('response', 'N/A')}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_multilingual_nlp()