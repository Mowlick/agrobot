"""
Enhanced Multilingual NLP Processor
Better symptom preservation and disease detection from text descriptions
"""

from typing import Dict, Optional
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

bidirectional_translator = None
to_english = None
from_english = None
detect_lang = None
nlp_engine = None

try:
    from enhanced_translator import (
        bidirectional_translator,
        to_english as et_to_english,
        from_english as et_from_english,
        detect_lang as et_detect_lang
    )
    to_english = et_to_english
    from_english = et_from_english
    detect_lang = et_detect_lang
    
    from custom_nlp_system import nlp_engine
    
    print("✓ Modules loaded successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")


class MultilingualNLPProcessor:
    """Enhanced multilingual processor with better symptom handling"""
    
    def __init__(self):
        self.translator = bidirectional_translator
        self.nlp = nlp_engine
        self.current_user_language = 'en'
        self.conversation_context = {
            'disease': None,
            'language': 'en'
        }
        
        # Enhanced symptom keyword mapping for all languages
        self.symptom_mappings = {
            # Color symptoms
            'पीला': 'yellow', 'पीली': 'yellow',
            'மஞ்சள்': 'yellow',
            'పసుపు': 'yellow',
            'മഞ്ഞ': 'yellow',
            
            'भूरा': 'brown', 'भूरी': 'brown',
            'பழுப்பு': 'brown',
            'గోధుమ': 'brown',
            'തവിട്ടു': 'brown',
            
            'काला': 'black', 'काली': 'black',
            'கருப்பு': 'black',
            'నల్పు': 'black',
            'കറുപ്പു': 'black',
            
            'सफेद': 'white', 'सफेदी': 'white',
            'வெள்ளை': 'white',
            'తెలుపు': 'white',
            'വെള്ള': 'white',
            
            # Surface symptoms
            'धब्बे': 'spots', 'धब्बा': 'spots',
            'புள்ளிகள்': 'spots',
            'మచ్చలు': 'spots',
            'പാടുകൾ': 'spots',
            
            # Condition symptoms
            'मुरझाना': 'wilting', 'मुरझा': 'wilting',
            'வாடுதல்': 'wilting',
            'విల్టింగ్': 'wilting',
            'വാടൽ': 'wilting',
            
            'सड़न': 'rot', 'सड़ना': 'rot',
            'அழுகல்': 'rot',
            'కుళ్ళు': 'rot',
            'ചീയൽ': 'rot',
            
            'फफूंद': 'mold', 'फंगस': 'mold',
            'பூஞ்சை': 'mold',
            'ఫంగస్': 'mold',
            'പൂപ്പൽ': 'mold',
            
            'जंग': 'rust',
            'துரு': 'rust',
            'తుప్పు': 'rust',
            'തുരുമ്പു': 'rust',
            
            'पाउडर': 'powdery', 'चूर्ण': 'powdery',
            'தூள்': 'powdery',
            'పొడి': 'powdery',
            'പൊടി': 'powdery',
            
            # Plant parts
            'पत्ती': 'leaf', 'पत्तियां': 'leaves', 'पत्ते': 'leaves',
            'இலை': 'leaf', 'இலைகள்': 'leaves',
            'ఆకు': 'leaf', 'ఆకులు': 'leaves',
            'ഇല': 'leaf', 'ഇലകൾ': 'leaves'
        }
    
    def enhance_translation(self, original_text, translated_text, user_lang):
        """Enhanced translation with explicit symptom preservation"""
        if user_lang == 'en':
            return translated_text
        
        # Find symptom keywords in original text
        found_symptoms = []
        text_lower = original_text.lower()
        
        for native_word, english_word in self.symptom_mappings.items():
            if native_word in text_lower:
                # Check if English equivalent is in translation
                if english_word not in translated_text.lower():
                    found_symptoms.append(english_word)
        
        # Enhance translation if symptoms found
        if found_symptoms:
            # Remove duplicates
            found_symptoms = list(set(found_symptoms))
            enhanced = f"{translated_text} showing {' and '.join(found_symptoms)}"
            print(f"[TRANSLATION+] Added symptoms: {found_symptoms}")
            return enhanced
        
        return translated_text
    
    def process_user_message(self, user_message, explicit_lang=None, image_prediction=None, context_disease=None):
        """Complete multilingual processing pipeline"""
        try:
            # Detect or use explicit language
            user_lang = explicit_lang or detect_lang(user_message)
            self.current_user_language = user_lang
            self.conversation_context['language'] = user_lang
            
            # Translate to English
            english_text = to_english(user_message, user_lang)
            
            # Enhance with symptom keywords
            english_text = self.enhance_translation(user_message, english_text, user_lang)
            
            print(f"[MLNLP] Original ({user_lang}): {user_message}")
            print(f"[MLNLP] Enhanced EN: {english_text}")
            print(f"[MLNLP] Context: {self.conversation_context.get('disease')}")
            
            # Process with NLP
            if image_prediction:
                disease_key = image_prediction.get('disease_key', '')
                confidence = image_prediction.get('confidence', 0.0)
                
                english_response = self.nlp.process_image_prediction(disease_key, confidence)
                self.conversation_context['disease'] = disease_key
            else:
                context_disease = context_disease or self.conversation_context.get('disease')
                
                # Process text query
                english_response = self.nlp.process_text_query(english_text, context_disease)
                
                # Update context if disease detected
                new_disease = self.nlp.memory.last_disease
                if new_disease:
                    print(f"[MLNLP] Disease updated: {new_disease}")
                    self.conversation_context['disease'] = new_disease
            
            print(f"[MLNLP] EN Response: {english_response[:100]}...")
            
            # Translate response back
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
            print(f"[MLNLP ERROR] {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'response': "Sorry, I encountered an error. Please try again.",
                'language': self.current_user_language or 'en',
                'error': str(e),
                'status': 'error'
            }
    
    def process_image_result(self, disease_key, confidence, user_lang='en'):
        """Process image prediction"""
        try:
            self.current_user_language = user_lang
            self.conversation_context['disease'] = disease_key
            self.conversation_context['language'] = user_lang
            
            english_response = self.nlp.process_image_prediction(disease_key, confidence)
            translated_response = from_english(english_response, user_lang)
            
            return {
                'response': translated_response,
                'language': user_lang,
                'disease': disease_key,
                'confidence': confidence,
                'status': 'success'
            }
        
        except Exception as e:
            print(f"[MLNLP ERROR] Image: {e}")
            return {
                'response': "Sorry, I encountered an error.",
                'language': user_lang,
                'error': str(e),
                'status': 'error'
            }
    
    def reset_conversation(self):
        """Reset conversation"""
        self.nlp.reset_conversation()
        self.conversation_context = {
            'disease': None,
            'language': self.current_user_language
        }
    
    def set_language(self, lang_code):
        """Set preferred language"""
        self.current_user_language = lang_code
        self.conversation_context['language'] = lang_code
    
    def get_context(self):
        """Get conversation context"""
        return self.conversation_context


# Global instance
multilingual_processor = MultilingualNLPProcessor()


# Convenience functions
def process_message(user_message, lang=None, image_prediction=None, context_disease=None):
    """Process user message"""
    return multilingual_processor.process_user_message(
        user_message=user_message,
        explicit_lang=lang,
        image_prediction=image_prediction,
        context_disease=context_disease
    )


def process_image(disease_key, confidence, lang='en'):
    """Process image prediction"""
    return multilingual_processor.process_image_result(disease_key, confidence, lang)


def reset():
    """Reset conversation"""
    multilingual_processor.reset_conversation()


# Test function
def test_system():
    """Test enhanced NLP system"""
    print("\n" + "="*60)
    print("TESTING ENHANCED TEXT-BASED DISEASE DETECTION")
    print("="*60)
    
    test_cases = [
        {
            'text': 'My tomato plant has yellow leaves with brown spots',
            'lang': 'en',
            'expected': 'early_blight or septoria'
        },
        {
            'text': 'Tomato leaves are curling and turning yellow',
            'lang': 'en',
            'expected': 'yellow_leaf_curl_virus'
        },
        {
            'text': 'White powdery coating on leaves',
            'lang': 'en',
            'expected': 'powdery_mildew'
        },
        {
            'text': 'टमाटर के पत्तों पर भूरे धब्बे हैं और पत्तियां पीली हो रही हैं',
            'lang': 'hi',
            'expected': 'early_blight'
        },
        {
            'text': 'தக்காளி இலைகளில் பழுப்பு புள்ளிகள் மற்றும் மஞ்சள் நிறம்',
            'lang': 'ta',
            'expected': 'early_blight'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input ({test['lang']}): {test['text']}")
        print(f"Expected: {test['expected']}")
        
        result = process_message(test['text'], test['lang'])
        
        print(f"Status: {result.get('status')}")
        print(f"English: {result.get('english_text', 'N/A')}")
        print(f"Response: {result.get('response', 'N/A')[:150]}...")
        
        if 'context' in result and result['context'].get('disease'):
            print(f"✓ Detected disease: {result['context']['disease']}")
        else:
            print("✗ No disease detected")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    test_system()