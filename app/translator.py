"""
Robust Language detection and translation module using deep-translator
"""

from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import re
from functools import lru_cache

# Set seed for consistent language detection
DetectorFactory.seed = 0

class LanguageHandler:
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam'
        }
        # In-memory cache for translations to reduce API calls
        self.translation_cache = {}

    def detect_language(self, text):
        """Detect the language of input text"""
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return 'en'
            
            # langdetect is reliable for longer text
            lang = detect(cleaned_text)
            
            # Map specific codes to our supported list
            if lang in ['hi', 'mr', 'ne', 'sa']: return 'hi'
            if lang == 'ta': return 'ta'
            if lang == 'te': return 'te'
            if lang == 'ml': return 'ml'
            
            return 'en' # Default fallback
                
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'en'

    def translate_text(self, text, target_lang='en', source_lang=None):
        """Translate text with caching"""
        if not text:
            return ""
            
        try:
            # 1. Check if translation is not needed
            if source_lang == target_lang:
                return text
            
            # 2. Check Cache (Key: text_source_target)
            cache_key = f"{text}_{source_lang}_{target_lang}"
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]

            # 3. Perform Translation
            # If source is unknown, let the translator auto-detect
            src = source_lang if source_lang else 'auto'
            
            translator = GoogleTranslator(source=src, target=target_lang)
            translated_text = translator.translate(text)
            
            # 4. Save to Cache
            self.translation_cache[cache_key] = translated_text
            
            return translated_text
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Fail gracefully by returning original

    def clean_text(self, text):
        """Clean text for processing"""
        return re.sub(r'\s+', ' ', text).strip()