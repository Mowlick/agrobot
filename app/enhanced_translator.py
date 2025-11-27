"""
Enhanced Bidirectional Language Translation Module
Supports translation TO English for processing and FROM English for output
"""

from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import re
from functools import lru_cache

# Set seed for consistent language detection
DetectorFactory.seed = 0

class BidirectionalTranslator:
    """
    Handles translation in both directions:
    1. User language → English (for NLP processing)
    2. English → User language (for response delivery)
    """
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ml': 'Malayalam'
        }
        
        # Language codes for deep-translator
        self.lang_codes = {
            'en': 'en',
            'hi': 'hi',
            'ta': 'ta',
            'te': 'te',
            'ml': 'ml'
        }
        
        # In-memory cache for translations
        self.translation_cache = {}
        
    def detect_language(self, text):
        """
        Detect the language of input text
        Returns language code: 'en', 'hi', 'ta', 'te', 'ml'
        """
        try:
            if not text or not isinstance(text, str):
                return 'en'
                
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return 'en'
            
            # Use langdetect for detection
            detected_lang = detect(cleaned_text)
            
            # Map to supported languages
            lang_mapping = {
                'en': 'en',
                'hi': 'hi', 'mr': 'hi', 'ne': 'hi', 'sa': 'hi',  # Hindi family
                'ta': 'ta',
                'te': 'te',
                'ml': 'ml'
            }
            
            return lang_mapping.get(detected_lang, 'en')
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'en'
    
    def translate_to_english(self, text, source_lang=None):
        """
        Translate user input TO English for NLP processing
        
        Args:
            text: User input text
            source_lang: Source language code (if known)
        
        Returns:
            English translation of the text
        """
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Auto-detect if source language not provided
            if not source_lang:
                source_lang = self.detect_language(text)
            
            # If already English, return as-is
            if source_lang == 'en':
                return text
            
            # Check cache
            cache_key = f"to_en_{text}_{source_lang}"
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]
            
            # Translate to English
            translator = GoogleTranslator(source=self.lang_codes[source_lang], target='en')
            english_text = translator.translate(text)
            
            # Cache result
            self.translation_cache[cache_key] = english_text
            
            return english_text
            
        except Exception as e:
            print(f"Translation to English error: {e}")
            return text  # Return original on error
    
    def translate_from_english(self, text, target_lang='en'):
        """
        Translate system response FROM English to user's language
        
        Args:
            text: English response text
            target_lang: Target language code
        
        Returns:
            Translated text in target language
        """
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # If target is English, return as-is
            if target_lang == 'en':
                return text
            
            # Check cache
            cache_key = f"from_en_{text}_{target_lang}"
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]
            
            # Translate from English
            translator = GoogleTranslator(source='en', target=self.lang_codes[target_lang])
            translated_text = translator.translate(text)
            
            # Cache result
            self.translation_cache[cache_key] = translated_text
            
            return translated_text
            
        except Exception as e:
            print(f"Translation from English error: {e}")
            return text  # Return original on error
    
    def bidirectional_translate(self, text, user_lang=None):
        """
        Complete bidirectional translation workflow:
        1. Detect user language
        2. Translate to English for processing
        3. Return both English version and detected language
        
        Args:
            text: User input
            user_lang: User's language (if known)
        
        Returns:
            dict with 'english_text' and 'detected_lang'
        """
        try:
            # Detect language if not provided
            if not user_lang:
                user_lang = self.detect_language(text)
            
            # Translate to English
            english_text = self.translate_to_english(text, user_lang)
            
            return {
                'english_text': english_text,
                'detected_lang': user_lang,
                'original_text': text
            }
            
        except Exception as e:
            print(f"Bidirectional translation error: {e}")
            return {
                'english_text': text,
                'detected_lang': 'en',
                'original_text': text
            }
    
    def translate_disease_info(self, disease_data, target_lang='en'):
        """
        Translate disease information (name, symptoms, treatment) to target language
        
        Args:
            disease_data: Dict with disease information in English
            target_lang: Target language code
        
        Returns:
            Translated disease data dict
        """
        try:
            if target_lang == 'en':
                return disease_data
            
            translated_data = {}
            
            for key, value in disease_data.items():
                if isinstance(value, str):
                    translated_data[key] = self.translate_from_english(value, target_lang)
                else:
                    translated_data[key] = value
            
            return translated_data
            
        except Exception as e:
            print(f"Disease info translation error: {e}")
            return disease_data
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()
    
    def get_language_name(self, lang_code):
        """Get full language name from code"""
        return self.supported_languages.get(lang_code, 'English')


# Global translator instance
bidirectional_translator = BidirectionalTranslator()


# Convenience functions for easy import
def to_english(text, source_lang=None):
    """Quick function to translate any text to English"""
    return bidirectional_translator.translate_to_english(text, source_lang)


def from_english(text, target_lang):
    """Quick function to translate English text to target language"""
    return bidirectional_translator.translate_from_english(text, target_lang)


def detect_lang(text):
    """Quick function to detect language"""
    return bidirectional_translator.detect_language(text)