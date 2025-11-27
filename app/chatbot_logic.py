"""
Controller logic for Chatbot interactions
"""

import re
import random
from config import DISEASE_TREATMENT, GENERAL_TIPS
from custom_nlp import translation_aware_nlp, AgroNLP
from custom_nlp import translation_aware_nlp

class AgroChatbot:
    def __init__(self):
        # Memory is now handled via the context string passed from app.py
        pass

    def process_query(self, user_input, user_id='default', context=None):
        """
        Main entry point for chat. 
        Delegates to TranslationAwareNLP for the heavy lifting.
        """
        # The translation_aware_nlp handles detection, translation, and logic
        result = translation_aware_nlp.process_multilingual_query(
            user_message=user_input, 
            explicit_lang=None, # Let it detect automatically
            context=context
        )
        
        return result['response']

    def get_disease_treatment(self, disease_name, lang='en'):
        """Helper to fetch treatment directly"""
        from config import DISEASE_TREATMENT
        
        # Normalize name
        name = disease_name.lower().replace("___", "_").replace(" ", "_")
        
        # Check specific key
        if name in DISEASE_TREATMENT:
            treatment_en = DISEASE_TREATMENT[name]['en']
        else:
            # Fallback for keys not matching exactly
            treatment_en = DISEASE_TREATMENT['unknown']['en']

        # If language is English, return immediately
        if lang == 'en':
            return treatment_en
            
        # Translate if needed
        from enhanced_translator import enhanced_lang_handler
        return enhanced_lang_handler.translate_text(treatment_en, lang, 'en')
        
        # Detect language
        response_lang = self.lang_handler.detect_language(user_input)
        
        # Translate to English for processing
        english_input = self.lang_handler.translate_text(user_input, 'en', response_lang)
        
        # Initialize session memory
        if user_id not in self.session_memory:
            self.session_memory[user_id] = []
        
        # Add to conversation history
        self.session_memory[user_id].append({
            'role': 'user',
            'message': english_input,
            'lang': response_lang
        })
        
        normalized_input = english_input.lower().strip()
        
        # Use multi-turn memory to get previous context if not provided
        if not context and self.session_memory[user_id]:
            prev_context = self.session_memory[user_id][-2] if len(self.session_memory[user_id]) > 1 else None
            if prev_context and 'disease' in prev_context.get('message', ''):
                context = prev_context['message']
        
        # Priority-based response routing
        if self.is_disease_query(normalized_input):
            response = self.handle_disease_query(normalized_input, context, response_lang)
        elif self.has_symptoms(normalized_input):
            response = self.diagnose_from_symptoms(english_input, response_lang)
        elif self.is_greeting(normalized_input):
            response = self.get_greeting_response(response_lang)
        elif self.is_general_question(normalized_input):
            response = self.get_general_response(response_lang)
        else:
            response = self.get_default_response(response_lang)
        
        # Record bot response
        self.session_memory[user_id].append({
            'role': 'bot',
            'message': response,
            'lang': response_lang
        })
        
        return response

    def is_disease_query(self, text):
        keywords = ['disease', 'detect', 'found', 'symptoms', 'treatment',
                   'cure', 'what is', 'tell me', 'explain', 'signs', 'prevent']
        return any(kw in text for kw in keywords)

    def handle_disease_query(self, text, context, lang):
        # Extract disease from context or session memory
        disease_name = None
        confidence = 0
        if context:
            disease_match = re.search(r'disease=([^,]+)', context)
            confidence_match = re.search(r'confidence=([\d.]+)%', context)
            if disease_match:
                disease_name = disease_match.group(1).strip()
                confidence = float(confidence_match.group(1)) if confidence_match else 0
        
        # If no context, try to infer from text
        if not disease_name:
            extraction = agro_nlp.extract_plant_and_symptoms(text)
            matches = agro_nlp.match_symptoms_to_diseases(extraction['plant'], extraction['symptoms'])
            if matches:
                disease_name = matches[0]['disease']
                confidence = matches[0]['confidence']
        
        if not disease_name:
            return self.lang_handler.translate_text(
                "I need more information. Please describe the symptoms or upload an image.",
                lang, 'en'
            )
        
        disease_info = agro_nlp.get_disease_info(disease_name, lang)
        
        if any(kw in text for kw in ['symptom', 'sign', 'look like']):
            response = disease_info.get('symptoms', 'Symptom information not available.')
        elif any(kw in text for kw in ['treatment', 'cure', 'fix', 'remedy']):
            response = disease_info.get('treatment', 'Treatment information not available.')
        elif any(kw in text for kw in ['prevent', 'avoid', 'stop']):
            response = disease_info.get('prevention', 'Prevention information not available.')
        else:
            response = f"The detected disease is {disease_name} with {confidence:.1f}% confidence."
            if disease_info.get('symptoms'):
                response += f"\n\nSymptoms: {disease_info['symptoms']}"
            if disease_info.get('treatment'):
                response += f"\n\nTreatment: {disease_info['treatment']}"
        
        return response  # Already in target lang from get_disease_info

    def has_symptoms(self, text):
        symptom_indicators = ['leaf', 'leaves', 'spot', 'yellow', 'brown',
                            'black', 'curl', 'wilt', 'rot', 'mold', 'rust']
        plant_names = ['tomato', 'potato', 'corn', 'grape', 'apple', 'pepper']
        
        has_symptom = any(ind in text for ind in symptom_indicators)
        has_plant = any(plant in text for plant in plant_names)
        
        return has_symptom and has_plant

    def diagnose_from_symptoms(self, text, lang):
        extraction = agro_nlp.extract_plant_and_symptoms(text)
        plant = extraction['plant']
        symptoms = extraction['symptoms']
        
        if not symptoms:
            return self.lang_handler.translate_text(
                "I couldn't identify specific symptoms. Please describe what you see on the plant (e.g., yellow spots, brown leaves, wilting).",
                lang, 'en'
            )
        
        matches = agro_nlp.match_symptoms_to_diseases(plant, symptoms)
        
        if not matches:
            return self.lang_handler.translate_text(
                "I couldn't match the symptoms to a known disease. Please upload an image for better diagnosis.",
                lang, 'en'
            )
        
        best_match = matches[0]
        disease_name = best_match['disease']
        disease_info = agro_nlp.get_disease_info(disease_name, lang)
        
        response = f"Based on the symptoms you described, this might be {disease_name.replace('___', ' - ')} with {best_match['confidence']:.1f}% confidence."
        
        if len(matches) > 1:
            response += f"\n\nOther possibilities: "
            response += ", ".join([m['disease'].replace('___', ' - ') for m in matches[1:]])
        
        if disease_info.get('treatment'):
            response += f"\n\nRecommended treatment: {disease_info['treatment']}"
        
        response += "\n\nFor accurate diagnosis, please upload a clear image of the affected plant."
        
        return response

    def is_greeting(self, text):
        greetings = ['hello', 'hi', 'hey', 'namaste', 'namaskar',
                    'good morning', 'good evening', 'greetings']
        return any(re.search(r'\b' + re.escape(g) + r'\b', text) for g in greetings)

    def get_greeting_response(self, lang):
        responses = agro_nlp.qa_database['greeting']['responses']
        return responses.get(lang, responses['en'])

    def is_general_question(self, text):
        keywords = ['fertilizer', 'seed', 'irrigation', 'soil', 'weather',
                   'plant', 'crop', 'water', 'grow']
        return any(kw in text for kw in keywords)

    def get_general_response(self, lang):
        tips = GENERAL_TIPS.get(lang, GENERAL_TIPS['en'])
        return random.choice(tips)

    def get_default_response(self, lang):
        responses = agro_nlp.qa_database.get('default', agro_nlp.qa_database['disease_help'])['responses']
        return responses.get(lang, responses['en'])

    def get_disease_treatment(self, disease_name, lang='en'):
        name = disease_name.lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip('_')
        
        mappings = {
            'pepper_bell_bacterial_spot': 'pepper_bacterial_spot',
            'pepper_bell_healthy': 'pepper_healthy',
        }
        
        disease_key = mappings.get(name, name)
        
        return DISEASE_TREATMENT.get(disease_key, DISEASE_TREATMENT['unknown']).get(lang, DISEASE_TREATMENT['unknown']['en'])