"""
Custom NLP System - No Pretrained Models
Handles both image-based and text-based disease interactions
Uses classical ML and rule-based approaches
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pickle
import os


class DiseaseKnowledgeBase:
    """
    Central knowledge base for all plant diseases
    Stores symptoms, treatments, severity, prevention
    """
    
    def __init__(self):
        self.load_disease_database()
        self.load_symptom_database()
        
    def load_disease_database(self):
        """Load comprehensive disease information"""
        from config import DISEASE_TREATMENT
        
        self.diseases = {}
        
        # Map all diseases with comprehensive info
        for disease_key, treatment_data in DISEASE_TREATMENT.items():
            # Parse disease name
            parts = disease_key.split('_')
            plant = parts[0] if parts else 'unknown'
            disease = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
            
            self.diseases[disease_key] = {
                'plant': plant,
                'disease': disease,
                'treatment': treatment_data.get('en', ''),
                'severity': self._infer_severity(disease_key),
                'symptoms': self._extract_symptoms(disease_key, disease),
                'prevention': self._generate_prevention(disease),
                'causes': self._generate_causes(disease)
            }
    
    def load_symptom_database(self):
        """Load symptom-to-disease mappings"""
        self.symptom_keywords = {
            'spots': ['spot', 'spots', 'dotted', 'marking', 'lesion', 'lesions'],
            'yellow': ['yellow', 'yellowing', 'chlorosis', 'pale'],
            'brown': ['brown', 'browning', 'tan', 'bronze'],
            'black': ['black', 'dark', 'necrotic', 'dead'],
            'white': ['white', 'powdery', 'milky', 'chalky'],
            'wilting': ['wilt', 'wilting', 'drooping', 'sagging', 'limp'],
            'curling': ['curl', 'curling', 'twisted', 'distorted', 'crinkled'],
            'mold': ['mold', 'mildew', 'fungus', 'fuzzy'],
            'rust': ['rust', 'rusty', 'orange', 'pustule'],
            'rot': ['rot', 'rotting', 'decay', 'decompose'],
            'blight': ['blight', 'blight disease'],
            'scab': ['scab', 'rough', 'crusty'],
            'mosaic': ['mosaic', 'mottled', 'pattern'],
            'ring': ['ring', 'circular', 'concentric'],
            'stunted': ['stunted', 'small', 'dwarf', 'reduced growth']
        }
        
        # Disease-symptom associations
        self.disease_symptoms = {
            'tomato_early_blight': ['spots', 'brown', 'yellow', 'ring'],
            'tomato_late_blight': ['spots', 'brown', 'black', 'wilting', 'rot'],
            'tomato_bacterial_spot': ['spots', 'brown', 'yellow'],
            'tomato_leaf_mold': ['yellow', 'mold', 'wilting'],
            'tomato_septoria_leaf_spot': ['spots', 'brown', 'yellow'],
            'tomato_spider_mites_two_spotted_spider_mite': ['spots', 'yellow', 'curling'],
            'tomato_target_spot': ['spots', 'brown', 'ring'],
            'tomato_tomato_yellow_leaf_curl_virus': ['yellow', 'curling', 'stunted'],
            'tomato_mosaic_virus': ['mosaic', 'yellow', 'curling'],
            'potato_early_blight': ['spots', 'brown', 'yellow', 'ring'],
            'potato_late_blight': ['spots', 'black', 'wilting', 'rot'],
            'corn_common_rust': ['rust', 'brown', 'spots'],
            'corn_cercospora_leaf_spot_gray_leaf_spot': ['spots', 'brown', 'yellow'],
            'corn_northern_leaf_blight': ['spots', 'brown'],
            'grape_black_rot': ['spots', 'black', 'rot'],
            'grape_esca_black_measles': ['spots', 'brown', 'wilting'],
            'grape_leaf_blight': ['spots', 'brown'],
            'apple_apple_scab': ['spots', 'brown', 'black', 'scab'],
            'apple_black_rot': ['rot', 'black', 'brown'],
            'apple_cedar_apple_rust': ['rust', 'yellow', 'spots'],
            'pepper_bacterial_spot': ['spots', 'brown', 'yellow'],
            'peach_bacterial_spot': ['spots', 'brown'],
            'cherry_powdery_mildew': ['mold', 'white'],
            'squash_powdery_mildew': ['mold', 'white'],
            'strawberry_leaf_scorch': ['brown', 'spots', 'wilting'],
            'orange_citrus_greening': ['yellow', 'stunted']
        }
    
    def _extract_symptoms(self, disease_key, disease_name):
        """Extract symptoms from disease name and database"""
        symptoms = []
        disease_lower = disease_name.lower()
        
        # Extract from name
        if 'spot' in disease_lower:
            symptoms.append('spots on leaves')
        if 'blight' in disease_lower:
            symptoms.append('rapid wilting and browning')
        if 'rust' in disease_lower:
            symptoms.append('rusty colored pustules')
        if 'mildew' in disease_lower:
            symptoms.append('white powdery growth')
        if 'rot' in disease_lower:
            symptoms.append('tissue decay and softening')
        if 'mosaic' in disease_lower:
            symptoms.append('mottled discoloration')
        if 'curl' in disease_lower:
            symptoms.append('leaf curling and distortion')
        
        return symptoms if symptoms else ['various visible symptoms']
    
    def _infer_severity(self, disease_key):
        """Infer disease severity"""
        severe_keywords = ['blight', 'rot', 'virus', 'wilt']
        moderate_keywords = ['spot', 'rust', 'mildew']
        
        disease_lower = disease_key.lower()
        
        if any(k in disease_lower for k in severe_keywords):
            return 'severe'
        elif any(k in disease_lower for k in moderate_keywords):
            return 'moderate'
        else:
            return 'mild'
    
    def _generate_prevention(self, disease_name):
        """Generate prevention tips"""
        disease_lower = disease_name.lower()
        
        tips = []
        if 'bacterial' in disease_lower or 'virus' in disease_lower:
            tips.append("Use disease-free seeds and transplants")
            tips.append("Practice crop rotation")
        if 'fungal' in disease_lower or 'mildew' in disease_lower or 'blight' in disease_lower:
            tips.append("Improve air circulation")
            tips.append("Avoid overhead watering")
            tips.append("Remove infected plant debris")
        
        if not tips:
            tips = [
                "Maintain proper plant spacing",
                "Regular monitoring for early detection",
                "Use resistant varieties when available"
            ]
        
        return '; '.join(tips)
    
    def _generate_causes(self, disease_name):
        """Generate disease causes"""
        disease_lower = disease_name.lower()
        
        if 'bacterial' in disease_lower:
            return "Caused by bacterial pathogens, spread through water, tools, and insects"
        elif 'virus' in disease_lower:
            return "Caused by viral infection, transmitted by insects or mechanical means"
        elif 'fungal' in disease_lower or 'mildew' in disease_lower or 'rust' in disease_lower or 'blight' in disease_lower:
            return "Caused by fungal pathogens, favored by humid conditions"
        else:
            return "Caused by various pathogenic organisms"
    
    def get_disease_info(self, disease_key):
        """Get comprehensive disease information"""
        return self.diseases.get(disease_key, self.diseases.get('unknown', {}))


class IntentClassifier:
    """
    Classifies user intent using rule-based and classical ML
    No pretrained models used
    """
    
    def __init__(self):
        self.intents = {
            'treatment': [
                'treat', 'treatment', 'cure', 'remedy', 'fix', 'solve',
                'medicine', 'fungicide', 'spray', 'chemical', 'control'
            ],
            'symptoms': [
                'symptom', 'sign', 'look like', 'appear', 'indication',
                'what is', 'describe', 'show', 'display'
            ],
            'prevention': [
                'prevent', 'avoid', 'stop', 'protect', 'prevention',
                'how to prevent', 'keep from'
            ],
            'severity': [
                'severe', 'serious', 'dangerous', 'bad', 'how bad',
                'damage', 'harmful', 'spread', 'contagious'
            ],
            'causes': [
                'cause', 'why', 'reason', 'how did', 'where from',
                'origin', 'source'
            ],
            'identification': [
                'what disease', 'identify', 'diagnose', 'tell me',
                'what is this', 'which disease'
            ],
            'general': [
                'hello', 'hi', 'help', 'thanks', 'thank you',
                'ok', 'yes', 'no'
            ]
        }
        
        # Build keyword-intent mapping
        self.keyword_map = {}
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                self.keyword_map[keyword] = intent
    
    def classify(self, text):
        """
        Classify intent using rule-based keyword matching
        Returns: intent string
        """
        text_lower = text.lower().strip()
        
        # Score each intent
        intent_scores = Counter()
        
        words = text_lower.split()
        for word in words:
            # Exact match
            if word in self.keyword_map:
                intent_scores[self.keyword_map[word]] += 2
            
            # Partial match
            for keyword, intent in self.keyword_map.items():
                if keyword in word or word in keyword:
                    intent_scores[intent] += 1
        
        # Get highest scoring intent
        if intent_scores:
            return intent_scores.most_common(1)[0][0]
        
        # Default to identification if no clear intent
        return 'general'


class EntityExtractor:
    """
    Extracts entities (plants, symptoms) from text
    Uses dictionary-based matching
    """
    
    def __init__(self):
        self.plants = {
            'tomato': ['tomato', 'tomatoes', 'tamatar', 'tomate'],
            'potato': ['potato', 'potatoes', 'aloo', 'batata'],
            'corn': ['corn', 'maize', 'makka', 'bhutta'],
            'apple': ['apple', 'apples', 'seb'],
            'grape': ['grape', 'grapes', 'angur', 'draksha'],
            'pepper': ['pepper', 'peppers', 'mirchi', 'chilli'],
            'peach': ['peach', 'peaches', 'aadu'],
            'cherry': ['cherry', 'cherries'],
            'strawberry': ['strawberry', 'strawberries'],
            'orange': ['orange', 'oranges', 'santra'],
            'blueberry': ['blueberry', 'blueberries'],
            'squash': ['squash', 'pumpkin'],
            'soybean': ['soybean', 'soybeans', 'soya'],
            'rice': ['rice', 'paddy', 'chawal'],
            'wheat': ['wheat', 'gehun']
        }
        
        self.symptoms = {
            'spots': ['spot', 'spots', 'dotted', 'marking', 'lesion'],
            'yellow': ['yellow', 'yellowing', 'pale', 'chlorotic'],
            'brown': ['brown', 'browning', 'tan'],
            'black': ['black', 'dark', 'necrotic'],
            'white': ['white', 'powdery', 'milky'],
            'wilting': ['wilt', 'wilting', 'drooping', 'sagging'],
            'curling': ['curl', 'curling', 'twisted', 'distorted'],
            'mold': ['mold', 'mildew', 'fungus', 'fuzzy'],
            'rust': ['rust', 'rusty', 'orange'],
            'rot': ['rot', 'rotting', 'decay'],
            'stunted': ['stunted', 'small', 'dwarf']
        }
    
    def extract(self, text):
        """
        Extract plant and symptoms from text
        Returns: dict with 'plant' and 'symptoms'
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Extract plant
        detected_plant = None
        for plant, aliases in self.plants.items():
            if any(alias in text_lower for alias in aliases):
                detected_plant = plant
                break
        
        # Extract symptoms
        detected_symptoms = []
        for symptom, aliases in self.symptoms.items():
            if any(alias in text_lower for alias in aliases):
                detected_symptoms.append(symptom)
        
        return {
            'plant': detected_plant,
            'symptoms': detected_symptoms
        }


class SymptomMatcher:
    """
    Matches extracted symptoms to diseases using scoring
    No ML models - pure rule-based logic
    """
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def match(self, plant, symptoms):
        """
        Match symptoms to diseases with scoring
        Returns: List of (disease_key, score, confidence) tuples
        """
        if not symptoms:
            return []
        
        matches = []
        
        for disease_key, disease_data in self.kb.diseases.items():
            # Skip if plant doesn't match
            if plant and disease_data['plant'] != plant:
                continue
            
            # Get disease symptoms
            disease_symptoms = self.kb.disease_symptoms.get(disease_key, [])
            
            # Calculate match score
            matching_symptoms = set(symptoms) & set(disease_symptoms)
            score = len(matching_symptoms)
            
            if score > 0:
                confidence = (score / len(symptoms)) * 100
                matches.append((disease_key, score, confidence))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:5]  # Return top 5 matches


class ResponseGenerator:
    """
    Generates responses using templates and knowledge base
    """
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        
        self.templates = {
            'treatment': "Treatment for {disease}: {treatment}",
            'symptoms': "Common symptoms of {disease}: {symptoms}",
            'prevention': "Prevention tips: {prevention}",
            'severity': "This disease is classified as {severity}.",
            'causes': "Causes: {causes}",
            'identification_success': "Based on your description, this appears to be {disease} (confidence: {confidence}%).",
            'identification_failure': "I couldn't identify a specific disease from your description. Please upload an image or provide more details.",
            'greeting': "Hello! I'm your plant disease assistant. You can upload an image or describe symptoms you're observing.",
            'thanks': "You're welcome! Let me know if you need any more help.",
            'no_context': "Please upload a plant image first or describe the symptoms you're seeing."
        }
    
    def generate(self, intent, disease_key=None, confidence=None, context=None):
        """Generate response based on intent and context"""
        
        if intent == 'general':
            if 'hello' in context or 'hi' in context:
                return self.templates['greeting']
            elif 'thank' in context:
                return self.templates['thanks']
            else:
                return self.templates['greeting']
        
        if not disease_key:
            return self.templates['no_context']
        
        disease_info = self.kb.get_disease_info(disease_key)
        
        if intent == 'treatment':
            return self.templates['treatment'].format(
                disease=disease_key.replace('_', ' ').title(),
                treatment=disease_info.get('treatment', 'No specific treatment information available.')
            )
        
        elif intent == 'symptoms':
            symptoms_str = ', '.join(disease_info.get('symptoms', ['No symptoms listed']))
            return self.templates['symptoms'].format(
                disease=disease_key.replace('_', ' ').title(),
                symptoms=symptoms_str
            )
        
        elif intent == 'prevention':
            return self.templates['prevention'].format(
                prevention=disease_info.get('prevention', 'General prevention practices recommended.')
            )
        
        elif intent == 'severity':
            return self.templates['severity'].format(
                severity=disease_info.get('severity', 'unknown')
            )
        
        elif intent == 'causes':
            return self.templates['causes'].format(
                causes=disease_info.get('causes', 'Causes not specified.')
            )
        
        elif intent == 'identification':
            if confidence:
                return self.templates['identification_success'].format(
                    disease=disease_key.replace('_', ' ').title(),
                    confidence=f"{confidence:.1f}"
                )
            else:
                return self.templates['identification_failure']
        
        else:
            # Default comprehensive response
            return f"{disease_key.replace('_', ' ').title()}: {disease_info.get('treatment', 'Information not available.')}"


class ConversationMemory:
    """
    Maintains conversation context and history
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset conversation memory"""
        self.last_disease = None
        self.last_intent = None
        self.last_symptoms = []
        self.last_plant = None
        self.conversation_history = []
    
    def update(self, intent, disease=None, symptoms=None, plant=None):
        """Update memory with new information"""
        self.last_intent = intent
        if disease:
            self.last_disease = disease
        if symptoms:
            self.last_symptoms = symptoms
        if plant:
            self.last_plant = plant
    
    def add_turn(self, user_input, bot_response):
        """Add conversation turn"""
        self.conversation_history.append({
            'user': user_input,
            'bot': bot_response
        })
    
    def get_context(self):
        """Get current context"""
        return {
            'disease': self.last_disease,
            'intent': self.last_intent,
            'symptoms': self.last_symptoms,
            'plant': self.last_plant
        }


class CustomNLPEngine:
    """
    Main NLP engine coordinating all components
    Handles both image-based and text-based interactions
    """
    
    def __init__(self):
        self.kb = DiseaseKnowledgeBase()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.symptom_matcher = SymptomMatcher(self.kb)
        self.response_generator = ResponseGenerator(self.kb)
        self.memory = ConversationMemory()
    
    def process_image_prediction(self, disease_key, confidence):
        """
        Process CNN prediction result
        Initialize conversation context
        """
        self.memory.reset()
        self.memory.update(
            intent='identification',
            disease=disease_key
        )
        
        disease_info = self.kb.get_disease_info(disease_key)
        
        response = f"Detected: {disease_key.replace('_', ' ').title()} (Confidence: {confidence:.1f}%)\n\n"
        response += f"Treatment: {disease_info.get('treatment', 'No treatment info available.')}"
        
        return response
    
    def process_text_query(self, user_text, context_disease=None):
        """
        Process user text query
        Handles both follow-up questions and symptom-based identification
        """
        user_text_lower = user_text.lower().strip()
        
        # Classify intent
        intent = self.intent_classifier.classify(user_text_lower)
        
        # Extract entities
        entities = self.entity_extractor.extract(user_text_lower)
        plant = entities['plant']
        symptoms = entities['symptoms']
        
        # Update memory
        self.memory.update(intent, symptoms=symptoms, plant=plant)
        
        # Scenario 1: Follow-up question about existing disease (only if no new symptoms)
        if context_disease and not symptoms and intent in ['treatment', 'symptoms', 'prevention', 'severity', 'causes']:
            response = self.response_generator.generate(intent, context_disease, context=user_text_lower)
            self.memory.add_turn(user_text, response)
            return response
        
        # Scenario 2: Text-based disease identification (NEW symptoms detected - override context)
        if symptoms:
            matches = self.symptom_matcher.match(plant, symptoms)
            
            if matches:
                top_match = matches[0]
                disease_key, score, confidence = top_match
                
                # Update memory with NEW disease
                self.memory.update(intent='identification', disease=disease_key)
                
                response = self.response_generator.generate(
                    'identification',
                    disease_key,
                    confidence,
                    user_text_lower
                )
                
                # Add treatment info
                disease_info = self.kb.get_disease_info(disease_key)
                response += f"\n\nTreatment: {disease_info.get('treatment', 'No treatment info.')}"
                
                self.memory.add_turn(user_text, response)
                return response
            else:
                response = "I couldn't identify a specific disease from your description. Can you provide more details about the symptoms?"
                self.memory.add_turn(user_text, response)
                return response
        
        # Scenario 3: General conversation
        response = self.response_generator.generate(intent, context=user_text_lower)
        self.memory.add_turn(user_text, response)
        return response
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.memory.reset()


# Global NLP engine instance
nlp_engine = CustomNLPEngine()