"""
Custom NLP System - No Pretrained Models
Handles both image-based and text-based disease interactions
Uses classical ML and rule-based approaches
FIXED: Intent classification and multilingual symptom detection
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
        """Load symptom-to-disease mappings with multilingual keywords"""
        # English keywords
        self.symptom_keywords = {
            'spots': ['spot', 'spots', 'dotted', 'marking', 'lesion', 'lesions', 'धब्बे', 'புள்ளிகள்', 'మచ్చలు', 'പാടുകൾ'],
            'yellow': ['yellow', 'yellowing', 'chlorosis', 'pale', 'पीला', 'पीली', 'மஞ்சள்', 'పసుపు', 'മഞ്ഞ'],
            'brown': ['brown', 'browning', 'tan', 'bronze', 'भूरा', 'பழுப்பு', 'గోధుమ', 'തവിട്ട്'],
            'black': ['black', 'dark', 'necrotic', 'dead', 'काला', 'கருப்பு', 'నలుపు', 'കറുപ്പ്'],
            'white': ['white', 'powdery', 'milky', 'chalky', 'सफेद', 'வெள்ளை', 'తెలుపు', 'വെള്ള'],
            'wilting': ['wilt', 'wilting', 'drooping', 'sagging', 'limp', 'मुरझाना', 'சூடு', 'வாடுதல்', 'విల్టింగ్', 'വാടൽ'],
            'curling': ['curl', 'curling', 'twisted', 'distorted', 'crinkled', 'मुड़ना', 'சுருள்', 'వంగు', 'ചുരുളൽ'],
            'mold': ['mold', 'mildew', 'fungus', 'fuzzy', 'फफूंद', 'பூஞ்சை', 'ఫంగస్', 'പൂപ്പൽ'],
            'rust': ['rust', 'rusty', 'orange', 'pustule', 'जंग', 'துரு', 'తుప్పు', 'തുരുമ്പ്'],
            'rot': ['rot', 'rotting', 'decay', 'decompose', 'सड़न', 'அழுகல்', 'కుళ్ళు', 'ചീയൽ'],
            'blight': ['blight', 'blight disease', 'ब्लाइट', 'நோய்', 'బ్లైట్', 'ബ്ലൈറ്റ്'],
            'scab': ['scab', 'rough', 'crusty', 'खुरदरा', 'சிரங்கு', 'స్కాబ్', 'ചൊറി'],
            'mosaic': ['mosaic', 'mottled', 'pattern', 'मोज़ेक', 'மொசைக்', 'మొజాయిక్', 'മൊസൈക്'],
            'ring': ['ring', 'circular', 'concentric', 'गोलाकार', 'வளையம்', 'వలయం', 'വളയം'],
            'stunted': ['stunted', 'small', 'dwarf', 'reduced growth', 'बौना', 'குட்டை', 'పెరుగుదల తగ్గుట', 'വളർച്ച കുറവ്'],
            'leaf': ['leaf', 'leaves', 'foliage', 'पत्ती', 'இலை', 'ఆకు', 'ഇല'],
            'powdery': ['powdery', 'powder', 'पाउडर', 'தூள்', 'పొడి', 'പൊടി'],
            'patches': ['patch', 'patches', 'area', 'धब्बा', 'திட்டுகள்', 'పాచెస్', 'പാച്ചുകൾ']
        }
        
        # Disease-symptom associations
        self.disease_symptoms = {
            'tomato_early_blight': ['spots', 'brown', 'yellow', 'ring', 'leaf'],
            'tomato_late_blight': ['spots', 'brown', 'black', 'wilting', 'rot', 'leaf'],
            'tomato_bacterial_spot': ['spots', 'brown', 'yellow', 'leaf'],
            'tomato_leaf_mold': ['yellow', 'mold', 'wilting', 'leaf'],
            'tomato_septoria_leaf_spot': ['spots', 'brown', 'yellow', 'leaf'],
            'tomato_spider_mites_two_spotted_spider_mite': ['spots', 'yellow', 'curling', 'leaf'],
            'tomato_target_spot': ['spots', 'brown', 'ring', 'leaf'],
            'tomato_tomato_yellow_leaf_curl_virus': ['yellow', 'curling', 'stunted', 'leaf'],
            'tomato_mosaic_virus': ['mosaic', 'yellow', 'curling', 'leaf'],
            'potato_early_blight': ['spots', 'brown', 'yellow', 'ring', 'leaf'],
            'potato_late_blight': ['spots', 'black', 'wilting', 'rot', 'leaf'],
            'corn_common_rust': ['rust', 'brown', 'spots', 'leaf'],
            'corn_cercospora_leaf_spot_gray_leaf_spot': ['spots', 'brown', 'yellow', 'leaf'],
            'corn_northern_leaf_blight': ['spots', 'brown', 'leaf'],
            'grape_black_rot': ['spots', 'black', 'rot'],
            'grape_esca_black_measles': ['spots', 'brown', 'wilting'],
            'grape_leaf_blight': ['spots', 'brown', 'leaf'],
            'apple_apple_scab': ['spots', 'brown', 'black', 'scab'],
            'apple_black_rot': ['rot', 'black', 'brown'],
            'apple_cedar_apple_rust': ['rust', 'yellow', 'spots'],
            'pepper_bacterial_spot': ['spots', 'brown', 'yellow', 'leaf'],
            'peach_bacterial_spot': ['spots', 'brown', 'leaf'],
            'cherry_powdery_mildew': ['mold', 'white', 'powdery', 'leaf'],
            'squash_powdery_mildew': ['mold', 'white', 'powdery', 'leaf'],
            'strawberry_leaf_scorch': ['brown', 'spots', 'wilting', 'leaf'],
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
        if 'mold' in disease_lower:
            symptoms.append('fungal growth on leaves')
        
        return symptoms if symptoms else ['various visible symptoms']
    
    def _infer_severity(self, disease_key):
        """Infer disease severity"""
        severe_keywords = ['blight', 'rot', 'virus', 'wilt']
        moderate_keywords = ['spot', 'rust', 'mildew', 'mold']
        
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
        if 'fungal' in disease_lower or 'mildew' in disease_lower or 'blight' in disease_lower or 'mold' in disease_lower:
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
        elif 'fungal' in disease_lower or 'mildew' in disease_lower or 'rust' in disease_lower or 'blight' in disease_lower or 'mold' in disease_lower:
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
    FIXED: More flexible keyword matching with partial matches
    """
    
    def __init__(self):
        self.intents = {
            'treatment': [
                'treat', 'treatment', 'cure', 'remedy', 'fix', 'solve',
                'medicine', 'fungicide', 'spray', 'chemical', 'control',
                'how to treat', 'what to do', 'manage', 'apply', 'इलाज', 'उपचार',
                'chikitsa', 'maruthuvam', 'chikitsaku', 'chikitsa cheyali', 'chikitsa cheyali'
            ],
            'symptoms': [
                'symptom', 'sign', 'look like', 'appear', 'indication',
                'what is', 'describe', 'show', 'display', 'characteristic',
                'what are', 'tell me about', 'lakshanam', 'lakshana', 'lakshanalu',
                'lakshanangal', 'लक्षण', 'रोग लक्षण', 'लक्षण क्या हैं', 'रोग के लक्षण'
            ],
            'prevention': [
                'prevent', 'avoid', 'stop', 'protect', 'prevention',
                'how to prevent', 'keep from', 'precaution', 'roktham', 'taduparirakshana',
                'tataṟkāvatippu', 'रोकथाम', 'बचाव', 'रोकने के उपाय', 'प्रतिबंध'
            ],
            'severity': [
                'severe', 'serious', 'dangerous', 'bad', 'how bad',
                'damage', 'harmful', 'spread', 'contagious', 'is this serious',
                'should i worry', 'critical', 'gambheer', 'gambhira', 'kathinam',
                'kathinamano', 'kathoram', 'गंभीर', 'खतरनाक', 'कितना गंभीर है', 'खतरा'
            ],
            'causes': [
                'cause', 'why', 'reason', 'how did', 'where from',
                'origin', 'source', 'what causes', 'kaaranam', 'karanam',
                'karanam enti', 'kāraṇaṅṅaḷ', 'कारण', 'वजह', 'क्यों होता है', 'कारण क्या है'
            ],
            'identification': [
                'what disease', 'identify', 'diagnose', 'tell me',
                'what is this', 'which disease', 'recognize', 'rognirdharanam',
                'rogam enti', 'rōgaṅṅaḷ', 'रोग की पहचान', 'यह कौन सी बीमारी है', 'पहचान'
            ],
            'greeting': [
                'hello', 'hi', 'hey', 'greetings', 'namaste', 'vanakkam',
                'namaskaram', 'नमस्ते', 'हैलो', 'வணக்கம்', 'నమస్కారం', 'നമസ്കാരം',
                'kaise ho', 'eppadi irukkinga', 'ela unnav', 'sukhamaano'
            ],
            'thanks': [
                'thanks', 'thank you', 'dhanyavad', 'nandri', 'kondadhu',
                'stuthi', 'धन्यवाद', 'थैंक यू', 'நன்றி', 'ధన్యవాదాలు', 'നന്ദി'
            ],
            'general': [
                'ok', 'yes', 'no', 'help', 'assist', 'sahayam', 'udhavi',
                'sahayam venam', 'sahayam veno', 'मदद', 'सहायता', 'உதவி', 'సహాయం', 'സഹായം'
            ]
        }
        
        # Build keyword-intent mapping with priorities
        self.keyword_map = {}
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword not in self.keyword_map:
                    self.keyword_map[keyword] = []
                self.keyword_map[keyword].append(intent)
    
    def classify(self, text):
        """
        Classify intent using rule-based keyword matching
        Returns: intent string
        FIXED: Better matching with partial word detection
        """
        text_lower = text.lower().strip()
        
        # Score each intent
        intent_scores = Counter()
        
        # Check for exact phrases first (higher weight)
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Multi-word phrases get higher score
                    weight = len(keyword.split()) * 3
                    intent_scores[intent] += weight
        
        # Check individual words
        words = text_lower.split()
        for word in words:
            # Exact match
            if word in self.keyword_map:
                for intent in self.keyword_map[word]:
                    intent_scores[intent] += 2
            
            # Partial match (substring)
            for keyword, intents in self.keyword_map.items():
                if len(keyword) > 3:  # Only for longer keywords
                    if keyword in word or word in keyword:
                        for intent in intents:
                            intent_scores[intent] += 1
        
        # Get highest scoring intent
        if intent_scores:
            top_intent = intent_scores.most_common(1)[0][0]
            print(f"[INTENT] Classified as '{top_intent}' with scores: {dict(intent_scores)}")
            return top_intent
        
        # Default to general if no clear intent
        print(f"[INTENT] No clear intent found, defaulting to 'general'")
        return 'general'


class EntityExtractor:
    """
    Extracts entities (plants, symptoms) from text
    Uses dictionary-based matching
    FIXED: Multilingual support and better pattern matching
    """
    
    def __init__(self):
        self.plants = {
            'tomato': ['tomato', 'tomatoes', 'tamatar', 'tomate', 'टमाटर', 'தக்காளி', 'టమాటో', 'തക്കാളി'],
            'potato': ['potato', 'potatoes', 'aloo', 'batata', 'आलू', 'உருளைக்கிழங்கு', 'బంగాళాదుంప', 'ഉരുളക്കിഴങ്ങ്'],
            'corn': ['corn', 'maize', 'makka', 'bhutta', 'मक्का', 'சோளம்', 'మొక్కజొన్న', 'ചോളം'],
            'apple': ['apple', 'apples', 'seb', 'सेब', 'ஆப்பிள்', 'ఆపిల్', 'ആപ്പിൾ'],
            'grape': ['grape', 'grapes', 'angur', 'draksha', 'अंगूर', 'திராட்சை', 'ద్రాక్ష', 'മുന്തിരി'],
            'pepper': ['pepper', 'peppers', 'mirchi', 'chilli', 'मिर्च', 'மிளகு', 'మిరప', 'മുളക്'],
            'peach': ['peach', 'peaches', 'aadu', 'आडू', 'பீச்', 'పీచ్', 'പീച്ച്'],
            'cherry': ['cherry', 'cherries', 'चेरी', 'செர்ரி', 'చెర్రీ', 'ചെറി'],
            'strawberry': ['strawberry', 'strawberries', 'स्ट्रॉबेरी', 'ஸ்ட்ராபெர்ரி', 'స్ట్రాబెర్రీ', 'സ്ട്രോബെറി'],
            'orange': ['orange', 'oranges', 'santra', 'संतरा', 'ஆரஞ்சு', 'ఆరంజ్', 'ഓറഞ്ച്'],
            'blueberry': ['blueberry', 'blueberries'],
            'squash': ['squash', 'pumpkin', 'कद्दू', 'பூசணி', 'గుమ్మడి', 'മത്തങ്ങ'],
            'soybean': ['soybean', 'soybeans', 'soya', 'सोया', 'சோயா', 'సోయా', 'സോയ'],
            'rice': ['rice', 'paddy', 'chawal', 'चावल', 'அரிசி', 'వరి', 'അരി'],
            'wheat': ['wheat', 'gehun', 'गेहूं', 'கோதுமை', 'గోధుమ', 'ഗോതമ്പ്']
        }
        
        self.symptoms = {
            'spots': ['spot', 'spots', 'dotted', 'marking', 'lesion', 'धब्बे', 'புள்ளிகள்', 'మచ్చలు', 'പാടുകൾ'],
            'yellow': ['yellow', 'yellowing', 'pale', 'chlorotic', 'पीला', 'पीली', 'मஞ்சள்', 'పసుపు', 'മഞ്ഞ'],
            'brown': ['brown', 'browning', 'tan', 'भूरा', 'பழுப்பு', 'గోధుమ', 'തവിട്ട്'],
            'black': ['black', 'dark', 'necrotic', 'काला', 'கருப்பு', 'నలుపు', 'കറുപ്പ്'],
            'white': ['white', 'powdery', 'milky', 'सफेद', 'வெள்ளை', 'తెలుపు', 'വെള്ള', 'पाउडर', 'தூள்', 'పొడి', 'പൊടി'],
            'wilting': ['wilt', 'wilting', 'drooping', 'sagging', 'मुरझाना', 'வாடுதல்', 'విల్టింగ్', 'വാടൽ'],
            'curling': ['curl', 'curling', 'twisted', 'distorted', 'मुड़ना', 'சுருள்', 'వంగు', 'ചുരുളൽ'],
            'mold': ['mold', 'mildew', 'fungus', 'fuzzy', 'फफूंद', 'பூஞ்சை', 'ఫంగస్', 'പൂപ്പൽ'],
            'rust': ['rust', 'rusty', 'orange', 'जंग', 'துரு', 'తుప్పు', 'തുരുമ്പ്'],
            'rot': ['rot', 'rotting', 'decay', 'सड़न', 'அழுகல்', 'కుళ్ళు', 'ചീയൽ'],
            'stunted': ['stunted', 'small', 'dwarf', 'बौना', 'குட்டை', 'పెరుగుదల తగ్గుట', 'വളർച്ച കുറവ്'],
            'patches': ['patch', 'patches', 'area', 'धब्बा', 'திட்டுகள்', 'పాచెస్', 'പാച്ചുകൾ']
        }
    
    def extract(self, text):
        """
        Extract plant and symptoms from text
        Returns: dict with 'plant' and 'symptoms'
        FIXED: Better Unicode handling for multilingual text and improved symptom matching
        """
        if not text or not isinstance(text, str):
            print("[ENTITY] No text or invalid input provided")
            return {'plant': None, 'symptoms': []}
            
        text_lower = text.lower().strip()
        print(f"[ENTITY] Processing text: {text_lower}")
        
        # Extract plant
        detected_plant = None
        for plant, aliases in self.plants.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    detected_plant = plant
                    print(f"[ENTITY] Found plant: {detected_plant} (from alias: {alias})")
                    break
            if detected_plant:
                break
        
        # Extract symptoms (check both full text and individual words)
        detected_symptoms = []
        text_words = set(text_lower.split())
        
        # First check for multi-word symptoms in the full text
        for symptom, aliases in self.symptoms.items():
            for alias in aliases:
                if ' ' in alias and alias in text_lower:
                    if symptom not in detected_symptoms:
                        detected_symptoms.append(symptom)
                        print(f"[ENTITY] Found multi-word symptom: {symptom} (from: {alias})")
                    break
        
        # Then check for single-word symptoms in individual words
        for word in text_words:
            for symptom, aliases in self.symptoms.items():
                if symptom in detected_symptoms:
                    continue  # Skip already detected symptoms
                    
                for alias in aliases:
                    if ' ' not in alias and alias == word:
                        detected_symptoms.append(symptom)
                        print(f"[ENTITY] Found single-word symptom: {symptom} (from: {word})")
                        break
        
        print(f"[ENTITY] Final extraction - plant: {detected_plant}, symptoms: {detected_symptoms}")
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
            # Skip if plant doesn't match (unless plant is unknown)
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
        
        print(f"[MATCHER] Found {len(matches)} matches for symptoms {symptoms}")
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
            'prevention': "Prevention tips for {disease}: {prevention}",
            'severity': "This disease ({disease}) is classified as {severity}. {severity_details}",
            'causes': "Causes of {disease}: {causes}",
            'identification_success': "Based on your description, this appears to be {disease} (confidence: {confidence}%).",
            'identification_failure': "I couldn't identify a specific disease from your description. Please upload an image or provide more details.",
            'greeting': "Hello! I'm your plant disease assistant. You can upload an image or describe symptoms you're observing.",
            'thanks': "You're welcome! Let me know if you need any more help.",
            'no_context': "Please upload a plant image first or describe the symptoms you're seeing."
        }
        
        self.severity_details = {
            'severe': 'This can cause significant crop damage if left untreated. Immediate action is recommended.',
            'moderate': 'This can affect plant health but is manageable with proper treatment.',
            'mild': 'This is a minor issue but should still be addressed to maintain plant health.'
        }
    
    def generate(self, intent, disease_key=None, confidence=None, context=None):
        """
        Generate response based on intent and context
        Enhanced to handle greetings and other intents more effectively
        """
        # Handle greetings and thanks first
        if intent == 'greeting':
            return self.templates['greeting']
        elif intent == 'thanks':
            return self.templates['thanks']
        
        # Handle general queries
        if intent == 'general':
            if context and ('hello' in context.lower() or 'hi ' in context.lower() or 'hey ' in context.lower()):
                return self.templates['greeting']
            elif context and 'thank' in context.lower():
                return self.templates['thanks']
            else:
                return self.templates['greeting']
        
        # Check if we have disease context for disease-specific queries
        if not disease_key and intent in ['treatment', 'symptoms', 'prevention', 'severity', 'causes']:
            return "Please first tell me which plant disease you're asking about or upload an image."
        
        if not disease_key:
            return self.templates['no_context']
        
        # Get disease information
        disease_info = self.kb.get_disease_info(disease_key)
        disease_display = disease_key.replace('_', ' ').title()
        
        # Generate appropriate response based on intent
        if intent == 'treatment':
            treatment = disease_info.get('treatment')
            if not treatment:
                treatment = "No specific treatment information available. " \
                          "General recommendations include removing affected parts and applying appropriate fungicides."
            return self.templates['treatment'].format(
                disease=disease_display,
                treatment=treatment
            )
        
        elif intent == 'symptoms':
            symptoms = disease_info.get('symptoms', ['No symptoms listed'])
            # Make sure we have a list
            if isinstance(symptoms, str):
                symptoms = [symptoms]
            symptoms_str = ', '.join(symptoms)
            return self.templates['symptoms'].format(
                disease=disease_display,
                symptoms=symptoms_str
            )
        
        elif intent == 'prevention':
            prevention = disease_info.get('prevention', 'General prevention practices recommended.')
            return self.templates['prevention'].format(
                disease=disease_display,
                prevention=prevention
            )
        
        elif intent == 'severity':
            severity = disease_info.get('severity', 'moderate')
            severity_detail = self.severity_details.get(severity, 
                'The severity of this disease is not specified.')
            return self.templates['severity'].format(
                disease=disease_display,
                severity=severity,
                severity_details=severity_detail
            )
        
        elif intent == 'causes':
            causes = disease_info.get('causes', 'Causes not specified.')
            return self.templates['causes'].format(
                disease=disease_display,
                causes=causes
            )
        
        elif intent == 'identification':
            if confidence:
                return self.templates['identification_success'].format(
                    disease=disease_display,
                    confidence=f"{confidence:.1f}"
                )
            else:
                return self.templates['identification_failure']
        
        # Default response if intent not recognized
        return ("I'm not sure how to respond to that. "
                "You can ask about symptoms, treatment, prevention, or causes of plant diseases.")


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
    FIXED: Better intent handling and context management
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
    
    def process_text_query(self, text, context_disease=None):
        """
        Process text query from user
        Handles both new queries and follow-up questions with improved intent handling
        """
        try:
            # Clean and normalize input
            text = text.strip()
            if not text:
                return "I didn't catch that. Could you please rephrase?"
                
            # Handle greetings and thanks first
            if any(word in text.lower() for word in ['hello', 'hi', 'hey', 'namaste', 'hola', 'greetings']):
                return self.response_generator.generate('greeting')
                
            if any(word in text.lower() for word in ['thank', 'thanks', 'dhanyavad', 'nandri']):
                return self.response_generator.generate('thanks')
            
            # Classify intent with more context
            intent = self.intent_classifier.classify(text)
            print(f"[NLP] Classified intent: {intent}")
            
            # Extract entities with improved handling for short texts
            entities = self.entity_extractor.extract(text)
            print(f"[NLP] Extracted entities: {entities}")
            
            # If we have a context disease from image or previous message, use it
            current_disease = context_disease or self.memory.last_disease
            
            # Update conversation memory with current context
            self.memory.update(
                intent=intent,
                disease=current_disease,
                symptoms=entities.get('symptoms', []),
                plant=entities.get('plant')
            )
            
            # Handle identification requests or when no disease context exists
            if intent == 'identification' or not current_disease:
                if entities.get('symptoms'):
                    # Find matching diseases based on symptoms
                    matches = self.symptom_matcher.match(
                        entities.get('plant'), 
                        entities['symptoms']
                    )
                    
                    if matches:
                        disease_key, score, confidence = matches[0]
                        self.memory.update(disease=disease_key)
                        
                        # Generate comprehensive response
                        response = self.response_generator.generate(
                            'identification_success',
                            disease_key=disease_key,
                            confidence=min(confidence, 99.9)  # Cap at 99.9%
                        )
                        
                        # Add more context if it's a new identification
                        if not context_disease:
                            response += f"\n\n{self.response_generator.generate('symptoms', disease_key=disease_key)}"
                            response += f"\n\n{self.response_generator.generate('treatment', disease_key=disease_key)}"
                        
                        return response
                    else:
                        return self.response_generator.generate('identification_failure')
                else:
                    # No symptoms provided, ask for more details
                    return "Could you please describe the symptoms you're seeing? For example: 'My tomato leaves have yellow spots'"
            
            # Handle other intents with known disease context
            if current_disease:
                # For follow-up questions, provide more detailed responses
                if intent in ['treatment', 'symptoms', 'prevention', 'severity', 'causes']:
                    response = self.response_generator.generate(
                        intent,
                        disease_key=current_disease,
                        context=text
                    )
                    
                    # For treatment questions, also include prevention tips
                    if intent == 'treatment':
                        response += f"\n\n{self.response_generator.generate('prevention', disease_key=current_disease)}"
                    
                    return response
                
                # For general questions about the current disease
                return self.response_generator.generate('general', disease_key=current_disease, context=text)
            
            # Default response for unhandled cases
            return ("I'm not sure how to help with that. You can ask about symptoms, "
                   "treatment, prevention, or causes of plant diseases. Or upload an image for identification.")
            
        except Exception as e:
            print(f"Error in process_text_query: {e}")
            import traceback
            traceback.print_exc()
            return "I encountered an error processing your request. Please try again with more details."
    
    def reset_conversation(self):
        """Reset the conversation memory"""
        self.memory.reset()
        return "Conversation reset. How can I help you today?"


# Create a global instance of the NLP engine
nlp_engine = CustomNLPEngine()