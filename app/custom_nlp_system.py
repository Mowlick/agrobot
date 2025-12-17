"""
Custom NLP System - Enhanced Disease Classification
Improved text-based disease detection from symptom descriptions
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
        """Load symptom-to-disease mappings with enhanced keywords"""
        # Enhanced symptom keywords with more variations
        self.symptom_keywords = {
            'spots': ['spot', 'spots', 'dotted', 'marking', 'lesion', 'lesions', 'blotch', 'blotches',
                     'speck', 'specks', 'धब्बे', 'புள்ளிகள்', 'మచ్చలు', 'പാടുകൾ'],
            'yellow': ['yellow', 'yellowing', 'chlorosis', 'pale', 'light green', 'lime',
                      'पीला', 'पीली', 'மஞ்சள்', 'పసుపు', 'മഞ്ഞ'],
            'brown': ['brown', 'browning', 'tan', 'bronze', 'rust colored', 'reddish brown',
                     'भूरा', 'பழுப்பு', 'గోధుమ', 'തവിട്ടു'],
            'black': ['black', 'dark', 'necrotic', 'dead', 'charred', 'blackened',
                     'काला', 'கருப்பு', 'నల్పు', 'കറുപ്പു'],
            'white': ['white', 'powdery', 'milky', 'chalky', 'whitish', 'pale white',
                     'सफेद', 'வெள்ளை', 'తెలుపు', 'വെള്ള'],
            'wilting': ['wilt', 'wilting', 'drooping', 'sagging', 'limp', 'droopy', 'hanging',
                       'मुरझाना', 'வாடுதல்', 'విల్టింగ్', 'വാടൽ'],
            'curling': ['curl', 'curling', 'twisted', 'distorted', 'crinkled', 'warped', 'bent',
                       'मुड़ना', 'சுருள்', 'వంగు', 'ചുരുൾ'],
            'mold': ['mold', 'mildew', 'fungus', 'fuzzy', 'moldy', 'fungal growth',
                    'फफूंद', 'பூஞ்சை', 'ఫంగస్', 'പൂപ്പൽ'],
            'rust': ['rust', 'rusty', 'orange', 'pustule', 'reddish', 'rust spots',
                    'जंग', 'துரு', 'తుప్పు', 'തുരുമ്പു'],
            'rot': ['rot', 'rotting', 'decay', 'decompose', 'mushy', 'soft', 'rotten',
                   'सड़न', 'அழுகல்', 'కుళ్ళు', 'ചീയൽ'],
            'blight': ['blight', 'blighted', 'blight disease',
                      'ब्लाइट', 'நோய்', 'బ్లైట్', 'ബ്ലൈറ്റ്'],
            'scab': ['scab', 'rough', 'crusty', 'scabby', 'roughness',
                    'खुरदरा', 'சிரங்கு', 'స్కాబ్', 'ചൊറി'],
            'mosaic': ['mosaic', 'mottled', 'pattern', 'patchy', 'variegated',
                      'मोज़ेक', 'மொசைக்', 'మొజాయిక్', 'മൊസൈക്'],
            'ring': ['ring', 'circular', 'concentric', 'rings', 'circles', 'target',
                    'गोलाकार', 'வளையம்', 'వలయం', 'വളയം'],
            'stunted': ['stunted', 'small', 'dwarf', 'reduced growth', 'undersized', 'tiny',
                       'बौना', 'குட்டை', 'పెరుగుదల తగ్గుట', 'വളർച്ച കുറവ്'],
            'leaf': ['leaf', 'leaves', 'foliage', 'leafy',
                    'पत्ती', 'இலை', 'ఆకు', 'ഇല'],
            'powdery': ['powdery', 'powder', 'dusty', 'powdered',
                       'पाउडर', 'தூள்', 'పొడి', 'പൊടി'],
            'patches': ['patch', 'patches', 'area', 'areas', 'sections',
                       'धब्बा', 'திட்டுகள்', 'పాచెస్', 'പാച്ചുകൾ'],
            'damage': ['damage', 'damaged', 'injury', 'hurt', 'affected'],
            'deformed': ['deformed', 'deformity', 'malformed', 'abnormal', 'irregular'],
            'holes': ['hole', 'holes', 'eaten', 'chewed', 'perforated'],
            'streaks': ['streak', 'streaks', 'stripe', 'stripes', 'lines'],
            'coating': ['coating', 'covered', 'layer', 'film', 'covering']
        }
        
        # Enhanced disease-symptom associations with more details
        self.disease_symptoms = {
            'tomato_early_blight': ['spots', 'brown', 'yellow', 'ring', 'leaf', 'concentric'],
            'tomato_late_blight': ['spots', 'brown', 'black', 'wilting', 'rot', 'leaf', 'water'],
            'tomato_bacterial_spot': ['spots', 'brown', 'yellow', 'leaf', 'small'],
            'tomato_leaf_mold': ['yellow', 'mold', 'wilting', 'leaf', 'fuzzy'],
            'tomato_septoria_leaf_spot': ['spots', 'brown', 'yellow', 'leaf', 'small', 'circular'],
            'tomato_spider_mites_two_spotted_spider_mite': ['spots', 'yellow', 'curling', 'leaf', 'stippling'],
            'tomato_target_spot': ['spots', 'brown', 'ring', 'leaf', 'target', 'concentric'],
            'tomato_tomato_yellow_leaf_curl_virus': ['yellow', 'curling', 'stunted', 'leaf', 'upward'],
            'tomato_mosaic_virus': ['mosaic', 'yellow', 'curling', 'leaf', 'mottled', 'pattern'],
            'potato_early_blight': ['spots', 'brown', 'yellow', 'ring', 'leaf', 'concentric'],
            'potato_late_blight': ['spots', 'black', 'wilting', 'rot', 'leaf', 'water'],
            'corn_common_rust': ['rust', 'brown', 'spots', 'leaf', 'orange', 'pustule'],
            'corn_cercospora_leaf_spot_gray_leaf_spot': ['spots', 'brown', 'yellow', 'leaf', 'gray', 'rectangular'],
            'corn_northern_leaf_blight': ['spots', 'brown', 'leaf', 'long', 'cigar'],
            'grape_black_rot': ['spots', 'black', 'rot', 'fruit', 'mummy'],
            'grape_esca_black_measles': ['spots', 'brown', 'wilting', 'stripe', 'tiger'],
            'grape_leaf_blight': ['spots', 'brown', 'leaf', 'irregular'],
            'apple_apple_scab': ['spots', 'brown', 'black', 'scab', 'rough', 'fruit'],
            'apple_black_rot': ['rot', 'black', 'brown', 'fruit', 'concentric'],
            'apple_cedar_apple_rust': ['rust', 'yellow', 'spots', 'orange', 'leaf'],
            'pepper_bacterial_spot': ['spots', 'brown', 'yellow', 'leaf', 'raised'],
            'peach_bacterial_spot': ['spots', 'brown', 'leaf', 'shot', 'hole'],
            'cherry_powdery_mildew': ['mold', 'white', 'powdery', 'leaf', 'coating'],
            'squash_powdery_mildew': ['mold', 'white', 'powdery', 'leaf', 'coating'],
            'strawberry_leaf_scorch': ['brown', 'spots', 'wilting', 'leaf', 'purple', 'margin'],
            'orange_citrus_greening': ['yellow', 'stunted', 'asymmetric', 'mottled']
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
    Classifies user intent using enhanced rule-based matching
    """
    
    def __init__(self):
        self.intents = {
            'treatment': [
                'treat', 'treatment', 'cure', 'remedy', 'fix', 'solve',
                'medicine', 'fungicide', 'spray', 'chemical', 'control',
                'how to treat', 'what to do', 'manage', 'apply', 'इलाज', 'उपचार',
                'chikitsa', 'maruthuvam', 'chikitsaku', 'how do i', 'what should i do'
            ],
            'symptoms': [
                'symptom', 'sign', 'look like', 'appear', 'indication',
                'what is', 'describe', 'show', 'display', 'characteristic',
                'what are', 'tell me about', 'lakshanam', 'lakshana',
                'लक्षण', 'रोग लक्षण', 'looks like', 'showing'
            ],
            'prevention': [
                'prevent', 'avoid', 'stop', 'protect', 'prevention',
                'how to prevent', 'keep from', 'precaution', 'roktham',
                'रोकथाम', 'बचाव', 'how can i prevent'
            ],
            'severity': [
                'severe', 'serious', 'dangerous', 'bad', 'how bad',
                'damage', 'harmful', 'spread', 'contagious', 'is this serious',
                'should i worry', 'critical', 'gambheer', 'गंभीर', 'खतरनाक'
            ],
            'causes': [
                'cause', 'why', 'reason', 'how did', 'where from',
                'origin', 'source', 'what causes', 'kaaranam', 'karanam',
                'कारण', 'वजह', 'क्यों होता है'
            ],
            'identification': [
                'what disease', 'identify', 'diagnose', 'tell me',
                'what is this', 'which disease', 'recognize', 'is this',
                'रोग की पहचान', 'यह कौन सी बीमारी है', 'my plant has',
                'i see', 'there are', 'having'
            ],
            'greeting': [
                'hello', 'hi', 'hey', 'greetings', 'namaste', 'vanakkam',
                'namaskaram', 'नमस्ते', 'हैलो'
            ],
            'thanks': [
                'thanks', 'thank you', 'dhanyavad', 'nandri',
                'धन्यवाद', 'थैंक यू'
            ],
            'general': [
                'ok', 'yes', 'no', 'help', 'assist', 'sahayam',
                'मदद', 'सहायता'
            ]
        }
        
        # Build keyword-intent mapping
        self.keyword_map = {}
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword not in self.keyword_map:
                    self.keyword_map[keyword] = []
                self.keyword_map[keyword].append(intent)
    
    def classify(self, text):
        """Enhanced intent classification with context awareness"""
        text_lower = text.lower().strip()
        
        # Quick greeting/thanks check
        if any(w in text_lower.split() for w in ['hello', 'hi', 'hey', 'namaste']):
            return 'greeting'
        if any(w in text_lower.split() for w in ['thanks', 'thank']):
            return 'thanks'
        
        # Score each intent
        intent_scores = Counter()
        
        # Check for exact phrases (higher weight)
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    weight = len(keyword.split()) * 3
                    intent_scores[intent] += weight
        
        # Check individual words
        words = text_lower.split()
        for word in words:
            if word in self.keyword_map:
                for intent in self.keyword_map[word]:
                    intent_scores[intent] += 2
            
            # Partial match for longer keywords
            for keyword, intents in self.keyword_map.items():
                if len(keyword) > 3:
                    if keyword in word or word in keyword:
                        for intent in intents:
                            intent_scores[intent] += 1
        
        # Special patterns for identification
        if any(pattern in text_lower for pattern in ['my plant', 'plant has', 'i see', 'there are', 'having', 'with']):
            intent_scores['identification'] += 5
        
        # Get highest scoring intent
        if intent_scores:
            top_intent = intent_scores.most_common(1)[0][0]
            print(f"[INTENT] Classified as '{top_intent}' with scores: {dict(intent_scores)}")
            return top_intent
        
        print(f"[INTENT] No clear intent found, defaulting to 'identification'")
        return 'identification'  # Changed default to identification for symptom descriptions


class EntityExtractor:
    """Enhanced entity extraction with fuzzy matching"""
    
    def __init__(self):
        self.plants = {
            'tomato': ['tomato', 'tomatoes', 'tamatar', 'टमाटर', 'தக்காளி'],
            'potato': ['potato', 'potatoes', 'aloo', 'आलू', 'உருளைக்கிழங்கு'],
            'corn': ['corn', 'maize', 'makka', 'मक्का', 'சோளம்'],
            'apple': ['apple', 'apples', 'seb', 'सेब', 'ஆப்பிள்'],
            'grape': ['grape', 'grapes', 'angur', 'अंगूर', 'திராட்சை'],
            'pepper': ['pepper', 'peppers', 'mirchi', 'chilli', 'मिर्च', 'மிளகு'],
            'peach': ['peach', 'peaches', 'aadu', 'आड़ू'],
            'cherry': ['cherry', 'cherries', 'चेरी'],
            'strawberry': ['strawberry', 'strawberries', 'स्ट्रॉबेरी'],
            'orange': ['orange', 'oranges', 'santra', 'संतरा', 'ஆரஞ்சு'],
            'blueberry': ['blueberry', 'blueberries'],
            'squash': ['squash', 'pumpkin', 'कद्दू', 'பூசணி'],
            'soybean': ['soybean', 'soybeans', 'soya', 'सोया'],
            'rice': ['rice', 'paddy', 'chawal', 'चावल', 'அரிசி'],
            'wheat': ['wheat', 'gehun', 'गेहूं', 'கோதுமை']
        }
        
        self.symptoms = {
            'spots': ['spot', 'spots', 'dotted', 'marking', 'lesion', 'blotch', 'speck'],
            'yellow': ['yellow', 'yellowing', 'pale', 'chlorotic', 'light green'],
            'brown': ['brown', 'browning', 'tan', 'bronze'],
            'black': ['black', 'dark', 'necrotic', 'blackened'],
            'white': ['white', 'powdery', 'milky', 'chalky', 'whitish'],
            'wilting': ['wilt', 'wilting', 'drooping', 'sagging', 'droopy'],
            'curling': ['curl', 'curling', 'twisted', 'distorted', 'warped'],
            'mold': ['mold', 'mildew', 'fungus', 'fuzzy', 'moldy'],
            'rust': ['rust', 'rusty', 'orange', 'pustule', 'reddish'],
            'rot': ['rot', 'rotting', 'decay', 'mushy', 'soft', 'rotten'],
            'stunted': ['stunted', 'small', 'dwarf', 'undersized', 'tiny'],
            'patches': ['patch', 'patches', 'area', 'areas'],
            'ring': ['ring', 'rings', 'circular', 'concentric', 'target'],
            'coating': ['coating', 'covered', 'layer', 'film'],
            'holes': ['hole', 'holes', 'eaten', 'chewed'],
            'streaks': ['streak', 'streaks', 'stripe', 'lines']
        }
    
    def extract(self, text):
        """Enhanced extraction with better pattern matching"""
        if not text or not isinstance(text, str):
            return {'plant': None, 'symptoms': []}
            
        text_lower = text.lower().strip()
        print(f"[ENTITY] Processing: {text_lower}")
        
        # Extract plant
        detected_plant = None
        for plant, aliases in self.plants.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    detected_plant = plant
                    print(f"[ENTITY] Found plant: {plant}")
                    break
            if detected_plant:
                break
        
        # Extract symptoms with enhanced matching
        detected_symptoms = []
        text_words = set(text_lower.split())
        
        # Multi-word symptom phrases
        for symptom, aliases in self.symptoms.items():
            for alias in aliases:
                if ' ' in alias and alias in text_lower:
                    if symptom not in detected_symptoms:
                        detected_symptoms.append(symptom)
                        print(f"[ENTITY] Found phrase symptom: {symptom} ('{alias}')")
                    break
        
        # Single-word symptoms
        for word in text_words:
            for symptom, aliases in self.symptoms.items():
                if symptom in detected_symptoms:
                    continue
                for alias in aliases:
                    if ' ' not in alias:
                        if alias == word or (len(alias) > 3 and (alias in word or word in alias)):
                            detected_symptoms.append(symptom)
                            print(f"[ENTITY] Found word symptom: {symptom} ('{word}' matches '{alias}')")
                            break
        
        # Pattern-based symptom extraction
        if not detected_symptoms:
            # Look for color + spots pattern
            colors = ['yellow', 'brown', 'black', 'white']
            if any(c in text_lower for c in colors) and any(s in text_lower for s in ['spot', 'patch', 'mark']):
                for color in colors:
                    if color in text_lower:
                        detected_symptoms.append(color)
                detected_symptoms.append('spots')
                print(f"[ENTITY] Detected color+spots pattern")
        
        # Remove duplicates
        unique_symptoms = []
        for s in detected_symptoms:
            if s not in unique_symptoms:
                unique_symptoms.append(s)
        
        print(f"[ENTITY] Final: plant={detected_plant}, symptoms={unique_symptoms}")
        return {
            'plant': detected_plant,
            'symptoms': unique_symptoms
        }


class SymptomMatcher:
    """Enhanced symptom matching with better scoring"""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        
        # Symptom groups for better matching
        self.symptom_groups = {
            'color_change': ['yellow', 'brown', 'black', 'white', 'orange'],
            'surface_damage': ['spots', 'patches', 'lesions', 'blotches'],
            'growth_issues': ['wilting', 'curling', 'stunted', 'deformed'],
            'fungal_signs': ['mold', 'mildew', 'rust', 'powdery', 'fuzzy'],
            'tissue_damage': ['rot', 'decay', 'necrotic', 'holes']
        }
    
    def _expand_symptoms(self, symptoms):
        """Expand symptoms using synonyms and related terms"""
        expanded = set(symptoms)
        
        # Add related symptoms from same group
        for symptom in symptoms:
            for group, group_symptoms in self.symptom_groups.items():
                if symptom in group_symptoms:
                    # Don't add all, just the most related ones
                    if symptom == 'yellow' and 'spots' in symptoms:
                        expanded.update(['chlorotic', 'pale'])
                    if symptom == 'mold':
                        expanded.update(['mildew', 'fungus', 'powdery'])
                    if symptom == 'spots':
                        expanded.update(['lesion', 'blotch'])
        
        return list(expanded)
    
    def match(self, plant, symptoms):
        """Enhanced symptom matching with improved scoring"""
        print(f"[MATCHER] Matching plant:{plant}, symptoms:{symptoms}")
        
        if not symptoms:
            return []
        
        expanded_symptoms = self._expand_symptoms(symptoms)
        print(f"[MATCHER] Expanded symptoms: {expanded_symptoms}")
        
        matches = []
        
        for disease_key, disease_data in self.kb.diseases.items():
            # Skip if plant doesn't match
            if plant and disease_data['plant'].lower() != plant.lower():
                continue
            
            disease_symptoms = self.kb.disease_symptoms.get(disease_key, [])
            if not disease_symptoms:
                continue
            
            # Calculate match score
            exact_matches = set(symptoms) & set(disease_symptoms)
            expanded_matches = set(expanded_symptoms) & set(disease_symptoms)
            
            match_count = len(exact_matches) + (len(expanded_matches) * 0.5)
            
            if match_count > 0:
                # Calculate confidence
                user_coverage = len(exact_matches) / len(symptoms)
                disease_coverage = len(exact_matches) / len(disease_symptoms)
                
                # Weighted score
                confidence = (user_coverage * 0.6 + disease_coverage * 0.4) * 100
                
                # Boost for exact plant match
                if plant and disease_data['plant'].lower() == plant.lower():
                    confidence *= 1.4
                
                # Boost for high symptom count
                if len(exact_matches) >= 2:
                    confidence *= 1.2
                
                # Boost for specific disease patterns
                disease_lower = disease_key.lower()
                if 'yellow' in symptoms and 'curl' in symptoms and 'curl' in disease_lower:
                    confidence = max(confidence, 85)
                if 'brown' in symptoms and 'spot' in symptoms and 'blight' in disease_lower:
                    confidence = max(confidence, 80)
                if 'white' in symptoms and 'powdery' in symptoms and 'mildew' in disease_lower:
                    confidence = max(confidence, 90)
                
                confidence = min(confidence, 100)
                
                print(f"[MATCHER] {disease_key}: score={match_count:.1f}, conf={confidence:.1f}%")
                matches.append((disease_key, match_count, confidence, exact_matches))
        
        # Sort by confidence then match count
        matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        if matches:
            print(f"[MATCHER] Top match: {matches[0][0]} ({matches[0][2]:.1f}%)")
            
            # Return top matches if confidence is reasonable
            if matches[0][2] >= 35:  # Lower threshold for better recall
                return matches[:3]
        
        return matches[:1] if matches else []


class ResponseGenerator:
    """Generates natural responses"""
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        
        self.templates = {
            'treatment': "**Treatment for {disease}:**\n{treatment}",
            'symptoms': "**Common symptoms of {disease}:**\n{symptoms}",
            'prevention': "**Prevention tips for {disease}:**\n{prevention}",
            'severity': "**{disease}** is classified as {severity}. {severity_details}",
            'causes': "**Causes of {disease}:**\n{causes}",
            'greeting': "Hello! I'm your plant disease assistant. You can describe symptoms or upload an image.",
            'thanks': "You're welcome! Let me know if you need more help.",
            'no_context': "Please describe the symptoms you're seeing or upload a plant image."
        }
        
        self.severity_details = {
            'severe': 'This can cause significant crop damage. Immediate action recommended.',
            'moderate': 'This can affect plant health but is manageable with proper treatment.',
            'mild': 'This is a minor issue but should still be addressed.'
        }
    
    def generate(self, intent, disease_key=None, confidence=None, context=None, matches=None):
        """Generate appropriate response"""
        
        if intent == 'greeting':
            return self.templates['greeting']
        elif intent == 'thanks':
            return self.templates['thanks']
        
        # Handle multiple matches
        if intent == 'identification' and matches and len(matches) > 1:
            response = "Based on your description, possible diseases:\n\n"
            for i, (dis_key, score, conf, symp) in enumerate(matches[:3], 1):
                dis_name = dis_key.replace('_', ' ').title()
                response += f"{i}. **{dis_name}** ({conf:.1f}% match)\n"
            response += "\nFor more accurate diagnosis, please provide more details or upload an image."
            return response
        
        # Need disease context
        if not disease_key and intent in ['treatment', 'symptoms', 'prevention', 'severity', 'causes']:
            return "Please tell me which disease you're asking about or describe the symptoms."
        
        if not disease_key:
            return self.templates['no_context']
        
        disease_info = self.kb.get_disease_info(disease_key)
        disease_display = disease_key.replace('_', ' ').title()
        
        if intent == 'treatment':
            treatment = disease_info.get('treatment', 'No specific treatment available.')
            return self.templates['treatment'].format(disease=disease_display, treatment=treatment)
        
        elif intent == 'symptoms':
            symptoms = disease_info.get('symptoms', [])
            if isinstance(symptoms, str):
                symptoms = [symptoms]
            return self.templates['symptoms'].format(disease=disease_display, symptoms=', '.join(symptoms))
        
        elif intent == 'prevention':
            prevention = disease_info.get('prevention', 'General prevention practices recommended.')
            return self.templates['prevention'].format(disease=disease_display, prevention=prevention)
        
        elif intent == 'severity':
            severity = disease_info.get('severity', 'moderate')
            details = self.severity_details.get(severity, '')
            return self.templates['severity'].format(
                disease=disease_display, severity=severity, severity_details=details
            )
        
        elif intent == 'causes':
            causes = disease_info.get('causes', 'Causes not specified.')
            return self.templates['causes'].format(disease=disease_display, causes=causes)
        
        elif intent == 'identification':
            if confidence:
                response = f"Based on your description: **{disease_display}** ({confidence:.1f}% confidence)\n\n"
                
                symptoms = disease_info.get('symptoms', [])
                if symptoms:
                    if isinstance(symptoms, str):
                        symptoms = [symptoms]
                    response += f"**Symptoms:** {', '.join(symptoms)}\n\n"
                
                treatment = disease_info.get('treatment')
                if treatment:
                    response += f"**Treatment:** {treatment}\n\n"
                
                prevention = disease_info.get('prevention')
                if prevention:
                    response += f"**Prevention:** {prevention}"
                
                return response
            else:
                return "I couldn't identify a specific disease. Please provide more details or upload an image."
        
        return "I'm not sure how to help with that. Try describing symptoms or uploading an image."


class ConversationMemory:
    """Maintains conversation context"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.last_disease = None
        self.last_intent = None
        self.last_symptoms = []
        self.last_plant = None
        self.conversation_history = []
    
    def update(self, intent, disease=None, symptoms=None, plant=None):
        self.last_intent = intent
        if disease:
            self.last_disease = disease
        if symptoms:
            self.last_symptoms = symptoms
        if plant:
            self.last_plant = plant
    
    def add_turn(self, user_input, bot_response):
        self.conversation_history.append({
            'user': user_input,
            'bot': bot_response
        })
    
    def get_context(self):
        return {
            'disease': self.last_disease,
            'intent': self.last_intent,
            'symptoms': self.last_symptoms,
            'plant': self.last_plant
        }


class CustomNLPEngine:
    """Main NLP engine with enhanced text-based disease detection"""
    
    def __init__(self):
        self.kb = DiseaseKnowledgeBase()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.symptom_matcher = SymptomMatcher(self.kb)
        self.response_generator = ResponseGenerator(self.kb)
        self.memory = ConversationMemory()
    
    def process_image_prediction(self, disease_key, confidence):
        """Process CNN prediction"""
        self.memory.reset()
        self.memory.update(intent='identification', disease=disease_key)
        
        disease_info = self.kb.get_disease_info(disease_key)
        
        response = f"Detected: **{disease_key.replace('_', ' ').title()}** ({confidence:.1f}% confidence)\n\n"
        response += f"**Treatment:** {disease_info.get('treatment', 'No info available.')}"
        
        return response
    
    def process_text_query(self, text, context_disease=None):
        """Enhanced text query processing with better disease detection"""
        try:
            print(f"\n[NLP] Processing: {text}")
            
            text = text.strip()
            if not text:
                return "Please enter a message."
            
            # Quick greeting/thanks check
            if any(w in text.lower() for w in ['hello', 'hi', 'hey', 'namaste']):
                return self.response_generator.generate('greeting')
            if any(w in text.lower() for w in ['thank', 'thanks']):
                return self.response_generator.generate('thanks')
            
            # Classify intent
            intent = self.intent_classifier.classify(text)
            print(f"[NLP] Intent: {intent}")
            
            # Extract entities
            entities = self.entity_extractor.extract(text)
            print(f"[NLP] Entities: {entities}")
            
            current_disease = context_disease or self.memory.last_disease
            print(f"[NLP] Context disease: {current_disease}")
            
            # Update memory
            self.memory.update(
                intent=intent,
                disease=current_disease,
                symptoms=entities.get('symptoms', []),
                plant=entities.get('plant')
            )
            
            # Handle identification or when symptoms are present
            if intent == 'identification' or (entities.get('symptoms') and not current_disease):
                if entities.get('symptoms'):
                    print(f"[NLP] Matching symptoms: {entities['symptoms']}")
                    
                    matches = self.symptom_matcher.match(
                        entities.get('plant'),
                        entities['symptoms']
                    )
                    
                    if matches:
                        top_match = matches[0]
                        disease_key, score, confidence, matched_symp = top_match
                        
                        self.memory.update(intent=intent, disease=disease_key)
                        
                        # Show multiple if low-medium confidence
                        if confidence < 60 or (confidence < 75 and len(matches) > 1):
                            return self.response_generator.generate(
                                'identification',
                                disease_key=disease_key,
                                confidence=confidence,
                                matches=matches
                            )
                        else:
                            # High confidence - detailed response
                            return self.response_generator.generate(
                                'identification',
                                disease_key=disease_key,
                                confidence=confidence
                            )
                    else:
                        return ("I couldn't match those symptoms to a specific disease. "
                               "Please provide more details or upload an image.")
                else:
                    return "Please describe the symptoms you're seeing (e.g., yellow spots, brown leaves)."
            
            # Handle other intents with disease context
            if current_disease:
                response = self.response_generator.generate(
                    intent,
                    disease_key=current_disease,
                    context=text
                )
                
                # Add prevention for treatment queries
                if intent == 'treatment':
                    response += f"\n\n{self.response_generator.generate('prevention', disease_key=current_disease)}"
                
                return response
            
            return ("I need more information. Please describe symptoms or ask about a specific disease.")
            
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return "I encountered an error. Please try rephrasing or provide more details."
    
    def reset_conversation(self):
        self.memory.reset()
        return "Conversation reset. How can I help you?"


# Global instance
nlp_engine = CustomNLPEngine()