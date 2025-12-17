"""
Comprehensive Test Suite for Text-Based Disease Detection
Tests the enhanced NLP system's ability to classify diseases from symptom descriptions
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from custom_nlp_system import nlp_engine
from multilingual_nlp_processor import process_message


def test_english_symptom_descriptions():
    """Test English symptom descriptions"""
    print("\n" + "="*80)
    print("TEST 1: ENGLISH SYMPTOM DESCRIPTIONS")
    print("="*80)
    
    test_cases = [
        {
            'description': 'Yellow leaves with brown spots on tomato',
            'expected_diseases': ['tomato_early_blight', 'tomato_septoria_leaf_spot'],
            'plant': 'tomato'
        },
        {
            'description': 'My tomato plant leaves are curling and turning yellow',
            'expected_diseases': ['tomato_tomato_yellow_leaf_curl_virus'],
            'plant': 'tomato'
        },
        {
            'description': 'White powdery coating on tomato leaves',
            'expected_diseases': ['cherry_powdery_mildew', 'squash_powdery_mildew'],
            'plant': None
        },
        {
            'description': 'Black spots with rot on tomato',
            'expected_diseases': ['tomato_late_blight'],
            'plant': 'tomato'
        },
        {
            'description': 'Orange rust spots on corn leaves',
            'expected_diseases': ['corn_common_rust'],
            'plant': 'corn'
        },
        {
            'description': 'Brown spots with concentric rings on potato leaves',
            'expected_diseases': ['potato_early_blight'],
            'plant': 'potato'
        },
        {
            'description': 'Tomato leaves have mottled yellow and green pattern',
            'expected_diseases': ['tomato_mosaic_virus'],
            'plant': 'tomato'
        },
        {
            'description': 'Small brown spots on pepper leaves',
            'expected_diseases': ['pepper_bacterial_spot'],
            'plant': 'pepper'
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Description: {test['description']}")
        print(f"Expected: {test['expected_diseases']}")
        
        try:
            result = process_message(test['description'], 'en')
            
            if result.get('status') == 'success':
                detected_disease = result.get('context', {}).get('disease')
                response = result.get('response', '')
                
                print(f"Detected: {detected_disease}")
                print(f"Response preview: {response[:100]}...")
                
                # Check if detected disease matches expected
                if detected_disease and any(exp in detected_disease for exp in test['expected_diseases']):
                    print("✓ PASS - Correct disease identified")
                    passed += 1
                else:
                    print(f"✗ FAIL - Expected one of {test['expected_diseases']}, got {detected_disease}")
                    failed += 1
            else:
                print(f"✗ FAIL - Processing error: {result.get('error')}")
                failed += 1
                
        except Exception as e:
            print(f"✗ FAIL - Exception: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"English Tests: {passed} passed, {failed} failed out of {len(test_cases)}")
    print(f"{'='*80}")
    
    return passed, failed


def test_multilingual_descriptions():
    """Test multilingual symptom descriptions"""
    print("\n" + "="*80)
    print("TEST 2: MULTILINGUAL SYMPTOM DESCRIPTIONS")
    print("="*80)
    
    test_cases = [
        {
            'description': 'टमाटर के पत्तों पर भूरे धब्बे हैं',
            'lang': 'hi',
            'expected': 'early_blight',
            'label': 'Hindi - Brown spots on tomato leaves'
        },
        {
            'description': 'तक்காளி இலைகளில் பழுப்பு புள்ளிகள்',
            'lang': 'ta',
            'expected': 'spot',
            'label': 'Tamil - Brown spots on tomato'
        },
        {
            'description': 'టమాటో ఆకులపై గోధుమ మచ్చలు',
            'lang': 'te',
            'expected': 'spot',
            'label': 'Telugu - Brown spots on tomato'
        },
        {
            'description': 'തക്കാളി ഇലകളിൽ തവിട്ടു പാടുകൾ',
            'lang': 'ml',
            'expected': 'spot',
            'label': 'Malayalam - Brown spots on tomato'
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test['label']} ---")
        print(f"Description: {test['description']}")
        print(f"Language: {test['lang']}")
        
        try:
            result = process_message(test['description'], test['lang'])
            
            if result.get('status') == 'success':
                english_text = result.get('english_text', '')
                detected_disease = result.get('context', {}).get('disease')
                
                print(f"Translated: {english_text}")
                print(f"Detected: {detected_disease}")
                
                if detected_disease and test['expected'] in detected_disease:
                    print("✓ PASS - Disease detected correctly")
                    passed += 1
                else:
                    print(f"✗ FAIL - Expected '{test['expected']}' in disease name")
                    failed += 1
            else:
                print(f"✗ FAIL - Processing error")
                failed += 1
                
        except Exception as e:
            print(f"✗ FAIL - Exception: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Multilingual Tests: {passed} passed, {failed} failed out of {len(test_cases)}")
    print(f"{'='*80}")
    
    return passed, failed


def test_entity_extraction():
    """Test entity extraction accuracy"""
    print("\n" + "="*80)
    print("TEST 3: ENTITY EXTRACTION")
    print("="*80)
    
    test_cases = [
        {
            'text': 'My tomato plant has yellow leaves',
            'expected_plant': 'tomato',
            'expected_symptoms': ['yellow']
        },
        {
            'text': 'Brown spots on potato',
            'expected_plant': 'potato',
            'expected_symptoms': ['brown', 'spots']
        },
        {
            'text': 'White powdery mildew on leaves',
            'expected_plant': None,
            'expected_symptoms': ['white', 'powdery', 'mold']
        },
        {
            'text': 'Corn with rusty orange spots',
            'expected_plant': 'corn',
            'expected_symptoms': ['rust', 'orange', 'spots']
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: {test['text']}")
        
        entities = nlp_engine.entity_extractor.extract(test['text'])
        
        plant_match = entities.get('plant') == test['expected_plant']
        symptoms = entities.get('symptoms', [])
        symptom_match = any(exp in symptoms for exp in test['expected_symptoms'])
        
        print(f"Expected plant: {test['expected_plant']}, Got: {entities.get('plant')}")
        print(f"Expected symptoms: {test['expected_symptoms']}, Got: {symptoms}")
        
        if plant_match and symptom_match:
            print("✓ PASS - Entities extracted correctly")
            passed += 1
        else:
            print("✗ FAIL - Entity extraction mismatch")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Entity Tests: {passed} passed, {failed} failed out of {len(test_cases)}")
    print(f"{'='*80}")
    
    return passed, failed


def test_intent_classification():
    """Test intent classification"""
    print("\n" + "="*80)
    print("TEST 4: INTENT CLASSIFICATION")
    print("="*80)
    
    test_cases = [
        {'text': 'How do I treat early blight?', 'expected': 'treatment'},
        {'text': 'What are the symptoms of late blight?', 'expected': 'symptoms'},
        {'text': 'How can I prevent tomato diseases?', 'expected': 'prevention'},
        {'text': 'Is this disease serious?', 'expected': 'severity'},
        {'text': 'What causes powdery mildew?', 'expected': 'causes'},
        {'text': 'My tomato has brown spots', 'expected': 'identification'},
        {'text': 'Hello', 'expected': 'greeting'},
        {'text': 'Thank you', 'expected': 'thanks'}
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: {test['text']}")
        
        intent = nlp_engine.intent_classifier.classify(test['text'])
        
        print(f"Expected: {test['expected']}, Got: {intent}")
        
        if intent == test['expected']:
            print("✓ PASS")
            passed += 1
        else:
            print("✗ FAIL")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Intent Tests: {passed} passed, {failed} failed out of {len(test_cases)}")
    print(f"{'='*80}")
    
    return passed, failed


def test_conversation_flow():
    """Test conversation flow and context management"""
    print("\n" + "="*80)
    print("TEST 5: CONVERSATION FLOW")
    print("="*80)
    
    # Reset conversation
    nlp_engine.reset_conversation()
    
    conversation = [
        {
            'message': 'My tomato plant has yellow leaves with brown spots',
            'check': lambda r: r.get('context', {}).get('disease') is not None,
            'label': 'Initial disease identification'
        },
        {
            'message': 'How do I treat it?',
            'check': lambda r: 'treatment' in r.get('response', '').lower(),
            'label': 'Follow-up treatment question'
        },
        {
            'message': 'Is this serious?',
            'check': lambda r: 'severe' in r.get('response', '').lower() or 'moderate' in r.get('response', '').lower(),
            'label': 'Severity question'
        }
    ]
    
    passed = 0
    failed = 0
    context_disease = None
    
    for i, turn in enumerate(conversation, 1):
        print(f"\n--- Turn {i}: {turn['label']} ---")
        print(f"User: {turn['message']}")
        
        try:
            result = process_message(turn['message'], 'en', context_disease=context_disease)
            
            # Update context for next turn
            if result.get('context', {}).get('disease'):
                context_disease = result['context']['disease']
            
            print(f"Response: {result.get('response', '')[:100]}...")
            print(f"Context disease: {context_disease}")
            
            if turn['check'](result):
                print("✓ PASS")
                passed += 1
            else:
                print("✗ FAIL")
                failed += 1
                
        except Exception as e:
            print(f"✗ FAIL - Exception: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Conversation Tests: {passed} passed, {failed} failed out of {len(conversation)}")
    print(f"{'='*80}")
    
    return passed, failed


def run_all_tests():
    """Run all test suites"""
    print("\n" + "█"*80)
    print("COMPREHENSIVE NLP SYSTEM TEST SUITE")
    print("█"*80)
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    p1, f1 = test_english_symptom_descriptions()
    p2, f2 = test_multilingual_descriptions()
    p3, f3 = test_entity_extraction()
    p4, f4 = test_intent_classification()
    p5, f5 = test_conversation_flow()
    
    total_passed = p1 + p2 + p3 + p4 + p5
    total_failed = f1 + f2 + f3 + f4 + f5
    total_tests = total_passed + total_failed
    
    # Summary
    print("\n" + "█"*80)
    print("FINAL SUMMARY")
    print("█"*80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ({(total_passed/total_tests*100):.1f}%)")
    print(f"Failed: {total_failed} ({(total_failed/total_tests*100):.1f}%)")
    print("█"*80 + "\n")
    
    if total_failed == 0:
        print("🎉 ALL TESTS PASSED! 🎉")
    elif total_passed / total_tests >= 0.8:
        print("✓ Good performance (>80% pass rate)")
    elif total_passed / total_tests >= 0.6:
        print("⚠ Moderate performance (60-80% pass rate) - needs improvement")
    else:
        print("✗ Poor performance (<60% pass rate) - significant improvements needed")


if __name__ == '__main__':
    run_all_tests()