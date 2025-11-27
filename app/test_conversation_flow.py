"""
Test script to demonstrate the fixed conversation flow
"""

def simulate_conversation():
    """Simulate the conversation flow you described"""
    
    print("=" * 80)
    print("SIMULATING CONVERSATION FLOW")
    print("=" * 80)
    
    # Import the processor
    try:
        from multilingual_nlp_processor import multilingual_processor
        from custom_nlp_system import nlp_engine
    except ImportError:
        print("Error: Could not import modules. Make sure all files are in place.")
        return
    
    # Reset to start fresh
    multilingual_processor.reset_conversation()
    
    # Turn 1: User greets
    print("\n--- Turn 1 ---")
    print("User: Hello")
    
    result1 = multilingual_processor.process_user_message("Hello", "en", None)
    print(f"Bot: {result1['response']}")
    print(f"Context disease: {result1['context'].get('disease')}")
    
    # Turn 2: Image upload (simulated)
    print("\n--- Turn 2 ---")
    print("User: [Uploads image]")
    print("System detects: tomato_spider_mites_two_spotted_spider_mite (45.2%)")
    
    result2 = multilingual_processor.process_image_result(
        'tomato_spider_mites_two_spotted_spider_mite',
        45.2,
        'en'
    )
    print(f"Bot: {result2['response']}")
    print(f"Context disease: {result2.get('disease')}")
    
    # Turn 3: User asks "What are the symptoms?" (should answer about spider mites)
    print("\n--- Turn 3 ---")
    print("User: What are the symptoms?")
    
    result3 = multilingual_processor.process_user_message(
        "What are the symptoms?",
        "en",
        None
    )
    print(f"Bot: {result3['response']}")
    print(f"Context disease: {result3['context'].get('disease')}")
    
    # Turn 4: User describes NEW symptoms (should detect NEW disease)
    print("\n--- Turn 4 ---")
    print("User: Leaves have white powdery patches")
    
    result4 = multilingual_processor.process_user_message(
        "Leaves have white powdery patches",
        "en",
        None
    )
    print(f"Bot: {result4['response']}")
    print(f"Context disease: {result4['context'].get('disease')}")
    
    # Turn 5: Follow-up about the NEW disease
    print("\n--- Turn 5 ---")
    print("User: How to treat it?")
    
    result5 = multilingual_processor.process_user_message(
        "How to treat it?",
        "en",
        None
    )
    print(f"Bot: {result5['response']}")
    print(f"Context disease: {result5['context'].get('disease')}")
    
    print("\n" + "=" * 80)
    print("EXPECTED BEHAVIOR:")
    print("=" * 80)
    print("Turn 1: Generic greeting response")
    print("Turn 2: Spider mites detection from image")
    print("Turn 3: Symptoms of spider mites (from context)")
    print("Turn 4: NEW disease detected (powdery mildew) - context changes")
    print("Turn 5: Treatment for powdery mildew (new context)")
    print("=" * 80)


def test_symptom_detection():
    """Test if symptoms are properly detected"""
    
    print("\n\n" + "=" * 80)
    print("TESTING SYMPTOM DETECTION")
    print("=" * 80)
    
    try:
        from custom_nlp_system import nlp_engine
    except ImportError:
        print("Error: Could not import NLP engine")
        return
    
    test_inputs = [
        "Leaves have white powdery patches",
        "My tomato has yellow leaves with brown spots",
        "Plant is wilting and has black rot",
        "Orange rusty pustules on corn leaves",
        "White mold on cherry leaves"
    ]
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {text}")
        
        # Extract entities
        entities = nlp_engine.entity_extractor.extract(text)
        print(f"Detected plant: {entities['plant']}")
        print(f"Detected symptoms: {entities['symptoms']}")
        
        # Match to diseases
        if entities['symptoms']:
            matches = nlp_engine.symptom_matcher.match(
                entities['plant'],
                entities['symptoms']
            )
            
            if matches:
                print(f"Top matches:")
                for disease, score, confidence in matches[:3]:
                    print(f"  - {disease} (score: {score}, confidence: {confidence:.1f}%)")
            else:
                print("No disease matches found")
        else:
            print("No symptoms detected")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Run the simulation
    simulate_conversation()
    
    # Test symptom detection
    test_symptom_detection()