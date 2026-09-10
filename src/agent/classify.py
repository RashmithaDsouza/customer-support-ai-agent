import json
from pathlib import Path
from src.agent.llm_client import call_llm

def get_intent_definitions():
    intents_json = Path("data/processed/intent_definitions.json")
    if not intents_json.exists():
        raise FileNotFoundError(f"{intents_json} not found.")
    
    with open(intents_json, "r", encoding="utf-8") as f:
        return json.load(f)

def classify_intent(customer_message):
    """
    Uses the LLM to classify the customer message into one of the 8 predefined intents.
    Returns a dictionary: {"intent": "...", "confidence": 0.95}
    """
    intents = get_intent_definitions()
    
    # Construct prompt
    system_prompt = (
        "You are an expert customer support intent classifier for AppleSupport.\n"
        "Your task is to classify the customer message into EXACTLY ONE of the following intents.\n\n"
    )
    
    for i, intent in enumerate(intents, 1):
        system_prompt += f"Intent: {intent['name']}\nDescription: {intent['description']}\n\n"
        
    system_prompt += (
        "Respond ONLY with a valid JSON object containing exactly two keys: 'intent' and 'confidence'.\n"
        "The 'intent' value MUST be one of the exact intent names listed above.\n"
        "The 'confidence' value MUST be a float between 0.0 and 1.0 indicating your confidence.\n"
        "Do not include markdown blocks or any other text outside the JSON."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Customer Message: {customer_message}"}
    ]
    
    response_text = call_llm(messages, max_tokens=150, temperature=0.0)
    
    # Clean possible markdown from response
    response_text = response_text.replace("```json", "").replace("```", "").strip()
    
    try:
        result = json.loads(response_text)
        # Ensure fallback if invalid intent is returned
        valid_names = [i["name"] for i in intents]
        if result.get("intent") not in valid_names:
            result["intent"] = "device_hardware_problems" # safe fallback
            result["confidence"] = 0.0
        return result
    except json.JSONDecodeError:
        return {"intent": "device_hardware_problems", "confidence": 0.0}
