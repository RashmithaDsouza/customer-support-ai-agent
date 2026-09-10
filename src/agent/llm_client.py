import os
import json
import urllib.request
import urllib.error
from pathlib import Path

# Custom minimal .env loader to avoid dotenv dependency issues
env_path = Path(".env")
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

class MissingAPIKeyError(Exception):
    pass

def call_llm(messages, max_tokens=200, temperature=0.1):
    """
    Minimal zero-dependency LLM client that supports OpenAI API format.
    Checks for mock mode and missing API keys.
    """
    mock_mode = os.getenv("MOCK_MODE", "true").lower() == "true"
    
    if mock_mode:
        all_text = " ".join(m.get("content", "").lower() for m in messages)
        
        # Classify intent mock
        if "classify the customer message" in all_text:
            if "battery" in all_text:
                return '{"intent": "battery_drain", "confidence": 0.95}'
            elif "update" in all_text:
                return '{"intent": "ios_update_problems", "confidence": 0.90}'
            else:
                return '{"intent": "app_or_system_crashes", "confidence": 0.80}'
                
        # Escalate mock
        elif "escalation routing manager" in all_text:
            return '{"action": "auto", "reason": "Standard inquiry handled by mock."}'
            
        # Generate reply mock
        else:
            return "This is a mock draft reply. Please check your settings."
            
    api_key = os.getenv("LLM_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise MissingAPIKeyError("LLM_API_KEY is missing or invalid in .env file. Enable MOCK_MODE=true or provide a key.")
        
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # We use OpenAI's chat completions endpoint (standard across many open models too)
    url = "https://api.openai.com/v1/chat/completions"
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        raise RuntimeError(f"LLM API HTTP Error {e.code}: {error_msg}")
    except Exception as e:
        raise RuntimeError(f"LLM API Request Failed: {str(e)}")
