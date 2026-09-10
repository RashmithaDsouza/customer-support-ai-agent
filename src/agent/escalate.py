import json
from src.agent.llm_client import call_llm

def decide_escalation(customer_message, intent, draft_reply):
    """
    Decides whether to auto-handle or escalate using hybrid rules + LLM reasoning.
    Returns a dictionary: {"action": "auto|escalate", "reason": "..."}
    """
    
    # Rule-based fast paths
    escalate_keywords = ["sue", "lawyer", "stolen", "hacked", "police", "fraud", "scam", "manager"]
    if any(keyword in customer_message.lower() for keyword in escalate_keywords):
        return {
            "action": "escalate",
            "reason": "Rule match: High-risk or sensitive keywords detected."
        }
        
    if intent in ["account_icloud_security", "device_hardware_problems"]:
        return {
            "action": "escalate",
            "reason": f"Rule match: Intent '{intent}' usually requires human verification or hardware support."
        }
        
    # LLM-based reasoning for ambiguous cases
    system_prompt = (
        "You are an escalation routing manager for AppleSupport.\n"
        "Analyze the customer's message and the drafted AI reply to decide if a human needs to intervene.\n\n"
        "You must output ONLY valid JSON with exactly two keys: 'action' and 'reason'.\n"
        "The 'action' MUST be either 'auto' (if the draft reply safely resolves the issue or asks for DM appropriately) "
        "or 'escalate' (if the issue is complex, unresolved, sensitive, or the customer is extremely angry).\n"
        "The 'reason' MUST be a concise string explaining why."
    )
    
    user_prompt = (
        f"Customer Message: {customer_message}\n"
        f"Intent: {intent}\n"
        f"Draft AI Reply: {draft_reply}\n\n"
        "Provide your JSON decision:"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response_text = call_llm(messages, max_tokens=150, temperature=0.1)
    response_text = response_text.replace("```json", "").replace("```", "").strip()
    
    try:
        result = json.loads(response_text)
        if result.get("action") not in ["auto", "escalate"]:
            result["action"] = "escalate" # safe fallback
        return result
    except json.JSONDecodeError:
        return {"action": "escalate", "reason": "Fallback: LLM failed to return valid JSON."}
