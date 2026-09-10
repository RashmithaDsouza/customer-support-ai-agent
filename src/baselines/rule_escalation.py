def decide_baseline_escalation(customer_message, intent):
    """
    Decides auto vs escalate using ONLY deterministic rules.
    No LLMs used.
    """
    escalate_keywords = ["sue", "lawyer", "stolen", "hacked", "police", "fraud", "scam", "manager"]
    if any(keyword in customer_message.lower() for keyword in escalate_keywords):
        return {
            "action": "escalate",
            "reason": "Rule match: High-risk or sensitive keywords detected."
        }
        
    if intent in ["account_icloud_security", "device_hardware_problems"]:
        return {
            "action": "escalate",
            "reason": f"Rule match: Intent '{intent}' requires human verification."
        }
        
    return {
        "action": "auto",
        "reason": "Default rule: No high-risk keywords or sensitive intents detected."
    }
