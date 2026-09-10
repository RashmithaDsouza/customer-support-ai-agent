from src.agent.llm_client import call_llm

def generate_reply(customer_message, retrieved_cases):
    """
    Generates a draft reply for the customer grounded ONLY in the retrieved historical resolutions.
    """
    
    system_prompt = (
        "You are an AI support agent for AppleSupport.\n"
        "Your task is to draft a helpful, professional, and concise reply to the customer.\n\n"
        "CRITICAL RULES:\n"
        "1. You MUST ground your reply ONLY in the provided Historical Cases.\n"
        "2. Do NOT invent policies, refunds, troubleshooting steps, or facts unsupported by the context.\n"
        "3. If the Historical Cases do not provide a clear resolution or troubleshooting step, "
        "your response should be a polite generic escalation (e.g., 'We'd like to look into this with you. Please DM us.').\n"
        "4. Adopt the tone of AppleSupport (helpful, empathetic, professional).\n"
    )
    
    context_text = "Historical Cases:\n\n"
    for i, case in enumerate(retrieved_cases, 1):
        context_text += f"Case {i}:\n"
        context_text += f"Customer: {case['customer_text']}\n"
        context_text += f"AppleSupport Reply: {case['reply_text']}\n\n"
        
    user_prompt = (
        f"{context_text}\n"
        f"Current Customer Message: {customer_message}\n\n"
        "Draft the reply now. Output ONLY the reply text, nothing else."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    reply = call_llm(messages, max_tokens=200, temperature=0.3)
    return reply.strip()
