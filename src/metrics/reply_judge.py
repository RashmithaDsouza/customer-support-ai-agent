import os
import json
from src.agent.llm_client import call_llm

def judge_reply(customer_message, retrieved_cases_text, generated_reply):
    """
    Uses LLM-as-judge to evaluate a generated reply.
    """
    mock_mode = os.getenv("MOCK_MODE", "true").lower() == "true"
    if mock_mode:
        return {
            "groundedness": 3,
            "correctness": 3,
            "tone": 4,
            "actionability": 3,
            "factual_consistency": 3,
            "overall_score": 3.2,
            "judge_reason": "[MOCK] API is offline. Returning default median mock scores."
        }
        
    system_prompt = (
        "You are an expert customer support evaluator.\n"
        "Score the AI's generated reply based ONLY on the provided Historical Cases evidence.\n"
        "Rate each dimension on a 1-5 scale (1=Poor, 5=Excellent).\n"
        "1. Groundedness: Is the reply rooted in the provided historical evidence?\n"
        "2. Correctness: Does the reply appropriately address the customer's issue?\n"
        "3. Tone: Is the tone helpful and professional?\n"
        "4. Actionability: Does it give clear next steps (or escalate politely)?\n"
        "5. Factual consistency: Did the AI invent any unsupported facts or policies?\n\n"
        "Return strictly a JSON object with these keys: "
        "'groundedness', 'correctness', 'tone', 'actionability', 'factual_consistency', 'overall_score', 'judge_reason'."
    )
    
    user_prompt = (
        f"Customer Message: {customer_message}\n\n"
        f"Historical Cases Evidence:\n{retrieved_cases_text}\n\n"
        f"Generated Reply: {generated_reply}\n\n"
        "JSON Score:"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response_text = call_llm(messages, max_tokens=200, temperature=0.0)
        # Clean possible markdown
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(response_text)
        return result
    except Exception as e:
        return {
            "groundedness": 1,
            "correctness": 1,
            "tone": 1,
            "actionability": 1,
            "factual_consistency": 1,
            "overall_score": 1,
            "judge_reason": f"[ERROR] LLM Judge Failed: {str(e)}"
        }
