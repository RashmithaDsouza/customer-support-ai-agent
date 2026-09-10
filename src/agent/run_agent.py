import sys
from pathlib import Path
import argparse

# Add project root to sys.path so it can be run directly
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.agent.classify import classify_intent
from src.agent.retrieve import retrieve_top_k
from src.agent.generate_reply import generate_reply
from src.agent.escalate import decide_escalation

def run_pipeline(customer_message):
    print("=" * 60)
    print(f"CUSTOMER MESSAGE:\n{customer_message}")
    print("=" * 60)
    
    # 1. Classify
    print("\n[1/4] Classifying intent...")
    classification = classify_intent(customer_message)
    intent = classification.get("intent", "Unknown")
    confidence = classification.get("confidence", 0.0)
    print(f"  -> Intent: {intent} (Confidence: {confidence:.2f})")
    
    # 2. Retrieve
    print("\n[2/4] Retrieving historical context...")
    cases = retrieve_top_k(customer_message, k=3)
    for i, case in enumerate(cases, 1):
        print(f"  -> Case {i} (Sim: {case['similarity']:.2f}): {case['customer_text'][:60]}...")
        
    # 3. Generate Reply
    print("\n[3/4] Generating grounded reply...")
    draft_reply = generate_reply(customer_message, cases)
    print(f"\nDRAFT REPLY:\n{draft_reply}\n")
    
    # 4. Escalate
    print("[4/4] Deciding escalation...")
    escalation = decide_escalation(customer_message, intent, draft_reply)
    action = escalation.get("action", "escalate")
    reason = escalation.get("reason", "Unknown")
    print(f"  -> Action: {action.upper()}")
    print(f"  -> Reason: {reason}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Run the AppleSupport AI Agent Pipeline")
    parser.add_argument("message", type=str, nargs="?", help="The customer message to process")
    args = parser.parse_args()
    
    if args.message:
        run_pipeline(args.message)
    else:
        print("No message provided. Please provide a message as an argument.")
        print("Example: python src/agent/run_agent.py \"My iphone battery is draining super fast after the update!\"")
        
if __name__ == "__main__":
    main()
