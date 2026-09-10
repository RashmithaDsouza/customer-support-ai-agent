import unittest
import os
from unittest.mock import patch
from src.agent.llm_client import call_llm, MissingAPIKeyError
from src.agent.classify import classify_intent
from src.agent.retrieve import retrieve_top_k
from src.agent.generate_reply import generate_reply
from src.agent.escalate import decide_escalation

class TestAgentPipeline(unittest.TestCase):
    
    def setUp(self):
        # Force mock mode for tests
        os.environ["MOCK_MODE"] = "true"
        os.environ["LLM_API_KEY"] = "your_api_key_here"
        
    def test_missing_api_key_handling(self):
        os.environ["MOCK_MODE"] = "false"
        os.environ["LLM_API_KEY"] = "your_api_key_here"
        with self.assertRaises(MissingAPIKeyError):
            call_llm([{"role": "user", "content": "hello"}])
            
    def test_classify_intent_mock(self):
        result = classify_intent("my battery is draining")
        self.assertIn("intent", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["intent"], "battery_drain")
        
    def test_retrieve_top_k(self):
        # This will trigger embedding generation if not cached
        results = retrieve_top_k("my phone is frozen", k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("customer_text", results[0])
        self.assertIn("reply_text", results[0])
        self.assertIn("similarity", results[0])
        
    def test_generate_reply_mock(self):
        cases = [{"customer_text": "frozen", "reply_text": "try restart"}]
        reply = generate_reply("my phone is frozen", cases)
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)
        
    def test_escalate_rules(self):
        # Test rule match
        res = decide_escalation("I will sue you if you dont fix this", "battery_drain", "Hi")
        self.assertEqual(res["action"], "escalate")
        self.assertIn("Rule match", res["reason"])
        
        # Test intent rule match
        res2 = decide_escalation("Can't login", "account_icloud_security", "Hi")
        self.assertEqual(res2["action"], "escalate")
        
    def test_escalate_mock(self):
        # Test mock LLM fallback
        res = decide_escalation("How do I update?", "ios_update_problems", "Go to settings")
        self.assertEqual(res["action"], "auto")

if __name__ == "__main__":
    unittest.main()
