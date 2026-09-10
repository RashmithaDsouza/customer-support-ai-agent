import unittest
import numpy as np
from src.baselines.majority_baseline import MajorityBaseline
from src.baselines.tfidf_baseline import TfidfBaseline
from src.baselines.canned_reply import generate_canned_reply
from src.baselines.rule_escalation import decide_baseline_escalation

class TestBaselines(unittest.TestCase):
    
    def test_majority_baseline(self):
        y_train = ["apple_music_itunes", "battery_drain", "battery_drain"]
        X_test = ["msg1", "msg2"]
        
        model = MajorityBaseline()
        model.fit(y_train)
        
        preds = model.predict(X_test)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], "battery_drain")
        self.assertEqual(preds[1], "battery_drain")
        
    def test_tfidf_baseline(self):
        X_train = ["my battery is dying", "update is stuck", "my battery dies fast"]
        y_train = ["battery_drain", "ios_update_problems", "battery_drain"]
        X_test = ["battery issue"]
        
        model = TfidfBaseline()
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        self.assertEqual(len(preds), 1)
        self.assertIsInstance(preds[0], str)
        
    def test_canned_reply(self):
        reply = generate_canned_reply("battery_drain")
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)
        self.assertIn("battery", reply.lower())
        
        reply2 = generate_canned_reply("unknown_intent_XYZ")
        self.assertIsInstance(reply2, str)
        self.assertTrue(len(reply2) > 0)
        
    def test_rule_escalation(self):
        # High risk
        res1 = decide_baseline_escalation("I will sue you", "app_or_system_crashes")
        self.assertEqual(res1["action"], "escalate")
        
        # High risk intent
        res2 = decide_baseline_escalation("how to login", "account_icloud_security")
        self.assertEqual(res2["action"], "escalate")
        
        # Low risk
        res3 = decide_baseline_escalation("how do I update my phone?", "ios_update_problems")
        self.assertEqual(res3["action"], "auto")

if __name__ == "__main__":
    unittest.main()
