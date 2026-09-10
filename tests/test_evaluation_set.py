import unittest
import json
import pandas as pd
from pathlib import Path

class TestEvaluationSetAndIntents(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.intents_file = Path("data/processed/intent_definitions.json")
        cls.eval_file = Path("data/evaluation/evaluation_set.csv")
        
    def test_intent_definitions_has_exactly_8_intents(self):
        """Test that exactly 8 valid intents are defined."""
        self.assertTrue(self.intents_file.exists(), "intent_definitions.json does not exist")
        
        with open(self.intents_file, "r", encoding="utf-8") as f:
            intents = json.load(f)
            
        self.assertEqual(len(intents), 8, "There must be exactly 8 intents defined")
        
        expected_names = {
            "ios_update_problems",
            "battery_drain",
            "keyboard_autocorrect_glitch",
            "app_or_system_crashes",
            "device_hardware_problems",
            "apple_music_itunes",
            "account_icloud_security",
            "shipping_store_support"
        }
        
        actual_names = {i["name"] for i in intents}
        self.assertEqual(actual_names, expected_names, "Intent names do not match expected list")
        
    def test_evaluation_csv_schema_and_constraints(self):
        """Test the schema and data constraints of evaluation_set.csv if it exists."""
        if not self.eval_file.exists():
            # If not yet created, just skip or pass, since the labeling hasn't started
            self.skipTest("evaluation_set.csv not yet created. Run create_evaluation_set.py first.")
            
        df = pd.read_csv(self.eval_file)
        
        # 1. Schema check
        expected_columns = ["evaluation_id", "customer_tweet_id", "customer_text", "intent", "ideal_action", "notes"]
        self.assertEqual(list(df.columns), expected_columns, "evaluation_set.csv columns do not match exactly")
        
        # Only run data constraints if there is data
        if df.empty:
            return
            
        # 2. Valid ideal_action values
        valid_actions = {"auto", "escalate"}
        actual_actions = set(df["ideal_action"].dropna().unique())
        self.assertTrue(actual_actions.issubset(valid_actions), "Found invalid ideal_action values. Must be 'auto' or 'escalate'")
        
        # 3. Unique evaluation IDs
        self.assertEqual(len(df["evaluation_id"]), len(df["evaluation_id"].unique()), "evaluation_ids are not unique")
        
        # 4. Unique customer tweet IDs
        self.assertEqual(len(df["customer_tweet_id"]), len(df["customer_tweet_id"].unique()), "customer_tweet_ids are not unique")
        
        # 5. Non-empty customer text
        # Check for NaN/Null
        self.assertFalse(df["customer_text"].isnull().any(), "Found null values in customer_text")
        
        # Check for empty strings after stripping
        empty_texts = df[df["customer_text"].astype(str).str.strip() == ""]
        self.assertTrue(empty_texts.empty, "Found empty strings in customer_text")

if __name__ == '__main__':
    unittest.main()
