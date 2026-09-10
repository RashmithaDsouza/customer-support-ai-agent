import json
from pathlib import Path

def main():
    intents = [
        {
            "name": "ios_update_problems",
            "description": "Issues related to downloading, installing, or post-installation bugs of iOS updates.",
            "examples": [
                "Since updating to iOS 11 my phone is extremely slow.",
                "I tried to update my phone but it won't even start back up."
            ]
        },
        {
            "name": "battery_drain",
            "description": "Complaints about battery life decreasing rapidly or devices dying unexpectedly.",
            "examples": [
                "My battery life has decreased sharply, I need to charge it twice daily!",
                "iPhone 5S keeps dying on 40%+ while on Snapchat."
            ]
        },
        {
            "name": "keyboard_autocorrect_glitch",
            "description": "Issues with the iOS keyboard, autocorrect errors, or the 'I' to 'A ?' or 'I.' glitch.",
            "examples": [
                "everytime I type an 'I' it corrects it to this and I can't stop it??",
                "Please fix the I glitch it showing up weird."
            ]
        },
        {
            "name": "app_or_system_crashes",
            "description": "Apps freezing, lagging, crashing, or the whole device becoming unresponsive.",
            "examples": [
                "My phone freezes when I play games even after the update.",
                "whenever I try to see the photos that I've just taken my iPhone shows me a black screen."
            ]
        },
        {
            "name": "device_hardware_problems",
            "description": "Physical damage, screen cracking, hardware malfunctions, or unexpected noises.",
            "examples": [
                "My iPhone 7 screen has accidentally cracked. What should I do?",
                "The mac isnt overheat but the fan is on max and running slow."
            ]
        },
        {
            "name": "apple_music_itunes",
            "description": "Problems with Apple Music subscriptions, iTunes purchases, skipped songs, or missing libraries.",
            "examples": [
                "Unable to use Apple Music after purchasing Apple Music Plan.",
                "WHY DOES AN ALBUM I BOUGHT ON ITUNES KEEP SKIPPING?"
            ]
        },
        {
            "name": "account_icloud_security",
            "description": "Issues with Apple ID, iCloud storage, password resets, hacking, or scam emails.",
            "examples": [
                "I forgot my itunes password and you email to say you will call me in 14 days?",
                "I got this email and it takes you to this site. This is a scam right?"
            ]
        },
        {
            "name": "shipping_store_support",
            "description": "Problems with product delivery, store appointments, or customer service interactions.",
            "examples": [
                "over an hour on hold, failing to speak to someone at Southampton store!",
                "not happy that on ordering it tells me signature required and then email tells me no signature"
            ]
        }
    ]

    output_path = Path("data/processed/intent_definitions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(intents, f, indent=4)
        
    print(f"Successfully generated {len(intents)} intent definitions at {output_path.absolute()}")

if __name__ == "__main__":
    main()
