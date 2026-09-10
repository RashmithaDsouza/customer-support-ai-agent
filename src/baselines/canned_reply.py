def generate_canned_reply(intent):
    """
    Returns a static canned reply depending on the intent.
    No LLMs, no retrieval.
    """
    templates = {
        "ios_update_problems": "We can help with your iOS update. Please ensure you are connected to Wi-Fi and have enough storage.",
        "battery_drain": "We'd like to help you optimize your battery life. Please check your Battery settings for more details.",
        "keyboard_autocorrect_glitch": "We are aware of the keyboard issue. Please try resetting your keyboard dictionary in Settings > General > Reset.",
        "app_or_system_crashes": "We'd like to look into this performance issue with you. Please restart your device and let us know if it persists.",
        "device_hardware_problems": "Physical damage or hardware issues often require an inspection. Please visit an Apple Store or contact Support directly.",
        "apple_music_itunes": "We can help with your Apple Music or iTunes account. Please try signing out and signing back in.",
        "account_icloud_security": "Security is important to us. Please visit iforgot.apple.com to manage your Apple ID or reset your password.",
        "shipping_store_support": "We can help check on your order or appointment. Please DM us your details."
    }
    
    return templates.get(intent, "We are here to help. Please DM us with more details so we can assist you.")
