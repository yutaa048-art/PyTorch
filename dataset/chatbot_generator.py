import random

def generate_chatbot_conversation(i: int) -> str:
    user_prompts = [
        "Jelaskan apa itu AI.",
        "Buatkan kode python.",
        "Apa cuaca hari ini?",
        "Tolong bantu saya memperbaiki bug.",
        "Siapa namamu?",
        "Jelaskan teori relativitas."
    ]
    
    bot_responses = [
        "AI adalah kecerdasan buatan.",
        "Berikut kodenya:\n```python\nprint('Hello')\n```",
        "Saya tidak bisa mengecek cuaca.",
        "Silakan berikan detail bug yang Anda alami.",
        "Saya adalah SentinelLM.",
        "Teori relativitas dikemukakan oleh Albert Einstein."
    ]
    
    user = random.choice(user_prompts)
    bot = random.choice(bot_responses)
    
    return f"""<|system|>
Anda adalah AI asisten bernama SentinelLM.
<|user|>
{user}
<|assistant|>
{bot}
"""

def generate_chatbot_dataset(count: int = 5000) -> list[str]:
    return [generate_chatbot_conversation(i) for i in range(count)]
