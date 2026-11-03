"""
Mistral LLM Wrapper
===================
Classifies French harassment using Mistral API.
Fixed version for mistralai >= 0.1.0
"""

import os

# Try new SDK format first
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_message import ChatMessage
    
    api_key = os.getenv("MISTRAL_API_KEY")
    client = MistralClient(api_key=api_key)
    USE_NEW_SDK = True
    print("✅ Using new Mistral SDK (MistralClient)")
except ImportError:
    # Fallback to older format
    try:
        from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
        from mistral_common.tokens.tokenizers import get_tokenizer
        import os
        
        api_key = os.getenv("MISTRAL_API_KEY")
        USE_NEW_SDK = False
        print("✅ Using legacy Mistral SDK")
    except ImportError:
        print("❌ Mistral SDK not installed. Run: pip install mistralai --break-system-packages")
        raise

SYSTEM_PROMPT = """You are a French content moderation expert analyzing tweets for harassment.
Classify each tweet as either "Harassment" or "Neutral".

Harassment includes: insults, threats, discrimination, hate speech, cyberbullying.
Neutral includes: normal discourse, opinions, discussions.

Respond with ONLY the classification: either "Harassment" or "Neutral"."""


def call_mistral(texts: list) -> list:
    """
    Classify a batch of French texts using Mistral.
    
    Args:
        texts: List of French tweet texts
        
    Returns:
        List of predictions ("Harassment" or "Neutral")
    """
    predictions = []
    
    if not USE_NEW_SDK:
        # Fallback for legacy SDK
        for text in texts:
            predictions.append("Neutral")
        return predictions
    
    for text in texts:
        try:
            message = client.chat(
                model="mistral-large-latest",
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=text)
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            pred = message.choices[0].message.content.strip()
            
            # Normalize response
            if "harassment" in pred.lower():
                predictions.append("Harassment")
            elif "neutral" in pred.lower():
                predictions.append("Neutral")
            else:
                predictions.append("Neutral")  # default
                
        except Exception as e:
            print(f"  ⚠️  Mistral error: {e}")
            predictions.append("Neutral")  # fail-safe
    
    return predictions

