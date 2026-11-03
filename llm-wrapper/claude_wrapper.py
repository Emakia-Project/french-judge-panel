"""
Claude LLM Wrapper - FIXED Model
"""
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Use correct Claude model (claude-opus-4-8 is the latest)
MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

SYSTEM = """You are a French harassment classifier.
Respond with ONLY "Harassment" or "Neutral". Nothing else."""

def call_claude(texts: list) -> list:
    """Classify texts using Claude API"""
    predictions = []
    
    for text in texts:
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=10,
                system=SYSTEM,
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            
            pred = message.content[0].text.strip().lower()
            if "harassment" in pred:
                predictions.append("Harassment")
            else:
                predictions.append("Neutral")
                
        except Exception as e:
            print(f"  ⚠️ Claude error: {e}")
            predictions.append("Neutral")
    
    return predictions

