"""
Gemini Vision Wrapper for Toxicity Classification

Uses Google's Gemini multimodal model to classify tweets using BOTH:
  - The tweet text
  - The tweet image (if a media_url is available)

This catches harassment that is image-based (memes, screenshots, visual slurs)
that text-only models would miss entirely.

Returns "Neutral" or "Harassment" to match the rest of the pipeline.

Weight in voting formula:  votes = 2×(CoreML) + Σ LLMᵢ + w_c × GeminiVision ≥ θ
Recommended w_c = 2  (same weight as CoreML, since it sees modality others miss)
"""

import os
import time
import base64
import requests
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

# Use the vision-capable model
MODEL_NAME = "gemini-2.0-flash"

# Prompt sent alongside the image (and/or text)
VISION_SYSTEM_PROMPT = """You are a content moderation classifier. 
Analyze the provided content — which may include text, an image, or both — and classify it as:
- "Harassment" if it contains hate speech, threats, harassment, bullying, discrimination, 
  slurs, violent imagery, or any toxic content (including visual content)
- "Neutral" if it is safe, non-harmful content

Respond with ONLY one word: "Harassment" or "Neutral"."""


def _fetch_image(url: str, timeout: int = 10):
    """
    Download an image from a URL and return a PIL Image.
    Returns None if the download fails or URL is empty.
    """
    if not url or not url.strip():
        return None
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"      ⚠️ Could not fetch image from {url[:60]}...: {e}")
        return None


def _classify_single(text: str, image_url: str = None, retries: int = 4, initial_delay: float = 2.0) -> str:
    """
    Classify a single tweet using Gemini Vision.

    Args:
        text:       Tweet text
        image_url:  URL of the tweet's media attachment (optional)
        retries:    Number of retry attempts on failure
        initial_delay: Base delay for exponential backoff

    Returns:
        "Harassment" or "Neutral"
    """
    model = genai.GenerativeModel(MODEL_NAME)

    # Build the message parts
    parts = [VISION_SYSTEM_PROMPT]

    # Attach image if available
    img = _fetch_image(image_url) if image_url else None
    if img:
        parts.append(img)
        parts.append(f"\nTweet text: {text.strip()[:800]}")
    else:
        # Text-only fallback (still uses the vision model for consistency)
        parts.append(f"\nTweet text: {text.strip()[:800]}")

    for attempt in range(retries):
        try:
            response = model.generate_content(parts)
            raw = response.text.strip().strip('"').lower()

            if "neutral" in raw:
                return "Neutral"
            elif "harassment" in raw:
                return "Harassment"
            else:
                # Ambiguous — fail safe to Harassment
                print(f"      ⚠️ Ambiguous Gemini Vision response: '{raw}', defaulting to Harassment")
                return "Harassment"

        except Exception as e:
            error_str = str(e).lower()
            delay = min(initial_delay * (2 ** attempt), 60)

            if any(term in error_str for term in ["rate", "quota", "429", "resource exhausted"]):
                print(f"      ⚠️ Gemini Vision rate limit (attempt {attempt+1}/{retries}), waiting {delay:.0f}s...")
                time.sleep(delay)
            elif any(term in error_str for term in ["overloaded", "503", "unavailable"]):
                print(f"      ⚠️ Gemini Vision overloaded (attempt {attempt+1}/{retries}), waiting {delay:.0f}s...")
                time.sleep(delay)
            elif attempt < retries - 1:
                print(f"      ⚠️ Gemini Vision error (attempt {attempt+1}/{retries}): {e}")
                time.sleep(delay)
            else:
                print(f"      ❌ Gemini Vision failed after {retries} attempts: {e}")
                return "Harassment"  # Fail safe

    return "Harassment"


def call_gemini_vision(texts: list, image_urls: list = None) -> list:
    """
    Classify a batch of tweets with Gemini Vision.

    Args:
        texts:       List of tweet text strings
        image_urls:  List of image URL strings (same length as texts).
                     Pass None or an empty list if no images are available.
                     Individual entries can be None/empty for text-only tweets.

    Returns:
        List of predictions: "Neutral" or "Harassment"
    """
    if image_urls is None:
        image_urls = [None] * len(texts)

    # Pad image_urls if shorter than texts (safety)
    if len(image_urls) < len(texts):
        image_urls = image_urls + [None] * (len(texts) - len(image_urls))

    predictions = []

    for i, (text, img_url) in enumerate(zip(texts, image_urls)):
        if not text or not text.strip():
            predictions.append("Harassment")  # Default for empty text
            continue

        prediction = _classify_single(text, img_url)
        predictions.append(prediction)

        # Respect Gemini rate limits between requests
        if i < len(texts) - 1:
            time.sleep(1.2)

    return predictions


# ── Quick self-test ────────────────────────────────────────────────────────────
def _test():
    test_cases = [
        ("I love this beautiful sunny day!", None),
        ("You are worthless and should disappear.", None),
        ("Just had the best coffee ever ☕", None),
    ]
    texts = [t for t, _ in test_cases]
    urls  = [u for _, u in test_cases]

    print("Testing Gemini Vision wrapper (text-only mode)...")
    results = call_gemini_vision(texts, urls)
    for (text, _), result in zip(test_cases, results):
        print(f"  [{result}]  {text[:60]}")


if __name__ == "__main__":
    _test()
