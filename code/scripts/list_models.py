"""Probe the configured Gemini account for available models + capabilities.

Run once to confirm the exact model id to put in EVIDENCE_MODEL (e.g. the
correct id for "3.5 Flash") and whether it supports thinking.

    python code/scripts/list_models.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("No GEMINI_API_KEY / GOOGLE_API_KEY in environment.")
        return 1
    try:
        from google import genai
    except ImportError:
        print("google-genai not installed. Run: pip install google-genai")
        return 1

    client = genai.Client(api_key=key)
    print(f"{'MODEL':45} {'INPUT_TOK':>10} {'OUTPUT_TOK':>11}  METHODS")
    print("-" * 100)
    rows = []
    for m in client.models.list():
        methods = ",".join(getattr(m, "supported_actions", []) or [])
        if "generateContent" not in methods:
            continue
        name = m.name.replace("models/", "")
        rows.append(name)
        print(f"{name:45} {str(getattr(m,'input_token_limit','?')):>10} "
              f"{str(getattr(m,'output_token_limit','?')):>11}  {methods}")

    print("\nFlash candidates:")
    for n in rows:
        if "flash" in n.lower():
            print("  -", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
