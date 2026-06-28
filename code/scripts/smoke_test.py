import os, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from google import genai
from google.genai import types

DATASET = Path(__file__).resolve().parents[2] / "dataset"
img = DATASET / "images/sample/case_001/img_1.jpg"
data = img.read_bytes()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
model = os.environ["EVIDENCE_MODEL"]
print("model:", model, "| image bytes:", len(data))

t0 = time.time()
resp = client.models.generate_content(
    model=model,
    contents=[
        types.Part.from_bytes(data=data, mime_type="image/jpeg"),
        "Claim: rear bumper has a dent. In 2 sentences, does this image show that? Name the visible object and part.",
    ],
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        temperature=0,
    ),
)
dt = time.time() - t0
print("=== TEXT ===")
print(resp.text)
um = resp.usage_metadata
print("=== USAGE ===")
print("prompt_tokens:", um.prompt_token_count, "| output_tokens:", um.candidates_token_count,
      "| thoughts:", getattr(um, "thoughts_token_count", None), "| total:", um.total_token_count)
print(f"latency: {dt:.1f}s")
