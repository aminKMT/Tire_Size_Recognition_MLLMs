import os
import re
import json
import time
import pandas as pd
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Setup ─────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT = "From this image, extract the tire size in the format like 265/45R18, 215/60R16, or 305/35ZR20. Only return one tire size that best matches this pattern. Do not include any explanation or extra text."

TIRE_REGEX = r'\d{3}/\d{2}[A-Z]?\d{2}'

IMAGE_DIR = os.path.join("images", "archive", "train")

# ── Load all image filenames ──────────────────────────────────
all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
print(f"Total images to label: {len(all_images)}")

results = []
failed = []

for i, filename in enumerate(all_images):
    image_path = os.path.join(IMAGE_DIR, filename)

    try:
        img = Image.open(image_path)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[PROMPT, img]
        )

        raw = response.text.strip()
        match = re.search(TIRE_REGEX, raw)
        predicted = match.group(0) if match else ""

        results.append({
            "original_filename": filename,
            "ground_truth": predicted,
            "zone": 1,
            "condition": "clean"
        })

        print(f"[{i+1}/{len(all_images)}] {filename} → {predicted}")

        # Small delay to avoid API rate limits
        time.sleep(0.5)

    except Exception as e:
        print(f"[{i+1}/{len(all_images)}] FAILED: {filename} → {e}")
        failed.append(filename)
        time.sleep(2)

# ── Save results ──────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv("all_labels.csv", index=False)
print(f"\nDone! Labeled {len(results)} images")
print(f"Failed: {len(failed)} images")
print(df.head())


# Labeling the images that gemini could not read them :
import os
import pandas as pd
from PIL import Image

IMAGE_DIR = os.path.join("images", "archive", "train")

# The 16 null images
# Using it as a test:

import pandas as pd

data = [
    {"original_filename": "image(112)_jpg.rf.FKCLE1wGtjXp8z51iUN3.jpg", "ground_truth": "195/55R15", "zone": 1, "condition": "clean"},
    {"original_filename": "image(13)_jpg.rf.qaAsuEoY3ZzuR8Qgq5Lv.jpg",  "ground_truth": "185/55R16", "zone": 1, "condition": "clean"},
    {"original_filename": "image(17)_jpg.rf.woQNfO2t4mWwYzpNeQF5.jpg",  "ground_truth": "195/55R15", "zone": 1, "condition": "clean"},
    {"original_filename": "image(201)_jpg.rf.bJRhKz1elRHLBKWBXhqb.jpg", "ground_truth": "205/45R16", "zone": 1, "condition": "clean"},
    {"original_filename": "image(434)_jpg.rf.m7K9jmMx5sMLIoMp33Zw.jpg", "ground_truth": "185/55R15", "zone": 1, "condition": "clean"},
    {"original_filename": "image(437)_jpg.rf.XWAO3YmvbpijSwmUtvx1.jpg", "ground_truth": "185/55R15", "zone": 1, "condition": "clean"},
    {"original_filename": "image(47)_jpg.rf.bXxi2MIvbshvbnS8yBqg.jpg",  "ground_truth": "185/55R16", "zone": 1, "condition": "clean"},
    {"original_filename": "image(470)_jpg.rf.zZZL9DrSGyo0TlLvoibZ.jpg", "ground_truth": "195/55R15", "zone": 1, "condition": "clean"},
    {"original_filename": "image(492)_jpg.rf.8jSQujHOwnZGC7KEMeeA.jpg", "ground_truth": "175/65R14", "zone": 1, "condition": "clean"},
    {"original_filename": "image(493)_jpg.rf.DdmEPkMeyZWFXrUDV99W.jpg", "ground_truth": "175/65R14", "zone": 1, "condition": "clean"},
    {"original_filename": "image(517)_jpg.rf.4sqqFLuq9w64jKYQQVuD.jpg", "ground_truth": "175/65R14", "zone": 1, "condition": "clean"},
    {"original_filename": "image(520)_jpg.rf.Q9HyvU8odLV5pF0lUTfz.jpg", "ground_truth": "175/65R14", "zone": 1, "condition": "clean"},
    {"original_filename": "image(542)_jpg.rf.MFyXKv1iBTJFgFiAoKRb.jpg", "ground_truth": "185/55R15", "zone": 1, "condition": "clean"},
    {"original_filename": "image(573)_jpg.rf.bxFxuYjmbns04zLCM75T.jpg", "ground_truth": "185/55R15", "zone": 1, "condition": "clean"},
]

df = pd.DataFrame(data)
df.to_csv("test_labels.csv", index=False)
print(f"Saved {len(df)} test images")
print(df)