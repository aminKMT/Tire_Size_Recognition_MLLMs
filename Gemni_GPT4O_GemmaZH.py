import re, os, time, base64, pandas as pd
from PIL import Image
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
import Levenshtein, ollama

# ── 1. LOAD API KEYS FROM .env FILE ──────────────────────────────────────────
# load_dotenv() reads the .env file and sets environment variables
# os.getenv() retrieves the key by name
load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 2. SHARED SETTINGS ────────────────────────────────────────────────────────
# Same prompt sent to all three models — ensures fair comparison
PROMPT = "From this image, extract the tire size in the format like 265/45R18, 215/60R16, or 305/35ZR20. Only return one tire size that best matches this pattern. Do not include any explanation or extra text."

# Regex pattern to extract tire size from model output.
# Models sometimes return extra text like "The tire size is 265/45R18"
# Regex pulls just the size pattern out of the full response.
TIRE_REGEX = r'\d{3}/\d{2}[A-Z]?\d{2}'

# ── 3. FILE PATHS ─────────────────────────────────────────────────────────────
# Folder where the 14 test images are stored
image_dir = r"C:\Users\akeramati\OneDrive - Widener University\Desktop\MLLM_Tier_Image\images\archive\train"

# CSV file with 14 rows: original_filename and ground_truth columns
test_df = pd.read_csv(r"C:\Users\akeramati\OneDrive - Widener University\Desktop\MLLM_Tier_Image\test_labels.csv")

# ── 4. METRIC FUNCTIONS ───────────────────────────────────────────────────────
def character_accuracy(pred, gt):
    """
    Calculates how many characters are correct using Levenshtein distance.
    Example: pred="185/55R16", gt="185/55R15"
    distance=1, len(gt)=9, accuracy = (1 - 1/9) * 100 = 88.9%
    """
    if not gt: return 0.0
    dist = Levenshtein.distance(pred, gt)
    return round(max(0.0, 1.0 - dist / len(gt)) * 100, 2)

# ── 5. HELPER: CONVERT IMAGE TO BASE64 STRING ─────────────────────────────────
def image_to_b64(image_path):
    """
    Reads image file as raw bytes and converts to Base64 text string.
    GPT and Gemma APIs cannot accept file paths — they need the image
    embedded as a text string inside the API request.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ── 6. PREDICTION FUNCTIONS (one per model) ───────────────────────────────────
def predict_gemini(image_path):
    """
    Sends image + prompt to Gemini 2.5 Flash via Google Generative AI API.
    Gemini accepts PIL Image objects directly — no need for Base64.
    """
    img = Image.open(image_path)
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[PROMPT, img]   # Gemini takes a list of [text, image]
    )
    raw = response.text.strip()
    match = re.search(TIRE_REGEX, raw)
    return match.group(0) if match else ""  # return tire size or empty string

def predict_gpt(image_path):
    """
    Sends image + prompt to GPT-4o via OpenAI API.
    GPT requires image as a Base64-encoded data URL inside the message content.
    """
    b64 = image_to_b64(image_path)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT}
        ]}],
        max_tokens=20   # tire size is short — no need for long output
    )
    raw = response.choices[0].message.content.strip()
    match = re.search(TIRE_REGEX, raw)
    return match.group(0) if match else ""

def predict_gemma(image_path):
    """
    Sends image + prompt to Gemma 3 4B running locally via Ollama.
    Ollama acts as a local server — no API key needed, runs on your GPU/CPU.
    Image must be Base64 encoded and passed in the 'images' field.
    This is zero-shot: the base Gemma model with no fine-tuning.
    """
    b64 = image_to_b64(image_path)
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [b64]   # Ollama decodes Base64 back to image internally
        }]
    )
    raw = response['message']['content'].strip()
    match = re.search(TIRE_REGEX, raw)
    return match.group(0) if match else ""

# ── 7. MAIN EVALUATION LOOP ───────────────────────────────────────────────────
# Loop through all 14 test images, send to each model, collect results
results = []

print("Evaluating Gemini, GPT-4o and Gemma on 14 test images...")
print("="*60)

for _, row in test_df.iterrows():
    # Build full image path by combining folder + filename from CSV
    image_path = f"{image_dir}\\{row['original_filename']}"
    ground_truth = row['ground_truth']  # correct tire size label

    # Run Gemini — sleep 1 second after to avoid hitting API rate limits
    try:
        gemini_pred = predict_gemini(image_path)
        time.sleep(1)
    except Exception as e:
        print(f"Gemini error: {e}")
        gemini_pred = ""

    # Run GPT-4o — sleep 1 second after to avoid hitting API rate limits
    try:
        gpt_pred = predict_gpt(image_path)
        time.sleep(1)
    except Exception as e:
        print(f"GPT error: {e}")
        gpt_pred = ""

    # Run Gemma locally via Ollama — no rate limit needed (runs on your machine)
    try:
        gemma_pred = predict_gemma(image_path)
    except Exception as e:
        print(f"Gemma error: {e}")
        gemma_pred = ""

    # Print result for this image across all three models
    print(f"GT: {ground_truth} | "
          f"Gemini: {gemini_pred} {'✅' if gemini_pred==ground_truth else '❌'} | "
          f"GPT: {gpt_pred} {'✅' if gpt_pred==ground_truth else '❌'} | "
          f"Gemma: {gemma_pred} {'✅' if gemma_pred==ground_truth else '❌'}")

    # Store all predictions and metrics for this image
    results.append({
        "ground_truth":    ground_truth,
        "gemini_pred":     gemini_pred,
        "gemini_correct":  gemini_pred == ground_truth,
        "gemini_char_acc": character_accuracy(gemini_pred, ground_truth),
        "gpt_pred":        gpt_pred,
        "gpt_correct":     gpt_pred == ground_truth,
        "gpt_char_acc":    character_accuracy(gpt_pred, ground_truth),
        "gemma_pred":      gemma_pred,
        "gemma_correct":   gemma_pred == ground_truth,
        "gemma_char_acc":  character_accuracy(gemma_pred, ground_truth),
    })

# ── 8. CALCULATE AND PRINT FINAL METRICS ─────────────────────────────────────
df = pd.DataFrame(results)
print("="*60)
print(f"Gemini         — Exact Match: {df['gemini_correct'].mean()*100:.1f}%  |  Char Accuracy: {df['gemini_char_acc'].mean():.1f}%")
print(f"GPT-4o         — Exact Match: {df['gpt_correct'].mean()*100:.1f}%  |  Char Accuracy: {df['gpt_char_acc'].mean():.1f}%")
print(f"Gemma (0-shot) — Exact Match: {df['gemma_correct'].mean()*100:.1f}%  |  Char Accuracy: {df['gemma_char_acc'].mean():.1f}%")

# ── 9. SAVE RESULTS TO CSV ────────────────────────────────────────────────────
# Save all predictions so we can build the final comparison table later
df.to_csv("eval_all_zeroshot_14.csv", index=False)
print("Saved to eval_all_zeroshot_14.csv")