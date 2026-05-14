# Tire Size Recognition using Multimodal LLMs

A comparative study of Multimodal Large Language Models (MLLMs) for tire size recognition from real-world images. This project replicates and extends the methodology of the paper *"A Comparative Study of Multimodal Large Language Models for Tire Size Recognition from Real-World Images"*.

---

## Project Overview

Tire size information (e.g., `185/55R15`) is printed on the sidewall of every tire. Automatically extracting this information from images is useful for automotive inspection, inventory management, and e-commerce applications.

This project evaluates three models in zero-shot settings and fine-tunes an open-source model using QLoRA to compare performance before and after training.

---

## Pipeline

```
Raw Images (494)
      ↓
Pseudo-Labeling with Gemini API         ← auto_label.py
      ↓
Train/Validation/Test Split
382 train | 96 val | 14 test
      ↓
Fine-Tuning with QLoRA                  ← Tire_Fine_Tuning.ipynb
(Gemma 3 4B, r=8 and r=16)
      ↓
Evaluation on 14 Test Images            ← Gemini_GPT4O_GemmaZH.py
      ↓
Results & Visualization                 ← plot_results.py
```

---

## Models Compared

| Model | Type | Parameters |
|---|---|---|
| Gemini 2.5 Flash | Zero-shot, Google API | ~100B+ |
| GPT-4o | Zero-shot, OpenAI API | ~200B+ |
| Gemma 3 4B (zero-shot) | Zero-shot, Local (Ollama) | 4B |
| Gemma 3 4B (fine-tuned r=8) | QLoRA fine-tuned, Local | 4B |
| Gemma 3 4B (fine-tuned r=16) | QLoRA fine-tuned, Local | 4B |

---

## Results

All models evaluated on the same **14 held-out test images**.

| Model | Exact Match | Char Accuracy |
|---|---|---|
| Gemini 2.5 Flash (zero-shot) | **92.9%** | **92.9%** |
| GPT-4o (zero-shot) | 7.1% | 7.1% |
| Gemma 3 4B (zero-shot) | 0.0% | 63.5% |
| Fine-tuned Gemma r=8 ✅ | 42.9% | 80.2% |
| Fine-tuned Gemma r=16 | 28.6% | 57.1% |

![Model Comparison](model_comparison.png)

---

## Key Findings

- **Fine-tuning works**: Zero-shot Gemma scored 0% exact match. After fine-tuning on just 382 images, it jumped to **42.9%** — a 6x improvement
- **Rank matters**: r=8 outperformed r=16, showing that higher LoRA rank causes overfitting on small datasets
- **API models dominate**: Gemini 2.5 Flash achieved 92.9% with zero fine-tuning, showing the power of large-scale vision pre-training
- **GPT-4o underperformed**: Mostly returned empty predictions on these tire sidewall images

---

## Fine-Tuning Details

- **Base model**: `google/gemma-3-4b-it`
- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit (load_in_4bit=True)
- **LoRA ranks tested**: r=8, r=16
- **Trainable parameters (r=8)**: 19,248,896 / 2,958,801,008 (0.65%)
- **Training data**: 382 pseudo-labeled images
- **Validation data**: 96 images
- **Hardware**: Google Colab Pro (A100 80GB)
- **Framework**: Unsloth + TRL SFTTrainer

---

## Metrics

- **Exact Match Accuracy**: Prediction must match ground truth character by character
- **Character Accuracy**: Based on Levenshtein edit distance — measures how many characters are correct

```python
def character_accuracy(pred, gt):
    dist = Levenshtein.distance(pred, gt)
    return max(0.0, 1.0 - dist / len(gt)) * 100
```

---

## Project Structure

```
├── auto_label.py              # Pseudo-labels 494 images using Gemini API
├── Gemini_GPT4O_GemmaZH.py   # Zero-shot evaluation of all 3 models
├── Tire_Fine_Tuning.ipynb     # QLoRA fine-tuning pipeline (Google Colab)
├── plot_results.py            # Bar chart visualization
├── test_labels.csv            # 14 manually labeled test images
├── annotations.csv            # 8 manually labeled images (initial set)
├── model_comparison.png       # Final results chart
└── .env                       # API keys (not committed)
```

---

## Setup

### 1. Install dependencies
```bash
pip install google-genai openai ollama python-Levenshtein pillow pandas python-dotenv
```

### 2. Add API keys to `.env`
```
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. Run zero-shot evaluation
```bash
python Gemini_GPT4O_GemmaZH.py
```

### 4. Fine-tuning
Open `Tire_Fine_Tuning.ipynb` in Google Colab with A100 GPU

---

## Dataset

- **Source**: Roboflow — Tyre Sidewall Text Detection dataset
- **Total images**: 494
- **Pseudo-labeled**: 478 (using Gemini 2.5 Flash)
- **Manually labeled test set**: 14 images
- **Image content**: Real-world tire sidewall photos

---

## Author

Amin Keramati
