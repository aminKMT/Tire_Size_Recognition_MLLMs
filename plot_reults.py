import matplotlib.pyplot as plt
import numpy as np

# ── 1. DATA ───────────────────────────────────────────────────────────────────
# All 5 models evaluated on the same 14 test images
models = [
    "Gemini 2.5\nFlash\n(zero-shot)",
    "GPT-4o\n(zero-shot)",
    "Gemma\n(zero-shot)",
    "Fine-tuned\nGemma r=8",
    "Fine-tuned\nGemma r=16",
]

exact_match   = [92.9, 7.1,  0.0,  42.9, 28.6]
char_accuracy = [92.9, 7.1, 63.5,  80.2, 57.1]

x = np.arange(len(models))   # [0, 1, 2, 3, 4] — position for each model on x-axis
width = 0.35                  # width of each bar

# ── 2. CREATE FIGURE ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

# Plot two bars side by side for each model
bars1 = ax.bar(x - width/2, exact_match,   width, label='Exact Match Accuracy',   color='steelblue',  alpha=0.85)
bars2 = ax.bar(x + width/2, char_accuracy, width, label='Character Accuracy',      color='darkorange', alpha=0.85)

# ── 3. ADD VALUE LABELS ON TOP OF EACH BAR ───────────────────────────────────
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ── 4. LABELS AND FORMATTING ──────────────────────────────────────────────────
ax.set_title('Tire Size Recognition — Model Comparison on 14 Test Images',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticset_ylimklabels(models, fontsize=10)
ax.(0, 110)     # y-axis from 0 to 110 to give room for labels
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)  # horizontal gridlines only

# ── 5. ADD A DIVIDER LINE BETWEEN ZERO-SHOT AND FINE-TUNED ───────────────────
# Visual separator to highlight zero-shot vs fine-tuned models
ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2.55, 105, 'Fine-tuned →', fontsize=9, color='gray')
ax.text(1.6,  105, '← Zero-shot', fontsize=9, color='gray')

# ── 6. SAVE AND SHOW ──────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved to model_comparison.png")