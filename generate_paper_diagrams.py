import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

out_dir = "ml_engine/research"
os.makedirs(out_dir, exist_ok=True)

def draw_box(ax, x, y, width, height, text):
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='#333333', facecolor='#f9f9f9')
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=11, family='sans-serif', color='#111111')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='#333333', shrink=0.05, width=1.5, headwidth=8))

# ─────────────────────────────────────────────
# Diagram 1: bm25_fusion_diagram.png
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
draw_box(ax, 0.3, 0.8, 0.4, 0.1, "User Query")
draw_arrow(ax, 0.5, 0.8, 0.3, 0.6)
draw_arrow(ax, 0.5, 0.8, 0.7, 0.6)
draw_box(ax, 0.05, 0.5, 0.4, 0.1, "Dense Embeddings\n(all-MiniLM-L6-v2)")
draw_box(ax, 0.55, 0.5, 0.4, 0.1, "Sparse Keyword Matching\n(BM25)")
draw_arrow(ax, 0.25, 0.5, 0.5, 0.3)
draw_arrow(ax, 0.75, 0.5, 0.5, 0.3)
draw_box(ax, 0.2, 0.2, 0.6, 0.1, "Weighted Score Fusion\n$\\alpha=0.7$ (Dense) | $\\beta=0.3$ (Sparse)")
draw_arrow(ax, 0.5, 0.2, 0.5, 0.05)
draw_box(ax, 0.3, -0.05, 0.4, 0.1, "Top-K Retrieved Context")
plt.title("Figure 1: Hybrid Retrieval (BM25 + Vector Fusion)", pad=20, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(out_dir, "bm25_fusion_diagram.png"), bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────────────────────────
# Diagram 2: semantic_metrics_pipeline.png
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
draw_box(ax, 0.1, 0.8, 0.35, 0.1, "Retrieved Medical Context")
draw_box(ax, 0.55, 0.8, 0.35, 0.1, "LLM Generated Answer")
draw_arrow(ax, 0.275, 0.8, 0.5, 0.6)
draw_arrow(ax, 0.725, 0.8, 0.5, 0.6)
draw_box(ax, 0.15, 0.5, 0.7, 0.1, "Generative Safety Evaluation\n(LLM-as-a-Judge & Cosine Similarity)")
draw_arrow(ax, 0.5, 0.5, 0.5, 0.3)
draw_box(ax, 0.2, 0.2, 0.6, 0.1, "Safety Constraints:\nGroundedness > 0.90 | Hallucination < 0.05")
plt.title("Figure 2: Generative Safety (LLM Judge & Semantic Metrics)", pad=20, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(out_dir, "semantic_metrics_pipeline.png"), bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────────────────────────
# Diagram 3: uncertainty_pipeline.png
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
draw_box(ax, 0.25, 0.8, 0.5, 0.1, "Clinical Sequence Predictions\n(Base + MAML Adapted)")
draw_arrow(ax, 0.5, 0.8, 0.5, 0.6)
draw_box(ax, 0.15, 0.5, 0.7, 0.1, "Epistemic Uncertainty Quantification\nVariance ($\\sigma^2$) | Confidence = $1 / (1 + \\sigma^2$)")
draw_arrow(ax, 0.5, 0.5, 0.5, 0.3)
draw_box(ax, 0.2, 0.2, 0.6, 0.1, "Edge-Case Fallback Trigger\nReject Generation if Confidence < 0.65")
plt.title("Figure 3: Epistemic Uncertainty Quantification", pad=20, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(out_dir, "uncertainty_pipeline.png"), bbox_inches='tight', dpi=300)
plt.close()

print(f"✅ Generated 3 Scopus-ready architectural diagrams in {out_dir}")
