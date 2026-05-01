import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math

out_dir = "ml_engine/research"
os.makedirs(out_dir, exist_ok=True)

def draw_box(ax, x, y, width, height, text, shape='box'):
    if shape == 'box':
        rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='#333333', facecolor='#f9f9f9', zorder=2)
        ax.add_patch(rect)
    elif shape == 'cylinder':
        # approximate cylinder with a box and ellipses
        ellipse_top = patches.Ellipse((x + width/2, y + height), width, height*0.2, linewidth=1.5, edgecolor='#333333', facecolor='#f9f9f9', zorder=2)
        ellipse_bot = patches.Ellipse((x + width/2, y), width, height*0.2, linewidth=1.5, edgecolor='#333333', facecolor='#f9f9f9', zorder=1)
        rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='#333333', facecolor='#f9f9f9', zorder=2)
        # hide top line of rect
        rect_cover = patches.Rectangle((x+0.01, y+0.01), width-0.02, height-0.02, linewidth=0, facecolor='#f9f9f9', zorder=3)
        ax.add_patch(ellipse_bot)
        ax.add_patch(rect)
        ax.add_patch(rect_cover)
        ax.add_patch(ellipse_top)
    
    # draw lines for cylinder sides
    if shape == 'cylinder':
        ax.plot([x, x], [y, y+height], color='#333333', linewidth=1.5, zorder=4)
        ax.plot([x+width, x+width], [y, y+height], color='#333333', linewidth=1.5, zorder=4)
        
    z_text = 5 if shape == 'cylinder' else 3
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=11, family='sans-serif', color='#111111', zorder=z_text)

def draw_arrow(ax, x1, y1, x2, y2, connectionstyle="arc3"):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='#333333', shrink=0.05, width=1.5, headwidth=8, connectionstyle=connectionstyle), zorder=1)

# ─────────────────────────────────────────────
# 1. system_architecture.png
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')
draw_box(ax, 0.2, 0.8, 0.6, 0.1, "Presentation Layer\n(Next.js)")
draw_arrow(ax, 0.5, 0.8, 0.5, 0.6)
draw_box(ax, 0.2, 0.5, 0.6, 0.1, "Orchestration Layer\n(FastAPI)")
draw_arrow(ax, 0.5, 0.5, 0.25, 0.3)
draw_arrow(ax, 0.5, 0.5, 0.75, 0.3)
draw_box(ax, 0.05, 0.2, 0.4, 0.1, "Inference Layer\n(PyTorch GRU + MAML)")
draw_box(ax, 0.55, 0.2, 0.4, 0.1, "Data Layer\n(Supabase + pgvector)", shape='cylinder')
plt.title("Figure 1: System Architecture", pad=20, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(out_dir, "system_architecture.png"), bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────────────────────────
# 2. request_pipeline.png
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
steps = [
    "Symptom\nIngestion",
    "API Gateway\nValidation",
    "MAML Temporal\nInference",
    "State-Conditioned\nRAG Retrieval",
    "LLM\nGeneration",
    "Explainable\nUI Output"
]
for i, step in enumerate(steps):
    x = 0.02 + i * 0.16
    draw_box(ax, x, 0.4, 0.14, 0.2, step)
    if i < len(steps) - 1:
        draw_arrow(ax, x + 0.14, 0.5, x + 0.16, 0.5)
plt.title("Figure 2: Request Pipeline Sequence", pad=20, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(out_dir, "request_pipeline.png"), bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────────────────────────
# 3. database_schema.png
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
draw_box(ax, 0.3, 0.7, 0.4, 0.15, "User Profiles", shape='cylinder')
draw_box(ax, 0.05, 0.2, 0.4, 0.15, "Temporal Symptom Logs\n(11-feature vectors)", shape='cylinder')
draw_box(ax, 0.55, 0.2, 0.4, 0.15, "Knowledge Base\n(pgvector embeddings)", shape='cylinder')
draw_arrow(ax, 0.4, 0.7, 0.25, 0.35)
ax.text(0.28, 0.55, "1 : N\n(Logs sequence)", ha='center', va='center', fontsize=9)
draw_arrow(ax, 0.6, 0.7, 0.75, 0.35)
ax.text(0.72, 0.55, "State Conditioned\nContext Retrieval", ha='center', va='center', fontsize=9)
plt.title("Figure 3: Database Schema & Relational Mapping", pad=20, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(out_dir, "database_schema.png"), bbox_inches='tight', dpi=300)
plt.close()

# ─────────────────────────────────────────────
# 4. continual_learning_loop.png
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')
nodes = [
    ("User Interaction\n& Feedback", 0.5, 0.85),
    ("Asynchronous\nTraining Buffer", 0.85, 0.5),
    ("Outer-Loop MAML\nMeta-Update", 0.65, 0.15),
    ("Shadow Mode\nEvaluation", 0.35, 0.15),
    ("Production\nWeight Rollout", 0.15, 0.5)
]

# Draw nodes
for text, x, y in nodes:
    draw_box(ax, x - 0.15, y - 0.075, 0.3, 0.15, text)

# Draw circular arrows
# 1 -> 2
draw_arrow(ax, 0.65, 0.85, 0.85, 0.575, connectionstyle="arc3,rad=-0.3")
# 2 -> 3
draw_arrow(ax, 0.85, 0.425, 0.65, 0.225, connectionstyle="arc3,rad=-0.3")
# 3 -> 4
draw_arrow(ax, 0.5, 0.15, 0.5, 0.15) # simple line over gap
draw_arrow(ax, 0.5, 0.15, 0.5, 0.15) 
ax.annotate('', xy=(0.35+0.15, 0.15), xytext=(0.65-0.15, 0.15), arrowprops=dict(facecolor='#333333', shrink=0.05, width=1.5, headwidth=8))
# 4 -> 5
draw_arrow(ax, 0.35, 0.225, 0.15, 0.425, connectionstyle="arc3,rad=-0.3")
# 5 -> 1
draw_arrow(ax, 0.15, 0.575, 0.35, 0.85, connectionstyle="arc3,rad=-0.3")

plt.title("Figure 4: Continual Learning Loop", pad=20, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(out_dir, "continual_learning_loop.png"), bbox_inches='tight', dpi=300)
plt.close()

print(f"✅ Generated 4 Scopus-ready architectural diagrams in {out_dir}")
