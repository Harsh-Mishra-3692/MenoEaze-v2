import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Load data ─────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "synthetic_menopause_data.csv")

df = pd.read_csv(path)

print("="*60)
print("DATASET VALIDATION REPORT")
print("="*60)

# ── 1. Basic stats ───────────────────────
print("\n📊 BASIC STATISTICS\n")
print(df.describe())

# ── 2. Missing values ────────────────────
print("\n📉 MISSING VALUES (%)\n")
missing = df.isna().mean() * 100
print(missing)

# ── 3. Correlation matrix ────────────────
features = [
    "hot_flash_score",
    "night_sweats_score",
    "sleep_quality",
    "mood_score",
    "fatigue_score",
    "anxiety_score",
    "physical_activity",
    "stress_level",
    "caffeine_intake",
    "severity"
]

corr = df[features].corr()

print("\n🔗 CORRELATION MATRIX\n")
print(corr)

# ── 4. Plots ────────────────────────────

# Histograms
df[features].hist(figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("distributions.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation.png")
plt.close()

# Severity over time (sample patients)
plt.figure(figsize=(10, 6))
for pid in df["patient_id"].unique()[:5]:
    subset = df[df["patient_id"] == pid]
    plt.plot(subset["day"], subset["severity"], label=f"Patient {pid}")

plt.legend()
plt.title("Severity Trends (Sample Patients)")
plt.savefig("temporal_trends.png")
plt.close()

print("\n✅ Plots saved:")
print("- distributions.png")
print("- correlation.png")
print("- temporal_trends.png")

print("\n" + "="*60)
