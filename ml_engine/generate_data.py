"""
generate_data.py
================
Final calibrated research-grade synthetic menopause dataset generator.

Fixes:
- severity saturation removed
- stronger feature correlations
- realistic variance
- balanced distribution
"""

import numpy as np
import pandas as pd
import os

SEED = 42
np.random.seed(SEED)

N_PATIENTS = 500
N_DAYS = 14

OUTPUT_FILE = "synthetic_menopause_data.csv"


# ── Utilities ──────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip01(x):
    return np.clip(x, 0, 1)


# ── Patient profile ─────────────────────────────────────
def sample_profile():
    subtype = np.random.choice(["mild", "moderate", "severe"], p=[0.4, 0.4, 0.2])

    if subtype == "mild":
        baseline = np.random.uniform(0.2, 0.4)
    elif subtype == "moderate":
        baseline = np.random.uniform(0.4, 0.65)
    else:
        baseline = np.random.uniform(0.65, 0.85)  # reduced upper bound

    return subtype, baseline


# ── Core generator ─────────────────────────────────────
def generate_patient(pid):
    age = np.clip(np.random.normal(50, 4), 40, 60)
    bmi = np.clip(np.random.normal(26, 4), 18, 38)

    subtype, baseline = sample_profile()

    rows = []

    history = [baseline] * 3
    regime = np.random.choice([0, 1])
    flare_days_left = 0

    for day in range(1, N_DAYS + 1):

        # ── flare episodes
        if flare_days_left > 0:
            flare_days_left -= 1
            flare = 1
        else:
            if np.random.rand() < 0.15:
                flare_days_left = np.random.randint(2, 5)
                flare = 1
            else:
                flare = 0

        # ── regime switching
        if np.random.rand() > 0.85:
            regime = 1 - regime

        # ── latent + cyclic
        latent_health = np.random.normal(0, 0.3)
        cycle = np.sin(2 * np.pi * day / 7)

        # ── temporal memory
        temporal_state = (
            0.5 * history[-1]
            + 0.3 * history[-2]
            + 0.2 * history[-3]
        )

        latent = (
            temporal_state
            + 0.3 * baseline
            + 0.2 * latent_health
            + 0.1 * cycle
            + 0.2 * flare
        )

        # ── symptoms
        hot_flash = sigmoid(1.3 * latent + np.random.normal(0, 0.35))
        night_sweats = sigmoid(1.25 * hot_flash + np.random.normal(0, 0.35))

        sleep = clip01(1 - night_sweats + np.random.normal(0, 0.25))

        stress = clip01(
            0.3 + 0.4 * regime + 0.25 * latent + np.random.normal(0, 0.3)
        )

        anxiety = clip01(0.55 * stress + np.random.normal(0, 0.3))
        fatigue = clip01(1 - sleep + np.random.normal(0, 0.3))
        mood = clip01(1 - stress + np.random.normal(0, 0.3))

        # behavioral (semi-independent)
        activity = clip01(np.random.normal(0.5, 0.35) - 0.15 * fatigue)
        caffeine = clip01(np.random.normal(0.5, 0.4))

        # ── FINAL CALIBRATED SEVERITY ────────────────────
        raw_score = (
            1.6 * hot_flash
            + 1.5 * night_sweats
            - 1.4 * sleep
            + 1.3 * stress
            + 1.2 * anxiety
            + 1.0 * fatigue
            + 0.8 * (fatigue * stress)   # interaction
            + 0.6 * latent_health
            + 0.3 * flare
        )

        # prevent sigmoid saturation
        raw_score = raw_score / 4.5

        severity = sigmoid(raw_score)

        # balance with baseline
        severity = 0.6 * severity + 0.4 * baseline

        # stronger noise → realistic variance
        severity += np.random.normal(0, 0.08)

        severity = clip01(severity)

        history.append(severity)
        history.pop(0)

        rows.append({
            "patient_id": pid,
            "day": day,
            "age": age,
            "bmi": bmi,
            "hot_flash_score": hot_flash,
            "night_sweats_score": night_sweats,
            "sleep_quality": sleep,
            "mood_score": mood,
            "fatigue_score": fatigue,
            "anxiety_score": anxiety,
            "physical_activity": activity,
            "stress_level": stress,
            "caffeine_intake": caffeine,
            "severity": severity,
        })

    return rows


# ── Missingness ─────────────────────────────────────────
def inject_missingness(df):
    for col in df.columns:
        if col in ["patient_id", "day"]:
            continue

        mask_random = np.random.rand(len(df)) < 0.04

        stress_mask = df["stress_level"] > 0.7
        mask_stress = (np.random.rand(len(df)) < 0.10) & stress_mask

        df.loc[mask_random | mask_stress, col] = np.nan

    return df


# ── Main ────────────────────────────────────────────────
def main():
    all_rows = []

    for pid in range(N_PATIENTS):
        all_rows.extend(generate_patient(pid))

    df = pd.DataFrame(all_rows)
    df = inject_missingness(df)

    path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    df.to_csv(path, index=False)

    print("━" * 60)
    print("Dataset generated successfully")
    print("Path   :", path)
    print("Shape  :", df.shape)
    print("Missing:", df.isna().sum().sum())
    print("━" * 60)


if __name__ == "__main__":
    main()
