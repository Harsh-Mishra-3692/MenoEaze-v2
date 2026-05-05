"use client";

import { useState } from "react";
import { supabase } from "@/lib/supabase";

type ResultType = {
  severity: number;
  confidence: number;
  answer: string;
};

type FormDataType = {
  hot_flashes: number;
  night_sweats: number;
  fatigue: number;
  physical_activity: number;
  mood: number;
  anxiety: number;
  stress: number;
  sleep_quality: number;
  caffeine: number;
  headaches: number;
  joint_stiffness: number;
  notes: string;
};

const FEATURES = [
  { key: "hot_flashes", label: "Hot Flashes", emoji: "🔥" },
  { key: "night_sweats", label: "Night Sweats", emoji: "💦" },
  { key: "fatigue", label: "Fatigue", emoji: "🥱" },
  { key: "sleep_quality", label: "Sleep Quality", emoji: "😴" },
  { key: "mood", label: "Mood", emoji: "🎭" },
  { key: "anxiety", label: "Anxiety", emoji: "😟" },
  { key: "stress", label: "Stress", emoji: "📈" },
  { key: "physical_activity", label: "Physical Activity", emoji: "🏃‍♀️" },
  { key: "caffeine", label: "Caffeine", emoji: "☕" },
  { key: "headaches", label: "Headaches", emoji: "🤕" },
  { key: "joint_stiffness", label: "Joint Stiffness", emoji: "🦴" },
] as const;

const TOTAL_STEPS = 3;

// Component to render a slider
const SliderFeature = ({ feature, value, onChange }: any) => (
  <div className="mb-4">
    <div className="flex justify-between items-center mb-1">
      <label className="text-sm font-medium text-gray-700 flex items-center gap-2">
        <span>{feature.emoji}</span> {feature.label}
      </label>
      <span className="text-sm font-bold text-purple-600">{value}/10</span>
    </div>
    <input
      type="range"
      min="1"
      max="10"
      value={value || 1}
      onChange={(e) => onChange(feature.key, parseInt(e.target.value))}
      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
    />
  </div>
);

const Step1 = ({ formData, updateField }: any) => (
  <div>
    {FEATURES.slice(0, 5).map((f) => (
      <SliderFeature key={f.key} feature={f} value={formData[f.key]} onChange={updateField} />
    ))}
  </div>
);

const Step2 = ({ formData, updateField }: any) => (
  <div>
    {FEATURES.slice(5, 11).map((f) => (
      <SliderFeature key={f.key} feature={f} value={formData[f.key]} onChange={updateField} />
    ))}
  </div>
);

const Step3 = ({ formData, updateField }: any) => (
  <div>
    <label className="block text-sm font-medium text-gray-700 mb-2">
      Notes / Emojis 📝
    </label>
    <textarea
      value={formData.notes}
      onChange={(e) => updateField("notes", e.target.value)}
      className="w-full border rounded-lg p-3 text-sm focus:ring-2 focus:ring-purple-500 outline-none"
      rows={4}
      placeholder="How are you feeling today?"
    />
  </div>
);

export default function SymptomForm({
  onSuccess,
  onComplete,
}: {
  onSuccess?: () => void;
  onComplete?: (result: ResultType) => void;
}) {
  const [step, setStep] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(false);

  // ---- FORM STATE ----
  const [formData, setFormData] = useState<FormDataType>({
    hot_flashes: 1,
    night_sweats: 1,
    fatigue: 1,
    physical_activity: 1,
    mood: 1,
    anxiety: 1,
    stress: 1,
    sleep_quality: 1,
    caffeine: 1,
    headaches: 1,
    joint_stiffness: 1,
    notes: "",
  });

  // ---- SAFE UPDATE ----
  const updateField = (key: keyof FormDataType, value: number | string) => {
    setFormData((prev) => ({
      ...prev,
      [key]:
        typeof value === "number"
          ? Math.max(1, Math.min(10, value))
          : value,
    }));
  };

  const validateStep = (): boolean => true;

  // ---- NEXT ----
  const handleNext = async () => {
    if (!validateStep()) return;

    if (step < TOTAL_STEPS) {
      setStep((s) => s + 1);
      return;
    }

    await handleSubmit();
  };

  // ---- SUBMIT ----
  const handleSubmit = async () => {
    try {
      setLoading(true);

      const { data: userData } = await supabase.auth.getUser();
      const userId = userData?.user?.id || "demo_user";

      // Build exact 11-feature vector in model order
      const featureVector: number[] = [
        formData.hot_flashes,
        formData.night_sweats,
        formData.fatigue,
        formData.physical_activity,
        formData.mood,
        formData.anxiety,
        formData.stress,
        formData.sleep_quality,
        formData.caffeine,
        formData.headaches,
        formData.joint_stiffness,
      ];

      // Compute severity as average / 10 (normalized 0-1)
      const severity =
        featureVector.reduce((sum, v) => sum + v, 0) / (featureVector.length * 10);

      // Extract emoji from notes (simple heuristic)
      const emojiRegex = /[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu;
      const emojiMatches = formData.notes.match(emojiRegex);
      const emojiStr = emojiMatches ? emojiMatches.join("") : "";

      // 1. Store in Supabase (schema: id, user_id, feature_vector, notes, emoji)
      const { error: insertError } = await supabase.from("symptom_logs").insert([{
        user_id: userId,
        feature_vector: featureVector,
        notes: formData.notes || null,
        emoji: emojiStr || null,
      }]);

      if (insertError) {
        console.error("Supabase insert error:", insertError);
      }

      // 2. Build result from stored data (no backend call on submit — 
      //    the dashboard/assistant will call the backend when needed)
      const result: ResultType = {
        severity: severity,
        confidence: 0,
        answer: severity > 0.6
          ? "Your symptoms appear elevated. Consider consulting a healthcare provider."
          : "Your symptoms are within a manageable range. Keep tracking for trends.",
      };

      onSuccess?.();
      onComplete?.(result);
    } catch (err) {
      console.error("Submission failed:", err);
    } finally {
      setLoading(false);
    }
  };

  // ---- UI ----
  return (
    <div className="space-y-6">
      <div className="text-sm text-gray-500 font-medium">
        Step {step} of {TOTAL_STEPS}
      </div>

      {step === 1 && <Step1 formData={formData} updateField={updateField} />}
      {step === 2 && <Step2 formData={formData} updateField={updateField} />}
      {step === 3 && <Step3 formData={formData} updateField={updateField} />}

      <div className="flex gap-3 mt-8">
        {step > 1 && (
          <button
            onClick={() => setStep((s) => s - 1)}
            className="px-5 py-2 border border-gray-200 text-gray-600 font-medium rounded-lg hover:bg-gray-50 transition"
          >
            Back
          </button>
        )}

        <button
          onClick={handleNext}
          disabled={loading}
          className="px-6 py-2 bg-gradient-to-r from-purple-600 to-pink-500 hover:from-purple-700 hover:to-pink-600 text-white font-medium rounded-lg shadow-sm disabled:opacity-50 transition ml-auto"
        >
          {loading
            ? "Processing..."
            : step === TOTAL_STEPS
              ? "Submit"
              : "Next"}
        </button>
      </div>
    </div>
  );
}