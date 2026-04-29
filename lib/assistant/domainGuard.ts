// lib/assistant/domainGuard.ts

const DOMAIN_KEYWORDS = [
    // Core menopause terms
    "menopause", "perimenopause", "postmenopause", "premenopause",
    "menopausal", "climacteric",

    // Hormonal
    "hormone", "hormonal", "estrogen", "progesterone", "testosterone",
    "hrt", "hormone therapy", "hormone replacement", "endocrine",

    // Vasomotor symptoms
    "hot flash", "hot flush", "night sweat", "vasomotor",
    "body temperature", "chills", "sweating",

    // Sleep & energy
    "sleep", "insomnia", "fatigue", "tired", "exhaustion",
    "restless", "waking up", "sleep quality",

    // Mood & cognitive
    "mood", "mood swing", "anxiety", "anxious", "depression",
    "irritability", "brain fog", "memory", "concentration",
    "emotional", "stress", "overwhelm", "crying",

    // Physical symptoms
    "bone", "osteoporosis", "joint pain", "muscle ache",
    "headache", "migraine", "weight gain", "bloating",
    "dry skin", "hair loss", "hair thinning",
    "heart palpitation", "dizziness",

    // Reproductive & urinary
    "irregular period", "missed period", "vaginal dryness",
    "libido", "urinary", "bladder", "incontinence",
    "breast tenderness", "period", "menstrual", "cycle",

    // Body changes
    "nausea", "cramp", "ache", "pain", "stiffness",
    "inflammation", "swelling", "tingling",

    // Wellness & remedies
    "supplement", "vitamin", "calcium", "magnesium", "omega",
    "herbal", "natural remedy", "yoga", "meditation",
    "exercise", "diet", "nutrition", "phytoestrogen",
    "black cohosh", "evening primrose", "flaxseed",
    "acupuncture", "wellness", "self-care",

    // General health context
    "symptom", "severity", "health", "well-being",
    "doctor", "gynecologist", "treatment", "relief",
    "aging", "midlife"
]

export function isMenopauseRelated(text: string): boolean {
    const lower = text.toLowerCase()
    return DOMAIN_KEYWORDS.some(keyword => lower.includes(keyword))
}