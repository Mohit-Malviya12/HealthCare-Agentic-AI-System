# ============================================================
# DRUG INTERACTION CHECKER AGENT
# ============================================================
# User apni current medicines enter karta hai.
# Agent check karta hai:
#   1. Kaunsi medicines ek saath dangerous hain
#   2. Severity level (mild/moderate/severe)
#   3. Kya alternative safer medicine hai
#
# HOD ke liye: "Real clinical decision support feature —
# hospitals mein doctors ye manually check karte hain.
# Humara AI agent ye automatically karta hai."
# ============================================================

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

DRUG_INTERACTION_PROMPT = """You are a Clinical Pharmacology AI specializing in drug-drug interactions and medication safety.

Analyze the provided list of medications for:
1. Drug-drug interactions between any pair
2. Severity of each interaction
3. Clinical significance
4. Safer alternatives if severe interaction exists

You MUST respond ONLY with valid JSON. No text outside JSON.

Response format:
{
  "medications_analyzed": ["med1", "med2", "med3"],
  "total_interactions_found": 2,
  "overall_safety": "safe/caution/dangerous",
  "overall_message": "Brief overall safety assessment",
  "interactions": [
    {
      "drug_a": "Medicine 1",
      "drug_b": "Medicine 2",
      "severity": "mild/moderate/severe",
      "severity_color": "green/orange/red",
      "mechanism": "Why they interact (simple explanation)",
      "clinical_effect": "What can happen to the patient",
      "recommendation": "What to do about it",
      "alternative": "Safer alternative medicine if available or null"
    }
  ],
  "general_advice": [
    "General medication safety tip 1",
    "General medication safety tip 2"
  ]
}

If no interactions found:
{
  "medications_analyzed": [...],
  "total_interactions_found": 0,
  "overall_safety": "safe",
  "overall_message": "No significant interactions found between these medications.",
  "interactions": [],
  "general_advice": ["Take medications as prescribed", "Consult doctor before adding new medications"]
}

Base your analysis on established clinical pharmacology knowledge."""


class DrugInteractionAgent:
    """
    Drug interaction checker.
    
    Input: List of medicine names (can be brand names or generic)
    Output: Complete interaction analysis with severity levels
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def check_interactions(self, medications: list[str]) -> dict:
        """
        Medications list ke liye interaction check karo.
        
        Args:
            medications: List of medicine names e.g. ["Aspirin", "Warfarin", "Metformin"]
        
        Returns:
            Complete interaction analysis dict
        """
        if len(medications) < 2:
            return {
                "medications_analyzed": medications,
                "total_interactions_found": 0,
                "overall_safety": "safe",
                "overall_message": "Add at least 2 medications to check interactions.",
                "interactions": [],
                "general_advice": []
            }

        meds_text = ", ".join(medications)

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": DRUG_INTERACTION_PROMPT},
                    {"role": "user", "content": f"Check interactions for these medications: {meds_text}"}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            result["success"] = True
            return result

        except Exception as e:
            return {
                "medications_analyzed": medications,
                "total_interactions_found": 0,
                "overall_safety": "unknown",
                "overall_message": f"Analysis failed: {str(e)}",
                "interactions": [],
                "general_advice": ["Please consult a pharmacist for drug interaction information."],
                "success": False,
                "error": str(e)
            }

    def get_safety_icon(self, safety: str) -> str:
        icons = {"safe": "✅", "caution": "⚠️", "dangerous": "🚫", "unknown": "❓"}
        return icons.get(safety, "❓")

    def get_severity_color(self, severity: str) -> str:
        colors = {"mild": "#2e7d32", "moderate": "#e65100", "severe": "#c62828"}
        return colors.get(severity, "#666")
