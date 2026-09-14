"""
Analysis of the VAD profile (Valence, Arousal, Dominance)
"""

import math

class VADProfileAnalyzer:
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def analyze(self, text):
        vad_result = self.lexicon.get_vad_vector(text)

        vad_result['valence_interpretation'] = self._interpret_valence(vad_result['valence'])
        vad_result['arousal_interpretation'] = self._interpret_arousal(vad_result['arousal'])
        vad_result['dominance_interpretation'] = self._interpret_dominance(vad_result['dominance'])
        vad_result['overall_interpretation'] = self._overall_interpretation(vad_result)

        return vad_result

    def _interpret_valence(self, valence):
        if valence >= 0.7:
            return "Strongly positive: optimism and joy"
        if valence >= 0.55:
            return "Positive: positive affect predominates"
        if valence >= 0.45:
            return "Neutral: balanced text"
        if valence >= 0.3:
            return "Negative: negative affect predominates"
        return "Strongly negative: pronounced negative affect"

    def _interpret_arousal(self, arousal):
        if arousal >= 0.7:
            return "High arousal: intense and dramatic"
        if arousal >= 0.55:
            return "Elevated arousal: dynamic"
        if arousal >= 0.45:
            return "Moderate arousal: balanced"
        if arousal >= 0.3:
            return "Reduced arousal: calm"
        return "Low arousal: minimal intensity"

    def _interpret_dominance(self, dominance):
        if dominance >= 0.7:
            return "High dominance: confident tone"
        if dominance >= 0.55:
            return "Confident narration"
        if dominance >= 0.45:
            return "Neutral stance"
        if dominance >= 0.3:
            return "Uncertain tone"
        return "Low dominance: helplessness"

    def _overall_interpretation(self, vad):
        if vad['coverage'] < 0.3:
            return f"✗ The lexicon covers {int(vad['coverage'] * 100)}% of the words"

        valence_desc = "positive" if vad['valence'] >= 0.5 else "negative" if vad['valence'] < 0.5 else "neutral"
        arousal_desc = "high" if vad['arousal'] >= 0.55 else "low" if vad['arousal'] < 0.45 else "moderate"

        return f"The text has {valence_desc} valence and {arousal_desc} arousal"

    def calculate_distance(self, vad_orig, vad_trans):
        distance = math.sqrt(
            (vad_orig['valence'] - vad_trans['valence']) ** 2 +
            (vad_orig['arousal'] - vad_trans['arousal']) ** 2 +
            (vad_orig['dominance'] - vad_trans['dominance']) ** 2
        )
        normalized = distance / 1.732

        if normalized < 0.2:
            interpretation = "Excellent match"
        elif normalized < 0.4:
            interpretation = "Good match"
        elif normalized < 0.6:
            interpretation = "Moderate match"
        else:
            interpretation = "Substantial divergence"

        return {
            'distance': round(distance, 4),
            'normalized_distance': round(normalized, 4),
            'interpretation': interpretation
        }