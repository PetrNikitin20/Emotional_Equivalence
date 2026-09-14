"""
Analysis of the emotional profile using eight Plutchik emotions
"""

class EmotionalProfileAnalyzer:
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.plutchik_emotions = lexicon.plutchik_emotions
        self.emotion_labels = {
            'anger': 'anger',
            'anticipation': 'anticipation',
            'disgust': 'disgust',
            'fear': 'fear',
            'joy': 'joy',
            'sadness': 'sadness',
            'surprise': 'surprise',
            'trust': 'trust'
        }

    def analyze(self, text, language='english'):
        emotion_vector, emotional_words = self.lexicon.get_emotion_vector(text, language)

        total = sum(emotion_vector.values())
        percentages = {e: round((emotion_vector[e] / total) * 100, 2) if total > 0 else 0
                       for e in emotion_vector}

        dominant = max(emotion_vector, key=emotion_vector.get) if max(emotion_vector.values()) > 0 else None

        interpretation = self._get_interpretation(percentages, dominant)

        return {
            'emotions': emotion_vector,
            'percentages': percentages,
            'dominant': dominant,
            'dominant_label': self.emotion_labels.get(dominant, 'undetermined') if dominant else 'neutral',
            'dominant_intensity': max(emotion_vector.values()) if max(emotion_vector.values()) > 0 else 0,
            'emotional_words_count': emotional_words,
            'interpretation': interpretation
        }

    def _get_interpretation(self, percentages, dominant):
        if not dominant:
            return "The text has a neutral emotional profile."

        if percentages.get(dominant, 0) > 50:
            intensity = "is dominant"
        elif percentages.get(dominant, 0) > 30:
            intensity = "is strongly expressed"
        else:
            intensity = "is present"

        return (
            f"The emotion '{self.emotion_labels[dominant]}' {intensity} in the text. "
            f"Intensity: {percentages.get(dominant, 0)}%"
        )
