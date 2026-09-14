"""
LLM advisor for analysing emotional preservation in translation
Uses Ollama with the Qwen2.5:7b model
"""

import requests
import json
import re
import time
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

class LLMAdvisor:
    
    def __init__(self, ollama_model="qwen2.5:7b"):
        self.ollama_model = ollama_model
        self.ollama_available = False
        self.ollama_url = "http://localhost:11434/api/generate"
        
        self.model_available = False
        self.sentiment_available = False
        self.en_sentiment_available = False
        
        print(f"Checking Ollama for model {ollama_model}...")
        
        try:
            tags_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if tags_response.status_code == 200:
                installed_models = tags_response.json().get("models", [])
                installed_names = [m.get("name", "") for m in installed_models]
                
                model_found = False
                for m in installed_names:
                    if self.ollama_model in m:
                        model_found = True
                        break
                
                if model_found:
                    print(f"✓ Model {self.ollama_model} found")
                    self.ollama_available = True
                    self.model_available = True
                else:
                    print(f"✗ Model {self.ollama_model} not found")
                    print(f"   Available models: {installed_names}")
        except Exception as e:
            print(f"✗ Ollama is unavailable: {e}")
        
        print("Loading sentiment-analysis models...")
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="blanchefort/rubert-base-cased-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            self.sentiment_available = True
            print("✓ Loaded the Russian sentiment model (rubert-base-cased-sentiment)")
        except Exception as e:
            print(f"✗ Could not load the Russian sentiment model: {e}")
        
        try:
            self.en_sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            self.en_sentiment_available = True
            print("✓ Loaded the English sentiment model")
        except Exception as e:
            print(f"✗ Could not load the English sentiment model: {e}")
        
        print("✓ The LLM advisor is ready.")
    
    def _call_ollama(self, prompt, max_tokens=600):
        if not self.ollama_available:
            return None
        
        for attempt in range(2):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5,
                            "num_predict": max_tokens,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1
                        }
                    },
                    timeout=180
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    print(f"Ollama error (attempt {attempt+1}): {response.status_code}")
                    time.sleep(3)
                    
            except requests.exceptions.Timeout:
                print(f"Ollama timeout (attempt {attempt+1})")
                time.sleep(3)
            except Exception as e:
                print(f"Ollama error (attempt {attempt+1}): {e}")
                time.sleep(3)
        
        return None
    
    def generate_recommendation(self, original_text, translated_text, eqa_score, 
                                 ed_score, pde_score, k_score, dominant_orig, dominant_trans):
        if self.ollama_available:
            result = self._generate_ollama_recommendation(
                original_text, translated_text, eqa_score,
                ed_score, pde_score, k_score, dominant_orig, dominant_trans
            )
            if result and len(result) > 50:
                return result
        
        return self._fallback_recommendation(eqa_score, ed_score, pde_score, k_score, 
                                             dominant_orig, dominant_trans)
    
    def _generate_ollama_recommendation(self, original_text, translated_text, eqa_score, 
                                         ed_score, pde_score, k_score, dominant_orig, dominant_trans):
        if len(original_text) > 500:
            orig_short = original_text[:250] + "\n...\n" + original_text[-250:]
            trans_short = translated_text[:250] + "\n...\n" + translated_text[-250:]
        else:
            orig_short = original_text
            trans_short = translated_text
        
        if eqa_score >= 0.8:
            eqa_text = "excellent translation quality; emotional tone is preserved exceptionally well"
        elif eqa_score >= 0.6:
            eqa_text = "good translation quality with minor emotional differences"
        elif eqa_score >= 0.4:
            eqa_text = "satisfactory quality with noticeable emotional distortion"
        else:
            eqa_text = "low quality with substantial emotional loss"
        
        if ed_score < 0.2:
            ed_text = "the emotional tone is an almost exact match"
        elif ed_score < 0.4:
            ed_text = "the emotional tone is close to the source"
        elif ed_score < 0.6:
            ed_text = "the emotional tone differs noticeably"
        else:
            ed_text = "the emotional tone is substantially distorted"
        
        if pde_score >= 0.8:
            pde_text = f"the dominant emotion '{dominant_orig}' is fully preserved"
        elif pde_score >= 0.6:
            pde_text = f"the dominant emotion '{dominant_orig}' is preserved but weakened"
        elif pde_score >= 0.4:
            pde_text = f"the dominant emotion shifted from '{dominant_orig}' to '{dominant_trans}'"
        else:
            pde_text = f"the dominant emotion is lost; the source emotion '{dominant_orig}' is replaced by '{dominant_trans}'"
        
        if k_score >= 0.8:
            k_text = "cultural markers are adapted excellently and their emotional subtext is preserved"
        elif k_score >= 0.6:
            k_text = "cultural markers are adapted well, with minor losses"
        elif k_score >= 0.4:
            k_text = "cultural markers are adapted satisfactorily, with noticeable semantic losses"
        else:
            k_text = "cultural markers are largely unadapted and their emotional subtext is lost"
        
        prompt = f"""You are an expert in cross-cultural communication and literary translation. Analyse the translation and give specific recommendations that account for differences between English-speaking and Russian-speaking audiences.

SOURCE TEXT (English):
{orig_short}

TRANSLATION (Russian):
{trans_short}

OVERALL QUALITY ASSESSMENT:
{eqa_text} (EQA = {eqa_score:.2f})

DETAILED METRICS:
1) Emotional tone: {ed_text} (ED = {ed_score:.2f}, where 0 is an exact match and 1 is complete divergence)
2) Preservation of the dominant emotion: {pde_text} (PDE = {pde_score:.2f})
3) Adaptation of cultural markers: {k_text} (K = {k_score:.2f})
4) Dominant emotion in the source: {dominant_orig}
5) Dominant emotion in the translation: {dominant_trans}

Write a detailed recommendation for the translator. Focus on:
1. Preserving emotional tone for a Russian-speaking readership
2. Specific alternative renderings for key emotionally charged passages
3. Strategies for adapting cultural markers without losing emotional subtext

Respond in English in one coherent paragraph of four to six sentences. Do not use bullet points. Be specific.

RECOMMENDATION:"""
        
        result = self._call_ollama(prompt, max_tokens=600)
        
        if result:
            result = re.sub(r'^RECOMMENDATIONS?:\s*', '', result)
            result = re.sub(r'^Recommendations?:\s*', '', result)
        
        return result
    
    def generate_marker_recommendation(self, marker, original_sentence, translated_sentence, loss):
        if self.ollama_available:
            result = self._generate_ollama_marker_recommendation(
                marker, original_sentence, translated_sentence, loss
            )
            if result and len(result) > 30:
                return result
        
        return self._fallback_marker_recommendation(marker, loss)
    
    def _generate_ollama_marker_recommendation(self, marker, original_sentence, translated_sentence, loss):
        orig_short = original_sentence[:300] + "..." if len(original_sentence) > 300 else original_sentence
        trans_short = translated_sentence[:300] + "..." if len(translated_sentence) > 300 else translated_sentence
        
        if loss > 0.8:
            severity = "critical loss of emotional tone"
        elif loss > 0.6:
            severity = "major loss of emotional tone"
        elif loss > 0.4:
            severity = "moderate loss of emotional tone"
        else:
            severity = "minor loss of emotional tone"
        
        prompt = f"""You are an expert in translating culture-specific expressions from English into Russian.

CULTURAL MARKER: "{marker}"
Severity: {severity} (loss {loss:.0%})

SOURCE TEXT:
{orig_short}

CURRENT TRANSLATION:
{trans_short}

Provide a detailed three- or four-sentence recommendation. Explain why the current rendering does not preserve the emotional subtext and propose a specific Russian alternative that retains the cultural and emotional meaning. Account for differences between English-speaking and Russian-speaking cultures. Respond in English.

RECOMMENDATION:"""
        
        result = self._call_ollama(prompt, max_tokens=350)
        
        if result:
            result = re.sub(r'^RECOMMENDATION:\s*', '', result)
        
        return result
    
    def analyze_emotional_shift(self, original_text, translated_text):
        result = {
            'original': {'label': 'undetermined', 'score': 0.5},
            'translated': {'label': 'undetermined', 'score': 0.5},
            'shift_detected': False,
            'shift_magnitude': 0,
            'interpretation': "The analysis is based on the system metrics."
        }
        
        if self.en_sentiment_available:
            try:
                orig_result = self.en_sentiment_pipeline(original_text[:512])[0]
                label_map = {'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral'}
                result['original']['label'] = label_map.get(orig_result['label'], orig_result['label'])
                result['original']['score'] = orig_result['score']
            except:
                pass
        
        if self.sentiment_available:
            try:
                trans_result = self.sentiment_pipeline(translated_text[:512])[0]
                label_map_ru = {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'}
                result['translated']['label'] = label_map_ru.get(trans_result['label'], trans_result['label'])
                result['translated']['score'] = trans_result['score']
            except:
                pass
        
        if result['original']['label'] != 'undetermined' and result['translated']['label'] != 'undetermined':
            result['shift_detected'] = result['original']['label'] != result['translated']['label']
            if result['shift_detected']:
                result['interpretation'] = f"The source text is predominantly {result['original']['label']}, whereas the translation is predominantly {result['translated']['label']}."
            else:
                result['interpretation'] = f"The emotional polarity is preserved ({result['original']['label']})."
        
        return result
    
    def _fallback_recommendation(self, eqa_score, ed_score, pde_score, k_score, dominant_orig, dominant_trans):
        if eqa_score >= 0.8:
            return f"The translation is of a high standard and preserves the emotional tone very well. {dominant_orig} is conveyed naturally. Cultural markers are adapted well; only isolated culture-specific expressions require review."
        
        elif eqa_score >= 0.6:
            text = f"The translation is generally good but would benefit from minor revision. "
            if ed_score > 0.35:
                text += f"Its overall emotional tone differs slightly from the source. "
            if pde_score < 0.7:
                text += f"The dominant emotion '{dominant_orig}' is not conveyed at full intensity; consider more expressive wording. "
            if k_score < 0.6:
                text += f"Some cultural markers require stronger adaptation; consider functional Russian equivalents instead of literal renderings. "
            return text
        
        elif eqa_score >= 0.4:
            text = f"The translation requires revision because its emotional tone is not preserved accurately. "
            if ed_score > 0.5:
                text += f"The overall tone differs substantially from the source. "
            if pde_score < 0.5:
                text += f"The dominant emotion shifted from '{dominant_orig}' to '{dominant_trans}'; restore the emphasis on the source emotion. "
            if k_score < 0.4:
                text += f"Cultural markers are largely unpreserved; use functional substitution. "
            return text
        
        else:
            return f"The translation requires substantial revision. Its emotional profile differs markedly from the source; compare it systematically with the source and restore the principal emotional emphases."
    
    def _fallback_marker_recommendation(self, marker, loss):
        if loss > 0.8:
            return f"The marker '{marker}' has almost completely lost its emotional force. Use a Russian functional equivalent that evokes a comparable response rather than a literal rendering."
        elif loss > 0.6:
            return f"The marker '{marker}' is conveyed with noticeable losses. Replace the literal rendering with a more idiomatic Russian expression that preserves the source's emotional nuance."
        elif loss > 0.4:
            return f"The marker '{marker}' requires refinement. The meaning is accurate, but part of the emotional nuance is lost; adjust the wording or word order to strengthen the effect."
        else:
            return f"The marker '{marker}' is conveyed well and retains its emotional tone. Check only whether the phrase sounds natural in the context of the full paragraph."
