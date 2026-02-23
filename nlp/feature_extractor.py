# nlp/feature_extractor.py
import re
import sys
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

class NLPFeatureExtractor:
    """Extract numerical features from text for regression models"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Simple keyword lists for LinkedIn
        self.engagement_keywords = [
            'insight', 'learn', 'share', 'excited', 'proud', 'happy',
            'announce', 'launch', 'grow', 'success', 'achievement',
            'team', 'collaborate', 'innovation', 'future', 'digital'
        ]
        
        self.cta_keywords = [
            'comment', 'share', 'like', 'thoughts', 'opinions',
            'questions', 'experience', 'advice', 'recommendations'
        ]
        
        # Industry categories for LinkedIn
        self.industry_categories = {
            'tech': ['ai', 'machine learning', 'software', 'data', 'cloud', 'cybersecurity'],
            'business': ['strategy', 'management', 'leadership', 'growth', 'marketing'],
            'finance': ['investment', 'banking', 'stocks', 'crypto', 'fintech'],
            'healthcare': ['medical', 'health', 'biotech', 'pharma', 'wellness']
        }
        
        # Initialize spaCy for advanced features
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Note: spaCy model not found. Installing...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive NLP features for regression input"""
        features = {}
        
        # 1. Basic Text Statistics
        features.update(self._extract_basic_stats(text))
        
        # 2. Readability Scores
        features.update(self._extract_readability_scores(text))
        
        # 3. Sentiment Analysis
        features.update(self._extract_sentiment_features(text))
        
        # 4. Content Quality Features
        features.update(self._extract_content_quality(text))
        
        # 5. Engagement Triggers
        features.update(self._extract_engagement_triggers(text))
        
        # 6. Topic Features
        features.update(self._extract_topic_features(text))
        
        # 7. Style Features
        features.update(self._extract_style_features(text))
        
        # 8. Named Entity Recognition
        features.update(self._extract_ner_features(text))
        
        return features
    
    def _extract_basic_stats(self, text: str) -> Dict[str, float]:
        """Basic text statistics"""
        words = text.split()
        sentences = sent_tokenize(text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words)/len(sentences) if sentences else 0,
            'unique_word_ratio': len(set(words))/len(words) if words else 0,
            'hashtag_count': len([w for w in words if w.startswith('#')]),
            'mention_count': len([w for w in words if w.startswith('@')]),
            'link_count': len(re.findall(r'https?://\S+', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper())/len(text) if text else 0,
        }
    
    def _extract_readability_scores(self, text: str) -> Dict[str, float]:
        """Calculate readability scores"""
        sentences = sent_tokenize(text)
        words = [word_tokenize(sent) for sent in sentences]
        total_words = sum(len(w) for w in words)
        total_sentences = len(sentences)
        
        # Flesch Reading Ease
        if total_words > 0 and total_sentences > 0:
            avg_sentence_length = total_words / total_sentences
            avg_syllables_per_word = self._count_syllables(text) / total_words
            flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        else:
            flesch = 0
        
        # Coleman-Liau Index
        if text:
            letters = sum(c.isalpha() for c in text)
            L = letters / len(text) * 100
            S = total_sentences / len(text.split()) * 100
            coleman = 0.0588 * L - 0.296 * S - 15.8
        else:
            coleman = 0
        
        return {
            'flesch_reading_ease': flesch,
            'coleman_liau_index': coleman,
            'readability_grade': self._flesch_to_grade_level(flesch)
        }
    
    def _count_syllables(self, text: str) -> int:
        """Approximate syllable count"""
        vowels = 'aeiouy'
        count = 0
        words = text.lower().split()
        
        for word in words:
            if len(word) <= 3:
                count += 1
                continue
                
            if word.endswith(('es', 'ed')):
                word = word[:-2]
            
            prev_char_vowel = False
            for char in word:
                if char in vowels:
                    if not prev_char_vowel:
                        count += 1
                    prev_char_vowel = True
                else:
                    prev_char_vowel = False
        
        return max(count, 1)
    
    def _flesch_to_grade_level(self, score: float) -> float:
        """Convert Flesch score to US grade level"""
        if score >= 90: return 5.0
        elif score >= 80: return 6.0
        elif score >= 70: return 7.0
        elif score >= 60: return 8.0
        elif score >= 50: return 9.0
        elif score >= 30: return 10.0
        else: return 12.0
    
    def _extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features"""
        blob = TextBlob(text)
        
        return {
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
            'positive_word_ratio': sum(1 for w in text.split() if TextBlob(w).sentiment.polarity > 0.1) / len(text.split()) if text.split() else 0,
            'negative_word_ratio': sum(1 for w in text.split() if TextBlob(w).sentiment.polarity < -0.1) / len(text.split()) if text.split() else 0,
        }
    
    def _extract_content_quality(self, text: str) -> Dict[str, float]:
        """Features indicating content quality"""
        words = text.lower().split()
        
        # Stop word ratio (lower is better for professional content)
        stopword_count = sum(1 for w in words if w in self.stop_words)
        
        # Power words (words that drive engagement)
        power_words = ['proven', 'secret', 'powerful', 'ultimate', 'essential',
                      'surprising', 'amazing', 'incredible', 'guaranteed']
        
        power_word_count = sum(1 for w in words if w in power_words)
        
        return {
            'stopword_ratio': stopword_count / len(words) if words else 0,
            'power_word_ratio': power_word_count / len(words) if words else 0,
            'content_density': (len(words) - stopword_count) / len(words) if words else 0,
        }
    
    def _extract_engagement_triggers(self, text: str) -> Dict[str, float]:
        """Features that trigger engagement"""
        text_lower = text.lower()
        
        engagement_score = sum(1 for kw in self.engagement_keywords if kw in text_lower)
        cta_score = sum(1 for kw in self.cta_keywords if kw in text_lower)
        
        # Question features
        has_question = int('?' in text)
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        question_word_count = sum(1 for w in question_words if w in text_lower)
        
        # Personal pronouns (engagement)
        personal_pronouns = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        pronoun_count = sum(1 for pp in personal_pronouns if pp.lower() in text_lower)
        
        return {
            'engagement_keyword_score': engagement_score,
            'cta_score': cta_score,
            'has_question': has_question,
            'question_word_count': question_word_count,
            'personal_pronoun_ratio': pronoun_count / len(text.split()) if text.split() else 0,
            'exclamation_count': text.count('!'),
            'emoticon_count': len(re.findall(r'[:;]-?[\)\(PpDd]', text)),
        }
    
    def _extract_topic_features(self, text: str) -> Dict[str, float]:
        """Extract topic-related features"""
        text_lower = text.lower()
        
        # Industry relevance
        industry_scores = {}
        for industry, keywords in self.industry_categories.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            industry_scores[f'industry_{industry}_score'] = score
        
        # Trending topics check (simple version)
        trending_topics = {
            'ai_ml': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network'],
            'remote_work': ['remote work', 'hybrid', 'work from home', 'flexible work'],
            'sustainability': ['sustainability', 'esg', 'green', 'climate', 'environmental']
        }
        
        trending_scores = {}
        for topic, keywords in trending_topics.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            trending_scores[f'trend_{topic}_score'] = score
        
        return {**industry_scores, **trending_scores}
    
    def _extract_style_features(self, text: str) -> Dict[str, float]:
        """Extract writing style features"""
        sentences = sent_tokenize(text)
        
        if len(sentences) == 0:
            return {
                'sentence_variation': 0,
                'paragraph_count': 0,
                'bullet_point_count': 0,
            }
        
        # Sentence length variation
        sentence_lengths = [len(sent.split()) for sent in sentences]
        sentence_variation = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0
        
        # Paragraph features
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        # Bullet points
        bullet_patterns = [r'^\s*[•\-\*]\s', r'^\s*\d+\.\s']
        bullet_count = 0
        for line in text.split('\n'):
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    bullet_count += 1
                    break
        
        return {
            'sentence_variation': sentence_variation,
            'paragraph_count': len(paragraphs),
            'bullet_point_count': bullet_count,
            'has_code_snippet': int('```' in text or '`' in text),
            'has_quote': int('"' in text or "'" in text),
        }
    
    def _extract_ner_features(self, text: str) -> Dict[str, float]:
        """Named Entity Recognition features using spaCy"""
        doc = self.nlp(text[:1000])  # Limit for speed
        
        ner_counts = {
            'ner_person_count': 0,
            'ner_org_count': 0,
            'ner_product_count': 0,
            'ner_location_count': 0,
            'ner_date_count': 0,
            'ner_money_count': 0,
        }
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                ner_counts['ner_person_count'] += 1
            elif ent.label_ == 'ORG':
                ner_counts['ner_org_count'] += 1
            elif ent.label_ == 'PRODUCT':
                ner_counts['ner_product_count'] += 1
            elif ent.label_ == 'GPE' or ent.label_ == 'LOC':
                ner_counts['ner_location_count'] += 1
            elif ent.label_ == 'DATE':
                ner_counts['ner_date_count'] += 1
            elif ent.label_ == 'MONEY':
                ner_counts['ner_money_count'] += 1
        
        # Normalize by text length
        word_count = len(text.split())
        if word_count > 0:
            for key in ner_counts:
                ner_counts[key] = ner_counts[key] / word_count
        
        return ner_counts

# Simple version (without spaCy)
class SimpleNLPFeatureExtractor:
    """Lightweight NLP feature extractor without heavy dependencies"""
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract key NLP features for regression"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            # Basic features
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            
            # Engagement features
            'has_question': 1 if '?' in text else 0,
            'has_exclamation': 1 if '!' in text else 0,
            'hashtag_count': len([w for w in words if w.startswith('#')]),
            'mention_count': len([w for w in words if w.startswith('@')]),
            
            # Content features
            'link_count': len(re.findall(r'https?://\S+', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper())/len(text) if text else 0,
            
            # Simple sentiment
            'positive_words': self._count_positive_words(text),
            'negative_words': self._count_negative_words(text),
            
            # LinkedIn specific
            'has_cta': 1 if any(word in text.lower() for word in 
                              ['comment', 'share', 'like', 'thoughts']) else 0,
        }
    
    def _count_positive_words(self, text: str) -> int:
        positive = ['great', 'good', 'excellent', 'amazing', 'wonderful', 
                   'happy', 'pleased', 'excited', 'proud', 'success']
        return sum(1 for word in text.lower().split() if word in positive)
    
    def _count_negative_words(self, text: str) -> int:
        negative = ['bad', 'poor', 'terrible', 'awful', 'difficult',
                   'challenging', 'hard', 'struggle', 'problem', 'issue']
        return sum(1 for word in text.lower().split() if word in negative)