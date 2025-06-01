import pickle
import numpy as np
import re
import html
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from ftfy import fix_text
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Download NLTK data if not already available
nltk.download('stopwords')
nltk.download('punkt')

# Define stop words
NEGATIONS = {"no", "nor", "not", "n't", "never", "none", "nobody", "nothing", "nowhere", "hardly", "scarcely", "barely", "wouldn't", "couldn't", "shouldn't", "won't", "can't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "without"}

CUSTOM_STOPWORDS = set(stopwords.words('english'))
STOP_WORDS = CUSTOM_STOPWORDS - NEGATIONS


class TrigramTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer=None, max_len=100, stop_words=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stop_words = stop_words if stop_words is not None else STOP_WORDS
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self


    def clean_text(self, text):
        """Applies all text cleaning steps once"""
        if not isinstance(text, str):
            return ""

        # Character replacements dictionary
        replacements = {
            '½': '0.5',
            '¼': '0.25',
            '¾': '0.75',
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'â€""': '-',
            'ï': '',
            'á': '',
            'ãª': '',
            'â': ''
        }

        # Apply all cleaning steps in sequence
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        text = text.lower()  # Convert to lowercase
        
        # Apply character replacements
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        text = re.sub(r"[^a-z0-9\s.,!?']", '', text)  # Keep basic characters
        text = fix_text(text)  # Fix text encoding issues
        text = html.unescape(text)  # Unescape HTML entities
        text = re.sub(r'\b\d+\b', '', text)  # Remove isolated numbers
        text = re.sub(r'([!?.,])\1+', r'\1', text)  # Normalize punctuation
        
        # Remove stop words
        tokens = text.split()
        filtered_text = ' '.join([word for word in tokens if word not in self.stop_words])

        return filtered_text

    def generate_trigrams(self, text):
        """Generate trigrams from cleaned text"""
        tokens = word_tokenize(text)
        stemmed = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        trigram_tuples = list(ngrams(stemmed, 3))  # Generate actual trigrams
        trigram_strings = [' '.join(triplet) for triplet in trigram_tuples]
        
        return trigram_strings  # Returns list of trigram phrases

    def transform(self, texts):
        """Full processing pipeline including trigrams and padding"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        trigram_lists = [self.generate_trigrams(text) for text in cleaned_texts]
        trigram_texts = [' '.join(trigrams) for trigrams in trigram_lists]

        # Convert trigrams to sequences
        sequences = self.tokenizer.texts_to_sequences(trigram_texts)

        # Pad sequences
        return pad_sequences(sequences, maxlen=self.max_len, padding='post')

def build_and_save_pipeline():
    """Builds and saves the preprocessing pipeline"""
    with open('fitted_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    pipeline = Pipeline([
        ('preprocessor', TrigramTextPreprocessor(
            tokenizer=tokenizer,
            max_len=100,
            stop_words=STOP_WORDS
        ))
    ])

    with open('text_preprocessing_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print("Preprocessing pipeline saved successfully!")

def predict_sentiment(texts, model_path='best_bilstm_model_final.keras'):
    """Make predictions using the full pipeline"""
    with open('text_preprocessing_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    model = tf.keras.models.load_model(model_path)
    processed = pipeline.transform(texts)
    predictions = model.predict(processed)

    results = []
    for text, pred in zip(texts, predictions):
        score = float(pred[0])
        # sentiment = "Neutral" if abs(score - 0.5) < 0.1 else ("Positive" if score > 0.5 else "Negative")
        sentiment = "Positive" if score > 0.5 else "Negative"
        results.append({
            'text': text,
            'score': score,
            'sentiment': sentiment,
            'processed': processed[0]  # Include processed output for debugging
        })
    
    return results

def demonstrate_processing(sample_text):
    """Shows complete processing steps for a single sample"""
    with open('fitted_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    preprocessor = TrigramTextPreprocessor(
        tokenizer=tokenizer,
        max_len=100,
        stop_words=STOP_WORDS
    )
    
    print(f"\nProcessing Demonstration for: '{sample_text}'")

    cleaned = preprocessor.clean_text(sample_text)
    print("\n1. Cleaned Text:")
    print(cleaned)

    tokens = word_tokenize(cleaned)
    stemmed = [preprocessor.stemmer.stem(t) for t in tokens]
    print("\n2. Tokenization and Stemming:")
    print(f"Tokens: {tokens}")
    print(f"Stemmed: {stemmed}")

    trigrams = preprocessor.generate_trigrams(cleaned)
    print("\n3. Trigram Generation:")
    print(trigrams)

    sequence = tokenizer.texts_to_sequences([' '.join(trigrams)])
    print("\n4. Tokenized Sequences:")
    print(sequence)

    padded = pad_sequences(sequence, maxlen=100, padding='post')
    print("\n5. Padded Input for Model:")
    print(padded[:10])

if __name__ == "__main__":
    sample_texts = [
    "The student studies studied studying",
    "I wouldn't recomend this product", 
    "This product works great and exceeded my expectations!",
    "Terrible quality, would not recommend to anyone.",
    "It was okay, nothing special but not bad either.",
    "Absolutely loved it! Will buy again.",
    "Not what I expected, pretty disappointing.",
    "Fantastic service and very friendly staff!",
    "I have no complaints, everything was fine.",
    "The worst experience I've had in a long time.",
    "It was not bad at all, quite enjoyable actually.",
    "Decent, but could've been better.",
    "I can't say I hated it, but I wouldn't do it again.",
    "Great value for the price!",
    "The packaging was poor but the product was excellent.",
    "Meh. Just average. Nothing stood out.",
    "Totally exceeded my expectations.",
    "Wouldn't say it's perfect, but it's close.",
    "Disgusting. Would never touch this again.",
    "Not impressed, but not the worst either.",
    "Superb quality and lightning-fast delivery!"
]


    build_and_save_pipeline()
    predictions = predict_sentiment(sample_texts)

    print("\nSentiment Analysis Results:")
    for result in predictions:
        print(f"\nText: {result['text']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Sentiment: {result['sentiment']}")
        print("-" * 80)

    demonstrate_processing(sample_texts[0])