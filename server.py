import html
import json
import os
import pickle
import re
import warnings
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import nltk
import numpy as np
import praw
import tensorflow as tf
from ftfy import fix_text
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.exceptions import InconsistentVersionWarning

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

NEGATIONS = {
    "no",
    "nor",
    "not",
    "n't",
    "never",
    "none",
    "nobody",
    "nothing",
    "nowhere",
    "hardly",
    "scarcely",
    "barely",
    "wouldn't",
    "couldn't",
    "shouldn't",
    "won't",
    "can't",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "without",
}

CUSTOM_STOPWORDS = set(stopwords.words("english"))
STOP_WORDS = CUSTOM_STOPWORDS - NEGATIONS


class TrigramTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer=None, max_len=100, stop_words=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stop_words = stop_words if stop_words is not None else STOP_WORDS
        self.stemmer = PorterStemmer()
        if tokenizer is not None:
            self.vocab_size = len(tokenizer.word_index) + 1

    def fit(self, X, y=None):
        return self

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        text = text.replace("½", "0.5").replace("¼", "0.25").replace("¾", "0.75")
        text = re.sub(r"[^a-z0-9\s.,!?']", "", text)
        text = fix_text(text)
        text = html.unescape(text)
        text = re.sub(r"\b\d+\b", "", text)
        text = re.sub(r"([!?.,])\1+", r"\1", text)
        return " ".join([word for word in text.split() if word not in self.stop_words])

    def generate_trigrams(self, text):
        tokens = word_tokenize(text)
        stemmed = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return [" ".join(trigram) for trigram in ngrams(stemmed, 3)]

    def transform(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        trigram_texts = [" ".join(self.generate_trigrams(text)) for text in cleaned_texts]
        sequences = self.tokenizer.texts_to_sequences(trigram_texts)
        if hasattr(self, "vocab_size"):
            sequences = [[index for index in sequence if index < self.vocab_size] for sequence in sequences]
        return pad_sequences(sequences, maxlen=self.max_len, padding="post")


import sys

sys.modules.setdefault("__main__", sys.modules.get("__main__", sys.modules[__name__]))
setattr(sys.modules["__main__"], "TrigramTextPreprocessor", TrigramTextPreprocessor)


class SentimentEngine:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.testing_dir = self.base_dir / "Testing"
        self.tokenizer_path = self.testing_dir / "fitted_tokenizer.pkl"
        self.pipeline_path = self.testing_dir / "text_preprocessing_pipeline.pkl"
        self.model_path = self.testing_dir / "best_bilstm_model_final.keras"
        self.reddit = self._build_reddit_client()

        with self.tokenizer_path.open("rb") as file:
            self.tokenizer = pickle.load(file)

        with self.pipeline_path.open("rb") as file:
            self.pipeline = pickle.load(file)

        self.preprocessor = self.pipeline.named_steps.get("preprocessor")
        if self.preprocessor is None:
            raise ValueError("Preprocessing pipeline is missing the preprocessor step")
        if not hasattr(self.preprocessor, "vocab_size"):
            self.preprocessor.vocab_size = len(self.tokenizer.word_index) + 1

        self.model = tf.keras.models.load_model(self.model_path)
        self.embedding_dim = None
        for layer in self.model.get_config().get("layers", []):
            if layer.get("class_name") == "Embedding":
                self.embedding_dim = layer.get("config", {}).get("input_dim")
                break

    def _build_reddit_client(self):
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "script:reddit-lens:v1.0")
        if not client_id or not client_secret:
            return None
        return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    def _bucket(self, score):
        if score > 0.5:
            return "Positive"
        return "Negative"

    def _confidence(self, score):
        return max(score, 1 - score) * 100

    def _normalize_texts(self, texts):
        return [text if isinstance(text, str) else "" for text in texts if text is not None]

    def _transform_texts(self, texts):
        normalized = self._normalize_texts(texts)
        try:
            processed = self.pipeline.transform(normalized)
        except Exception:
            processed = self.preprocessor.transform(normalized)
        if self.embedding_dim:
            processed = np.array(
                [[index if index < self.embedding_dim else 0 for index in row] for row in processed],
                dtype=processed.dtype,
            )
        return processed

    def predict_texts(self, texts):
        normalized = self._normalize_texts(texts)
        if not normalized:
            return []
        processed = self._transform_texts(normalized)
        predictions = self.model.predict(processed, verbose=0)
        results = []
        for text, prediction in zip(normalized, predictions):
            score = float(prediction[0])
            results.append(
                {
                    "text": text,
                    "sentiment_score": score,
                    "sentiment": self._bucket(score),
                    "confidence": round(self._confidence(score), 2),
                }
            )
        return results

    def _summary(self, records):
        total = len(records)
        if not total:
            return {
                "total": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "positive_pct": 0,
                "neutral_pct": 0,
                "negative_pct": 0,
                "avg_score": 0,
                "avg_confidence": 0,
            }

        counts = Counter(item["sentiment"] for item in records)
        avg_score = sum(item["sentiment_score"] for item in records) / total
        avg_confidence = sum(item["confidence"] for item in records) / total
        return {
            "total": total,
            "positive": counts.get("Positive", 0),
            "neutral": counts.get("Neutral", 0),
            "negative": counts.get("Negative", 0),
            "positive_pct": round((counts.get("Positive", 0) / total) * 100, 2),
            "neutral_pct": round((counts.get("Neutral", 0) / total) * 100, 2),
            "negative_pct": round((counts.get("Negative", 0) / total) * 100, 2),
            "avg_score": round(avg_score, 4),
            "avg_confidence": round(avg_confidence, 2),
        }

    def _keywords(self, texts, top_n=12):
        counter = Counter()
        for text in texts:
            cleaned = self.preprocessor.clean_text(text)
            tokens = word_tokenize(cleaned)
            stemmed = [self.preprocessor.stemmer.stem(token) for token in tokens if token.isalpha() and token not in self.preprocessor.stop_words]
            counter.update(stemmed)
        return [{"term": term, "count": count} for term, count in counter.most_common(top_n)]

    def _insight(self, items):
        summary = self._summary(items)
        if not summary["total"]:
            return "No usable text was provided."
        if summary["positive_pct"] >= 55:
            return f"The sample leans positive at {summary['positive_pct']:.0f}%, with the strongest approval in the model output."
        if summary["negative_pct"] >= 55:
            return f"The sample leans negative at {summary['negative_pct']:.0f}%, which suggests strong criticism or dissatisfaction."
        return f"The sample is mixed, with {summary['positive_pct']:.0f}% positive and {summary['negative_pct']:.0f}% negative signals."

    def analyze_custom_text(self, texts):
        normalized = self._normalize_texts(texts)
        if not normalized:
            return {"ok": False, "error": "No text was provided."}

        predictions = self.predict_texts(normalized)
        items = []
        for index, result in enumerate(predictions, start=1):
            items.append(
                {
                    "id": f"text-{index}",
                    "title": f"Custom text {index}",
                    "content": result["text"],
                    "label": f"Text {index}",
                    "sentiment": result["sentiment"],
                    "sentiment_score": result["sentiment_score"],
                    "confidence": result["confidence"],
                    "author": "User input",
                }
            )

        return {
            "ok": True,
            "mode": "custom_text",
            "title": "Custom text analysis",
            "insight": self._insight(items),
            "summary": self._summary(items),
            "top_keywords": self._keywords(normalized),
            "items": items,
            "request": {"mode": "custom_text"},
        }

    def _sorter(self, subreddit, sort_by):
        if sort_by == "new":
            return subreddit.new(limit=None)
        if sort_by == "top":
            return subreddit.top(limit=None)
        return subreddit.hot(limit=None)

    def analyze_subreddit(self, subreddit_name, limit=20, sort_by="hot"):
        if not self.reddit:
            return {"ok": False, "error": "Reddit credentials are not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."}

        subreddit_name = subreddit_name.strip()
        if not subreddit_name:
            return {"ok": False, "error": "A subreddit name is required."}

        subreddit = self.reddit.subreddit(subreddit_name)
        posts = self._sorter(subreddit, sort_by)
        raw_texts = []
        items = []

        for post in posts:
            if len(items) >= limit:
                break
            content = f"{post.title} {post.selftext or ''}".strip()
            raw_texts.append(content)
            items.append(
                {
                    "id": post.id,
                    "title": post.title,
                    "content": content,
                    "subreddit": subreddit_name,
                    "author": str(post.author) if post.author else "unknown",
                    "upvotes": int(getattr(post, "score", 0)),
                    "comments": int(getattr(post, "num_comments", 0)),
                    "created_utc": float(getattr(post, "created_utc", 0)),
                    "url": f"https://www.reddit.com{post.permalink}",
                }
            )

        predictions = self.predict_texts(raw_texts)
        items = [{**item, **prediction} for item, prediction in zip(items, predictions)]

        return {
            "ok": True,
            "mode": "subreddit",
            "title": f"r/{subreddit_name} sentiment analysis",
            "insight": self._insight(items),
            "summary": self._summary(items),
            "top_keywords": self._keywords(raw_texts),
            "items": items,
            "request": {"mode": "subreddit", "subreddit": subreddit_name, "limit": limit, "sort_by": sort_by},
        }

    def analyze_post_comments(self, post_url, limit=50):
        if not self.reddit:
            return {"ok": False, "error": "Reddit credentials are not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."}

        match = re.search(r"comments/([a-z0-9]+)/", post_url, re.IGNORECASE)
        if not match:
            return {"ok": False, "error": "Could not extract the Reddit post id from the supplied URL."}

        submission = self.reddit.submission(id=match.group(1))
        submission.comments.replace_more(limit=0)
        comments = list(submission.comments.list())[:limit]

        raw_texts = []
        items = []
        for comment in comments:
            if not hasattr(comment, "body"):
                continue
            body = comment.body or ""
            raw_texts.append(body)
            items.append(
                {
                    "id": comment.id,
                    "title": body[:120] if body else "Comment",
                    "content": body,
                    "author": str(comment.author) if comment.author else "unknown",
                    "upvotes": int(getattr(comment, "score", 0)),
                    "replies": len(comment.replies) if hasattr(comment, "replies") else 0,
                    "created_utc": float(getattr(comment, "created_utc", 0)),
                    "parent_post_title": submission.title,
                    "url": f"https://www.reddit.com{comment.permalink}",
                }
            )

        predictions = self.predict_texts(raw_texts)
        items = [{**item, **prediction} for item, prediction in zip(items, predictions)]

        return {
            "ok": True,
            "mode": "post_comments",
            "title": f"Comments from: {submission.title}",
            "insight": self._insight(items),
            "summary": self._summary(items),
            "top_keywords": self._keywords(raw_texts),
            "items": items,
            "request": {"mode": "post_comments", "post_url": post_url, "limit": limit},
        }

    def compare_subreddits(self, subreddit_names, limit_per_sub=20, sort_by="hot"):
        if not self.reddit:
            return {"ok": False, "error": "Reddit credentials are not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."}

        comparison_items = []
        all_texts = []
        flattened = []

        for subreddit_name in subreddit_names:
            data = self.analyze_subreddit(subreddit_name, limit=limit_per_sub, sort_by=sort_by)
            if not data.get("ok"):
                return data
            comparison_items.append(
                {
                    "subreddit": subreddit_name,
                    "avg_sentiment": data["summary"]["avg_score"],
                    "positive_percentage": data["summary"]["positive_pct"],
                    "post_count": data["summary"]["total"],
                }
            )
            all_texts.extend(item["content"] for item in data["items"])
            for item in data["items"]:
                flattened.append({**item, "group": subreddit_name})

        return {
            "ok": True,
            "mode": "compare",
            "title": "Subreddit comparison",
            "insight": self._insight(flattened),
            "summary": self._summary(flattened),
            "comparisons": comparison_items,
            "top_keywords": self._keywords(all_texts),
            "items": flattened,
            "request": {"mode": "compare", "subreddits": subreddit_names, "limit_per_sub": limit_per_sub, "sort_by": sort_by},
        }


ENGINE = SentimentEngine()


class RequestHandler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_OPTIONS(self):
        self._json({}, status=204)

    def do_GET(self):
        if self.path == "/api/health":
            self._json({"ok": True, "message": "Reddit Lens API is running."})
            return
        self._json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self):
        if self.path != "/api/analyze":
            self._json({"ok": False, "error": "Not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            self._json({"ok": False, "error": "Invalid JSON body"}, status=400)
            return

        mode = body.get("mode", "custom_text")
        try:
            if mode == "subreddit":
                response = ENGINE.analyze_subreddit(body.get("subreddit", ""), limit=int(body.get("limit", 20)), sort_by=body.get("sort_by", "hot"))
            elif mode == "post_comments":
                response = ENGINE.analyze_post_comments(body.get("post_url", ""), limit=int(body.get("limit", 40)))
            elif mode == "compare":
                subreddits = body.get("subreddits", [])
                if isinstance(subreddits, str):
                    subreddits = [item.strip() for item in subreddits.split(",") if item.strip()]
                response = ENGINE.compare_subreddits(subreddits, limit_per_sub=int(body.get("limit_per_sub", 20)), sort_by=body.get("sort_by", "hot"))
            else:
                texts = body.get("texts") or []
                if isinstance(texts, str):
                    texts = [item.strip() for item in texts.split("\n") if item.strip()]
                if not texts and body.get("text"):
                    texts = [body.get("text")]
                response = ENGINE.analyze_custom_text(texts)

            self._json(response, status=200 if response.get("ok") else 400)
        except Exception as error:
            self._json({"ok": False, "error": str(error)}, status=500)


def main():
    port = int(os.getenv("PORT", "8001"))
    server = ThreadingHTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"Reddit Lens API running on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
