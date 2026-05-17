import html
import json
import os
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
import torch 
from ftfy import fix_text
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import InconsistentVersionWarning
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


BASE_DIR = Path(__file__).resolve().parent
TESTING_DIR = BASE_DIR / "Testing"
MODEL_DIR = TESTING_DIR / "best model"
TOKENIZER_SOURCE = "cardiffnlp/twitter-roberta-base-sentiment"

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

STOP_WORDS = set(stopwords.words("english")) - NEGATIONS


class RobertaTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None):
        self.stop_words = stop_words if stop_words is not None else STOP_WORDS
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = re.sub(r"http\S+|www\S+|https\S+", " http ", text)
        text = re.sub(r"@\w+", " @user ", text)
        text = text.replace("½", "0.5").replace("¼", "0.25").replace("¾", "0.75")
        text = fix_text(text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def transform(self, texts):
        return [self.clean_text(text) for text in texts]


class SentimentEngine:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.testing_dir = TESTING_DIR
        self.model_dir = MODEL_DIR
        self.tokenizer_source = TOKENIZER_SOURCE
        self.preprocessor = RobertaTextPreprocessor()
        self._ensure_tokenizer_assets()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True)
        self.id2label = {int(key): str(value).lower() for key, value in (getattr(self.model.config, "id2label", {}) or {}).items()}
        self.label_to_id = {value: key for key, value in self.id2label.items()}
        self.reddit = self._build_reddit_client()

    def test_reddit_connection(self):
        if not self.reddit:
            return {"ok": False, "message": "Reddit credentials are not configured."}

        try:
            subreddit = self.reddit.subreddit("python")
            next(subreddit.hot(limit=1))
            return {"ok": True, "message": "Reddit API connection is working."}
        except Exception as error:
            return {"ok": False, "message": f"Reddit API connection failed: {error}"}

    def _ensure_tokenizer_assets(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_file = self.model_dir / "tokenizer.json"
        if tokenizer_file.exists():
            return

        previous_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        try:
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_source)
            tokenizer.save_pretrained(self.model_dir)
        finally:
            if previous_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = previous_offline

    def _build_reddit_client(self):
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "script:reddit-lens-roberta:v1.0")
        if not client_id or not client_secret:
            return None
        return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    def _normalize_texts(self, texts):
        return [text if isinstance(text, str) else "" for text in texts if text is not None]

    def _classify_score(self, probabilities):
        positive = probabilities[self.label_to_id.get("positive")]
        negative = probabilities[self.label_to_id.get("negative")]
        return float(positive - negative)

    def _sentiment_from_probabilities(self, probabilities):
        best_index = int(np.argmax(probabilities))
        return self.id2label.get(best_index, f"label_{best_index}").capitalize()

    def _confidence(self, probabilities):
        return round(float(np.max(probabilities)) * 100, 2)

    def predict_texts(self, texts):
        normalized = self._normalize_texts(texts)
        if not normalized:
            return []

        cleaned = self.preprocessor.transform(normalized)
        encoded = self.tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():               # add import torch at top of file
            outputs = self.model(**encoded)
        probability_rows = softmax(outputs.logits.cpu().numpy(), axis=-1)

        results = []
        for text, probabilities in zip(normalized, probability_rows):
            sentiment_score = self._classify_score(probabilities)
            results.append(
                {
                    "text": text,
                    "sentiment_score": round(sentiment_score, 4),
                    "sentiment": self._sentiment_from_probabilities(probabilities),
                    "confidence": self._confidence(probabilities),
                    "probabilities": {
                        self.id2label.get(index, f"label_{index}").capitalize(): round(float(value) * 100, 2)
                        for index, value in enumerate(probabilities)
                    },
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
            tokens = word_tokenize(cleaned.lower())
            counter.update(token for token in tokens if token.isalpha() and token not in self.preprocessor.stop_words)
        return [{"term": term, "count": count} for term, count in counter.most_common(top_n)]

    def _insight(self, items):
        summary = self._summary(items)
        if not summary["total"]:
            return "No usable text was provided."
        if summary["positive_pct"] >= 55:
            return f"The sample leans positive at {summary['positive_pct']:.0f}%, with stronger approval in the RoBERTa output."
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
                    "title": f"Text {index}",
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
            self._json({"ok": True, "message": "Reddit Lens RoBERTa API is running.", "reddit": ENGINE.test_reddit_connection()})
            return
        if self.path == "/api/reddit-test":
            self._json(ENGINE.test_reddit_connection())
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
    print(f"Reddit Lens RoBERTa API running on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()