"""Machine-learning phishing email detector.

URLs are extracted as text only and are never opened or executed.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

SUSPICIOUS_KEYWORDS = {
    "urgent", "verify", "verification", "suspended", "suspend", "password",
    "credential", "login", "click", "immediately", "limited time", "refund",
    "invoice", "payment", "gift card", "security alert", "confirm", "unlock",
    "wire transfer", "social security", "bank account", "crypto", "penalty",
}
URL_PATTERN = re.compile(r"(?:(?:https?|ftp)://|www\.)[^\s<>\"']+", re.IGNORECASE)
IP_URL_PATTERN = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/|$)", re.IGNORECASE)
SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"}
SUSPICIOUS_TLDS = {".top", ".click", ".zip", ".review", ".country", ".work", ".download"}

@dataclass(frozen=True)
class EmailFeatures:
    url_count: int
    ip_url_count: int
    shortened_url_count: int
    suspicious_url_count: int
    text_length: int
    special_char_count: int
    uppercase_count: int
    exclamation_count: int
    suspicious_phrase_count: int
    urgent_phrase_count: int
    sensitive_request_count: int

    def as_array(self) -> list[float]:
        return [float(getattr(self, field)) for field in self.__dataclass_fields__]


def extract_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text or "")


def _url_domain(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return (parsed.hostname or "").lower()


def extract_features(text: str) -> EmailFeatures:
    text = text or ""
    lower = text.lower()
    urls = extract_urls(text)
    domains = [_url_domain(url) for url in urls]
    suspicious_url_count = sum(
        bool(
            IP_URL_PATTERN.search(url)
            or any(url.lower().endswith(tld) for tld in SUSPICIOUS_TLDS)
            or "@" in urlparse(url if "://" in url else f"http://{url}").path
        )
        for url in urls
    )
    suspicious_phrases = sum(1 for phrase in SUSPICIOUS_KEYWORDS if phrase in lower)
    urgent_phrases = sum(1 for phrase in ("urgent", "immediately", "within 24 hours", "act now", "last warning", "final notice") if phrase in lower)
    sensitive_requests = sum(1 for phrase in ("password", "credit card", "bank account", "security code", "one-time code", "social security") if phrase in lower)
    return EmailFeatures(
        url_count=len(urls), ip_url_count=sum(bool(IP_URL_PATTERN.search(url)) for url in urls),
        shortened_url_count=sum(domain in SHORTENER_DOMAINS for domain in domains),
        suspicious_url_count=suspicious_url_count, text_length=len(text),
        special_char_count=sum(not char.isalnum() and not char.isspace() for char in text),
        uppercase_count=sum(char.isupper() for char in text), exclamation_count=text.count("!"),
        suspicious_phrase_count=suspicious_phrases, urgent_phrase_count=urgent_phrases,
        sensitive_request_count=sensitive_requests,
    )


class EngineeredFeatureTransformer(BaseEstimator, TransformerMixin):
    """Convert email text to a small numeric matrix for FeatureUnion."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return csr_matrix(np.asarray([extract_features(str(item)).as_array() for item in X], dtype=float))


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1, max_features=5000)),
        ("engineered", EngineeredFeatureTransformer()),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def load_dataset(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"email_text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset must contain email_text and label columns.")
    data = data[["email_text", "label"]].dropna()
    data["email_text"] = data["email_text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data = data[(data["email_text"] != "") & data["label"].isin({"phishing", "safe"})]
    data = data.drop_duplicates(subset=["email_text"]).reset_index(drop=True)
    if data["label"].nunique() < 2:
        raise ValueError("Dataset must contain both phishing and safe examples.")
    return data


def train_model(data: pd.DataFrame) -> Pipeline:
    return build_pipeline().fit(data["email_text"], data["label"])


def save_model(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def explain_prediction(email_text: str, model: Pipeline) -> dict[str, Any]:
    features = extract_features(email_text)
    label = str(model.predict([email_text])[0])
    probabilities = model.predict_proba([email_text])[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(label)])
    indicators: list[str] = []
    if features.url_count: indicators.append(f"Contains {features.url_count} URL(s)")
    if features.ip_url_count: indicators.append("Contains an IP-address-based URL")
    if features.shortened_url_count: indicators.append("Contains a URL-shortening service")
    if features.suspicious_url_count: indicators.append("Contains a suspicious URL structure")
    if features.urgent_phrase_count: indicators.append("Uses urgent or threatening language")
    if features.sensitive_request_count: indicators.append("Requests sensitive information")
    if features.suspicious_phrase_count: indicators.append(f"Contains {features.suspicious_phrase_count} suspicious phrase(s)")
    if features.exclamation_count >= 3: indicators.append("Uses repeated exclamation marks")
    if not indicators: indicators.append("No strong hand-crafted warning indicators were detected")
    return {"label": label, "confidence": confidence, "indicators": indicators, "features": features}
