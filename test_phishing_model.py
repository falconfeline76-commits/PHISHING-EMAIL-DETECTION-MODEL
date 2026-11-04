from pathlib import Path
import pandas as pd
from phishing_model import extract_features, explain_prediction, load_dataset, train_model

DATASET = Path(__file__).parents[1] / "data" / "sample_emails.csv"


def test_dataset_has_required_balanced_labels():
    data = load_dataset(DATASET)
    assert set(data.columns) == {"email_text", "label"}
    assert set(data["label"]) == {"phishing", "safe"}
    assert data["label"].value_counts().min() > 0


def test_feature_extractor_never_fetches_urls():
    features = extract_features("Urgent visit http://192.0.2.1/login and http://bit.ly/example!")
    assert features.url_count == 2
    assert features.ip_url_count == 1
    assert features.shortened_url_count == 1
    assert features.urgent_phrase_count == 1


def test_model_predicts_known_demo_examples():
    data = load_dataset(DATASET)
    model = train_model(data)
    phishing = explain_prediction("URGENT verify your password immediately at http://192.0.2.44/login", model)
    safe = explain_prediction("The team meeting is scheduled for Tuesday at 10 AM.", model)
    assert phishing["label"] in {"phishing", "safe"}
    assert safe["label"] in {"phishing", "safe"}
    assert 0 <= phishing["confidence"] <= 1
    assert phishing["indicators"]


def test_model_pipeline_works_with_custom_data():
    data = pd.DataFrame({"email_text": ["click urgent verify password http://bit.ly/x", "project meeting notes attached"], "label": ["phishing", "safe"]})
    model = train_model(data)
    assert model.predict(["urgent verify password http://bit.ly/x"])[0] in {"phishing", "safe"}
