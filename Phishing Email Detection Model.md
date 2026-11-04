# Phishing Email Detection Model

A Python machine-learning application that classifies email text as **phishing** or **safe** using Scikit-learn, TF-IDF text features, engineered email indicators, and Logistic Regression.

The project includes a Tkinter desktop interface, a sample CSV dataset, model training and evaluation, explainable prediction indicators, model persistence with Joblib, and automated tests.

## Features

- Phishing or safe email classification
- TF-IDF word and bigram features
- Engineered features for:
  - Suspicious keywords
  - Urgent language
  - Sensitive-information requests
  - URL count
  - IP-address URLs
  - URL shorteners
  - Suspicious URL structures
  - Text length and punctuation
  - Uppercase characters and exclamation marks
- Logistic Regression classification pipeline
- Stratified train-test evaluation
- Saved model loading with Joblib
- Explainable prediction indicators
- Tkinter desktop GUI
- JSON metrics output and confusion-matrix visualization
- Pytest test suite

URLs are extracted as text only. The application never opens, visits, downloads from, or executes URLs found in email content.

## Technology Stack

- Python 3.11+
- Scikit-learn
- Pandas
- NumPy
- SciPy
- Joblib
- Matplotlib
- Seaborn
- Tkinter
- pytest

## Project Structure

```text
phishing-email-detection-model/
├── app.py                    # Tkinter desktop application
├── phishing_model.py         # Features, pipeline, training, prediction
├── train_model.py            # Evaluation, metrics, and visualization
├── requirements.txt          # Python dependencies
├── data/
│   └── sample_emails.csv     # Small synthetic demonstration dataset
├── tests/
│   └── test_phishing_model.py
├── models/                   # Generated model and evaluation files
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/phishing-email-detection-model.git
cd phishing-email-detection-model
```

Replace `your-username` with your GitHub username.

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Tkinter is included with most Python installations. On some Linux distributions, install it separately:

```bash
sudo apt install python3-tk
```

## Train and Evaluate the Model

Run the training script:

```bash
python train_model.py
```

This creates:

```text
models/phishing_model.joblib
models/metrics.json
models/confusion_matrix.png
```

The training script uses a stratified train-test split and reports accuracy, precision, recall, F1-score, class counts, and dataset size.

## Run the Application

```bash
python app.py
```

The application automatically loads the saved model if it exists. If no model exists, it trains one from the sample dataset.

To classify an email:

1. Paste email text into the text area.
2. Select **Classify Email**.
3. Review the predicted class and model confidence.
4. Review the indicators that influenced the interpretation.

The **Train / Refresh Model** button retrains the model and updates the saved evaluation files.

## Dataset

The included `data/sample_emails.csv` file is a small synthetic demonstration dataset with two required columns:

```text
email_text,label
```

Allowed labels are:

- `phishing`
- `safe`

The loader removes missing rows, removes duplicate email text, validates labels, and checks that both classes are present.

The sample dataset is included for immediate testing only. Its evaluation results are not representative of real-world phishing detection performance. For meaningful research, use a larger, representative, ethically sourced dataset and document its original source and license before publishing results.

## Testing

Run the automated tests with:

```bash
pytest -q
```

The tests cover dataset validation, feature extraction, URL non-fetching behavior, model training, prediction output, and custom pipeline usage.

## Machine Learning Design

The model uses a Scikit-learn `Pipeline` containing:

1. A `FeatureUnion`.
2. A `TfidfVectorizer` for word and bigram text features.
3. A custom engineered-feature transformer.
4. A Logistic Regression classifier.

The vectorizer is fitted only on the training split, which prevents text-feature leakage from the test data.

## Limitations

This project is an educational prototype and is not a replacement for enterprise email security systems. It does not inspect email attachments, authenticate sender domains, verify SPF/DKIM/DMARC, query threat-intelligence services, open URLs, or guarantee that any prediction is correct.

A machine-learning prediction should be treated as an additional signal. Users should verify suspicious messages through trusted channels and avoid clicking unexpected links or sharing credentials.

## License

This project is provided for educational and cybersecurity research purposes.
