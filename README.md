Phishing Email Detection Model

A Python machine-learning application that classifies email messages as Phishing or Safe.

The project uses Scikit-learn, TF-IDF text features, engineered email indicators, and Logistic Regression. It includes a Tkinter desktop interface for entering email content and viewing predictions.

Features

•
Classifies emails as phishing or safe

•
Uses TF-IDF text analysis

•
Detects suspicious keywords and urgent language

•
Detects requests for passwords, codes, and financial information

•
Analyzes URLs without opening or visiting them

•
Detects IP-based URLs and URL shorteners

•
Provides prediction confidence and explanation indicators

•
Saves and loads the trained model with Joblib

•
Generates evaluation metrics and a confusion matrix

•
Includes automated tests with pytest

Technologies

•
Python 3.11+

•
Scikit-learn

•
Pandas

•
NumPy

•
SciPy

•
Joblib

•
Tkinter

•
Matplotlib

•
Seaborn

•
pytest

Project Structure

Plain Text


phishing-email-detection-model/
├── app.py
├── phishing_model.py
├── train_model.py
├── requirements.txt
├── data/
│   └── sample_emails.csv
├── tests/
│   └── test_phishing_model.py
└── README.md



Installation

Clone the repository:

Bash


git clone https://github.com/your-username/phishing-email-detection-model.git
cd phishing-email-detection-model



Create a virtual environment:

Windows

Bash


python -m venv .venv
.venv\Scripts\activate



macOS/Linux

Bash


python3 -m venv .venv
source .venv/bin/activate



Install the dependencies:

Bash


pip install -r requirements.txt



Train the Model

Bash


python train_model.py



This creates the trained model, evaluation metrics, and confusion matrix inside the models folder.

Run the Application

Bash


python app.py



Paste an email into the application and select Classify Email to view the prediction and explanation.

Run Tests

Bash


pytest -q



Dataset

The included data/sample_emails.csv file contains example phishing and safe emails for demonstration and testing.

The sample dataset is small and synthetic. Its results are not representative of real-world phishing detection performance. For production use, train the model with a larger, diverse, and properly licensed dataset.

Safety Notes

•
Email URLs are analyzed as text only.

•
The application never opens or visits URLs.

•
Predictions are not guaranteed to be correct.

•
Do not share passwords or sensitive personal information while testing.

•
Suspicious emails should be verified through a trusted communication channel.

License

This project is provided for educational purposes.

