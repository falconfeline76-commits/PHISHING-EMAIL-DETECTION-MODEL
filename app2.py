from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from phishing_model import explain_prediction, load_dataset, load_model, save_model, train_model
from train_model import train_and_evaluate

BASE_DIR = Path(__file__).parent
DATASET = BASE_DIR / "data" / "sample_emails.csv"
MODEL = BASE_DIR / "models" / "phishing_model.joblib"
METRICS = BASE_DIR / "models" / "metrics.json"

class PhishingDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phishing Email Detection Model")
        self.geometry("1050x720")
        self.minsize(850, 600)
        self.configure(bg="#081426")
        self.model = None
        self._styles()
        self._ui()
        self._load_or_train()

    def _styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#081426")
        style.configure("Card.TFrame", background="#10243d")
        style.configure("TLabel", background="#081426", foreground="#e8f1fa", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#081426", foreground="#ffffff", font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel", background="#081426", foreground="#9db3c9", font=("Segoe UI", 11))
        style.configure("Section.TLabel", background="#10243d", foreground="#ffffff", font=("Segoe UI", 12, "bold"))
        style.configure("TButton", padding=8, font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background="#1477b8", foreground="white")
        style.map("Accent.TButton", background=[("active", "#1b91d6")])

    def _ui(self):
        root = ttk.Frame(self, padding=24)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="PHISHING EMAIL DETECTION MODEL", style="Title.TLabel").pack(anchor="w")
        ttk.Label(root, text="Machine-learning classification with understandable prediction indicators", style="Subtitle.TLabel").pack(anchor="w", pady=(3, 16))
        notice = tk.Label(root, text="SAFETY NOTICE  •  Email text is analyzed locally. URLs are extracted as text only and are never opened or executed.", bg="#123454", fg="#bde4ff", padx=12, pady=10, anchor="w", font=("Segoe UI", 10, "bold"))
        notice.pack(fill="x", pady=(0, 14))

        input_card = ttk.Frame(root, style="Card.TFrame", padding=14)
        input_card.pack(fill="both", expand=True)
        ttk.Label(input_card, text="Email content", style="Section.TLabel").pack(anchor="w")
        self.email_text = tk.Text(input_card, height=12, wrap="word", bg="#0b1c30", fg="#edf5fb", insertbackground="white", relief="flat", padx=10, pady=10, font=("Segoe UI", 11))
        self.email_text.pack(fill="both", expand=True, pady=(8, 10))
        actions = ttk.Frame(input_card, style="Card.TFrame")
        actions.pack(fill="x")
        ttk.Button(actions, text="Classify Email", style="Accent.TButton", command=self.classify).pack(side="left")
        ttk.Button(actions, text="Clear", command=lambda: self.email_text.delete("1.0", "end")).pack(side="left", padx=8)
        ttk.Button(actions, text="Train / Refresh Model", command=self.train_refresh).pack(side="right")
        ttk.Button(actions, text="Load Email File", command=self.load_file).pack(side="right", padx=8)

        result_card = ttk.Frame(root, style="Card.TFrame", padding=14)
        result_card.pack(fill="x", pady=(14, 0))
        ttk.Label(result_card, text="Prediction and explanation", style="Section.TLabel").pack(anchor="w")
        self.result_label = tk.Label(result_card, text="Enter email content and select Classify Email.", bg="#10243d", fg="#b6c8d9", font=("Segoe UI", 17, "bold"), anchor="w")
        self.result_label.pack(fill="x", pady=(10, 5))
        self.confidence_label = ttk.Label(result_card, text="")
        self.confidence_label.pack(anchor="w")
        self.indicators = tk.Listbox(result_card, height=5, bg="#0b1c30", fg="#d9e7f2", relief="flat", highlightthickness=0, font=("Segoe UI", 10))
        self.indicators.pack(fill="x", pady=(8, 0))
        self.status_var = tk.StringVar(value="Loading model…")
        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", pady=(10, 0))

    def _load_or_train(self):
        try:
            if MODEL.exists(): self.model = load_model(MODEL)
            else:
                self.model = train_model(load_dataset(DATASET))
                save_model(self.model, MODEL)
            self.status_var.set("Model ready. Sample dataset metrics are demonstration-only.")
        except Exception as exc:
            self.status_var.set("Model unavailable.")
            messagebox.showerror("Model error", str(exc))

    def classify(self):
        text = self.email_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Missing email", "Enter email text before classification.")
            return
        if self.model is None:
            messagebox.showerror("Model unavailable", "Train or load a model first.")
            return
        result = explain_prediction(text, self.model)
        label = result["label"].upper()
        color = "#ff727c" if label == "PHISHING" else "#58df9a"
        self.result_label.configure(text=f"{label}", fg=color)
        self.confidence_label.configure(text=f"Model confidence: {result['confidence']:.1%}")
        self.indicators.delete(0, "end")
        for indicator in result["indicators"]: self.indicators.insert("end", "• " + indicator)

    def train_refresh(self):
        try:
            metrics = train_and_evaluate(str(DATASET), str(MODEL), str(METRICS), str(BASE_DIR / "models" / "confusion_matrix.png"))
            self.model = load_model(MODEL)
            self.status_var.set(f"Model refreshed using {metrics['dataset_rows']} sample rows. Metrics are not real-world performance guarantees.")
            messagebox.showinfo("Training complete", f"Model saved. Dataset accuracy on the held-out sample split: {metrics['accuracy']:.1%}\n\nUse larger, representative data before drawing conclusions.")
        except Exception as exc:
            messagebox.showerror("Training error", str(exc))

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self.email_text.delete("1.0", "end")
            self.email_text.insert("1.0", Path(path).read_text(encoding="utf-8", errors="replace"))

if __name__ == "__main__":
    PhishingDetectorApp().mainloop()
