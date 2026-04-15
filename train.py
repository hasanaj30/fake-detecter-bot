import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🚀 TRAINING STARTED")

# =========================
# NLTK SETUP
# =========================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# =========================
# CLEAN TEXT FUNCTION
# =========================
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# =========================
# LOAD DATA
# =========================
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Fake shape:", fake.shape)
print("True shape:", true.shape)

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine
df = pd.concat([fake, true])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title + text
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

# 🔥 IMPORTANT: Reduce size for speed + stability
df = df.sample(n=15000, random_state=42)

# Clean text
df["content"] = df["content"].apply(clean_text)

print("✅ Text cleaning done")

# =========================
# SPLIT DATA
# =========================
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TF-IDF VECTORIZER
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("✅ Vectorization done")

# =========================
# MODEL (BEST FOR TEXT)
# =========================
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

print("✅ Model training done")

# =========================
# EVALUATE
# =========================
pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)

print(f"🎯 Accuracy: {acc*100:.2f}%")

# =========================
# SAVE MODEL
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model & vectorizer saved successfully")