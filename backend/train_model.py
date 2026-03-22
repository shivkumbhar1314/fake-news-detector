print("Training script started")

import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")

df = pd.read_csv("train.csv")

print("Dataset loaded")
print("Rows:", len(df))

df = df.fillna("")

df["content"] = df["title"] + " " + df["author"] + " " + df["text"]

X = df["content"]
y = df["label"]

print("Vectorizing text")

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

print("Splitting data")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model")

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)

print("Accuracy:", score)

os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model saved successfully")