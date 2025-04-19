import pandas as pd
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ✅ Load dataset
df = pd.read_csv("data/imdb_reviews.csv")

# ✅ Map sentiments to binary
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ✅ Clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    return text.lower()

df['text'] = df['review'].apply(clean_text)

# ✅ Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# ✅ Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(X_train, y_train)
print("✅ Accuracy:", pipeline.score(X_test, y_test))

# ✅ Save model
os.makedirs("backend/model", exist_ok=True)
joblib.dump(pipeline, "backend/model/sentiment_model.pkl")
print("✅ Model saved to backend/model/sentiment_model.pkl")
