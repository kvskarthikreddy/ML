import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Sample dataset
data = {
    "text": ["I love this product", "This is terrible", "Amazing quality!", "Not good", "Excellent service", "Worst experience"],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Create and train the model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df["text"], df["label"])

# Save the model
joblib.dump(model, "sentiment_model.pkl")

print("Model trained and saved!")
