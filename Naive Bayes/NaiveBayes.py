import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Define dataset with 'text' and 'label' columns
data = {
    'text': [
        'Free money now!!!',
        'Hi, how are you?',
        'Win a free ticket to Bahamas',
        'Your invoice is attached',
        'Call me when you are free',
        'Congratulations, you won!',
        'Let’s grab coffee tomorrow.',
        'Get paid to work from home!'
    ],
    'label': [
        'spam',  # Labels: 'spam' and 'ham' (ham is non-spam)
        'ham',
        'spam',
        'ham',
        'ham',
        'spam',
        'ham',
        'spam'
    ]
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Define the feature (X) and target (y) variables
X = df['text']
y = df['label']

# Split data into training and test sets (25% test size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize TF-IDF Vectorizer (converts text to numerical format)
tfidf = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform the test data using the already fitted TF-IDF Vectorizer
X_test_tfidf = tfidf.transform(X_test)

# Initialize the Naive Bayes model for classification
model = MultinomialNB()

# Fit the model on the training data
model.fit(X_train_tfidf, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test_tfidf)

# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report (precision, recall, F1-score)
print('Classification Report:')
print(classification_report(y_test, y_pred))
