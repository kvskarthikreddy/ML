from transformers import pipeline

# Load pre-trained emotion detection model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

def predict_emotion(text):
    result = emotion_classifier(text)
    return {entry['label']: entry['score'] for entry in result[0]}
