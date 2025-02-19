from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


try:
    model = joblib.load("sentiment_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    sentiment_class = ""

    if request.method == "POST":
        text_input = request.form.get("text_input", "").strip()

        if not text_input:
            prediction = "Please enter some text!"
            sentiment_class = "negative"
        elif model:
            pred = model.predict([text_input])[0]
            if pred == 1:
                prediction = "Positive 😊"
                sentiment_class = "positive"
            else:
                prediction = "Negative 😞"
                sentiment_class = "negative"
        else:
            prediction = "Model not loaded. Please try again later."
            sentiment_class = "negative"

    return render_template("index.html", prediction=prediction, sentiment_class=sentiment_class)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

