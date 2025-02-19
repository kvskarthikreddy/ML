from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("sentiment_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        text_input = request.form["text_input"]
        prediction = model.predict([text_input])[0]
        prediction = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
