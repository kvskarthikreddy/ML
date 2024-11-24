import numpy as np
from flask import Flask, request, render_template
import pickle

# Create Flask app
flask_app = Flask(__name__)

# Load the trained model
model = pickle.load(open("Random Forest with Flask/model.pkl", "rb"))

# Home route to render the main page
@flask_app.route("/")
def Home():
    return render_template("index.html")

# Predict route to handle form submissions and make predictions
@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect and convert the input features from the form
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        
        # Predict using the trained model
        prediction = model.predict(features)
        
        # Return the result to the user
        return render_template("index.html", prediction_text="The flower species is: {}".format(prediction[0]))
    
    except Exception as e:
        return render_template("index.html", prediction_text="Error in prediction: {}".format(str(e)))

if __name__ == "__main__":
    flask_app.run(debug=True)

