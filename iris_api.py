from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to disk if not already saved
if not os.path.exists("model.pkl"):
    joblib.dump(model, "model.pkl")

    # Load model
    model = joblib.load("model.pkl")

    # Initialize Flask app
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "🌼 Iris Classifier API is running!"

        @app.route('/predict', methods=['POST'])
        def predict():
            data = request.get_json(force=True)
                 
# Extract features
            try:
               features = [data['sepal_length'],data['sepal_width'],data['petal_length'],data['petal_width']]
            except KeyError as e:
               return jsonify({'error': f'Missing key: {str(e)}'}), 400
# Make prediction
            prediction = model.predict([features])[0]
            species = target_names[prediction]
            return jsonify({'predicted_species': species})
if __name__ == '__main__':
   app.run(debug=True)