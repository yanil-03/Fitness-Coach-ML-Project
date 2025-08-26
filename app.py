from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import db_utils, encoders_util
app = Flask(__name__)

# Load the model once
diet_model = joblib.load('rf_diet_model.pkl')
exercise_model = joblib.load('rf_ex_model.pkl')


@app.route('/')
def home():
    return render_template('frontend.html')  # Render the frontend page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # input_data = np.array(data['input']).reshape(1, -1)
        if "user_features" not in data:
            return jsonify({"error": "Invalid input format. 'user_features' key missing"}), 400

        print("data received")
        features = data["user_features"]

        # Convert to numpy array for model prediction
        input_features = np.array([
            features["Sex"],
            features["Age"],
            features["Height"],
            features["Weight"],
            features["Hypertension"],
            features["Diabetes"],
            features["BMI"],
            features["Level"],
            features["Fitness_Goal"],
            features["Fitness_Type"]
        ]).reshape(1, -1)

        #prediction
        exercise_pred = exercise_model.predict(input_features)[0]
        diet_pred = diet_model.predict(input_features)[0]

        #decoding
        decoded_exercise = encoders_util.execise_lb.inverse_transform([exercise_pred])[0]
        decoded_diet = encoders_util.diet_lb.inverse_transform([diet_pred])[0]
        

        db_utils.save_user_prediction(input_features, decoded_exercise, decoded_diet)
        
        # Convert numpy array of outputs to a list for JSON serialization
        return jsonify({
            'exercise_prediction': decoded_exercise,
            'diet_prediction': decoded_diet
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)


