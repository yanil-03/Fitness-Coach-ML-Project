# from flask import Flask, request, jsonify
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load your trained ML models (replace with actual model paths)
# exercise_model = pickle.load(open("exercise_model.pkl", "rb"))
# diet_model = pickle.load(open("diet_model.pkl", "rb"))

# # Example label decoding (replace with your encoder if you used LabelEncoder)
# exercise_decoder = {
#     0: "Walking / Light Yoga",
#     1: "Cardio + Jogging",
#     2: "Strength Training",
#     3: "Mixed Functional Workouts"
# }

# diet_decoder = {
#     0: "Low-calorie diet (high fiber, avoid sugar & fried food)",
#     1: "Balanced maintenance diet (protein, carbs, fats in moderation)",
#     2: "High-protein diet (chicken, paneer, lentils, nuts, eggs)"
# }


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()

#         if "user_features" not in data:
#             return jsonify({"error": "Invalid input format. 'user_features' key missing"}), 400

#         features = data["user_features"]

#         # Convert to numpy array for model prediction
#         input_features = np.array([
#             features["Sex"],
#             features["Age"],
#             features["Height"],
#             features["Weight"],
#             features["Hypertension"],
#             features["Diabetes"],
#             features["BMI"],
#             features["Level"],
#             features["Fitness_Goal"],
#             features["Fitness_Type"]
#         ]).reshape(1, -1)

#         # Predictions
#         exercise_pred = int(exercise_model.predict(input_features)[0])
#         diet_pred = int(diet_model.predict(input_features)[0])

#         # Decode labels
#         decoded_exercise = exercise_decoder.get(exercise_pred, "Custom Exercise Plan")
#         decoded_diet = diet_decoder.get(diet_pred, "Custom Diet Plan")

#         return jsonify({
#             "exercise_prediction": exercise_pred,
#             "diet_prediction": diet_pred,
#             "decoded_exercise": decoded_exercise,
#             "decoded_diet": decoded_diet
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True)
import encoders_util, db_utils
import numpy as np
import pandas as pd


df = db_utils.load_diet_data()
df_en = df.copy()
# print(df.columns)
df_en['Sex'] = encoders_util.sex_lb.fit_transform(df_en['Sex'])

print(df['Sex'].value_counts(),'\n', df_en['Sex'].value_counts())


df_en['Hypertension'] = encoders_util.sex_lb.fit_transform(df_en['Hypertension'])

print(df['Hypertension'].value_counts(),'\n', df_en['Hypertension'].value_counts())


df_en['Diabetes'] = encoders_util.sex_lb.fit_transform(df_en['Diabetes'])

print(df['Diabetes'].value_counts(),'\n', df_en['Diabetes'].value_counts())


df_en['Level'] = encoders_util.sex_lb.fit_transform(df_en['Level'])

print(df['Level'].value_counts(),'\n', df_en['Level'].value_counts())


df_en['Fitness Goal'] = encoders_util.sex_lb.fit_transform(df_en['Fitness Goal'])

print(df['Fitness Goal'].value_counts(),'\n', df_en['Fitness Goal'].value_counts())


df_en['Fitness Type'] = encoders_util.sex_lb.fit_transform(df_en['Fitness Type'])

print(df['Fitness Type'].value_counts(),'\n', df_en['Fitness Type'].value_counts())