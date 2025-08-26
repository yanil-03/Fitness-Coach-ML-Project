import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import db_utils

# Load data
# df = pd.read_csv('diet.csv')
df = db_utils.load_diet_data()
# print(df.head(3))
# Function to label encode categorical columns except targets

def encoder(df, cols):
    for col in cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            # Save the LabelEncoder with a specific name
            filename = f"{'lb_'}{col}.pkl"
            joblib.dump(le, filename)
    return df

# Encode target columns separately with separate LabelEncoders
df = encoder(df, df.columns)


# Prepare features and targets
y_exercise = df['Exercises']
X = df.drop(columns=['ID', 'Exercises', 'Diet'])
print(X.columns)

y_diet = df['Diet']

# Split dataset
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_diet, test_size=0.2, random_state=42)

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X, y_exercise, test_size=0.2, random_state=42)


# Best parameters for Diet: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

rf_diet = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

rf_diet.fit(X_train_d, y_train_d)

# print("Best parameters for Diet:", grid_diet.best_params_)

# Evaluate Diet Model
y_pred_d = rf_diet.predict(X_test_d)
print("Diet Accuracy:", accuracy_score(y_test_d, y_pred_d))
print(classification_report(y_test_d, y_pred_d))


# Best parameters for Exercises: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
rf_ex = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf_ex.fit(X_train_e, y_train_e)

# Evaluate Exercises Model
y_pred_e = rf_ex.predict(X_test_e)
print("Exercises Accuracy:", accuracy_score(y_test_e, y_pred_e))
print(classification_report(y_test_e, y_pred_e))



joblib.dump(rf_diet, "rf_diet_model.pkl")
joblib.dump(rf_ex, "rf_ex_model.pkl")
print("Models Saved Successfully!!")