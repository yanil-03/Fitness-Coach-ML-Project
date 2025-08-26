import joblib, os

lb_names = []
# Function that loads all pkl files of the encoders with prefix lb 
def load_encoders():
    loaded_objects = {}
    folder_path = '.'
    for filename in os.listdir(folder_path):
        # Check filename starts with lb and ends with .pkl
        if filename.startswith('lb') and filename.endswith('.pkl'):
            full_path = os.path.join(folder_path, filename)
            obj = joblib.load(full_path)
            loaded_objects[filename] = obj
            lb_names.append(filename)
            # print(f"Loaded: {filename}")
    return loaded_objects

# encoder = load_encoders()
# diabetes_lb =encoder['lb_Diabetes.pkl']
# goal_lb =encoder['lb_Fitness Goal.pkl']
# type_lb =encoder['lb_Fitness Type.pkl']
# hypertension_lb = encoder['lb_Hypertension.pkl']
# level_lb = encoder['lb_Level.pkl']
# sex_lb = encoder['lb_Sex.pkl']
# execise_lb = encoder['lb_Exercises.pkl']
# diet_lb = encoder['lb_Diet.pkl']

encoder = load_encoders()

try:
    diabetes_lb = encoder['lb_Diabetes.pkl']
except KeyError:
    print("Error: 'lb_Diabetes.pkl' encoder not found.")
    diabetes_lb = None

try:
    goal_lb = encoder['lb_Fitness Goal.pkl']
except KeyError:
    print("Error: 'lb_Fitness Goal.pkl' encoder not found.")
    goal_lb = None

try:
    type_lb = encoder['lb_Fitness Type.pkl']
except KeyError:
    print("Error: 'lb_Fitness Type.pkl' encoder not found.")
    type_lb = None

try:
    hypertension_lb = encoder['lb_Hypertension.pkl']
except KeyError:
    print("Error: 'lb_Hypertension.pkl' encoder not found.")
    hypertension_lb = None

try:
    level_lb = encoder['lb_Level.pkl']
except KeyError:
    print("Error: 'lb_Level.pkl' encoder not found.")
    level_lb = None

try:
    sex_lb = encoder['lb_Sex.pkl']
except KeyError:
    print("Error: 'lb_Sex.pkl' encoder not found.")
    sex_lb = None

try:
    execise_lb = encoder['lb_Exercises.pkl']
except KeyError:
    print("Error: 'lb_Exercises.pkl' encoder not found.")
    execise_lb = None

try:
    diet_lb = encoder['lb_Diet.pkl']
except KeyError:
    print("Error: 'lb_Diet.pkl' encoder not found.")
    diet_lb = None
