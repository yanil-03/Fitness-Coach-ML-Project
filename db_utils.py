from sqlalchemy import create_engine
import pandas as pd
import encoders_util, secret_config



# Create and return SQL connection engine
def get_engine():
    return create_engine(secret_config.config)

# Load and convert the diet.csv file to sql table
# df = pd.read_csv('diet.csv')
# engine = get_engine() 
# # df.to_sql('diet_table', con=engine, if_exists='replace', index=False)
# chunksize = 1000  # adjust as needed
# for chunk in pd.read_csv('diet.csv', chunksize=chunksize):
#     chunk.to_sql('diet_table', con=engine, if_exists='append', index=False)


# print("Converted!!!")

# Load data from diet_table
def load_diet_data():
    engine = get_engine()
    query = 'SELECT * FROM diet_table'
    return pd.read_sql(query, engine)

# print(load_diet_data().head())

# Save prediction record
def save_user_prediction(input_data, pred_exercise, pred_diet):
    engine = get_engine()
    # df = pd.DataFrame([data_dict])
    feature_columns = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']

    try:
        df = pd.DataFrame(input_data, columns=feature_columns)
        df['Predicted Exercise'] = pred_exercise
        df['Predicted Diet'] = pred_diet

        df['Diabetes'] = encoders_util.diabetes_lb.inverse_transform(df['Diabetes'].astype(int))
        df['Fitness Goal'] = encoders_util.goal_lb.inverse_transform(df['Fitness Goal'].astype(int))
        df['Fitness Type'] = encoders_util.type_lb.inverse_transform(df['Fitness Type'].astype(int))
        df['Hypertension'] = encoders_util.hypertension_lb.inverse_transform(df['Hypertension'].astype(int))
        df['Level'] = encoders_util.level_lb.inverse_transform(df['Level'].astype(int))
        df['Sex'] = encoders_util.sex_lb.inverse_transform(df['Sex'].astype(int))

        df.to_sql('user_predictions', engine, if_exists='append', index=False)
        print(f"Saved record: \n\n{df}")

    except Exception as e:
        print(f"Error saving user prediction record: {e}")
    # df = pd.DataFrame(input_data, columns= feature_columns)
    # df['Predicted Exercise'] = pred_exercise
    # df['Predicted Diet'] = pred_diet

    # df['Diabetes'] = encoders_util.diabetes_lb.inverse_transform(df['Diabetes'].astype(int))
    # df['Fitness Goal'] = encoders_util.goal_lb.inverse_transform(df['Fitness Goal'].astype(int))
    # df['Fitness Type'] = encoders_util.type_lb.inverse_transform(df['Fitness Type'].astype(int))
    # df['Hypertension'] = encoders_util.hypertension_lb.inverse_transform(df['Hypertension'].astype(int))
    # df['Level'] = encoders_util.level_lb.inverse_transform(df['Level'].astype(int))
    # df['Sex'] = encoders_util.sex_lb.inverse_transform(df['Sex'].astype(int))
    
    # df.to_sql('user_predictions', engine, if_exists='append', index=False)
    # print(f"Saved record: \n\n{df}")