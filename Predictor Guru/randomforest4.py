from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

app = Flask(__name__)
CORS(app)

# Load and preprocess the data
file_path = '/Users/vedant/Desktop/college_finder/output_file.csv'
df = pd.read_csv(file_path)

# Cleaning and preprocessing
def clean_seat_type(seat_type):
    categories_to_modify = ['OPEN', 'OBC', 'NT1', 'NT2', 'NT3', 'VJ', 'SC', 'ST']
    gender = 'All'
    original_seat_type = seat_type
    
    if seat_type.startswith('G'):
        gender = 'All'
        seat_type = seat_type[1:]
    elif seat_type.startswith('L'):
        gender = 'Female'
        seat_type = seat_type[1:]
    
    if len(seat_type) > 2 and seat_type[1:-1] in categories_to_modify:
        return seat_type[1:-1], gender, original_seat_type
    return seat_type, gender, original_seat_type

df['MODIFIED SEAT TYPE'], df['GENDER'], df['ORIGINAL SEAT TYPE'] = zip(*df['SEAT TYPE'].apply(clean_seat_type))

# Function to encode categorical variables
def encode_categorical(df, column):
    le = LabelEncoder()
    df[f'ENCODED_{column}'] = le.fit_transform(df[column])
    return le

# Encode categorical variables
institute_le = encode_categorical(df, 'INSTITUTE NAME')
branch_le = encode_categorical(df, 'BRANCH NAME')
seat_type_le = encode_categorical(df, 'MODIFIED SEAT TYPE')
district_le = encode_categorical(df, 'DISTRICT')

# Prepare features and target
X = df[['ENCODED_INSTITUTE NAME', 'ENCODED_BRANCH NAME', 'ENCODED_MODIFIED SEAT TYPE', 'ENCODED_DISTRICT']]
y = df['CUT OFF (RANK)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoders
with open('rank_predictor_model.pkl', 'wb') as file:
    pickle.dump({
        'model': model,
        'institute_le': institute_le,
        'branch_le': branch_le,
        'seat_type_le': seat_type_le,
        'district_le': district_le
    }, file)

# Load the model and label encoders
with open('rank_predictor_model.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    loaded_model = loaded_data['model']
    institute_le = loaded_data['institute_le']
    branch_le = loaded_data['branch_le']
    seat_type_le = loaded_data['seat_type_le']
    district_le = loaded_data['district_le']

# Sort options
seat_type_options = ['All'] + sorted(df['MODIFIED SEAT TYPE'].unique().tolist())
branch_options = ['All'] + sorted(df['BRANCH NAME'].unique().tolist())
district_options = ['All'] + sorted(df['DISTRICT'].unique().tolist())
college_options = ['All'] + sorted(df['INSTITUTE NAME'].unique().tolist())
gender_options = ['All', 'Male', 'Female']

def search_colleges(rank, seat_types, branches, districts, college_name, gender, range_=300):
    df_filtered = df.copy()
    
    if seat_types and 'All' not in seat_types:
        df_filtered = df_filtered[df_filtered['MODIFIED SEAT TYPE'].isin(seat_types)]
    
    if branches and 'All' not in branches:
        df_filtered = df_filtered[df_filtered['BRANCH NAME'].isin(branches)]
    
    if districts and 'All' not in districts:
        df_filtered = df_filtered[df_filtered['DISTRICT'].isin(districts)]
    
    if college_name and college_name != 'All':
        df_filtered = df_filtered[df_filtered['INSTITUTE NAME'].str.contains(college_name, case=False, na=False)]
    
    if gender != 'All':
        # Use the original seat type for gender filtering
        if gender == 'Female':
            df_filtered = df_filtered[df_filtered['ORIGINAL SEAT TYPE'].str.startswith('L')]
        else:  # Male
            df_filtered = df_filtered[~df_filtered['ORIGINAL SEAT TYPE'].str.startswith('L')]
    
    result_df = df_filtered[(df_filtered['CUT OFF (RANK)'] >= rank - range_) & (df_filtered['CUT OFF (RANK)'] <= rank + range_)]
    
    if len(result_df) < 10:
        result_df = df_filtered.head(10)  # Ensure at least 10 results
    
    return result_df

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/cet')
def cet():
    return render_template('index.html', 
                           seat_type_options=seat_type_options, 
                           branch_options=branch_options, 
                           district_options=district_options, 
                           college_options=college_options,
                           gender_options=gender_options)

@app.route('/search', methods=['POST'])
def search():
    rank = int(request.form['rank'])
    seat_types = request.form.getlist('seat_types')
    branches = request.form.getlist('branches')
    districts = request.form.getlist('districts')
    college_name = request.form.get('college_name', 'All')
    gender = request.form.get('gender', 'All')
    
    result_df = search_colleges(rank, seat_types, branches, districts, college_name, gender)
    
    # Clean the DataFrame by removing newlines
    result_df = result_df.applymap(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    
    # Predict ranks for the result colleges
    predictions = []
    for _, row in result_df.iterrows():
        features = np.array([[
            institute_le.transform([row['INSTITUTE NAME']])[0] if row['INSTITUTE NAME'] in institute_le.classes_ else -1,
            branch_le.transform([row['BRANCH NAME']])[0] if row['BRANCH NAME'] in branch_le.classes_ else -1,
            seat_type_le.transform([row['MODIFIED SEAT TYPE']])[0] if row['MODIFIED SEAT TYPE'] in seat_type_le.classes_ else -1,
            district_le.transform([row['DISTRICT']])[0] if row['DISTRICT'] in district_le.classes_ else -1
        ]])
        
        # Use SimpleImputer to handle any -1 values (unseen categories)
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        features = imputer.fit_transform(features)
        
        predicted_rank = loaded_model.predict(features)[0]
        predictions.append(round(predicted_rank))
    
    result_df['Predicted Rank'] = predictions
    
    # Select only the columns we want to display
    columns_to_display = ['INSTITUTE CODE', 'INSTITUTE NAME', 'BRANCH NAME', 'MODIFIED SEAT TYPE', 'ORIGINAL SEAT TYPE', 'GENDER', 'DISTRICT', 'CUT OFF (RANK)', 'Predicted Rank']
    result_df = result_df[columns_to_display]
    
    # Convert DataFrame to HTML with additional parameters
    result_table = result_df.to_html(index=False,
                                     classes='table table-striped table-bordered',
                                     justify='center',
                                     escape=False)
    
    return render_template('results.html', table=result_table)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

if __name__ == '__main__':
    app.run(debug=True)