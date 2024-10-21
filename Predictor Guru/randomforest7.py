import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Load and preprocess the data
file_path = '/Users/vedant/Desktop/college_finder/output_file.csv'
df = pd.read_csv(file_path)

# Cleaning and preprocessing
def clean_seat_type(seat_type):
    categories_to_modify = ['OPEN', 'OBC', 'NT1', 'NT2', 'NT3', 'VJ', 'SC', 'ST']
    if len(seat_type) > 2 and seat_type[1:-1] in categories_to_modify:
        return seat_type[1:-1]
    return seat_type

df['MODIFIED SEAT TYPE'] = df['SEAT TYPE'].apply(clean_seat_type)

# Prepare features and target
X = df[['INSTITUTE NAME', 'BRANCH NAME', 'MODIFIED SEAT TYPE', 'DISTRICT']]
y = df['CUT OFF (RANK)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['INSTITUTE NAME', 'BRANCH NAME', 'MODIFIED SEAT TYPE', 'DISTRICT'])
    ])

# Create pipeline for Decision Tree
dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Train and evaluate the model
print("Training Decision Tree model...")

# Perform cross-validation
cv_scores = cross_val_score(dt_pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
mean_cv_score = -cv_scores.mean()
print(f"Decision Tree CV MAE: {mean_cv_score:.2f} (+/- {cv_scores.std() * 2:.2f})")

# Train on full training set and evaluate on test set
dt_pipeline.fit(X_train, y_train)
y_pred = dt_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Decision Tree Test MAE: {mae:.2f}")
print(f"Decision Tree Test RMSE: {rmse:.2f}")

# Train the model on the full dataset
dt_pipeline.fit(X, y)

# Save the model
with open('decision_tree_rank_predictor_model.pkl', 'wb') as file:
    pickle.dump(dt_pipeline, file)

# Flask application
app = Flask(__name__)
CORS(app)

# Load the saved model
with open('decision_tree_rank_predictor_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Prepare options for dropdowns
seat_type_options = ['All'] + sorted(df['MODIFIED SEAT TYPE'].unique().tolist())
branch_options = ['All'] + sorted(df['BRANCH NAME'].unique().tolist())
district_options = ['All'] + sorted(df['DISTRICT'].unique().tolist())
college_options = ['All'] + sorted(df['INSTITUTE NAME'].unique().tolist())

def search_colleges(rank, seat_types, branches, districts, college_name, range_=300):
    df_filtered = df.copy()
    
    if seat_types and 'All' not in seat_types:
        df_filtered = df_filtered[df_filtered['MODIFIED SEAT TYPE'].isin(seat_types)]
    
    if branches and 'All' not in branches:
        df_filtered = df_filtered[df_filtered['BRANCH NAME'].isin(branches)]
    
    if districts and 'All' not in districts:
        df_filtered = df_filtered[df_filtered['DISTRICT'].isin(districts)]
    
    if college_name and college_name != 'All':
        df_filtered = df_filtered[df_filtered['INSTITUTE NAME'].str.contains(college_name, case=False, na=False)]
    
    result_df = df_filtered[(df_filtered['CUT OFF (RANK)'] >= rank - range_) & (df_filtered['CUT OFF (RANK)'] <= rank + range_)]
    
    if len(result_df) < 10:
        result_df = df_filtered.head(10)  # Ensure at least 10 results
    
    return result_df

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/cet')
def cet():
    return render_template('index.html', seat_type_options=seat_type_options, branch_options=branch_options, district_options=district_options, college_options=college_options)

@app.route('/search', methods=['POST'])
def search():
    rank = int(request.form['rank'])
    seat_types = request.form.getlist('seat_types')
    branches = request.form.getlist('branches')
    districts = request.form.getlist('districts')
    college_name = request.form.get('college_name', 'All')
    
    result_df = search_colleges(rank, seat_types, branches, districts, college_name)
    
    # Clean the DataFrame by removing newlines
    result_df = result_df.applymap(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    
    # Predict ranks for the result colleges
    predictions = loaded_model.predict(result_df[['INSTITUTE NAME', 'BRANCH NAME', 'MODIFIED SEAT TYPE', 'DISTRICT']])
    result_df['Predicted Rank'] = predictions.round().astype(int)
    
    # Select only the columns we want to display
    columns_to_display = ['INSTITUTE CODE', 'INSTITUTE NAME', 'BRANCH NAME', 'MODIFIED SEAT TYPE', 'DISTRICT', 'CUT OFF (RANK)', 'Predicted Rank']
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