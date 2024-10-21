from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the Excel file
file_path = '/Users/vedant/Desktop/college_finder/output_file.csv'
df = pd.read_csv(file_path)

# Cleaning and preprocessing
def clean_seat_type(seat_type):
    categories_to_modify = ['OPEN', 'OBC', 'NT1', 'NT2', 'NT3', 'VJ', 'SC', 'ST']
    if len(seat_type) > 2 and seat_type[1:-1] in categories_to_modify:
        return seat_type[1:-1]
    return seat_type

df['MODIFIED SEAT TYPE'] = df['SEAT TYPE'].apply(clean_seat_type)

seat_type_options = ['All'] + df['MODIFIED SEAT TYPE'].unique().tolist()
branch_options = ['All'] + df['BRANCH NAME'].unique().tolist()
district_options = ['All'] + df['DISTRICT'].unique().tolist()

def search_colleges(rank, seat_types, branches, districts, range_=300):
    if seat_types and 'All' not in seat_types:
        df_filtered = df[df['MODIFIED SEAT TYPE'].isin(seat_types)]
    else:
        df_filtered = df
    
    if branches and 'All' not in branches:
        df_filtered = df_filtered[df_filtered['BRANCH NAME'].isin(branches)]
    
    if districts and 'All' not in districts:
        df_filtered = df_filtered[df_filtered['DISTRICT'].isin(districts)]
    
    result_df = df_filtered[(df_filtered['CUT OFF (RANK)'] >= rank - range_) & (df_filtered['CUT OFF (RANK)'] <= rank + range_)]
    
    if len(result_df) < 10:
        result_df = df_filtered.head(10)  # Ensure at least 10 results
    
    return result_df

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/cet')
def cet():
    return render_template('index.html', seat_type_options=seat_type_options, branch_options=branch_options, district_options=district_options)

@app.route('/search', methods=['POST'])
def search():
    rank = int(request.form['rank'])
    seat_types = request.form.getlist('seat_types')
    branches = request.form.getlist('branches')
    districts = request.form.getlist('districts')
    
    result_df = search_colleges(rank, seat_types, branches, districts)
    
    # Clean the DataFrame by removing newlines
    result_df = result_df.applymap(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    
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