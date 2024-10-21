import pandas as pd

# Load the Excel file
excel_file = '/Users/vedant/Desktop/Random_Forest/JoSAA5.xlsx'

# Convert the Excel file to a DataFrame
df = pd.read_excel(excel_file)

# Save the DataFrame as a CSV file
csv_file = 'jee_file5.csv'
df.to_csv(csv_file, index=False)

print(f"File converted to CSV and saved as {csv_file}")
