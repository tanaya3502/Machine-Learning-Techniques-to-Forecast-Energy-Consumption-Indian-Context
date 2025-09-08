import pandas as pd

# Load the dataset
df = pd.read_csv('C:/Users/Shree/Desktop/dataset_tk.csv')

# Step 1: Remove the time part from the first column
# Assuming the first column is unnamed, otherwise use df['ColumnName']
df.iloc[:, 0] = df.iloc[:, 0].str.split(' ').str[0]  # Keeps only the date part

# Step 2: Save the cleaned DataFrame to a new CSV file
output_file_path = 'C:/Users/Shree/Desktop/cleaned_dataset.csv'
df.to_csv(output_file_path, index=False)

print(f"Cleaned dataset saved to {output_file_path}")
