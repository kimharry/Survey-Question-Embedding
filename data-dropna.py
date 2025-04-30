import pandas as pd
import sent2vec

# Load the CSV file
data = pd.read_csv('data/raw-data.csv')

# drop the first column
data = data.drop(data.columns[0], axis=1)

# drop the column if half of the answers are N/A
for column in data.columns:
    if data[column].isna().sum() > len(data) / 2:
        data = data.drop(column, axis=1)

data.to_csv('data/na-removed-data.csv', index=False)