import pandas as pd

# Load the CSV file
data = pd.read_csv('data/raw-data.csv')

# drop the first column
data = data.drop(data.columns[0], axis=1)

# drop columns related to stackoverflow
data = data.drop(columns=['NEWSOSites', 'SOVisitFreq', 'SOAccount', 'SOPartFreq', 'SOHow', 'SOComm'])

# drop the rows if half of the answers are N/A
for i, row in data.iterrows():
    if row.isna().sum() > len(row) / 20:
        data = data.drop(i, axis=0)

# drop the columns if half of the answers are N/A
for column in data.columns:
    if data[column].isna().sum() > len(data) / 2:
        data = data.drop(column, axis=1)

data.to_csv('data/na-removed-data.csv', index=False)