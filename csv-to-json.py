import pandas as pd
import json
import ast

vectorized = pd.read_csv('data/vectorized-data.csv')
indices = vectorized.index

temp = {}
for i in indices:
    temp[i] = list(map(ast.literal_eval, vectorized.iloc[i].to_list()))

# dict to json
with open('data/vectorized-data.json', 'w') as f:
    json.dump(temp, f)
