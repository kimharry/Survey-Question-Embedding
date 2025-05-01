import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the CSV file
data = pd.read_csv('data/na-removed-data.csv')

vectorized = pd.DataFrame()

# Load the model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Get the embeddings per column
for column in data.columns:
    vec = model.encode(data[column].apply(lambda x: str(x) if isinstance(x, float) else x).to_list())
    print(f"Embedding shape for {column}: {vec.shape}")
    
    # Convert each vector to a list and add it as a new column
    vectorized[column] = vec.tolist()

vectorized.to_csv('data/vectorized-data.csv', index=False)
