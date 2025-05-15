import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sent1 = "Do you believe AI is a threat to your current job?"
sent2 = "Which AI ethical responsibilities are most important to you?  Select all that apply."

model = SentenceTransformer('all-MiniLM-L12-v2')

vec1 = model.encode(sent1)
vec2 = model.encode(sent2)

# cosine similarity and euclidean distance
print("Cosine similarity:", cosine_similarity([vec1], [vec2]))
print("Euclidean distance:", np.linalg.norm(vec1 - vec2))
