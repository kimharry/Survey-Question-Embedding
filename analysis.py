import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from clustering import *

# 데이터 로드
df_original = pd.read_csv('data/na-removed-data.csv')
df_vectorized = pd.read_json('data/vectorized-data.json', orient='index')

# print(df_original.head())
# print(df_original.info())
# print(df_vectorized.head())
# print(df_vectorized.info())


q1 = 'AIThreat'
q2 = 'AIEthics'

unique_answers_q1 = df_original[q1].unique()
unique_answers_q2 = df_original[q2].unique()

q1_embedding = question_embedding(unique_answers_q1, df_original, df_vectorized, q1, len(df_original))
q2_embedding = question_embedding(unique_answers_q2, df_original, df_vectorized, q2, len(df_original))

# Cosine similarity
similarity = cosine_similarity([q1_embedding], [q2_embedding])
print(f"Cosine similarity between {q1} and {q2}: {similarity}")

# Euclidean distance
distance = np.linalg.norm(q1_embedding - q2_embedding)
print(f"Euclidean distance between {q1} and {q2}: {distance}")

q1_embedding_pca = question_embedding_pca(unique_answers_q1, df_original, df_vectorized, q1, len(df_original))
q2_embedding_pca = question_embedding_pca(unique_answers_q2, df_original, df_vectorized, q2, len(df_original))

# Cosine similarity
similarity_pca = cosine_similarity([q1_embedding_pca], [q2_embedding_pca])
print(f"Cosine similarity between {q1} and {q2} (PCA): {similarity_pca}")

# Euclidean distance
distance_pca = np.linalg.norm(q1_embedding_pca - q2_embedding_pca)
print(f"Euclidean distance between {q1} and {q2} (PCA): {distance_pca}")

q1_embedding_vae = question_embedding_vae(unique_answers_q1, df_original, df_vectorized, q1, len(df_original))
q2_embedding_vae = question_embedding_vae(unique_answers_q2, df_original, df_vectorized, q2, len(df_original))

# Cosine similarity
similarity_vae = cosine_similarity([q1_embedding_vae], [q2_embedding_vae])
print(f"Cosine similarity between {q1} and {q2} (VAE): {similarity_vae}")

# Euclidean distance
distance_vae = np.linalg.norm(q1_embedding_vae - q2_embedding_vae)
print(f"Euclidean distance between {q1} and {q2} (VAE): {distance_vae}")