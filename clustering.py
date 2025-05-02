import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ast import literal_eval
from datetime import datetime

# 데이터 로드
df_original = pd.read_csv('data/na-removed-data.csv')
df_vectorized = pd.read_csv('data/vectorized-data.csv')

# 벡터 열을 리스트로 변환
for col in df_vectorized.columns:
    df_vectorized[col] = df_vectorized[col].apply(literal_eval)

chosen_question = 'AISelect'
unique_answers = df_original[chosen_question].unique()
total_respondents = len(df_original)

# PCA 버전 클러스터링 함수
def process_answer_pca(answer, df_original, df_vectorized):
    indices = df_original[df_original[chosen_question] == answer].index
    subset = df_vectorized.loc[indices]
    other_questions = [col for col in df_vectorized.columns if col != chosen_question]
    X = np.hstack([np.vstack(subset[q]) for q in other_questions])
    pca = PCA(n_components=30)
    X_reduced = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_reduced)
    centroids = kmeans.cluster_centers_
    sum_centroids = np.sum(centroids, axis=0)
    num_respondents = len(indices)
    pca2 = PCA(n_components=2)
    X_2d = pca2.fit_transform(X)
    labels = kmeans.labels_
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], label=f'cluster {i}')
    plt.legend()
    plt.title(f'PCA version: "{chosen_question}" question of "{answer}" clustering')
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'clustering_visualization_pca_{answer.replace(", ", "_").replace(" ", "_")}_{timestamp}.png'
    plt.savefig(filename)
    plt.close()
    return sum_centroids, num_respondents

# PCA 없는 버전 클러스터링 함수
def process_answer_no_pca(answer, df_original, df_vectorized):
    indices = df_original[df_original[chosen_question] == answer].index
    subset = df_vectorized.loc[indices]
    other_questions = [col for col in df_vectorized.columns if col != chosen_question]
    X = np.hstack([np.vstack(subset[q]) for q in other_questions])
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    sum_centroids = np.sum(centroids, axis=0)
    num_respondents = len(indices)
    labels = kmeans.labels_
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'cluster {i}')
    plt.legend()
    plt.title(f'No PCA version: "{chosen_question}" question of "{answer}" clustering')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'clustering_visualization_no_pca_{answer.replace(", ", "_").replace(" ", "_")}_{timestamp}.png'
    plt.savefig(filename)
    plt.close()
    return sum_centroids, num_respondents

# PCA 버전 벡터 생성
results_pca = []
for answer in unique_answers:
    result = process_answer_pca(answer, df_original, df_vectorized)
    results_pca.append(result)

question_vector_pca = np.zeros(30)
for sum_centroids, num_respondents in results_pca:
    question_vector_pca += sum_centroids * (num_respondents / total_respondents)

print(f"PCA version: Question vector of {chosen_question}: {question_vector_pca}")

# PCA 없는 버전 벡터
other_questions = [col for col in df_vectorized.columns if col != chosen_question]
X_all = np.hstack([np.vstack(df_vectorized[q]) for q in other_questions])
vector_dim = X_all.shape[1]
question_vector_no_pca = np.zeros(vector_dim)

results_no_pca = []
for answer in unique_answers:
    result = process_answer_no_pca(answer, df_original, df_vectorized)
    results_no_pca.append(result)

for sum_centroids, num_respondents in results_no_pca:
    question_vector_no_pca += sum_centroids * (num_respondents / total_respondents)
print(f"No PCA version: Question vector of {chosen_question}: {question_vector_no_pca}")

# 비교를 위한 PCA 축소
pca_global = PCA(n_components=30)
pca_global.fit(X_all)
question_vector_no_pca_reduced = pca_global.transform(question_vector_no_pca.reshape(1, -1))[0]

delta_vector = question_vector_no_pca_reduced - question_vector_pca
euclidean_distance = np.linalg.norm(delta_vector)
cos_sim = cosine_similarity([question_vector_pca], [question_vector_no_pca_reduced])[0, 0]
print(f"Euclidean distance: {euclidean_distance:.6f}")
print(f"Cosine similarity: {cos_sim:.6f}")

plt.figure(figsize=(10, 4))
x = np.arange(30)
bar_width = 0.35
plt.bar(x - bar_width/2, question_vector_pca, width=bar_width, label='PCA version')
plt.bar(x + bar_width/2, question_vector_no_pca_reduced, width=bar_width, label='No PCA version (reduced)')
plt.xlabel('Dimension')
plt.ylabel('Value')
plt.title('PCA vs Non-PCA vector comparison')
plt.legend()
plt.tight_layout()
plt.savefig('vector_comparison.png')
plt.close()

np.savetxt('question_vector_pca.txt', question_vector_pca)
np.savetxt('question_vector_no_pca.txt', question_vector_no_pca)
print("저장됨")


# Wasserstein distance
from scipy.stats import wasserstein_distance
wasserstein_dist = wasserstein_distance(question_vector_pca, question_vector_no_pca)
print(f"Wasserstein distance: {wasserstein_dist:.6f}")