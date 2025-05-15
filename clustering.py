import numpy as np
from sklearn.cluster import KMeans
from VAE import VAE
from sklearn.decomposition import PCA
import torch

def question_embedding(unique_answers, df_original, df_vectorized, chosen_question, total_respondents):
    sum_centroids = np.zeros(df_vectorized.shape[1])

    for answer in unique_answers:
        indices = df_original[df_original[chosen_question] == answer].index
        subset = df_vectorized.iloc[indices]
        other_questions = [col for col in df_vectorized.columns if col != chosen_question]
        X = np.hstack([np.vstack(subset[q]) for q in other_questions])
        n_clusters = len(unique_answers)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        sum_centroids += np.sum(np.sum(centroids, axis=0).reshape(104, -1), axis=1) * (len(indices) / total_respondents)

    return sum_centroids

def question_embedding_pca(unique_answers, df_original, df_vectorized, chosen_question, total_respondents):
    sum_centroids = np.zeros(df_vectorized.shape[1])

    for answer in unique_answers:
        indices = df_original[df_original[chosen_question] == answer].index
        subset = df_vectorized.loc[indices]
        other_questions = [col for col in df_vectorized.columns if col != chosen_question]
        X = np.hstack([np.vstack(subset[q]) for q in other_questions])
        pca = PCA(n_components=30)
        X_reduced = pca.fit_transform(X)
        n_clusters = len(df_original[chosen_question].unique())
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_reduced)
        centroids = kmeans.cluster_centers_
        sum_centroids += np.sum(centroids, axis=0) * (len(indices) / total_respondents)

    return sum_centroids

def question_embedding_vae(unique_answers, df_original, df_vectorized, chosen_question, total_respondents):
    sum_centroids = np.zeros(df_vectorized.shape[1])
    for answer in unique_answers:
        indices = df_original[df_original[chosen_question] == answer].index
        subset = df_vectorized.loc[indices]
        other_questions = [col for col in df_vectorized.columns if col != chosen_question]
        X = np.hstack([np.vstack(subset[q]) for q in other_questions])
        vae = VAE()
        vae.load_state_dict(torch.load('vae_best.pth'))
        vae.eval()
        X_reduced = vae.encode(X)
        n_clusters = len(df_original[chosen_question].unique())
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_reduced)
        centroids = kmeans.cluster_centers_
        sum_centroids += np.sum(centroids, axis=0) * (len(indices) / total_respondents)

    return sum_centroids