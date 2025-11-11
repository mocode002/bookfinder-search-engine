from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

# ---------------- Word2Vec / TF-IDF with NearestNeighbors ----------------
def build_nn_index(vectors, metric='cosine', algorithm='brute'):
    nn = NearestNeighbors(metric=metric, algorithm=algorithm)
    nn.fit(vectors)
    return nn

def search_nn(query_vec, nn_model, top_k=5):
    distances, indices = nn_model.kneighbors(query_vec, n_neighbors=top_k)
    similarities = 1 - distances[0]  # convert distance → similarity
    return indices[0], similarities

# ---------------- SentenceTransformer semantic search ----------------
def semantic_search(query, model, embeddings, top_k=5):
    query_vec = model.encode([query])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]
