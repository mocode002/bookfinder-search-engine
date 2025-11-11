import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.search import search_nn, semantic_search
from src.result import show_search_results
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# Load preprocessed dataset
print("Loading preprocessed corpus...")
df = pd.read_pickle("datasets/processed/books_prepared.pkl")

# ===================== Load Models =====================
print("Loading TF-IDF vectorizer...")
tfidf_vec, _ = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_matrix = tfidf_vec.transform(df['cleaned_text'])

print("Loading Word2Vec model and vectors...")
w2v_model = Word2Vec.load("models/w2v_model")
w2v_vectors = np.load("models/w2v_vectors.npy")
nn_w2v = NearestNeighbors(metric='cosine', algorithm='brute')
nn_w2v.fit(w2v_vectors)

print("Loading SentenceTransformer model and embeddings...")
st_model = SentenceTransformer('all-MiniLM-L6-v2')
st_embeddings = np.load("models/sentence_embeddings.npy")

# ===================== Search Functions =====================
def search_tfidf(query, vectorizer, tfidf_matrix, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]

# ===================== Run Searches =====================
query = "girl who steals books during the world war"
print(f"\n🔍 Query: {query}\n")

# ---- TF-IDF ----
print("Top results with TF-IDF:")
tfidf_indices, tfidf_similarities = search_tfidf(query, tfidf_vec, tfidf_matrix, top_k=3)
show_search_results(tfidf_indices, tfidf_similarities, df)

# ---- Word2Vec ----
print("Top results with Word2Vec:")
query_tokens = query.split()
query_vec = np.array(
    [w2v_model.wv[t] for t in query_tokens if t in w2v_model.wv.key_to_index]
).mean(axis=0).reshape(1, -1)
w2v_indices, w2v_similarities = search_nn(query_vec, nn_w2v, top_k=3)
show_search_results(w2v_indices, w2v_similarities, df)

# ---- SentenceTransformer ----
print("Top results with SentenceTransformer:")
st_indices, st_similarities = semantic_search(query, st_model, st_embeddings, top_k=3)
show_search_results(st_indices, st_similarities, df)

print("\n✅ Search completed!")
