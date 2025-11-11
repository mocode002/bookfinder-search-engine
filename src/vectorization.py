import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# ---------------- TF-IDF ----------------
def vectorize_tfidf(corpus, max_features=5000, save_path=None):
    if save_path and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            vectorizer, tfidf_matrix = pickle.load(f)
        return vectorizer, tfidf_matrix
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((vectorizer, tfidf_matrix), f)
    
    return vectorizer, tfidf_matrix

# ---------------- Word2Vec ----------------
def train_word2vec(tokenized_texts, vector_size=100, window=5, min_count=2, save_path=None):
    if save_path and os.path.exists(save_path):
        w2v_model = Word2Vec.load(save_path)
        return w2v_model
    
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size,
                         window=window, min_count=min_count, workers=4, sg=1)
    
    if save_path:
        w2v_model.save(save_path)
    return w2v_model

def document_vector(w2v_model, doc_tokens):
    words = [w for w in doc_tokens if w in w2v_model.wv.key_to_index]
    if len(words) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(w2v_model.wv[words], axis=0)

def get_w2v_vectors(w2v_model, tokenized_texts, save_path=None):
    if save_path and os.path.exists(save_path):
        return np.load(save_path)
    
    vectors = np.array([document_vector(w2v_model, doc) for doc in tokenized_texts])
    if save_path:
        np.save(save_path, vectors)
    return vectors

# ---------------- SentenceTransformer ----------------
def get_sentence_embeddings(corpus, model_name='all-MiniLM-L6-v2', save_path=None):
    if save_path and os.path.exists(save_path):
        embeddings = np.load(save_path)
        model = SentenceTransformer(model_name)
        return model, embeddings
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus, show_progress_bar=True)
    
    if save_path:
        np.save(save_path, embeddings)
    
    return model, embeddings
