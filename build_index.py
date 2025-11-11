# ===================== Imports =====================
from src.data_loader import load_data
from src.preprocessing import prepare_corpus
from src.vectorization import vectorize_tfidf, train_word2vec, get_w2v_vectors, get_sentence_embeddings
from src.search import build_nn_index
import os

# ===================== 1. Load & Clean Data =====================
print("Loading and cleaning data...")
df = load_data(r"datasets\book_data.csv")

# ===================== 2. Preprocess Text =====================
print("Preprocessing text...")
df = prepare_corpus(df)

# Save the processed DataFrame for later use
os.makedirs("!datasets/processed", exist_ok=True)
df.to_pickle("!datasets/processed/books_prepared.pkl")

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

# ===================== 3. TF-IDF =====================
tfidf_path = 'models/tfidf_vectorizer.pkl'
print("Vectorizing TF-IDF...")
tfidf_vec, tfidf_matrix = vectorize_tfidf(
    df['cleaned_text'], 
    max_features=5000, 
    save_path=tfidf_path
)

# ===================== 4. Word2Vec =====================
w2v_model_path = 'models/w2v_model'
w2v_vectors_path = 'models/w2v_vectors.npy'

tokenized_texts = [doc.split() for doc in df['cleaned_text']]

print("Training / Loading Word2Vec model...")
w2v_model = train_word2vec(tokenized_texts, save_path=w2v_model_path)
w2v_vectors = get_w2v_vectors(w2v_model, tokenized_texts, save_path=w2v_vectors_path)

# Build NearestNeighbors index
nn_w2v = build_nn_index(w2v_vectors)

# ===================== 5. Sentence Transformers =====================
st_embeddings_path = 'models/sentence_embeddings.npy'

print("Generating / Loading Sentence Transformer embeddings...")
st_model, st_embeddings = get_sentence_embeddings(
    df['transformer_text'], 
    model_name='all-MiniLM-L6-v2', 
    save_path=st_embeddings_path
)

print("\n✅ All models and embeddings have been created and saved successfully!")
