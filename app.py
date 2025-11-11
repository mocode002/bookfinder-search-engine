import gradio as gr
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# ===================== 1. Load Dataset =====================
print("Loading dataset...")
books_df = pd.read_pickle("datasets/processed/books_prepared.pkl")

# ===================== Load Models =====================
print("Loading TF-IDF vectorizer...")
tfidf_vec, _ = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_matrix = tfidf_vec.transform(books_df['cleaned_text'])

print("Loading Word2Vec model and vectors...")
w2v_model = Word2Vec.load("models/w2v_model")
w2v_vectors = np.load("models/w2v_vectors.npy")
nn_w2v = NearestNeighbors(metric='cosine', algorithm='brute')
nn_w2v.fit(w2v_vectors)

print("Loading SentenceTransformer model and embeddings...")
st_model = SentenceTransformer('all-MiniLM-L6-v2')
st_embeddings = np.load("models/sentence_embeddings.npy")



# ===================== 3. Search Functions =====================
def search_tfidf(query, vectorizer, tfidf_matrix, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]


def search_w2v(query, model, nn, top_k=5):
    """Word2Vec semantic search using nearest neighbors."""
    tokens = query.split()
    vectors = [model.wv[t] for t in tokens if t in model.wv.key_to_index]
    if not vectors:
        return [], []
    query_vec = np.mean(vectors, axis=0).reshape(1, -1)
    distances, indices = nn.kneighbors(query_vec, n_neighbors=top_k)
    return indices[0], 1 - distances[0]


def semantic_search(query, model, embeddings, top_k=5):
    query_vec = model.encode([query])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices, similarities[top_indices]


def search_books(query, mode="tfidf", top_k=5):
    """Unified interface for all modes."""
    if not query.strip():
        return [], []

    if mode == "tfidf":
        return search_tfidf(query, tfidf_vec, tfidf_matrix, top_k)
    elif mode == "w2v":
        return search_w2v(query, w2v_model, nn_w2v, top_k)
    elif mode == "sentence":
        return semantic_search(query, st_model, st_embeddings, top_k)
    else:
        raise ValueError("Invalid mode. Choose from tfidf, w2v, or sentence.")


# ===================== 4. Format Results =====================
def format_results_html(top_indices, top_similarities, books_df):
    """
    Generate HTML to display search results with images and info.
    """
    if len(top_indices) == 0:
        return "<p style='color:red;'>No results found.</p>"

    html_output = ""
    for rank, i in enumerate(top_indices):
        row = books_df.iloc[i]
        img_url = row.get("image_url")
        if img_url and isinstance(img_url, str) and img_url.startswith("http"):
            img_html = f'<img src="{img_url}" width="120" style="border-radius:10px; margin-right:15px;">'
        else:
            img_html = '<div style="width:120px;height:180px;background:#ccc;border-radius:10px;display:inline-block;margin-right:15px;"></div>'
        title = f"<b>{row.get('title', 'Unknown Title')}</b>"
        score = f"<i>Similarity:</i> {top_similarities[rank]:.3f}"
        genres = ", ".join(row['genres']) if isinstance(row.get('genres'), list) else row.get('genres', 'N/A')
        description = row.get('description', '')
        description = (description[:250] + "...") if isinstance(description, str) and len(description) > 0 else "No description available."
        html_output += f"""
        <div style="display:flex;align-items:flex-start;margin-bottom:20px;">
            {img_html}
            <div>
                <div style="font-size:16px;">{title}</div>
                <div style="color:gray;font-size:14px;">{score}</div>
                <div style="color:#555;font-size:13px;margin-top:5px;"><b>Genres:</b> {genres}</div>
                <div style="margin-top:8px;font-size:13px;">{description}</div>
            </div>
        </div>
        <hr style="border:0;border-top:1px solid #eee;margin:10px 0;">
        """
    return html_output


# ===================== 5. Gradio Function =====================
def gradio_search(query, mode, top_k):
    top_indices, top_similarities = search_books(query, mode=mode, top_k=int(top_k))
    html = format_results_html(top_indices, top_similarities, books_df)
    return html


# ===================== 6. Gradio UI =====================
with gr.Blocks() as demo:
    gr.Markdown("<h2>📚 Book Search Engine</h2><p>Search your favorite books using different embedding models.</p>")

    with gr.Row():
        query_input = gr.Textbox(label="Enter your search query", placeholder="e.g., girl who steals books during the world war")
        mode_input = gr.Dropdown(label="Search Mode", choices=["tfidf", "w2v", "sentence"], value="tfidf")
        top_k_input = gr.Slider(label="Number of results", minimum=1, maximum=20, step=1, value=6)

    search_button = gr.Button("🔍 Search")
    results_output = gr.HTML()

    search_button.click(gradio_search, inputs=[query_input, mode_input, top_k_input], outputs=results_output)

demo.launch(share=True)
