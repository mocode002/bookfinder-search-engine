# 📚 Bookfinder Search Engine

This project implements a **book search engine** that allows users to query a large collection of books and retrieve relevant results using multiple search strategies: **TF-IDF**, **Word2Vec**, and **Sentence Transformers**. The project includes preprocessing, vectorization, similarity search, and both CLI and web-based interfaces.

---

## 🔹 Features

* Load and preprocess a large dataset of books with metadata and descriptions.
* Text cleaning and tokenization (lemmatization, stopword removal, punctuation cleanup).
* Three search modes:
  
  1. **TF-IDF** : traditional keyword-based search.
  2. **Word2Vec** : semantic search using averaged word embeddings.
  3. **Sentence Transformers** : advanced semantic search using pre-trained sentence embeddings.
* Caching of vectorizers, embeddings, and models to speed up repeated runs.
* CLI-based search with text output.
* Interactive web interface with **Gradio** showing book images, title, genres, description, and similarity scores.

---

## 🗂 Project Structure

```
bookfinder/
│
├── datasets/
│   ├── book_data.csv/            # Raw data
│   └── processed/
│       └── books_prepared.pkl    # Preprocessed and cleaned dataset
│
├── models/
│   ├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
│   ├── w2v_model                 # Saved Word2Vec model
│   ├── w2v_vectors.npy           # Word2Vec document vectors
│   └── sentence_embeddings.npy   # Sentence Transformer embeddings
│
├── notebooks/
│   └── base.py                   # Exploration notebook
│
├── src/
│   ├── data_loader.py            # Loading raw dataset
│   ├── preprocessing.py          # Text cleaning, preprocessing
│   ├── vectorization.py          # TF-IDF, Word2Vec, Sentence Transformer
│   ├── search.py                 # Search algorithms
│   └── result.py                 # CLI-friendly display functions
│
├── build_index.py                # Build and save all embeddings/models
├── run_search.py                 # CLI search interface
├── app.py                        # Gradio web interface
└── README.md                     # Project documentation
```

---

## ⚡ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/mocode002/bookfinder-search-engine.git
cd bookfinder
```

2. **Install dependencies:**

```bash
Later
```

3. **Prepare data:**

* Place your dataset CSV (`book_data.csv`) inside `datasets/`.
* The CSV should have at least `title`, `description`, `genres`, `image_url` columns.

1. **Build models and embeddings (first run only):**

```bash
uv run build_index.py
```

This will:

* Preprocess the text.
* Train TF-IDF vectorizer, Word2Vec model, and Sentence Transformer embeddings.
* Save all models and vectors in the `models/` folder.

---

## 🖥 Running the Project

### 1. Command-line interface

```bash
uv run run_search.py
```

* Prompts are displayed in the terminal.
* Outputs the **top results** with title, genres, description, and similarity score.

### 2. Gradio web interface

```bash
uv run app.py
```

* Opens a local web page.
* Allows typing a query, selecting search mode (TF-IDF / Word2Vec / Sentence Transformer), and choosing `top_k`.
* Shows results with images, title, genres, description, and similarity.

---

## 🔍 Search Modes

| Mode         | Description                                                       |
| ------------ | ----------------------------------------------------------------- |
| **tfidf**    | Classic keyword-based search using TF-IDF cosine similarity.      |
| **w2v**      | Semantic search using Word2Vec embeddings.                        |
| **sentence** | Advanced semantic search using pre-trained Sentence Transformers. |



---

## ✅ Notes

* **Caching:** The first run may take a while to build models and embeddings. Subsequent runs will load from disk.
* **Extensibility:** You can add more search modes, or integrate with additional datasets.
* **Web UI:** Gradio provides an interactive demo suitable for presentations or testing queries.

---

## 📚 Example Queries

* `"boy raised in the jungle"`
* `"girl who steals books during the world war"`
* `"detective solves a murder mystery"`

Results will display **top k books** with similarity scores, descriptions, and genres.