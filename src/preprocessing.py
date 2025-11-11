import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocessing for TF-IDF and Word2Vec"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def clean_for_transformer(text):
    """Preprocessing for SentenceTransformer"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!:?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_corpus(df):
    """Add preprocessed columns"""
    tqdm.pandas(desc="Preprocessing text")
    df['cleaned_text'] = (df['title'] + " " + df['description']).progress_apply(preprocess_text)
    df['transformer_text'] = (df['title'] + " : " + df['description']).progress_apply(clean_for_transformer)
    return df
