import pandas as pd
from langdetect import detect, DetectorFactory
from tqdm import tqdm

DetectorFactory.seed = 0

def detect_language_safe(text):
    try:
        return detect(text)
    except:
        return "unknown"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Rename columns
    df = df.rename(columns={
        'book_authors': 'authors',
        'book_desc': 'description',
        'book_edition': 'edition',
        'book_format': 'format',
        'book_isbn': 'isbn',
        'book_pages': 'num_pages',
        'book_rating': 'avg_rating',
        'book_rating_count': 'num_ratings',
        'book_review_count': 'num_reviews',
        'book_title': 'title',
        'genres': 'genres',
        'image_url': 'image_url'
    })
    
    # Drop missing critical info
    df = df.dropna(subset=['description', 'image_url']).reset_index(drop=True)
    
    # Detect English descriptions
    tqdm.pandas(desc="Detecting language")
    df['lang'] = df['description'].progress_apply(detect_language_safe)
    df = df[df['lang'] == 'en'].reset_index(drop=True)
    
    # Clean numeric columns
    df['num_pages'] = df['num_pages'].astype(str).str.extract(r'(\d+)').astype(float)
    
    # Clean genres
    df['genres'] = df['genres'].fillna('Unknown')
    df['genres'] = df['genres'].apply(lambda x: [g.strip().lower() for g in x.split('|') if g])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)
    
    return df
