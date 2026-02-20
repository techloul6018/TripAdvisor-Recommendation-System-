import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def load_and_clean_data(reviews_path, places_path):
    """Charge, filtre les langues et nettoie les outliers."""
    df_reviews = pd.read_csv(reviews_path)
    df_places = pd.read_csv(places_path)
    
    # Nettoyage
    df_reviews = df_reviews[df_reviews['langue'] == 'en'].dropna(subset=['review'])
    
    df_merged = pd.merge(
        df_reviews[['idplace', 'review']], 
        df_places[['id', 'nom', 'typeR']], 
        left_on='idplace', right_on='id'
    )
    
    # Filtrage IQR
    df_merged['word_count'] = df_merged['review'].apply(lambda x: len(str(x).split()))
    Q1, Q3 = df_merged['word_count'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_cleaned = df_merged[(df_merged['word_count'] >= 10) & 
                           (df_merged['word_count'] <= Q3 + 1.5 * IQR)].copy()
    
    # Agrégation
    df_grouped = df_cleaned.groupby('id').agg({
        'nom': 'first', 'typeR': 'first', 'review': lambda x: " ".join(list(x))
    }).reset_index()
    
    return df_grouped

def train_models(df_grouped):
    """Prépare BM25 et Word2Vec."""
    # BM25
    corpus = [doc.split() for doc in df_grouped['review']]
    bm25 = BM25Okapi(corpus)
    
    # Word2Vec
    sentences = corpus
    model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    
    # Pré-calcul des vecteurs de lieux
    place_vectors = np.array([_get_vec(text, model_w2v) for text in df_grouped['review']])
    
    return bm25, model_w2v, place_vectors

def _get_vec(text, model):
    """Fonction interne pour vectoriser un texte."""
    words = text.split()
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

def recommend(query_index, df_grouped, bm25, place_vectors, n=5):
    """Retourne les recos BM25 et W2V pour comparaison."""
    # Logic BM25
    query_tokens = df_grouped.iloc[query_index]['review'].split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_bm25 = bm25_scores.argsort()[::-1][1:n+1]
    
    # Logic W2V
    query_vec = place_vectors[query_index].reshape(1, -1)
    sims = cosine_similarity(query_vec, place_vectors).flatten()
    top_w2v = sims.argsort()[::-1][1:n+1]
    
    return df_grouped.iloc[top_bm25], df_grouped.iloc[top_w2v]