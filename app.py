import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender import load_and_clean_data, train_models, recommend, _get_vec

# Configuration de la page
st.set_page_config(page_title="TripAdvisor Reco", layout="wide", page_icon="🌍")
st.title("🏨 TripAdvisor Smart Recommender")

# Chargement des données et des modèles (mis en cache)
@st.cache_resource
def init_app():
    # Remplace par tes vrais chemins de fichiers
    df = load_and_clean_data('data/reviews83325.csv', 'data/Tripadvisor.csv')
    bm25, w2v, vectors = train_models(df)
    return df, bm25, w2v, vectors

df, bm25, w2v_model, place_vectors = init_app()

# --- INTERFACE ---

tab1, tab2 = st.tabs(["🔍 Recherche par mots-clés", "📍 Similitude par lieu"])

# Onglet 1 : Recherche sémantique (Word2Vec)
with tab1:
    st.subheader("Quelle ambiance recherchez-vous ?")
    user_query = st.text_input("Exemple : 'luxury hotel with spa', 'authentic italian pasta', 'quiet romantic place'", "")
    
    n_results = st.slider("Nombre de résultats", 3, 15, 5, key="slider_query")
    
    if st.button("Rechercher", key="btn_query"):
        if user_query:
            # Vectorisation de la requête utilisateur
            query_vec = _get_vec(user_query, w2v_model).reshape(1, -1)
            
            # Calcul de la similarité cosinus
            similarities = cosine_similarity(query_vec, place_vectors).flatten()
            top_indices = similarities.argsort()[::-1][:n_results]
            
            results = df.iloc[top_indices]
            
            st.success(f"Voici les lieux les plus proches de : '{user_query}'")
            st.table(results[['nom', 'typeR']])
        else:
            st.warning("Veuillez entrer des mots-clés.")

# Onglet 2 : Recherche par lieu (ton code précédent)
with tab2:
    st.subheader("Trouver des lieux similaires à un établissement")
    place_list = df['nom'].tolist()
    selected_place = st.selectbox("Choisissez un établissement :", place_list)
    
    n_reco = st.slider("Nombre de recommandations", 3, 8, 3, key="slider_place")
    
    if st.button("Générer des recommandations", key="btn_place"):
        query_idx = df[df['nom'] == selected_place].index[0]
        reco_bm25, reco_w2v = recommend(query_idx, df, bm25, place_vectors, n=n_reco)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Précision (BM25)")
            st.dataframe(reco_bm25[['nom', 'typeR']])
        with col2:
            st.subheader("🧠 Ambiance (Word2Vec)")
            st.dataframe(reco_w2v[['nom', 'typeR']])