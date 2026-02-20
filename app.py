import streamlit as st
from src.recommender import load_and_clean_data, train_models, recommend

# Configuration de la page
st.set_page_config(page_title="TripAdvisor Reco", layout="wide")
st.title("🏨 TripAdvisor Smart Recommender")

# Chargement des données (avec cache pour éviter de recharger à chaque clic)
@st.cache_resource
def init_app():
    df = load_and_clean_data('data/reviews83325.csv', 'data/Tripadvisor.csv')
    bm25, w2v, vectors = train_models(df)
    return df, bm25, vectors

df, bm25, vectors = init_app()

# Barre latérale pour la recherche
st.sidebar.header("Paramètres")
place_list = df['nom'].tolist()
selected_place = st.sidebar.selectbox("Choisissez un lieu :", place_list)

# Récupération de l'index
query_idx = df[df['nom'] == selected_place].index[0]

# Bouton pour lancer la recommandation
if st.button("Trouver des lieux similaires"):
    reco_bm25, reco_w2v = recommend(query_idx, df, bm25, vectors)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Recommandations BM25")
        st.write("*(Basé sur les mots-clés précis)*")
        st.table(reco_bm25[['nom', 'typeR']])
        
    with col2:
        st.subheader("🧠 Recommandations Word2Vec")
        st.write("*(Basé sur l'ambiance sémantique)*")
        st.table(reco_w2v[['nom', 'typeR']])