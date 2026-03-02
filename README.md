# 🌍 TripAdvisor Recommendation System

Ce projet implante un moteur de recommandation hybride pour les établissements TripAdvisor (hôtels, restaurants, attractions) en utilisant des techniques de Traitement du Langage Naturel (NLP) et de Recherche d'Information (IR). L'intelligence du système repose sur l'analyse sémantique des avis laissés par les utilisateurs.

---

## 🚀 Fonctionnalités

* **Nettoyage Intelligent** : Filtrage par langue (anglais) et suppression des avis non significatifs via la méthode des quartiles (IQR).
* **Extraction de Mots-Clés** : Utilisation de **TF-IDF** pour condenser des milliers d'avis en 100 mots-clés ultra-pertinents par lieu.
* **Moteur BM25** : Algorithme de ranking probabiliste pour une recherche par mots-clés ultra-précise.
* **Plongements Lexicaux (Word2Vec)** : Recommandation basée sur la similarité cosinus pour capturer "l'ambiance" d'un lieu (ex: chercher "romantique" sans que le mot soit forcément dans la description).
* **Évaluation à Double Niveau** : Mesure de la performance par *Ranking Error* (correspondance de type et de métadonnées).

---

## 📊 Méthodologie

### 1. Préparation & Agrégation

Les avis sont regroupés par identifiant de lieu. Pour éviter les bruits statistiques, nous appliquons un filtrage sur la distribution du nombre de mots : $$\text{Seuil} = Q_3 + 1.5 \times \text{IQR}$$


### 2. Algorithmes de Recommandation

Le projet compare deux approches majeures :

* **BM25 (Best Matching 25)** : Idéal pour trouver des lieux partageant des caractéristiques textuelles spécifiques.
* **Word2Vec** : Transforme chaque lieu en un vecteur dans un espace de dimension 100. La similarité est calculée par :
$$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$



### 3. Évaluation

Le système est testé sur sa capacité à classer en premier des lieux similaires :

* **Level 1** : Même type (Hôtel, Restaurant, etc.).
* **Level 2** : Mêmes étiquettes (Cuisine italienne, Spa, Luxe, etc.).

---

## 🛠 Installation et Données

1. **Cloner le projet** (le projet utilise **Git LFS** pour les datasets) :
   ```bash
   git clone [https://github.com/votre-username/tripadvisor-recommender.git](https://github.com/votre-username/tripadvisor-recommender.git)
   cd tripadvisor-recommender

---

## 🖥 Utilisation

Ouvrez le notebook `dev.ipynb` ou importez les fonctions de recommandation :

```python
# Exemple de recommandation par lieu similaire avec les 2 methodes
reco_bm25, reco_w2v = recommend(idx, df, bm25, vectors)

```
Vous pouvez aussi utiliser la commande "streamlit run app.py" pour avoir le projet sur votre navigateur en local, avec une application streamlit pour trouver une surprise ! 
---

## 📈 Résultats d'Évaluation

Les tests sur 200 requêtes montrent une excellente convergence :

* **Mode de l'erreur** : 0 (le système place très souvent un lieu pertinent en première position).
* **Distribution** : Une forte concentration des résultats pertinents dans le Top 3 des recommandations.

---
