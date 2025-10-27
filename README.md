# M2RS-0.1

Démonstration _expérimentale_ (disponible [ici](https://m2rs-01.streamlit.app/)) de la faisabilité d'un **moteur de recherche inversé**, appliqué à un fond de **photographies d'objets d'art**, prises pendant l'entre-deux-guerres. Toutes les technologies utilisées sont **open source** :

- [Python](https://www.python.org/) : langage de programmation de référence dans le domaine de l'intelligence artificielle, qui profite d'un vaste écosystème de bibliothèques.
- [Pandas](https://pandas.pydata.org/) : bibliothèque de référence de manipulation et d'analyse des données.
- [Numpy](https://numpy.org/) : bibliothèque de calcul scientifique.
- [PyTorch](https://pytorch.org/) : bibliothèque de machine learning optimisée pour les calculs complexes.
- [Transformers](https://huggingface.co/docs/transformers/index) : bibliothèque qui centralise les modèles d'apprentissage profonds préentraînés les plus importants. Proposée par [HuggingFace](https://huggingface.co/).
- [ChromaDB](https://www.trychroma.com/) : base de données vectorielle optimisée pour l'intelligence artificielle.
- [Streamlit](https://streamlit.io/) : bibliothèque de génération d'interfaces dédiée à l'intelligence artificielle.

## 🚀 Démarrage rapide
```shell
# Si UV n'est pas installé
curl -LsSf https://astral.sh/uv/install.sh | sh

# Cloner le dépôt
git clone https://github.com/lavantoine/m2rs-0.1.git
cd m2rs-0.1

# Installer les dépendances et initialiser l'environnement
uv sync

# Lancer l'application
source .venv/bin/activate
streamlit run app.py
```

