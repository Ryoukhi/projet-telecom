# MLOps — Prédiction du risque d'impayé télécom

Projet de bout en bout pour étudier et industrialiser la prédiction de `default_risk` à partir du jeu Telecom Credit Scoring Data (Nigeria). Le périmètre et les livrables sont détaillés dans le [dossier de cadrage](docs/00_roadmap.md).

## Documentation

- [Feuille de route et critères de sortie des milestones](docs/00_roadmap.md)
- [Contexte métier et hypothèses](docs/01_context.md)
- [Fiche courte du dataset](docs/dataset.md)
- [Architecture cible](docs/02_architecture.md)
- [Installation sous Windows / Git Bash](docs/03_setup.md)

## Environnement de développement

Python 3.10 ou 3.11 est accepté par le projet. Sous Git Bash, depuis la racine :

```bash
python -m venv .venv-m0
source .venv-m0/Scripts/activate
python -m pip install -e ".[dev]"
```

Le groupe `dev` installe les outils de qualité sans télécharger d'emblée les bibliothèques ML et services MLOps. Ajouter les extras au début de chaque milestone : `.[data]` (DVC local), `.[ml]`, `.[pipeline]`, `.[serving]`, `.[notebook]`. `.[cloud]` n'est requis que si le remote DVC choisi utilise S3.

Ne pas lancer `dvc pull` tant qu'un remote contenant les données n'a pas été configuré (voir [guide de setup](docs/03_setup.md)). Les secrets et fichiers de données ne doivent pas être commités.

## Vérifications

```bash
python --version
python -m pip --version
ruff check src tests
black --check src tests
```

La suite de tests sera exécutée dès que les tests du projet seront ajoutés. Les données, notebooks, rapports, modèles et instructions de lancement seront documentés au fil des milestones.

## Déploiement Streamlit Community Cloud

L'application Streamlit charge directement le modèle `models/model.pkl`; l'API FastAPI et Docker ne sont pas nécessaires pour cette démo. L'utilisateur peut charger sa base client au format CSV ou Excel (`.xlsx`/`.xls`), rechercher un identifiant, puis sélectionner le client pour afficher automatiquement son profil et sa prédiction. En local, le fichier `data/raw/telecom_credit_scoring_nigeria.csv` est chargé par défaut s'il existe.

1. Vérifier que `models/model.pkl` existe. Pour le régénérer, lancer `python -m src.training.export_best_model` dans l'environnement du projet.
2. Ajouter le fichier du modèle au dépôt Git : il est nécessaire au démarrage de l'application. La base client n'est pas intégrée à l'application cloud : elle est chargée par l'utilisateur au format CSV ou Excel. Ne pas ajouter `.env`, les données brutes ou des secrets.
3. Pousser le dépôt sur GitHub.
4. Dans [Streamlit Community Cloud](https://share.streamlit.io/), créer une application en sélectionnant le dépôt, la branche et `src/serving/streamlit_app.py` comme fichier d'entrée.
5. Choisir Python 3.11 dans les paramètres avancés. Les dépendances sont lues depuis `requirements.txt`.

L'application n'utilise pas de secret Streamlit. Le fichier `models/model.pkl` fait environ 1,4 Mo dans l'environnement de développement actuel.
