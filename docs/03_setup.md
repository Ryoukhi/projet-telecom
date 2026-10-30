# Guide de setup

## Prérequis

- Git for Windows avec Git Bash.
- Python 3.10 ou 3.11. Le projet accepte `>=3.10,<3.12`.
- Accès à PyPI pour installer les dépendances.
- Docker seulement à partir du milestone serving.

## Environnement local (Git Bash)

Depuis la racine du dépôt :

```bash
which python
python --version
python -m pip --version
```

Créer un environnement neuf à côté de l'ancien pour ne pas perdre son état :

```bash
python -m venv .venv-m0
source .venv-m0/Scripts/activate
python --version
python -m pip show setuptools
```

Le virtualenv `.venv` préexistant contient `pyvenv.cfg` avec Python 3.10.11. Dans le diagnostic de ce dépôt, `.venv/Scripts/python.exe` échoue à démarrer en référençant `C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe`. Si le nouvel environnement montre la même erreur, vérifier le chemin affiché par `which python` et réinstaller/réparer Python depuis l'installateur officiel avant de recréer le venv.

## Installation par étapes

Le projet sépare maintenant les dépendances par usage afin que le setup initial n'installe pas tous les services MLOps et notebooks :

```bash
python -m pip install -e ".[dev]"
```

Cette commande installe le projet editable et les outils de qualité (pytest, Ruff, Black, pre-commit). Pip crée un environnement de build isolé et y installe setuptools et wheel, déclarés dans `pyproject.toml`. Les dépendances ML et services sont ajoutées au milestone correspondant :

```bash
python -m pip install -e ".[data]"       # DVC local (pour initialiser DVC au M0)
python -m pip install -e ".[ml]"         # pandas, NumPy, scikit-learn, XGBoost
python -m pip install -e ".[pipeline]"   # MLflow, Prefect, Evidently
python -m pip install -e ".[serving]"    # FastAPI, Uvicorn
python -m pip install -e ".[notebook]"   # Jupyter et ipykernel
```

L'extra `.[cloud]` ajoute le support S3 de DVC uniquement si ce remote est retenu. Pour un remote local, il n'est pas nécessaire. Après avoir ajouté des extras, pip peut relire les dépendances mais réutilise les paquets déjà installés.

Pour identifier une installation lente, relancer une seule fois avec `-v` et noter le dernier paquet/étape affiché :

```bash
python -m pip install -v -e ".[dev]" --timeout 30 --retries 1
```

Si pip est bloqué sur un téléchargement, examiner la connexion/proxy et le nom du paquet. S'il résout les versions, fournir la dernière ligne et la durée d'attente avant d'ajouter des contraintes. Ne pas utiliser un lockfile vide comme preuve de reproductibilité.

## Qualité locale

```bash
pre-commit install
pre-commit run --all-files
ruff check src tests
black --check src tests
```

Les tests seront ajoutés avec le code métier. Le workflow CI indique explicitement quand aucun test n'existe encore.

## DVC et données

Initialiser DVC à partir de l'extra `data` lorsqu'il est installé :

```bash
dvc init
mkdir -p ../dvc-storage
dvc remote add -d local ../dvc-storage
dvc remote list
```

Le remote de stockage de démonstration est en dehors du dépôt. Ne pas committer de secrets. Ajouter les données avec `dvc add data/raw/...` seulement après avoir confirmé la source, la licence et la définition de la cible. Il n'y a pas encore de données DVC enregistrées; `dvc pull` ne fonctionnera pas tant qu'un remote contenant les artefacts n'est pas configuré et accessible.

## Variables d'environnement

Copier `.env.example` vers `.env` pour un usage local, puis renseigner uniquement les valeurs nécessaires. `.env` est ignoré par Git. Ne jamais y mettre des secrets partagés ou les committer.

## Lockfile et reproductibilité

`requirements.lock` présent dans le dépôt est vide et non suivi par Git; il ne verrouille rien. Le produire après validation des groupes d'installation et choisir une méthode de lock/versioning commune à l'équipe. La CI installe le projet avec pip depuis `pyproject.toml`.
