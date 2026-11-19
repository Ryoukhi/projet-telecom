"""Génère une base CSV réduite, prête à charger dans l'application Streamlit."""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "telecom_credit_scoring_nigeria.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "serving" / "clients.csv"

CLIENT_COLUMNS = [
    "customer_id",
    "operator",
    "credit_score",
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "max_loan_amount_ngn",
    "account_type",
]
NUMERIC_COLUMNS = [
    "credit_score",
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "max_loan_amount_ngn",
]


def read_source(path: Path) -> pd.DataFrame:
    """Charge un CSV ou un classeur Excel."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
            raise RuntimeError(
                f"La lecture de {suffix} nécessite le paquet {engine}. "
                f"Installez-le avec : python -m pip install {engine}"
            ) from exc
    raise ValueError("Le fichier source doit être au format CSV, XLSX ou XLS.")


def generate_database(source: Path, output: Path, limit: int | None = None) -> tuple[int, int]:
    """Nettoie les clients et écrit uniquement les colonnes attendues par l'app."""
    if not source.is_file():
        raise FileNotFoundError(f"Fichier source introuvable : {source}")

    clients = read_source(source)
    missing = sorted(set(CLIENT_COLUMNS) - set(clients.columns))
    if missing:
        raise ValueError("Colonnes requises absentes : " + ", ".join(missing))

    # La sélection explicite exclut notamment phone_number et les champs non utilisés.
    clients = clients[CLIENT_COLUMNS].copy()
    clients["customer_id"] = clients["customer_id"].astype("string").str.strip()
    for column in ("operator", "account_type"):
        clients[column] = clients[column].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        clients[column] = pd.to_numeric(clients[column], errors="coerce")

    before = len(clients)
    clients = clients.dropna(subset=CLIENT_COLUMNS)
    clients = clients[clients["customer_id"].ne("")]
    clients = clients.drop_duplicates(subset=["customer_id"], keep="first")
    cleaned_count = len(clients)

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit doit être un entier positif.")
        clients = clients.head(limit)

    output.parent.mkdir(parents=True, exist_ok=True)
    clients.to_csv(output, index=False, encoding="utf-8-sig")
    return before - cleaned_count, len(clients)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crée une base clients CSV sans numéro de téléphone pour l'app Streamlit."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV/Excel source")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV généré")
    parser.add_argument(
        "--limit",
        type=int,
        help="Nombre maximal de clients à exporter (par défaut : tous les clients valides)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    removed, exported = generate_database(source, output, args.limit)
    print(f"Base créée : {output}")
    print(f"Clients exportés : {exported:,}")
    print(f"Lignes supprimées (valeurs manquantes ou doublons) : {removed:,}")
    print("Le fichier contient uniquement les colonnes attendues et aucun numéro de téléphone.")


if __name__ == "__main__":
    main()
